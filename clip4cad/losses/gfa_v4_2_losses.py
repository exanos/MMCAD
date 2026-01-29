"""
GFA v4.2 Loss Functions - Query Distillation + Distribution Matching

Key changes from v4:
1. DISTRIBUTION MATCHING (L_dist): Match batch statistics of Q_self to T_feat
   - Encourages Q_self to have similar mean/variance to T_feat
   - Regularizes the feature space

2. Loss weights adjusted for curriculum learning:
   - Stage 1: Heavy query distillation (lambda_query=1.5)
   - Stage 2: Balanced with hard negatives

Loss terms:
1. GUIDED: Text-guided geometry contrastive (primary)
2. SELF: Self-grounded geometry contrastive
3. QUERY: Query-level cosine distillation
4. EMBED: Embedding cosine distillation
5. DIST: Distribution matching (NEW)
6. DETAIL: Fine-grained contrastive (Stage 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class GFAv4_2Loss(nn.Module):
    """
    GFA v4.2 Loss Function with Distribution Matching.

    Works with curriculum learning - as conditioning dropout increases,
    the distribution matching helps maintain query alignment.
    """

    def __init__(
        self,
        lambda_self: float = 0.1,
        lambda_query: float = 1.5,
        lambda_embed: float = 0.3,
        lambda_dist: float = 0.3,       # NEW: Distribution matching
        lambda_detail: float = 0.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_query = lambda_query
        self.lambda_embed = lambda_embed
        self.lambda_dist = lambda_dist
        self.lambda_detail = lambda_detail
        self.label_smoothing = label_smoothing

    def infonce_3way(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        z_c: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """Standard 3-way InfoNCE loss."""
        B = z_a.shape[0]
        device = z_a.device
        labels = torch.arange(B, device=device)

        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        z_c = F.normalize(z_c, dim=-1)

        loss = 0.0
        for zi, zj in [(z_a, z_c), (z_b, z_c), (z_a, z_b)]:
            logits = (zi.float() @ zj.float().T) / tau.float()
            loss += (
                F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing) +
                F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
            ) / 2

        return loss / 3

    def infonce_2way(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """2-way InfoNCE loss."""
        B = z_a.shape[0]
        device = z_a.device
        labels = torch.arange(B, device=device)

        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        logits = (z_a.float() @ z_b.float().T) / tau.float()

        return (
            F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing) +
            F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
        ) / 2

    def hard_negative_infonce(
        self,
        z_geo: torch.Tensor,
        z_text: torch.Tensor,
        hard_neg_indices: Optional[List[List[int]]],
        tau: torch.Tensor,
        hard_neg_scale: float = 2.0
    ) -> torch.Tensor:
        """InfoNCE with hard negative emphasis."""
        B = z_geo.shape[0]
        device = z_geo.device
        labels = torch.arange(B, device=device)

        z_geo = F.normalize(z_geo, dim=-1)
        z_text = F.normalize(z_text, dim=-1)

        logits = (z_geo.float() @ z_text.float().T) / tau.float()

        if hard_neg_indices is not None:
            for i, negs in enumerate(hard_neg_indices):
                if negs is not None and len(negs) > 0:
                    for neg_idx in negs:
                        if neg_idx != i and neg_idx < B:
                            logits[i, neg_idx] = logits[i, neg_idx] * hard_neg_scale

        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

    def query_distillation_loss(
        self,
        Q_brep: torch.Tensor,
        Q_pc: torch.Tensor,
        T_feat: torch.Tensor,
        conf: torch.Tensor
    ) -> torch.Tensor:
        """Per-slot cosine loss weighted by confidence."""
        Q_brep_norm = F.normalize(Q_brep, dim=-1)
        Q_pc_norm = F.normalize(Q_pc, dim=-1)
        T_norm = F.normalize(T_feat.detach(), dim=-1)

        cos_brep = (Q_brep_norm * T_norm).sum(dim=-1)  # (B, K)
        cos_pc = (Q_pc_norm * T_norm).sum(dim=-1)

        conf_detached = conf.detach()
        loss_brep = ((1 - cos_brep) * conf_detached).sum() / (conf_detached.sum() + 1e-8)
        loss_pc = ((1 - cos_pc) * conf_detached).sum() / (conf_detached.sum() + 1e-8)

        return (loss_brep + loss_pc) / 2

    def embedding_distillation_loss(
        self,
        z_brep_self: torch.Tensor,
        z_brep_guided: torch.Tensor,
        z_pc_self: torch.Tensor,
        z_pc_guided: torch.Tensor
    ) -> torch.Tensor:
        """Cosine loss between self and guided embeddings."""
        loss_brep = 1 - F.cosine_similarity(z_brep_self, z_brep_guided.detach(), dim=-1).mean()
        loss_pc = 1 - F.cosine_similarity(z_pc_self, z_pc_guided.detach(), dim=-1).mean()
        return (loss_brep + loss_pc) / 2

    def distribution_matching_loss(
        self,
        Q_brep: torch.Tensor,
        Q_pc: torch.Tensor,
        T_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Match the distribution of Q_self to T_feat.

        This encourages Q_self to have similar statistical properties
        (mean, variance) to T_feat across the batch.

        Particularly important during curriculum learning when conditioning
        is being gradually removed.
        """
        T_feat_detached = T_feat.detach()

        # Per-slot mean across batch: (K, d)
        Q_brep_mean = Q_brep.mean(dim=0)
        Q_pc_mean = Q_pc.mean(dim=0)
        T_mean = T_feat_detached.mean(dim=0)

        # Per-slot std across batch: (K, d)
        Q_brep_std = Q_brep.std(dim=0)
        Q_pc_std = Q_pc.std(dim=0)
        T_std = T_feat_detached.std(dim=0)

        # Match mean
        loss_mean_brep = F.mse_loss(Q_brep_mean, T_mean)
        loss_mean_pc = F.mse_loss(Q_pc_mean, T_mean)

        # Match std (with smaller weight)
        loss_std_brep = F.mse_loss(Q_brep_std, T_std)
        loss_std_pc = F.mse_loss(Q_pc_std, T_std)

        loss_brep = loss_mean_brep + 0.5 * loss_std_brep
        loss_pc = loss_mean_pc + 0.5 * loss_std_pc

        return (loss_brep + loss_pc) / 2

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        hard_negatives: Optional[List[List[int]]] = None,
        stage: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute all losses.

        Args:
            outputs: Model forward outputs
            hard_negatives: Optional hard negative indices
            stage: Training stage (1 or 2)

        Returns:
            total_loss: Scalar loss
            loss_dict: Individual losses for logging
        """
        losses = {}
        device = outputs['tau'].device
        tau = outputs['tau']

        # ─────────────────────────────────────────────────────────────────────
        # 1. GUIDED CONTRASTIVE (Primary)
        # ─────────────────────────────────────────────────────────────────────
        z_brep = outputs.get('z_brep')
        z_pc = outputs.get('z_pc')
        z_text = outputs.get('z_text')

        if z_brep is not None and z_pc is not None and z_text is not None:
            losses['guided'] = self.infonce_3way(z_brep, z_pc, z_text, tau)
        elif z_brep is not None and z_text is not None:
            losses['guided'] = self.infonce_2way(z_brep, z_text, tau)
        elif z_pc is not None and z_text is not None:
            losses['guided'] = self.infonce_2way(z_pc, z_text, tau)
        else:
            losses['guided'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # 2. SELF CONTRASTIVE
        # ─────────────────────────────────────────────────────────────────────
        z_brep_self = outputs.get('z_brep_self')
        z_pc_self = outputs.get('z_pc_self')

        if z_brep_self is not None and z_pc_self is not None and z_text is not None:
            losses['self'] = self.infonce_3way(z_brep_self, z_pc_self, z_text, tau)
        elif z_brep_self is not None and z_text is not None:
            losses['self'] = self.infonce_2way(z_brep_self, z_text, tau)
        elif z_pc_self is not None and z_text is not None:
            losses['self'] = self.infonce_2way(z_pc_self, z_text, tau)
        else:
            losses['self'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # 3. QUERY DISTILLATION
        # ─────────────────────────────────────────────────────────────────────
        T_feat = outputs.get('T_feat')
        Q_brep_self = outputs.get('Q_brep_self')
        Q_pc_self = outputs.get('Q_pc_self')
        conf_text = outputs.get('confidence')

        if T_feat is not None and Q_brep_self is not None and Q_pc_self is not None:
            losses['query'] = self.query_distillation_loss(
                Q_brep_self, Q_pc_self, T_feat, conf_text
            )
        else:
            losses['query'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # 4. EMBEDDING DISTILLATION
        # ─────────────────────────────────────────────────────────────────────
        if z_brep is not None and z_brep_self is not None and z_pc is not None and z_pc_self is not None:
            losses['embed'] = self.embedding_distillation_loss(
                z_brep_self, z_brep, z_pc_self, z_pc
            )
        else:
            losses['embed'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # 5. DISTRIBUTION MATCHING (NEW in v4.2)
        # ─────────────────────────────────────────────────────────────────────
        if T_feat is not None and Q_brep_self is not None and Q_pc_self is not None:
            losses['dist'] = self.distribution_matching_loss(
                Q_brep_self, Q_pc_self, T_feat
            )
        else:
            losses['dist'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # 6. DETAIL CONTRASTIVE (Stage 2 only)
        # ─────────────────────────────────────────────────────────────────────
        if self.lambda_detail > 0:
            z_brep_d = outputs.get('z_brep_detail')
            z_pc_d = outputs.get('z_pc_detail')

            if z_brep_d is not None and z_pc_d is not None and z_text is not None:
                if hard_negatives is not None:
                    losses['detail'] = self.hard_negative_infonce(
                        z_brep_d, z_text, hard_negatives, tau * 0.7
                    )
                else:
                    losses['detail'] = self.infonce_3way(z_brep_d, z_pc_d, z_text, tau * 0.7)
            else:
                losses['detail'] = torch.tensor(0.0, device=device)
        else:
            losses['detail'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # TOTAL
        # ─────────────────────────────────────────────────────────────────────
        losses['total'] = (
            1.0 * losses['guided'] +
            self.lambda_self * losses['self'] +
            self.lambda_query * losses['query'] +
            self.lambda_embed * losses['embed'] +
            self.lambda_dist * losses['dist'] +
            self.lambda_detail * losses['detail']
        )

        # Convert to float for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in losses.items()}

        return losses['total'], loss_dict

    def update_weights(
        self,
        lambda_self: Optional[float] = None,
        lambda_query: Optional[float] = None,
        lambda_embed: Optional[float] = None,
        lambda_dist: Optional[float] = None,
        lambda_detail: Optional[float] = None
    ):
        """Update loss weights (for stage transitions)."""
        if lambda_self is not None:
            self.lambda_self = lambda_self
        if lambda_query is not None:
            self.lambda_query = lambda_query
        if lambda_embed is not None:
            self.lambda_embed = lambda_embed
        if lambda_dist is not None:
            self.lambda_dist = lambda_dist
        if lambda_detail is not None:
            self.lambda_detail = lambda_detail


# =============================================================================
# Helper Functions
# =============================================================================

def compute_self_grounding_quality(
    z_guided: torch.Tensor,
    z_self: torch.Tensor
) -> float:
    """
    Compute cosine similarity between guided and self-grounded embeddings.

    Target: > 0.85 after training
    """
    z_guided = F.normalize(z_guided, dim=-1)
    z_self = F.normalize(z_self, dim=-1)
    cosine_sim = (z_guided * z_self).sum(dim=-1).mean()
    return cosine_sim.item()


def compute_query_alignment(
    T_feat: torch.Tensor,
    Q_self: torch.Tensor,
    confidence: Optional[torch.Tensor] = None
) -> float:
    """
    Compute query alignment between text features and self-generated queries.

    Target: > 0.7 after training
    """
    T_feat_norm = F.normalize(T_feat, dim=-1)
    Q_self_norm = F.normalize(Q_self, dim=-1)

    cos_sim = (T_feat_norm * Q_self_norm).sum(dim=-1)  # (B, K)

    if confidence is not None:
        weighted_sim = (cos_sim * confidence).sum() / (confidence.sum() + 1e-8)
        return weighted_sim.item()
    else:
        return cos_sim.mean().item()
