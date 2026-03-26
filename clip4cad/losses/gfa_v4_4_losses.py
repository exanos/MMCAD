"""
GFA v4.4 Loss Functions - Topology-Aware with Contrastive Query Loss

Key components:
1. L_guided: Primary contrastive alignment (InfoNCE 3-way)
2. L_self: Self-grounding contrastive
3. L_query_contrastive: Instance-level query matching (KEY INNOVATION!)
4. L_query_cosine: Slot-level query matching
5. L_embed: Embedding distillation (z_self → z_guided)
6. L_grounding: Attention pattern alignment (KL divergence)
7. L_detail: Hard negative contrastive (Stage 2)

The contrastive query loss is the key innovation:
- Forces Q_self[i] to match T_feat[i] at the instance level
- Not just cosine similarity, but contrastive to distinguish from other samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class GFAv44Loss(nn.Module):
    """
    Loss for GFA v4.4 with topology-aware encoding.

    Key insight: The contrastive query loss forces Q_self to match T_feat
    at the instance level, not just the embedding level.
    """

    def __init__(
        self,
        lambda_self: float = 0.1,
        lambda_query_contrastive: float = 1.5,
        lambda_query_cosine: float = 0.3,
        lambda_embed: float = 0.3,
        lambda_grounding: float = 0.5,
        lambda_detail: float = 0.0,
        query_tau: float = 0.1,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_query_contrastive = lambda_query_contrastive
        self.lambda_query_cosine = lambda_query_cosine
        self.lambda_embed = lambda_embed
        self.lambda_grounding = lambda_grounding
        self.lambda_detail = lambda_detail
        self.query_tau = query_tau
        self.label_smoothing = label_smoothing

    def update_weights(self, **kwargs):
        """Update loss weights dynamically (e.g., for stage transitions)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        hard_negatives: Optional[List] = None,
        stage: int = 1
    ) -> tuple:
        """
        Compute total loss and individual components.

        Args:
            outputs: Model outputs dict
            hard_negatives: List of hard negative indices per sample (Stage 2)
            stage: Training stage (1 or 2)

        Returns:
            total_loss: Scalar loss
            losses: Dict of individual loss components
        """
        losses = {}
        tau = outputs['tau']
        device = tau.device

        # ─────────────────────────────────────────────────────────────────────
        # 1. GUIDED CONTRASTIVE (Primary)
        # ─────────────────────────────────────────────────────────────────────
        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)
        z_text = F.normalize(outputs['z_text'], dim=-1)

        losses['guided'] = self._infonce_3way(z_brep, z_pc, z_text, tau)

        # ─────────────────────────────────────────────────────────────────────
        # 2. SELF CONTRASTIVE
        # ─────────────────────────────────────────────────────────────────────
        z_brep_self = F.normalize(outputs['z_brep_self'], dim=-1)
        z_pc_self = F.normalize(outputs['z_pc_self'], dim=-1)

        losses['self'] = self._infonce_3way(z_brep_self, z_pc_self, z_text, tau)

        # ─────────────────────────────────────────────────────────────────────
        # 3. CONTRASTIVE QUERY LOSS (Instance-level) - KEY INNOVATION!
        # ─────────────────────────────────────────────────────────────────────
        T_feat = outputs['T_feat'].detach()
        conf = outputs['confidence'].detach()

        losses['query_con'] = (
            self._contrastive_query_loss(outputs['Q_brep_self'], T_feat, conf) +
            self._contrastive_query_loss(outputs['Q_pc_self'], T_feat, conf)
        ) / 2

        # ─────────────────────────────────────────────────────────────────────
        # 4. COSINE QUERY LOSS (Slot-level)
        # ─────────────────────────────────────────────────────────────────────
        losses['query_cos'] = self._cosine_query_loss(
            outputs['Q_brep_self'], outputs['Q_pc_self'], T_feat, conf
        )

        # ─────────────────────────────────────────────────────────────────────
        # 5. EMBEDDING DISTILLATION
        # ─────────────────────────────────────────────────────────────────────
        losses['embed'] = (
            1 - F.cosine_similarity(
                outputs['z_brep_self'], outputs['z_brep'].detach(), dim=-1
            ).mean() +
            1 - F.cosine_similarity(
                outputs['z_pc_self'], outputs['z_pc'].detach(), dim=-1
            ).mean()
        ) / 2

        # ─────────────────────────────────────────────────────────────────────
        # 6. GROUNDING ALIGNMENT
        # ─────────────────────────────────────────────────────────────────────
        losses['grounding'] = (
            F.kl_div(
                torch.log(outputs['G_brep_self'] + 1e-8),
                outputs['G_brep_guided'].detach(),
                reduction='batchmean'
            ) +
            F.kl_div(
                torch.log(outputs['G_pc_self'] + 1e-8),
                outputs['G_pc_guided'].detach(),
                reduction='batchmean'
            )
        ) / 2

        # ─────────────────────────────────────────────────────────────────────
        # 7. DETAIL LOSS (Stage 2 with Hard Negatives)
        # ─────────────────────────────────────────────────────────────────────
        if self.lambda_detail > 0 and 'z_brep_detail' in outputs:
            z_brep_d = F.normalize(outputs['z_brep_detail'], dim=-1)
            z_pc_d = F.normalize(outputs['z_pc_detail'], dim=-1)

            if hard_negatives is not None:
                losses['detail'] = self._hard_negative_loss(
                    z_brep_d, z_text, hard_negatives, tau * 0.7
                )
            else:
                losses['detail'] = self._infonce_3way(z_brep_d, z_pc_d, z_text, tau)
        else:
            losses['detail'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # TOTAL
        # ─────────────────────────────────────────────────────────────────────
        losses['total'] = (
            1.0 * losses['guided'] +
            self.lambda_self * losses['self'] +
            self.lambda_query_contrastive * losses['query_con'] +
            self.lambda_query_cosine * losses['query_cos'] +
            self.lambda_embed * losses['embed'] +
            self.lambda_grounding * losses['grounding'] +
            self.lambda_detail * losses['detail']
        )

        return losses['total'], losses

    def _infonce_3way(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        z_c: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """3-way InfoNCE loss for (BRep, PC, Text) alignment."""
        B = z_a.shape[0]
        device = z_a.device
        labels = torch.arange(B, device=device)

        loss = 0.0
        for zi, zj in [(z_a, z_c), (z_b, z_c), (z_a, z_b)]:
            logits = (zi.float() @ zj.float().T) / tau.float()
            loss += (
                F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing) +
                F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
            ) / 2

        return loss / 3

    def _contrastive_query_loss(
        self,
        Q_self: torch.Tensor,
        T_feat: torch.Tensor,
        conf: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive query loss: Q_self[i] should match T_feat[i], not T_feat[j].

        This is the KEY innovation - forces queries to match at instance level.
        """
        B = Q_self.shape[0]
        device = Q_self.device

        # Confidence-weighted pooling
        Q_pooled = (Q_self * conf.unsqueeze(-1)).sum(dim=1)
        Q_pooled = Q_pooled / (conf.sum(dim=1, keepdim=True) + 1e-8)

        T_pooled = (T_feat * conf.unsqueeze(-1)).sum(dim=1)
        T_pooled = T_pooled / (conf.sum(dim=1, keepdim=True) + 1e-8)

        Q_norm = F.normalize(Q_pooled, dim=-1)
        T_norm = F.normalize(T_pooled, dim=-1)

        logits = Q_norm @ T_norm.T / self.query_tau
        labels = torch.arange(B, device=device)

        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def _cosine_query_loss(
        self,
        Q_brep: torch.Tensor,
        Q_pc: torch.Tensor,
        T_feat: torch.Tensor,
        conf: torch.Tensor
    ) -> torch.Tensor:
        """Per-slot cosine similarity loss."""
        Q_brep_norm = F.normalize(Q_brep, dim=-1)
        Q_pc_norm = F.normalize(Q_pc, dim=-1)
        T_norm = F.normalize(T_feat, dim=-1)

        cos_brep = (Q_brep_norm * T_norm).sum(dim=-1)
        cos_pc = (Q_pc_norm * T_norm).sum(dim=-1)

        loss_brep = ((1 - cos_brep) * conf).sum() / (conf.sum() + 1e-8)
        loss_pc = ((1 - cos_pc) * conf).sum() / (conf.sum() + 1e-8)

        return (loss_brep + loss_pc) / 2

    def _hard_negative_loss(
        self,
        z_geo: torch.Tensor,
        z_text: torch.Tensor,
        hard_negatives: List,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive with boosted hard negatives."""
        B = z_geo.shape[0]
        device = z_geo.device
        labels = torch.arange(B, device=device)

        logits = z_geo @ z_text.T / tau

        # Boost hard negatives
        if hard_negatives is not None:
            for i, negs in enumerate(hard_negatives):
                if negs is not None:
                    for neg_idx in negs:
                        if neg_idx < B and neg_idx != i:
                            logits[i, neg_idx] = logits[i, neg_idx] * 2.0

        return F.cross_entropy(logits, labels)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_self_grounding_quality(
    z_guided: torch.Tensor,
    z_self: torch.Tensor
) -> float:
    """Compute cosine similarity between guided and self embeddings."""
    z_g = F.normalize(z_guided, dim=-1)
    z_s = F.normalize(z_self, dim=-1)
    return (z_g * z_s).sum(dim=-1).mean().item()


def compute_query_alignment(
    Q_self: torch.Tensor,
    T_feat: torch.Tensor,
    conf: torch.Tensor
) -> float:
    """Compute weighted cosine similarity between queries."""
    Q_norm = F.normalize(Q_self, dim=-1)
    T_norm = F.normalize(T_feat, dim=-1)
    cos = (Q_norm * T_norm).sum(dim=-1)
    weighted = (cos * conf).sum() / (conf.sum() + 1e-8)
    return weighted.item()


def compute_retrieval_metrics(
    z_geo: torch.Tensor,
    z_text: torch.Tensor,
    ks: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute text→geometry retrieval metrics.

    Args:
        z_geo: (N, d) geometry embeddings
        z_text: (N, d) text embeddings
        ks: k values for R@k

    Returns:
        Dict with R@k metrics
    """
    z_g = F.normalize(z_geo, dim=-1)
    z_t = F.normalize(z_text, dim=-1)

    # Text → Geometry similarity
    sim = z_t @ z_g.T  # (N, N)
    N = sim.shape[0]

    metrics = {}
    for k in ks:
        # Get top-k indices per query
        _, topk_idx = sim.topk(k, dim=1)
        # Check if correct index is in top-k
        correct = (topk_idx == torch.arange(N, device=sim.device).unsqueeze(1)).any(dim=1)
        metrics[f'R@{k}'] = correct.float().mean().item()

    return metrics
