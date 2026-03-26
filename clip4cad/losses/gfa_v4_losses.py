"""
GFA v4 Loss Functions - Query Distillation + Attention Distillation

Key changes from v2.4:
1. QUERY DISTILLATION (L_query): Direct cosine supervision of Q_self → T_feat
   - This is THE KEY FIX that forces self-queries to match text queries
   - λ_query = 1.5 (Stage 1), 0.8 (Stage 2) - highest weight!

2. ATTENTION DISTILLATION (L_attn): KL divergence on slot attention patterns
   - Forces A_self (slot attention weights) to match G_guided (grounding matrix)
   - λ_attn = 0.5 (Stage 1), 0.3 (Stage 2)

3. EMBEDDING DISTILLATION reduced: λ_embed = 0.3 (was 1.0 in v2.4)
   - Query distillation handles alignment now

4. CONFIDENCE ALIGNMENT removed: Confidence comes from slot attention

Loss terms:
1. GUIDED: Text-guided geometry contrastive (primary)
2. SELF: Self-grounded geometry contrastive (increased in Stage 2)
3. QUERY: Query-level cosine distillation (THE KEY!)
4. ATTN: Attention pattern KL divergence
5. EMBED: Embedding cosine distillation (reduced)
6. DETAIL: Fine-grained contrastive (Stage 2 only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class GFAv4Loss(nn.Module):
    """
    GFA v4 Loss Function with Query Distillation.

    KEY INSIGHT: Directly supervise Q_self to match T_feat at the QUERY level,
    not just the embedding level. This forces the slot attention to produce
    queries in the same semantic space as the text parser.
    """

    def __init__(
        self,
        lambda_self: float = 0.05,          # Very low early - don't let self compete
        lambda_query: float = 1.5,          # HIGH - this is THE KEY FIX!
        lambda_attn: float = 0.5,           # Attention pattern distillation
        lambda_embed: float = 0.3,          # Reduced (query distill handles alignment)
        lambda_detail: float = 0.0,         # Stage 2 only
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_query = lambda_query
        self.lambda_attn = lambda_attn
        self.lambda_embed = lambda_embed
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

        # Scale hard negatives
        if hard_neg_indices is not None:
            for i, negs in enumerate(hard_neg_indices):
                if negs is not None and len(negs) > 0:
                    for neg_idx in negs:
                        if neg_idx != i and neg_idx < B:
                            logits[i, neg_idx] = logits[i, neg_idx] * hard_neg_scale

        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

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
        # 3. QUERY DISTILLATION (NEW - THE KEY FIX!)
        # Direct cosine supervision: Q_self should match T_feat
        # ─────────────────────────────────────────────────────────────────────

        T_feat = outputs.get('T_feat')
        Q_brep_self = outputs.get('Q_brep_self')
        Q_pc_self = outputs.get('Q_pc_self')
        conf_text = outputs.get('confidence')

        query_losses = []

        if T_feat is not None and Q_brep_self is not None:
            # Check for NaN in inputs
            if not torch.isnan(Q_brep_self).any():
                # Normalize for cosine similarity
                T_feat_norm = F.normalize(T_feat.detach(), dim=-1)
                Q_brep_norm = F.normalize(Q_brep_self, dim=-1)

                # Per-slot cosine similarity: (B, K)
                cos_brep = (T_feat_norm * Q_brep_norm).sum(dim=-1)
                cos_brep = torch.nan_to_num(cos_brep, nan=0.0)

                # Weight by text confidence (focus on active slots)
                if conf_text is not None:
                    conf = conf_text.detach().clamp(min=0.0, max=1.0)
                    loss_query_brep = ((1 - cos_brep) * conf).sum() / (conf.sum() + 1e-8)
                else:
                    loss_query_brep = (1 - cos_brep).mean()

                if not (torch.isnan(loss_query_brep) or torch.isinf(loss_query_brep)):
                    query_losses.append(loss_query_brep)

        if T_feat is not None and Q_pc_self is not None:
            # Check for NaN in inputs
            if not torch.isnan(Q_pc_self).any():
                T_feat_norm = F.normalize(T_feat.detach(), dim=-1)
                Q_pc_norm = F.normalize(Q_pc_self, dim=-1)

                cos_pc = (T_feat_norm * Q_pc_norm).sum(dim=-1)
                cos_pc = torch.nan_to_num(cos_pc, nan=0.0)

                if conf_text is not None:
                    conf = conf_text.detach().clamp(min=0.0, max=1.0)
                    loss_query_pc = ((1 - cos_pc) * conf).sum() / (conf.sum() + 1e-8)
                else:
                    loss_query_pc = (1 - cos_pc).mean()

                if not (torch.isnan(loss_query_pc) or torch.isinf(loss_query_pc)):
                    query_losses.append(loss_query_pc)

        if len(query_losses) > 0:
            losses['query'] = sum(query_losses) / len(query_losses)
            # Final NaN check
            if torch.isnan(losses['query']) or torch.isinf(losses['query']):
                losses['query'] = torch.tensor(0.0, device=device)
        else:
            losses['query'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # 4. ATTENTION DISTILLATION (NEW)
        # KL divergence between slot attention and guided grounding
        # ─────────────────────────────────────────────────────────────────────

        A_brep_self = outputs.get('A_brep_self')
        A_pc_self = outputs.get('A_pc_self')
        G_brep_guided = outputs.get('G_brep_guided')
        G_pc_guided = outputs.get('G_pc_guided')

        attn_losses = []

        if A_brep_self is not None and G_brep_guided is not None:
            # Ensure same shape - A_self and G_guided should both be (B, K, N)
            if A_brep_self.shape == G_brep_guided.shape:
                G_guided_safe = G_brep_guided.detach().clamp(min=1e-8, max=1.0)
                A_self_safe = A_brep_self.clamp(min=1e-8, max=1.0)
                # Clamp log to avoid -inf
                log_A = A_self_safe.log().clamp(min=-20)
                loss_attn_brep = F.kl_div(
                    log_A, G_guided_safe, reduction='batchmean'
                )
                # Check for NaN/inf and use 0 if invalid
                if torch.isnan(loss_attn_brep) or torch.isinf(loss_attn_brep):
                    loss_attn_brep = torch.tensor(0.0, device=device)
                attn_losses.append(loss_attn_brep)

        if A_pc_self is not None and G_pc_guided is not None:
            if A_pc_self.shape == G_pc_guided.shape:
                G_guided_safe = G_pc_guided.detach().clamp(min=1e-8, max=1.0)
                A_self_safe = A_pc_self.clamp(min=1e-8, max=1.0)
                # Clamp log to avoid -inf
                log_A = A_self_safe.log().clamp(min=-20)
                loss_attn_pc = F.kl_div(
                    log_A, G_guided_safe, reduction='batchmean'
                )
                # Check for NaN/inf and use 0 if invalid
                if torch.isnan(loss_attn_pc) or torch.isinf(loss_attn_pc):
                    loss_attn_pc = torch.tensor(0.0, device=device)
                attn_losses.append(loss_attn_pc)

        if len(attn_losses) > 0:
            losses['attn'] = sum(attn_losses) / len(attn_losses)
        else:
            losses['attn'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # 5. EMBEDDING DISTILLATION (Cosine - reduced weight)
        # ─────────────────────────────────────────────────────────────────────

        embed_losses = []

        if z_brep is not None and z_brep_self is not None:
            z_brep_norm = F.normalize(z_brep.detach(), dim=-1)
            z_brep_self_norm = F.normalize(z_brep_self, dim=-1)
            loss_embed_brep = 1 - (z_brep_norm * z_brep_self_norm).sum(dim=-1).mean()
            embed_losses.append(loss_embed_brep)

        if z_pc is not None and z_pc_self is not None:
            z_pc_norm = F.normalize(z_pc.detach(), dim=-1)
            z_pc_self_norm = F.normalize(z_pc_self, dim=-1)
            loss_embed_pc = 1 - (z_pc_norm * z_pc_self_norm).sum(dim=-1).mean()
            embed_losses.append(loss_embed_pc)

        if len(embed_losses) > 0:
            losses['embed'] = sum(embed_losses) / len(embed_losses)
        else:
            losses['embed'] = torch.tensor(0.0, device=device)

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
            elif z_brep_d is not None and z_text is not None:
                if hard_negatives is not None:
                    losses['detail'] = self.hard_negative_infonce(
                        z_brep_d, z_text, hard_negatives, tau * 0.7
                    )
                else:
                    losses['detail'] = self.infonce_2way(z_brep_d, z_text, tau * 0.7)
            else:
                losses['detail'] = torch.tensor(0.0, device=device)
        else:
            losses['detail'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # TOTAL
        # ─────────────────────────────────────────────────────────────────────

        # Ensure no individual loss is NaN
        for key in ['guided', 'self', 'query', 'attn', 'embed', 'detail']:
            if torch.isnan(losses[key]) or torch.isinf(losses[key]):
                losses[key] = torch.tensor(0.0, device=device)

        losses['total'] = (
            1.0 * losses['guided'] +
            self.lambda_self * losses['self'] +
            self.lambda_query * losses['query'] +      # KEY: Query alignment
            self.lambda_attn * losses['attn'] +        # Attention alignment
            self.lambda_embed * losses['embed'] +
            self.lambda_detail * losses['detail']
        )

        # Final safeguard - if total is NaN, use only guided loss
        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            losses['total'] = losses['guided']

        # Convert to float for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in losses.items()}

        return losses['total'], loss_dict

    def update_weights(
        self,
        lambda_self: Optional[float] = None,
        lambda_query: Optional[float] = None,
        lambda_attn: Optional[float] = None,
        lambda_embed: Optional[float] = None,
        lambda_detail: Optional[float] = None
    ):
        """Update loss weights (for stage transitions)."""
        if lambda_self is not None:
            self.lambda_self = lambda_self
        if lambda_query is not None:
            self.lambda_query = lambda_query
        if lambda_attn is not None:
            self.lambda_attn = lambda_attn
        if lambda_embed is not None:
            self.lambda_embed = lambda_embed
        if lambda_detail is not None:
            self.lambda_detail = lambda_detail


def compute_self_grounding_quality(
    z_guided: torch.Tensor,
    z_self: torch.Tensor
) -> float:
    """
    Compute cosine similarity between guided and self-grounded embeddings.

    Target: > 0.85 after training with v4 query distillation
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

    This is the key metric for v4 - if this is high, self-grounding will work.

    Target: > 0.7 after training
    """
    T_feat_norm = F.normalize(T_feat, dim=-1)
    Q_self_norm = F.normalize(Q_self, dim=-1)

    # Per-slot cosine similarity
    cos_sim = (T_feat_norm * Q_self_norm).sum(dim=-1)  # (B, K)

    if confidence is not None:
        # Weighted by confidence
        weighted_sim = (cos_sim * confidence).sum() / (confidence.sum() + 1e-8)
        return weighted_sim.item()
    else:
        return cos_sim.mean().item()
