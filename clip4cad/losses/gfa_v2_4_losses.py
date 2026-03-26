"""
GFA v2.4 Loss Functions

Key changes from v2:
1. MSE distillation instead of cosine (stronger gradient)
2. Confidence alignment loss (match activation patterns)
3. Rebalanced weights for shared encoder architecture

Loss terms:
1. GUIDED: Text-guided geometry contrastive (primary)
2. SELF: Self-grounded geometry contrastive (reduced weight early)
3. DISTILL: Grounding matrix KL divergence
4. EMBED_DISTILL: Embedding MSE (KEY - much stronger than cosine!)
5. CONF_ALIGN: Confidence pattern alignment (NEW)
6. DETAIL: Fine-grained contrastive (Stage 2 only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class GFAv2_4Loss(nn.Module):
    """
    GFA v2.4 Loss Function with MSE distillation and confidence alignment.

    KEY CHANGES:
    - MSE instead of cosine for embedding distillation (stronger gradient)
    - Confidence alignment to match activation patterns
    - Lower lambda_self early to prevent competing solutions
    """

    def __init__(
        self,
        lambda_self: float = 0.05,          # Very low early - don't let self compete
        lambda_distill: float = 0.5,        # High - grounding alignment
        lambda_embed_distill: float = 1.0,  # Very high - MSE embedding alignment
        lambda_conf_align: float = 0.2,     # Match confidence patterns
        lambda_detail: float = 0.0,         # Stage 2 only
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_distill = lambda_distill
        self.lambda_embed_distill = lambda_embed_distill
        self.lambda_conf_align = lambda_conf_align
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

        # Scale hard negatives - filter to valid batch indices
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

        # ─────────────────────────────────────────────────────────────────
        # 1. GUIDED CONTRASTIVE (Primary)
        # ─────────────────────────────────────────────────────────────────

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

        # ─────────────────────────────────────────────────────────────────
        # 2. SELF CONTRASTIVE (Reduced weight early)
        # ─────────────────────────────────────────────────────────────────

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

        # ─────────────────────────────────────────────────────────────────
        # 3. GROUNDING DISTILLATION (KL divergence)
        # ─────────────────────────────────────────────────────────────────

        G_brep_guided = outputs.get('G_brep_guided')
        G_pc_guided = outputs.get('G_pc_guided')
        G_brep_self = outputs.get('G_brep_self')
        G_pc_self = outputs.get('G_pc_self')

        distill_losses = []

        if G_brep_guided is not None and G_brep_self is not None:
            G_guided_safe = G_brep_guided.detach().clamp(min=1e-8)
            G_self_safe = G_brep_self.clamp(min=1e-8)
            loss_distill_brep = F.kl_div(
                G_self_safe.log(), G_guided_safe, reduction='batchmean'
            )
            distill_losses.append(loss_distill_brep)

        if G_pc_guided is not None and G_pc_self is not None:
            G_guided_safe = G_pc_guided.detach().clamp(min=1e-8)
            G_self_safe = G_pc_self.clamp(min=1e-8)
            loss_distill_pc = F.kl_div(
                G_self_safe.log(), G_guided_safe, reduction='batchmean'
            )
            distill_losses.append(loss_distill_pc)

        if len(distill_losses) > 0:
            losses['distill'] = sum(distill_losses) / len(distill_losses)
        else:
            losses['distill'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────
        # 4. EMBEDDING DISTILLATION (MSE - stronger than cosine!)
        # ─────────────────────────────────────────────────────────────────

        embed_distill_losses = []

        if z_brep is not None and z_brep_self is not None:
            # MSE loss - much stronger gradient than cosine!
            loss_embed_brep = F.mse_loss(z_brep_self, z_brep.detach())
            embed_distill_losses.append(loss_embed_brep)

        if z_pc is not None and z_pc_self is not None:
            loss_embed_pc = F.mse_loss(z_pc_self, z_pc.detach())
            embed_distill_losses.append(loss_embed_pc)

        if len(embed_distill_losses) > 0:
            losses['embed_distill'] = sum(embed_distill_losses) / len(embed_distill_losses)
        else:
            losses['embed_distill'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────
        # 5. CONFIDENCE ALIGNMENT (NEW - match activation patterns)
        # ─────────────────────────────────────────────────────────────────

        conf_text = outputs.get('confidence')
        conf_brep_self = outputs.get('confidence_brep_self')
        conf_pc_self = outputs.get('confidence_pc_self')

        conf_align_losses = []

        if conf_text is not None and conf_brep_self is not None:
            loss_conf_brep = F.mse_loss(conf_brep_self, conf_text.detach())
            conf_align_losses.append(loss_conf_brep)

        if conf_text is not None and conf_pc_self is not None:
            loss_conf_pc = F.mse_loss(conf_pc_self, conf_text.detach())
            conf_align_losses.append(loss_conf_pc)

        if len(conf_align_losses) > 0:
            losses['conf_align'] = sum(conf_align_losses) / len(conf_align_losses)
        else:
            losses['conf_align'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────
        # 6. DETAIL CONTRASTIVE (Stage 2 only)
        # ─────────────────────────────────────────────────────────────────

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

        # ─────────────────────────────────────────────────────────────────
        # TOTAL
        # ─────────────────────────────────────────────────────────────────

        losses['total'] = (
            1.0 * losses['guided'] +
            self.lambda_self * losses['self'] +
            self.lambda_distill * losses['distill'] +
            self.lambda_embed_distill * losses['embed_distill'] +
            self.lambda_conf_align * losses['conf_align'] +
            self.lambda_detail * losses['detail']
        )

        # Convert to float for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in losses.items()}

        return losses['total'], loss_dict

    def update_weights(
        self,
        lambda_self: Optional[float] = None,
        lambda_distill: Optional[float] = None,
        lambda_embed_distill: Optional[float] = None,
        lambda_conf_align: Optional[float] = None,
        lambda_detail: Optional[float] = None
    ):
        """Update loss weights (for stage transitions)."""
        if lambda_self is not None:
            self.lambda_self = lambda_self
        if lambda_distill is not None:
            self.lambda_distill = lambda_distill
        if lambda_embed_distill is not None:
            self.lambda_embed_distill = lambda_embed_distill
        if lambda_conf_align is not None:
            self.lambda_conf_align = lambda_conf_align
        if lambda_detail is not None:
            self.lambda_detail = lambda_detail


def compute_self_grounding_quality(
    z_guided: torch.Tensor,
    z_self: torch.Tensor
) -> float:
    """
    Compute cosine similarity between guided and self-grounded embeddings.

    Target: > 0.9 after training (with v2.4 shared encoder, should be high!)
    """
    z_guided = F.normalize(z_guided, dim=-1)
    z_self = F.normalize(z_self, dim=-1)
    cosine_sim = (z_guided * z_self).sum(dim=-1).mean()
    return cosine_sim.item()
