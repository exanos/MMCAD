"""
GFA v2 Loss Functions

Simplified 4-term loss:
1. GUIDED: Text-guided geometry contrastive (what worked in v1)
2. SELF: Self-encoded geometry contrastive (KEY FIX!)
3. DISTILL: Self learns grounding pattern from guided (auxiliary)
4. DETAIL: InfoNCE with hard negatives (Stage 2 only)

REMOVED from v1:
- Consistency loss (over-constraining)
- Local contrastive (false negatives)
- Diversity loss (not needed with hierarchical agg)

Based on CLIP4CAD_GFA_v2_Architecture.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class GFAv2Loss(nn.Module):
    """
    GFA v2 Loss Function

    KEY INSIGHT: Self-grounding must learn DIRECTLY via contrastive loss,
    not just by mimicking text-grounding patterns (distillation alone fails).
    """

    def __init__(
        self,
        lambda_self: float = 0.1,
        lambda_distill: float = 0.3,
        lambda_embed_distill: float = 0.5,
        lambda_detail: float = 0.0,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            lambda_self: Weight for self contrastive loss
            lambda_distill: Weight for grounding distillation
            lambda_embed_distill: Weight for embedding distillation (KEY FIX!)
            lambda_detail: Weight for detail contrastive (Stage 2)
            label_smoothing: Label smoothing for cross-entropy (0 = no smoothing)
        """
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_distill = lambda_distill
        self.lambda_embed_distill = lambda_embed_distill
        self.lambda_detail = lambda_detail
        self.label_smoothing = label_smoothing

    def infonce_3way(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        z_c: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard 3-way InfoNCE loss.

        Aligns: A<->C, B<->C, A<->B

        Args:
            z_a: First modality embeddings (B, d) - e.g., B-Rep
            z_b: Second modality embeddings (B, d) - e.g., PC
            z_c: Third modality embeddings (B, d) - e.g., Text
            tau: Temperature scalar

        Returns:
            Average loss over all pairs
        """
        B = z_a.shape[0]
        device = z_a.device
        labels = torch.arange(B, device=device)

        # Normalize
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        z_c = F.normalize(z_c, dim=-1)

        loss = 0.0
        num_pairs = 0

        for zi, zj in [(z_a, z_c), (z_b, z_c), (z_a, z_b)]:
            # Use FP32 for logits computation to avoid overflow
            logits = (zi.float() @ zj.float().T) / tau.float()

            loss += (
                F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing) +
                F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
            ) / 2
            num_pairs += 1

        return loss / num_pairs

    def infonce_2way(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        2-way InfoNCE loss for single modality pair.
        """
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
        """
        InfoNCE with hard negative emphasis.

        Hard negatives get scaled logits (effectively lower temperature).

        Args:
            z_geo: Geometry embeddings (B, d)
            z_text: Text embeddings (B, d)
            hard_neg_indices: List of hard negative indices per sample
            tau: Base temperature
            hard_neg_scale: Scale factor for hard negative logits

        Returns:
            Loss with hard negative emphasis
        """
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
                        # Only scale if index is valid for this batch size
                        if neg_idx != i and neg_idx < B:
                            logits[i, neg_idx] = logits[i, neg_idx] * hard_neg_scale

        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

    def grounding_distillation_loss(
        self,
        G_self: torch.Tensor,
        G_guided: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence between self-grounding and text-guided grounding.

        Self-grounding should attend to similar regions as text-grounding.

        Args:
            G_self: Self-grounding matrix (B, K, N)
            G_guided: Text-guided grounding matrix (B, K, N), detached

        Returns:
            KL divergence loss
        """
        # Clamp to avoid log(0)
        G_self_safe = G_self.clamp(min=1e-8)
        G_guided_safe = G_guided.clamp(min=1e-8)

        # KL divergence: D_KL(guided || self) = sum(guided * log(guided/self))
        # Using F.kl_div which expects log-probabilities for input
        loss = F.kl_div(
            G_self_safe.log(),
            G_guided_safe,
            reduction='batchmean'
        )

        return loss

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
            hard_negatives: Optional hard negative indices for detail loss
            stage: Training stage (1 or 2)

        Returns:
            total_loss: Scalar loss
            loss_dict: Individual losses for logging
        """
        losses = {}
        device = outputs['tau'].device
        tau = outputs['tau']

        # ─────────────────────────────────────────────────────────────────
        # 1. GUIDED CONTRASTIVE (Primary - what works)
        # ─────────────────────────────────────────────────────────────────

        z_brep = outputs.get('z_brep')
        z_pc = outputs.get('z_pc')
        z_text = outputs.get('z_text')

        # Handle cases where some modalities are missing
        if z_brep is not None and z_pc is not None and z_text is not None:
            losses['guided'] = self.infonce_3way(z_brep, z_pc, z_text, tau)
        elif z_brep is not None and z_text is not None:
            losses['guided'] = self.infonce_2way(z_brep, z_text, tau)
        elif z_pc is not None and z_text is not None:
            losses['guided'] = self.infonce_2way(z_pc, z_text, tau)
        else:
            losses['guided'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────
        # 2. SELF CONTRASTIVE (KEY FIX!)
        # ─────────────────────────────────────────────────────────────────

        z_brep_self = outputs.get('z_brep_self')
        z_pc_self = outputs.get('z_pc_self')

        # Self-encoded should ALSO align with text!
        # This is DIRECT retrieval training, not just distillation
        if z_brep_self is not None and z_pc_self is not None and z_text is not None:
            losses['self'] = self.infonce_3way(z_brep_self, z_pc_self, z_text, tau)
        elif z_brep_self is not None and z_text is not None:
            losses['self'] = self.infonce_2way(z_brep_self, z_text, tau)
        elif z_pc_self is not None and z_text is not None:
            losses['self'] = self.infonce_2way(z_pc_self, z_text, tau)
        else:
            losses['self'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────
        # 3. GROUNDING DISTILLATION (Auxiliary)
        # ─────────────────────────────────────────────────────────────────

        G_brep_guided = outputs.get('G_brep_guided')
        G_pc_guided = outputs.get('G_pc_guided')
        G_brep_self = outputs.get('G_brep_self')
        G_pc_self = outputs.get('G_pc_self')

        distill_losses = []

        if G_brep_guided is not None and G_brep_self is not None:
            loss_distill_brep = self.grounding_distillation_loss(
                G_brep_self, G_brep_guided.detach()
            )
            distill_losses.append(loss_distill_brep)

        if G_pc_guided is not None and G_pc_self is not None:
            loss_distill_pc = self.grounding_distillation_loss(
                G_pc_self, G_pc_guided.detach()
            )
            distill_losses.append(loss_distill_pc)

        if len(distill_losses) > 0:
            losses['distill'] = sum(distill_losses) / len(distill_losses)
        else:
            losses['distill'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────
        # 4. EMBEDDING DISTILLATION (KEY FIX!)
        # Force z_self to be close to z_guided
        # ─────────────────────────────────────────────────────────────────

        embed_distill_losses = []

        if z_brep is not None and z_brep_self is not None:
            loss_embed_brep = 1 - F.cosine_similarity(
                z_brep_self, z_brep.detach(), dim=-1
            ).mean()
            embed_distill_losses.append(loss_embed_brep)

        if z_pc is not None and z_pc_self is not None:
            loss_embed_pc = 1 - F.cosine_similarity(
                z_pc_self, z_pc.detach(), dim=-1
            ).mean()
            embed_distill_losses.append(loss_embed_pc)

        if len(embed_distill_losses) > 0:
            losses['embed_distill'] = sum(embed_distill_losses) / len(embed_distill_losses)
        else:
            losses['embed_distill'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────
        # 5. DETAIL CONTRASTIVE (Optional - for hard negatives)
        # ─────────────────────────────────────────────────────────────────

        if self.lambda_detail > 0:
            z_brep_d = outputs.get('z_brep_detail')
            z_pc_d = outputs.get('z_pc_detail')

            if z_brep_d is not None and z_text is not None:
                if hard_negatives is not None:
                    # Use harder temperature for detail loss
                    losses['detail'] = self.hard_negative_infonce(
                        z_brep_d, z_text, hard_negatives, tau * 0.7
                    )
                else:
                    losses['detail'] = self.infonce_2way(z_brep_d, z_text, tau * 0.7)
            elif z_pc_d is not None and z_text is not None:
                if hard_negatives is not None:
                    losses['detail'] = self.hard_negative_infonce(
                        z_pc_d, z_text, hard_negatives, tau * 0.7
                    )
                else:
                    losses['detail'] = self.infonce_2way(z_pc_d, z_text, tau * 0.7)
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
        lambda_detail: Optional[float] = None
    ):
        """Update loss weights (for stage transitions)."""
        if lambda_self is not None:
            self.lambda_self = lambda_self
        if lambda_distill is not None:
            self.lambda_distill = lambda_distill
        if lambda_embed_distill is not None:
            self.lambda_embed_distill = lambda_embed_distill
        if lambda_detail is not None:
            self.lambda_detail = lambda_detail


def compute_self_grounding_quality(
    z_guided: torch.Tensor,
    z_self: torch.Tensor
) -> float:
    """
    Compute cosine similarity between guided and self-grounded embeddings.

    This is the KEY metric for self-grounding quality.
    Target: > 0.9 after training

    Args:
        z_guided: Text-guided embeddings (B, d)
        z_self: Self-grounded embeddings (B, d)

    Returns:
        Average cosine similarity
    """
    z_guided = F.normalize(z_guided, dim=-1)
    z_self = F.normalize(z_self, dim=-1)
    cosine_sim = (z_guided * z_self).sum(dim=-1).mean()
    return cosine_sim.item()
