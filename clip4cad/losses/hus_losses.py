"""
HUS Loss: Simplified Hierarchical Contrastive Loss

This loss has only 3 terms (vs GFA's 8):
1. Unified (PRIMARY) - weight=1.0 (this is what retrieval uses!)
2. Global (regularizer) - weight=0.2 (ensures coarse structure)
3. Detail (regularizer) - weight=0.2 (ensures fine-grained, increases in stage 2)

Key insight: Unified is the only embedding used for retrieval.
Global/Detail are auxiliary losses to ensure the hierarchical structure exists.

No cross-level consistency loss (ablations showed it didn't help).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any


class HUSLoss(nn.Module):
    """
    Simplified hierarchical loss for CLIP4CAD-HUS.

    Unified is PRIMARY (what retrieval/generation uses).
    Global + Detail are mild regularizers.
    Detail weight increases in Stage 2 when hard negatives are used.

    v2: Added label smoothing for numerical stability.
    """

    def __init__(
        self,
        lambda_unified: float = 1.0,
        lambda_global: float = 0.2,
        lambda_detail: float = 0.2,
        label_smoothing: float = 0.05,  # Light smoothing (0.1 was too aggressive)
    ):
        """
        Args:
            lambda_unified: Weight for unified contrastive loss
            lambda_global: Weight for global contrastive loss
            lambda_detail: Weight for detail contrastive loss
            label_smoothing: Smoothing factor for cross-entropy (0.1 recommended)
        """
        super().__init__()
        self.lambda_unified = lambda_unified
        self.lambda_global = lambda_global
        self.lambda_detail = lambda_detail
        self.label_smoothing = label_smoothing

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        hard_negatives: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical contrastive loss.

        Args:
            outputs: Model outputs with keys:
                - z_brep, z_pc, z_text (unified)
                - z_brep_global, z_pc_global, z_text_global
                - z_brep_detail, z_pc_detail, z_text_detail
                - tau_unified, tau_global, tau_detail
            hard_negatives: Optional dict with hard negative info for detail level

        Returns:
            Dict with 'total', 'unified', 'global', 'detail' losses
        """
        losses = {}

        # =====================================================================
        # PRIMARY: Unified Contrastive
        # =====================================================================

        losses['unified'] = self.infonce_3way(
            outputs['z_brep'], outputs['z_pc'], outputs['z_text'],
            outputs['tau_unified']
        )

        # =====================================================================
        # REGULARIZER: Global Contrastive (coarse discrimination)
        # =====================================================================

        losses['global'] = self.infonce_3way(
            outputs['z_brep_global'], outputs['z_pc_global'], outputs['z_text_global'],
            outputs['tau_global']
        )

        # =====================================================================
        # REGULARIZER: Detail Contrastive (fine-grained, with optional hard negatives)
        # =====================================================================

        if hard_negatives is not None and 'indices' in hard_negatives:
            losses['detail'] = self.hard_negative_infonce(
                outputs['z_brep_detail'], outputs['z_pc_detail'], outputs['z_text_detail'],
                hard_negatives, outputs['tau_detail']
            )
        else:
            losses['detail'] = self.infonce_3way(
                outputs['z_brep_detail'], outputs['z_pc_detail'], outputs['z_text_detail'],
                outputs['tau_detail']
            )

        # =====================================================================
        # TOTAL: Unified dominant, others mild
        # =====================================================================

        losses['total'] = (
            self.lambda_unified * losses['unified'] +
            self.lambda_global * losses['global'] +
            self.lambda_detail * losses['detail']
        )

        return losses

    def infonce_3way(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        z_c: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard 3-way InfoNCE loss with numerical stability.

        Computes symmetric cross-entropy for all pairs:
        - z_a <-> z_c (brep-text)
        - z_b <-> z_c (pc-text)
        - z_a <-> z_b (brep-pc)

        Args:
            z_a: Embeddings (B, d) - typically B-Rep
            z_b: Embeddings (B, d) - typically PC
            z_c: Embeddings (B, d) - typically Text
            tau: Temperature scalar

        Returns:
            Scalar loss
        """
        # Cast to FP32 for numerical stability (critical for AMP training)
        z_a = F.normalize(z_a.float(), dim=-1)
        z_b = F.normalize(z_b.float(), dim=-1)
        z_c = F.normalize(z_c.float(), dim=-1)
        tau = tau.float().clamp(min=0.02)  # Min 0.02 for stability

        B = z_a.shape[0]
        labels = torch.arange(B, device=z_a.device)

        loss = 0
        # Text-centric pairs (more important, weighted 2x)
        for zi, zj in [(z_a, z_c), (z_b, z_c)]:
            logits = (zi @ zj.T / tau).clamp(-100, 100)
            # Symmetric cross-entropy with label smoothing
            loss += F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            loss += F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)

        # Geometry-geometry pair (less important, can be unstable)
        logits_geo = (z_a @ z_b.T / tau).clamp(-100, 100)
        loss += F.cross_entropy(logits_geo, labels, label_smoothing=self.label_smoothing)
        loss += F.cross_entropy(logits_geo.T, labels, label_smoothing=self.label_smoothing)

        return loss / 6  # Average over 3 pairs × 2 directions

    def hard_negative_infonce(
        self,
        z_brep: torch.Tensor,
        z_pc: torch.Tensor,
        z_text: torch.Tensor,
        hard_negs: Dict[str, Any],
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE with hard negative emphasis and numerical stability.

        For fine-grained discrimination (32 vs 64 teeth), we emphasize
        hard negatives by applying higher weight to their logits.

        Args:
            z_brep, z_pc, z_text: Embeddings (B, d)
            hard_negs: Dict with 'indices' (batch indices of hard negatives)
            tau: Temperature scalar

        Returns:
            Scalar loss
        """
        # Cast to FP32 for numerical stability
        z_brep = F.normalize(z_brep.float(), dim=-1)
        z_pc = F.normalize(z_pc.float(), dim=-1)
        z_text = F.normalize(z_text.float(), dim=-1)
        tau = tau.float().clamp(min=0.01)

        B = z_brep.shape[0]
        labels = torch.arange(B, device=z_brep.device)

        hard_neg_indices = hard_negs.get('indices', None)  # List of lists
        hard_neg_weight = hard_negs.get('weight', 2.0)  # Weight multiplier

        if hard_neg_indices is None:
            return self.infonce_3way(z_brep, z_pc, z_text, tau)

        loss = 0
        for zi, zj in [(z_brep, z_text), (z_pc, z_text), (z_brep, z_pc)]:
            # Compute logits with clamping
            logits = (zi @ zj.T / tau).clamp(-100, 100)

            # Apply hard negative weighting
            weighted_logits = logits.clone()
            for i in range(B):
                if i < len(hard_neg_indices) and hard_neg_indices[i] is not None:
                    negs = hard_neg_indices[i]
                    if len(negs) > 0:
                        # Convert to tensor if needed
                        if not isinstance(negs, torch.Tensor):
                            negs = torch.tensor(negs, device=logits.device, dtype=torch.long)
                        # Filter valid indices
                        negs = negs[negs < B]
                        if len(negs) > 0:
                            # Increase logit magnitude for hard negatives (makes them harder)
                            weighted_logits[i, negs] = logits[i, negs] * hard_neg_weight

            # Clamp weighted logits too
            weighted_logits = weighted_logits.clamp(-100, 100)

            # Symmetric cross-entropy with weighted logits and label smoothing
            loss += (F.cross_entropy(weighted_logits, labels, label_smoothing=self.label_smoothing) +
                     F.cross_entropy(weighted_logits.T, labels, label_smoothing=self.label_smoothing)) / 2

        return loss / 3

    def set_stage2_weights(
        self,
        lambda_unified: float = 1.0,
        lambda_global: float = 0.5,
        lambda_detail: float = 1.0
    ):
        """
        Update weights for Stage 2 training.

        Stage 2: Unified becomes primary, detail increased for fine-grained discrimination.

        Args:
            lambda_unified: Unified loss weight (typically 1.0 for stage 2)
            lambda_global: Global loss weight (typically 0.5 for stage 2)
            lambda_detail: Detail loss weight (typically 1.0 for stage 2)
        """
        self.lambda_unified = lambda_unified
        self.lambda_global = lambda_global
        self.lambda_detail = lambda_detail


class HUSLossConfig:
    """Configuration for HUS loss weights by training stage."""

    @staticmethod
    def stage1() -> Dict[str, float]:
        """Stage 1: Unified-dominant (learn the main retrieval space)."""
        return {
            'lambda_unified': 1.0,   # PRIMARY - what retrieval uses
            'lambda_global': 0.2,    # Mild regularizer
            'lambda_detail': 0.2,    # Mild regularizer
            'label_smoothing': 0.05, # Light smoothing
        }

    @staticmethod
    def stage2() -> Dict[str, float]:
        """Stage 2: Still unified-dominant, detail increased for hard negatives."""
        return {
            'lambda_unified': 1.0,   # Still primary
            'lambda_global': 0.1,    # Reduce further
            'lambda_detail': 0.5,    # Increase for fine-grained with hard negs
            'label_smoothing': 0.05,
        }
