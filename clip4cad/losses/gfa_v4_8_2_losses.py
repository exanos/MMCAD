"""
GFA v4.8.2 Loss Functions - Smooth Curriculum Training

Key improvements over v4.8.1:
1. Smooth cosine -> KL blend (no hard switch)
2. Configurable cosine weight that decays over epochs
3. ATP ramps up gradually (not sudden)
4. CU ramps up over warmup
5. Adaptive hard negative boost (1.1-1.5x based on similarity)
6. Temperature annealing (tau: 0.07 -> 0.05)
7. Label smoothing 0.1 in all stages

Three training stages with smooth transitions:
- Stage 0: Anchor BRep to PC (smooth cosine emphasis)
- Stage 1: Add Text + Codebook (gradual code loss ramp-up)
- Stage 2: Gap Closing (ATP/CU/hard_neg ramp-up)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GFAv482LossSmooth(nn.Module):
    """
    Smooth curriculum loss for GFA v4.8.2.

    Key differences from GFAv481Loss:
    1. Smooth transitions between loss components
    2. Dynamic weight scheduling based on epoch within stage
    3. Temperature annealing
    4. Adaptive hard negative boosting
    5. Label smoothing in all stages
    """

    def __init__(
        self,
        lambda_recon: float = 0.5,
        lambda_align: float = 0.5,
        lambda_uniform: float = 0.3,
        lambda_code: float = 0.3,
        lambda_diversity: float = 0.1,
        lambda_hard_neg: float = 0.3,
        label_smoothing: float = 0.1,
        tau_start: float = 0.07,
        tau_end: float = 0.05,
    ):
        """
        Args:
            lambda_recon: Weight for reconstruction loss
            lambda_align: Weight for alignment loss
            lambda_uniform: Weight for centroid uniformity loss
            lambda_code: Weight for code alignment loss
            lambda_diversity: Weight for diversity loss
            lambda_hard_neg: Weight for hard negative loss (Stage 2)
            label_smoothing: Label smoothing for contrastive loss (default 0.1)
            tau_start: Initial temperature (default 0.07)
            tau_end: Final temperature (default 0.05)
        """
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_align = lambda_align
        self.lambda_uniform = lambda_uniform
        self.lambda_code = lambda_code
        self.lambda_diversity = lambda_diversity
        self.lambda_hard_neg = lambda_hard_neg
        self.label_smoothing = label_smoothing
        self.tau_start = tau_start
        self.tau_end = tau_end

        # Stage configuration
        self.stage0_epochs = 12
        self.stage1_epochs = 15
        self.stage2_epochs = 10

        # Warmup epochs within stages
        self.stage1_code_warmup = 3  # Code loss ramps up over 3 epochs
        self.stage2_atp_warmup = 2   # ATP/CU ramps up over 2 epochs

    def get_temperature(self, global_epoch: int, total_epochs: int) -> float:
        """
        Anneal temperature from tau_start to tau_end over training.

        Only starts annealing after Stage 0 to avoid destabilizing early training.

        Args:
            global_epoch: Current epoch across all stages
            total_epochs: Total training epochs
        """
        # Don't anneal during Stage 0 - keep tau_start
        if global_epoch <= self.stage0_epochs:
            return self.tau_start

        # Anneal only during Stages 1 and 2
        remaining_epochs = total_epochs - self.stage0_epochs
        progress_in_stages = min(1.0, (global_epoch - self.stage0_epochs) / max(1, remaining_epochs))

        # Slower annealing: only go to 0.06 instead of 0.05, and use sqrt for slower decay
        tau_end_adjusted = 0.06  # Less aggressive than 0.05
        tau = tau_end_adjusted + (self.tau_start - tau_end_adjusted) * (1 - math.sqrt(progress_in_stages))
        return max(tau, tau_end_adjusted)

    def get_stage_weights(self, epoch_in_stage: int, stage: int) -> Dict[str, float]:
        """
        Get smoothly interpolated loss weights based on epoch within stage.

        Args:
            epoch_in_stage: Epoch number within current stage (1-indexed)
            stage: Training stage (0, 1, or 2)

        Returns:
            Dictionary of loss weights for this epoch
        """
        weights = {
            'contrastive': 1.0,
            'cosine': 1.0,
            'align': self.lambda_align,
            'code': 0.0,
            'diversity': self.lambda_diversity,
            'uniform': 0.0,
            'hard_neg': 0.0,
            'recon': self.lambda_recon,
            'atp': 0.0,
        }

        if stage == 0:
            # Stage 0: Cosine 1.0 -> 0.5, contrastive 0.5 -> 1.0
            progress = min(1.0, epoch_in_stage / self.stage0_epochs)
            weights['cosine'] = 1.0 - 0.5 * progress  # 1.0 -> 0.5
            weights['contrastive'] = 0.5 + 0.5 * progress  # 0.5 -> 1.0
            weights['align'] = self.lambda_align

        elif stage == 1:
            # Stage 1: Code loss ramps 0 -> full over warmup epochs
            code_progress = min(1.0, epoch_in_stage / self.stage1_code_warmup)
            weights['code'] = self.lambda_code * code_progress

            # Cosine -> KL blend (CONSERVATIVE: don't go full KL, it's unstable)
            # Cosine weight: 0.5 -> 0.35 over stage (keep more cosine for stability)
            stage_progress = min(1.0, epoch_in_stage / self.stage1_epochs)
            weights['cosine'] = 0.5 - 0.15 * stage_progress  # 0.5 -> 0.35 (was 0.2)

            # Use soft code alignment (cosine) early, blend in KL later
            # CAP at 0.4 to prevent KL explosion in later epochs
            weights['use_kl_blend'] = min(0.4, stage_progress * 0.5)  # 0 -> 0.4 max (was 0 -> 1.0)

        elif stage == 2:
            # Stage 2: ATP/CU/hard_neg ramp 0 -> full over warmup
            atp_progress = min(1.0, epoch_in_stage / self.stage2_atp_warmup)
            weights['atp'] = self.lambda_align * atp_progress
            weights['uniform'] = self.lambda_uniform * atp_progress
            weights['hard_neg'] = self.lambda_hard_neg * atp_progress

            # Full code loss
            weights['code'] = self.lambda_code
            weights['use_kl_blend'] = 1.0  # Full KL

            # Cosine minimal
            weights['cosine'] = 0.1

        return weights

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        stage: int,
        epoch_in_stage: int = 1,
        global_epoch: int = 1,
        total_epochs: int = 37,
        hard_negatives: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss based on training stage with smooth transitions.

        Args:
            outputs: Model outputs dict
            stage: Training stage (0, 1, or 2)
            epoch_in_stage: Current epoch within stage (1-indexed)
            global_epoch: Global epoch across all stages
            total_epochs: Total training epochs
            hard_negatives: List of hard negative indices (Stage 2)

        Returns:
            total_loss: Scalar loss
            losses: Dict of individual loss components
        """
        # Get dynamic weights for this epoch
        weights = self.get_stage_weights(epoch_in_stage, stage)

        # Get annealed temperature
        tau_annealed = self.get_temperature(global_epoch, total_epochs)

        if stage == 0:
            return self._stage0_loss(outputs, weights, tau_annealed)
        elif stage == 1:
            return self._stage1_loss(outputs, weights, tau_annealed)
        else:
            return self._stage2_loss(outputs, weights, tau_annealed, hard_negatives)

    def _stage0_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        weights: Dict[str, float],
        tau: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Stage 0: Anchor BRep to PC with smooth cosine emphasis.

        Changes from v4.8.1:
        - Label smoothing enabled
        - Cosine weight decays 1.0 -> 0.5 over stage
        - Contrastive ramps up 0.5 -> 1.0
        """
        losses = {}
        device = outputs['tau'].device

        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)

        B = z_brep.shape[0]
        labels = torch.arange(B, device=device)

        # Contrastive: BRep -> PC
        logits = (z_brep.float() @ z_pc.detach().float().T) / tau
        losses['contrastive'] = F.cross_entropy(
            logits, labels, label_smoothing=self.label_smoothing
        )

        # Alignment: MSE
        z_brep_raw = outputs['z_brep_raw']
        z_pc_raw = outputs['z_pc_raw']
        losses['align'] = (z_brep_raw - z_pc_raw.detach()).pow(2).sum(dim=-1).mean()

        # Cosine similarity
        cosine_sim = F.cosine_similarity(z_brep_raw, z_pc_raw.detach(), dim=-1)
        losses['cosine'] = (1 - cosine_sim).mean()

        # Reconstruction
        losses['recon'] = F.mse_loss(outputs['recon'], outputs['recon_target'])

        # Total with dynamic weights
        losses['total'] = (
            weights['contrastive'] * losses['contrastive'] +
            weights['align'] * losses['align'] +
            weights['cosine'] * losses['cosine'] +
            weights['recon'] * losses['recon']
        )

        return losses['total'], losses

    def _stage1_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        weights: Dict[str, float],
        tau: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Stage 1: 3-way contrastive + smooth code alignment.

        Changes from v4.8.1:
        - Code loss ramps up over 3 epochs
        - Smooth cosine -> KL blend
        - Label smoothing enabled
        """
        losses = {}
        device = outputs['tau'].device

        z_text = F.normalize(outputs['z_text'], dim=-1)
        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)

        z_text_raw = F.normalize(outputs['z_text_raw'], dim=-1)
        z_brep_raw = F.normalize(outputs['z_brep_raw'], dim=-1)
        z_pc_raw = F.normalize(outputs['z_pc_raw'], dim=-1)

        # 3-way contrastive
        losses['contrastive'] = self._infonce_3way(z_text, z_brep, z_pc, tau)

        # Direct cosine alignment
        cos_tb = (z_text_raw * z_brep_raw).sum(dim=-1).mean()
        cos_tp = (z_text_raw * z_pc_raw).sum(dim=-1).mean()
        cos_bp = (z_brep_raw * z_pc_raw).sum(dim=-1).mean()
        losses['cosine'] = (3 - cos_tb - cos_tp - cos_bp) / 3

        # Code alignment with cosine -> KL blend
        w_text = outputs['w_text']
        w_brep = outputs['w_brep']
        w_pc = outputs['w_pc']

        kl_blend = weights.get('use_kl_blend', 0.0)

        code_loss_cosine = 0.0
        code_loss_kl = 0.0

        for level in ['category', 'type']:
            # Cosine-based code alignment (soft)
            w_t = F.normalize(w_text[level], dim=-1)
            w_b = F.normalize(w_brep[level], dim=-1)
            w_p = F.normalize(w_pc[level], dim=-1)

            cos_tb_code = (w_t * w_b).sum(dim=-1).mean()
            cos_tp_code = (w_t * w_p).sum(dim=-1).mean()
            cos_bp_code = (w_b * w_p).sum(dim=-1).mean()
            code_loss_cosine += (3 - cos_tb_code - cos_tp_code - cos_bp_code) / 3

            # KL-based code alignment (hard)
            kl_brep = F.kl_div(
                (w_brep[level] + 1e-8).log(),
                w_text[level].detach(),
                reduction='batchmean'
            )
            kl_pc = F.kl_div(
                (w_pc[level] + 1e-8).log(),
                w_text[level].detach(),
                reduction='batchmean'
            )
            code_loss_kl += (kl_brep + kl_pc) / 2

        code_loss_cosine /= 2
        code_loss_kl = (code_loss_kl / 2).clamp(max=2.0)  # Aggressive clamp (was 10.0)

        # Blend cosine -> KL (cosine-dominant for stability)
        losses['code'] = (1 - kl_blend) * code_loss_cosine + kl_blend * code_loss_kl
        losses['code'] = losses['code'].clamp(max=1.5)  # Overall clamp on code loss

        # Diversity
        avg_cat = (
            w_text['category'].mean(0) +
            w_brep['category'].mean(0) +
            w_pc['category'].mean(0)
        ) / 3
        entropy = -(avg_cat * (avg_cat + 1e-8).log()).sum()
        max_entropy = math.log(avg_cat.shape[0])
        losses['diversity'] = 1 - (entropy / max_entropy)

        # Reconstruction (reduced)
        losses['recon'] = F.mse_loss(outputs['recon'], outputs['recon_target'])

        # Total with dynamic weights
        losses['total'] = (
            weights['contrastive'] * losses['contrastive'] +
            weights['cosine'] * losses['cosine'] +
            weights['code'] * losses['code'] +
            weights['diversity'] * losses['diversity'] +
            weights['recon'] * 0.1 * losses['recon']
        )

        return losses['total'], losses

    def _stage2_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        weights: Dict[str, float],
        tau: float,
        hard_negatives: Optional[List] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Stage 2: Gap closing with smooth ramp-up.

        Changes from v4.8.1:
        - ATP/CU/hard_neg ramp up over 2 epochs
        - Adaptive hard negative boost (1.1-1.5x based on similarity)
        - Label smoothing enabled
        """
        losses = {}
        device = outputs['tau'].device

        z_text = F.normalize(outputs['z_text'], dim=-1)
        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)

        z_text_raw = outputs['z_text_raw']
        z_brep_raw = outputs['z_brep_raw']
        z_pc_raw = outputs['z_pc_raw']

        # 3-way contrastive
        losses['contrastive'] = self._infonce_3way(z_text, z_brep, z_pc, tau)

        # ATP: Align True Pairs
        align_brep = (z_brep_raw - z_text_raw.detach()).pow(2).sum(dim=-1).mean()
        align_pc = (z_pc_raw - z_text_raw.detach()).pow(2).sum(dim=-1).mean()
        losses['align'] = (align_brep + align_pc) / 2

        # CU: Centroid Uniformity
        centroids = (z_text_raw + z_brep_raw + z_pc_raw) / 3
        losses['uniform'] = self._centroid_uniformity(centroids)

        # Code alignment (full KL)
        w_text = outputs['w_text']
        w_brep = outputs['w_brep']
        w_pc = outputs['w_pc']

        code_loss = 0.0
        for level in ['category', 'type', 'spatial']:
            kl_brep = F.kl_div(
                (w_brep[level] + 1e-8).log(),
                w_text[level].detach(),
                reduction='batchmean'
            )
            kl_pc = F.kl_div(
                (w_pc[level] + 1e-8).log(),
                w_text[level].detach(),
                reduction='batchmean'
            )
            code_loss += (kl_brep + kl_pc) / 2
        losses['code'] = (code_loss / 3).clamp(max=10.0)

        # Diversity
        avg_cat = (
            w_text['category'].mean(0) +
            w_brep['category'].mean(0) +
            w_pc['category'].mean(0)
        ) / 3
        entropy = -(avg_cat * (avg_cat + 1e-8).log()).sum()
        max_entropy = math.log(avg_cat.shape[0])
        losses['diversity'] = 1 - (entropy / max_entropy)

        # Adaptive hard negatives
        if weights['hard_neg'] > 0 and hard_negatives is not None:
            losses['hard_neg'] = self._adaptive_hard_negative_loss(
                z_brep, z_text, hard_negatives, tau * 0.7
            )
        else:
            losses['hard_neg'] = torch.tensor(0.0, device=device)

        # Total with dynamic weights
        losses['total'] = (
            weights['contrastive'] * losses['contrastive'] +
            weights['atp'] * losses['align'] +
            weights['uniform'] * losses['uniform'] +
            weights['code'] * losses['code'] +
            weights['diversity'] * losses['diversity'] +
            weights['hard_neg'] * losses['hard_neg']
        )

        return losses['total'], losses

    def _infonce_3way(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        z_c: torch.Tensor,
        tau: float
    ) -> torch.Tensor:
        """3-way InfoNCE loss with label smoothing."""
        B = z_a.shape[0]
        device = z_a.device
        labels = torch.arange(B, device=device)

        loss = 0.0
        for zi, zj in [(z_a, z_c), (z_b, z_c), (z_a, z_b)]:
            logits = (zi.float() @ zj.float().T) / tau
            loss += (
                F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing) +
                F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
            ) / 2

        return loss / 3

    def _centroid_uniformity(self, centroids: torch.Tensor) -> torch.Tensor:
        """Centroid uniformity loss using RBF kernel."""
        B = centroids.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=centroids.device)

        dists_sq = torch.cdist(centroids, centroids, p=2).pow(2)
        mask = ~torch.eye(B, dtype=torch.bool, device=centroids.device)
        rbf = torch.exp(-2 * dists_sq)
        rbf_sum = rbf[mask].sum()

        return torch.log(rbf_sum / (B * (B - 1)) + 1e-8)

    def _adaptive_hard_negative_loss(
        self,
        z_geo: torch.Tensor,
        z_text: torch.Tensor,
        hard_negatives: List,
        tau: float
    ) -> torch.Tensor:
        """
        Adaptive hard negative contrastive loss.

        Boost factor is 1.1-1.5x based on similarity (more similar = higher boost).
        """
        B = z_geo.shape[0]
        labels = torch.arange(B, device=z_geo.device)
        logits = z_geo @ z_text.T / tau

        # Compute similarity for adaptive boost
        with torch.no_grad():
            sim_matrix = z_geo @ z_text.T

        for i, negs in enumerate(hard_negatives):
            if negs is not None:
                for neg_idx in negs:
                    if neg_idx < B and neg_idx != i:
                        # Adaptive boost: 1.1 to 1.5 based on similarity
                        sim = sim_matrix[i, neg_idx].item()
                        # Higher similarity -> higher boost
                        boost = 1.1 + 0.4 * max(0, min(1, (sim + 1) / 2))
                        logits[i, neg_idx] *= boost

        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)


# =============================================================================
# Re-export v4.8.1 utilities for compatibility
# =============================================================================
from clip4cad.losses.gfa_v4_8_1_losses import (
    GFAv481Loss as GFAv482Loss,  # Legacy alias
    compute_modality_gap,
    compute_true_pair_cosine,
    compute_brep_pc_metrics,
    compute_code_diversity,
    compute_active_codes,
    mine_hard_negatives_by_code,
)


def get_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.01
):
    """
    Create a learning rate scheduler with linear warmup followed by cosine decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        steps_per_epoch: Number of steps per epoch
        min_lr_ratio: Minimum LR as ratio of initial LR (default 0.01)

    Returns:
        LambdaLR scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


__all__ = [
    'GFAv482LossSmooth',
    'GFAv482Loss',  # Legacy alias
    'compute_modality_gap',
    'compute_true_pair_cosine',
    'compute_brep_pc_metrics',
    'compute_code_diversity',
    'compute_active_codes',
    'mine_hard_negatives_by_code',
    'get_warmup_cosine_scheduler',
]
