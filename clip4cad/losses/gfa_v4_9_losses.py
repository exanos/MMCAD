"""
GFA v4.9 Loss Functions - Simple InfoNCE

Key Design Principles:
1. Single clear objective - InfoNCE only
2. No competing losses (no code alignment, no gap closing, no ATP/CU)
3. MARGIN is the key metric: margin = pos_sim - neg_sim

The ONE metric that matters: MARGIN
    Healthy training:
      Epoch 1:  margin = 0.02  (barely above random)
      Epoch 5:  margin = 0.15  (learning!)
      Epoch 10: margin = 0.30  (good separation)
      Epoch 20: margin = 0.50+ (strong discrimination)

    ALARM: margin stays near 0 for >3 epochs -> model collapsed
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GFAv49Loss(nn.Module):
    """
    v4.9.2 Loss - Pure InfoNCE with strong uniformity.

    KEY INSIGHT from v4.9.1 failure:
    - Training margin = 0.42 but Eval margin = 0.00
    - Cosine alignment loss caused batch-level overfitting
    - Model learned to push ALL texts to similar positions

    v4.9.2 FIXES:
    1. NO cosine alignment (removed - it caused collapse)
    2. STRONG uniformity (5x higher weight)
    3. LOW label smoothing (10x lower)
    4. PURE InfoNCE focus
    5. Variance logging (catch collapse early)

    Stages:
    - Stage 0: BRep ↔ PC symmetric InfoNCE + uniformity
    - Stage 1: 3-way InfoNCE with strong uniformity
    - Stage 2: Same + hard negative boosting
    """

    def __init__(
        self,
        label_smoothing: float = 0.01,      # 10x lower (was 0.1)
        cosine_weight: float = 0.0,          # REMOVED (was 0.5)
        uniformity_weight: float = 0.5,      # 5x higher (was 0.1)
        text_weight: float = 1.5,            # Slightly higher (was 2.0)
        variance_weight: float = 0.1,        # NEW: explicit variance reg
    ):
        """
        Args:
            label_smoothing: Label smoothing for cross-entropy (low!)
            cosine_weight: DEPRECATED - kept for compatibility
            uniformity_weight: Weight for uniformity loss (HIGH!)
            text_weight: Multiplier for text-related losses
            variance_weight: Weight for variance regularization
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.cosine_weight = cosine_weight  # Ignored in v4.9.2
        self.uniformity_weight = uniformity_weight
        self.text_weight = text_weight
        self.variance_weight = variance_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        stage: int = 1,
        hard_negatives: Optional[List] = None,
        hard_neg_boost: float = 1.5,
        epoch: int = 1,
        total_epochs: int = 20,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss based on training stage.

        Args:
            outputs: Model outputs dict with z_brep, z_pc, (z_text)
            stage: Training stage (0, 1, or 2)
            hard_negatives: List of hard negative indices (Stage 2)
            hard_neg_boost: Boost factor for hard negatives
            epoch: Current epoch (for curriculum)
            total_epochs: Total epochs in stage (for curriculum)

        Returns:
            total_loss: Scalar loss
            losses: Dict of individual loss components
        """
        if stage == 0:
            return self._stage0_loss(outputs, epoch, total_epochs)
        elif stage == 1:
            return self._stage1_loss(outputs, epoch, total_epochs)
        else:
            return self._stage2_loss(outputs, hard_negatives, hard_neg_boost)

    def _variance_loss(self, z: torch.Tensor, min_var: float = 0.5) -> torch.Tensor:
        """
        Variance regularization - prevent embeddings from collapsing.

        If variance < min_var, penalize. This explicitly prevents all
        embeddings from becoming identical.
        """
        var = z.var(dim=0).mean()  # Variance across batch, mean across dims
        return F.relu(min_var - var)  # Only penalize if below threshold

    def _stage0_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        epoch: int = 1,
        total_epochs: int = 8,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Stage 0: BRep ↔ PC symmetric InfoNCE (v4.9.2 - no cosine alignment).
        Goal: Anchor BRep to pre-trained PC encoder using ONLY InfoNCE.
        """
        losses = {}

        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)
        tau = outputs['tau']

        B = z_brep.shape[0]
        labels = torch.arange(B, device=z_brep.device)

        # Symmetric InfoNCE (THE ONLY contrastive signal)
        logits_b2p = (z_brep.float() @ z_pc.float().T) / tau
        logits_p2b = logits_b2p.T

        loss_b2p = F.cross_entropy(logits_b2p, labels, label_smoothing=self.label_smoothing)
        loss_p2b = F.cross_entropy(logits_p2b, labels, label_smoothing=self.label_smoothing)

        losses['infonce'] = (loss_b2p + loss_p2b) / 2

        # Strong uniformity loss (prevent collapse)
        losses['uniformity'] = self._uniformity_loss(z_brep)

        # Variance regularization (explicit collapse prevention)
        losses['variance'] = self._variance_loss(z_brep)

        # Total (NO cosine, NO MSE - pure InfoNCE + regularization)
        losses['total'] = (
            losses['infonce'] +
            self.uniformity_weight * losses['uniformity'] +
            self.variance_weight * losses['variance']
        )

        # Metrics
        with torch.no_grad():
            sim_matrix = z_brep @ z_pc.T
            pos_sim = sim_matrix.diag().mean()
            neg_mask = ~torch.eye(B, dtype=torch.bool, device=z_brep.device)
            neg_sim = sim_matrix[neg_mask].mean()
            losses['margin'] = pos_sim - neg_sim
            losses['pos_sim'] = pos_sim
            losses['neg_sim'] = neg_sim

            # Variance tracking (should stay > 0.3)
            losses['z_brep_var'] = z_brep.var(dim=0).mean()
            losses['z_pc_var'] = z_pc.var(dim=0).mean()

        return losses['total'], losses

    def _uniformity_loss(self, z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
        """
        Uniformity loss to prevent representation collapse.
        Encourages embeddings to be uniformly distributed on hypersphere.
        """
        z = F.normalize(z, dim=-1)
        sq_pdist = torch.cdist(z, z, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    def _stage1_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        epoch: int = 1,
        total_epochs: int = 20,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Stage 1: 3-way alignment with PURE InfoNCE (v4.9.2).

        KEY CHANGE: NO cosine alignment - it caused batch-level collapse!

        The model should learn discrimination ONLY through InfoNCE.
        Uniformity and variance losses prevent collapse.
        """
        losses = {}

        z_text = F.normalize(outputs['z_text'], dim=-1)
        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)
        tau = outputs['tau']

        B = z_text.shape[0]
        labels = torch.arange(B, device=z_text.device)
        neg_mask = ~torch.eye(B, dtype=torch.bool, device=z_text.device)

        # ============================================================
        # CONTRASTIVE LOSSES (PURE InfoNCE - no shortcuts!)
        # ============================================================

        # Text ↔ BRep
        logits_t2b = (z_text.float() @ z_brep.float().T) / tau
        loss_t2b = (
            F.cross_entropy(logits_t2b, labels, label_smoothing=self.label_smoothing) +
            F.cross_entropy(logits_t2b.T, labels, label_smoothing=self.label_smoothing)
        ) / 2

        # Text ↔ PC
        logits_t2p = (z_text.float() @ z_pc.float().T) / tau
        loss_t2p = (
            F.cross_entropy(logits_t2p, labels, label_smoothing=self.label_smoothing) +
            F.cross_entropy(logits_t2p.T, labels, label_smoothing=self.label_smoothing)
        ) / 2

        # BRep ↔ PC
        logits_b2p = (z_brep.float() @ z_pc.float().T) / tau
        loss_b2p = (
            F.cross_entropy(logits_b2p, labels, label_smoothing=self.label_smoothing) +
            F.cross_entropy(logits_b2p.T, labels, label_smoothing=self.label_smoothing)
        ) / 2

        losses['infonce_t2b'] = loss_t2b
        losses['infonce_t2p'] = loss_t2p
        losses['infonce_b2p'] = loss_b2p

        # Weighted sum (text gets slightly more weight)
        losses['infonce'] = (
            self.text_weight * loss_t2b +
            self.text_weight * loss_t2p +
            loss_b2p
        ) / (2 * self.text_weight + 1)

        # ============================================================
        # REGULARIZATION (prevent collapse)
        # ============================================================

        # Strong uniformity
        losses['uniformity_text'] = self._uniformity_loss(z_text)
        losses['uniformity_brep'] = self._uniformity_loss(z_brep)
        losses['uniformity_pc'] = self._uniformity_loss(z_pc)

        # Variance regularization (explicit)
        losses['variance_text'] = self._variance_loss(z_text)
        losses['variance_brep'] = self._variance_loss(z_brep)

        # ============================================================
        # TOTAL LOSS (NO cosine alignment!)
        # ============================================================
        losses['total'] = (
            losses['infonce'] +
            self.uniformity_weight * (
                losses['uniformity_text'] +
                losses['uniformity_brep'] +
                losses['uniformity_pc']
            ) / 3 +
            self.variance_weight * (
                losses['variance_text'] +
                losses['variance_brep']
            ) / 2
        )

        # ============================================================
        # METRICS (expanded for debugging)
        # ============================================================
        with torch.no_grad():
            # Text-BRep margin
            sim_tb = z_text @ z_brep.T
            pos_sim_tb = sim_tb.diag().mean()
            neg_sim_tb = sim_tb[neg_mask].mean()
            losses['margin_tb'] = pos_sim_tb - neg_sim_tb
            losses['pos_sim_tb'] = pos_sim_tb
            losses['neg_sim_tb'] = neg_sim_tb

            # Text-PC margin
            sim_tp = z_text @ z_pc.T
            pos_sim_tp = sim_tp.diag().mean()
            neg_sim_tp = sim_tp[neg_mask].mean()
            losses['margin_tp'] = pos_sim_tp - neg_sim_tp

            # BRep-PC margin
            sim_bp = z_brep @ z_pc.T
            pos_sim_bp = sim_bp.diag().mean()
            neg_sim_bp = sim_bp[neg_mask].mean()
            losses['margin_bp'] = pos_sim_bp - neg_sim_bp

            # Average margin
            losses['margin'] = (losses['margin_tb'] + losses['margin_tp'] + losses['margin_bp']) / 3

            # CRITICAL: Track variance (should stay > 0.3)
            losses['z_text_var'] = z_text.var(dim=0).mean()
            losses['z_brep_var'] = z_brep.var(dim=0).mean()
            losses['z_pc_var'] = z_pc.var(dim=0).mean()

            # Track uniformity values (should be negative, lower = more uniform)
            losses['uniformity'] = (
                losses['uniformity_text'] +
                losses['uniformity_brep'] +
                losses['uniformity_pc']
            ) / 3

        return losses['total'], losses

    def _stage2_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        hard_negatives: Optional[List] = None,
        hard_neg_boost: float = 1.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Stage 2: 3-way InfoNCE + hard negative boosting.
        Goal: Fine-tune with hard negatives for better discrimination.
        """
        # Start with Stage 1 loss
        total_loss, losses = self._stage1_loss(outputs)

        # Add hard negative loss if provided
        if hard_negatives is not None:
            z_brep = F.normalize(outputs['z_brep'], dim=-1)
            z_text = F.normalize(outputs['z_text'], dim=-1)
            tau = outputs['tau']

            B = z_brep.shape[0]
            labels = torch.arange(B, device=z_brep.device)
            logits = z_brep @ z_text.T / (tau * 0.8)  # Slightly lower temp for hard negs

            # Boost hard negatives
            for i, negs in enumerate(hard_negatives):
                if negs is not None:
                    for neg_idx in negs:
                        if neg_idx < B and neg_idx != i:
                            logits[i, neg_idx] *= hard_neg_boost

            loss_hard = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            losses['hard_neg'] = loss_hard

            # Combine
            total_loss = total_loss + 0.3 * loss_hard
            losses['total'] = total_loss
        else:
            losses['hard_neg'] = torch.tensor(0.0, device=outputs['z_brep'].device)

        return total_loss, losses


def compute_retrieval_metrics(
    z_query: torch.Tensor,
    z_gallery: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10)
) -> Dict[str, float]:
    """
    Compute retrieval metrics (Recall@K).

    Args:
        z_query: Query embeddings (N, d)
        z_gallery: Gallery embeddings (N, d)
        ks: Tuple of K values for R@K

    Returns:
        Dict with R@K for each k
    """
    z_query = F.normalize(z_query, dim=-1)
    z_gallery = F.normalize(z_gallery, dim=-1)

    N = z_query.shape[0]
    sim = z_query @ z_gallery.T  # (N, N)

    # Get ranks
    ranks = (sim.argsort(dim=1, descending=True) == torch.arange(N, device=sim.device).unsqueeze(1)).float().argmax(dim=1)

    metrics = {}
    for k in ks:
        metrics[f'R@{k}'] = (ranks < k).float().mean().item() * 100

    # MRR
    metrics['MRR'] = (1.0 / (ranks.float() + 1)).mean().item()

    return metrics


def compute_batch_r1(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """
    Compute R@1 within a single batch - useful for early collapse detection.

    If batch R@1 is near 1/B (random chance), the model is not discriminating.
    If batch R@1 is 0%, something is very wrong.

    Args:
        z_a: First modality embeddings (B, d)
        z_b: Second modality embeddings (B, d)

    Returns:
        R@1 as a percentage
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)

    B = z_a.shape[0]
    sim = z_a @ z_b.T  # (B, B)

    # For each row, check if diagonal is the max
    max_indices = sim.argmax(dim=1)
    correct = (max_indices == torch.arange(B, device=z_a.device)).float()

    return correct.mean().item() * 100


def diagnose_embeddings(
    z_text: torch.Tensor,
    z_brep: torch.Tensor,
    z_pc: torch.Tensor,
) -> Dict[str, float]:
    """
    Comprehensive embedding diagnostics - use this to catch collapse.

    Returns dict with:
    - variance: Should be > 0.3 for each modality
    - uniformity: Should be < -1.0 for well-spread embeddings
    - batch_r1: Should be > 1/B (random chance)
    - margin: pos_sim - neg_sim
    """
    z_text = F.normalize(z_text, dim=-1)
    z_brep = F.normalize(z_brep, dim=-1)
    z_pc = F.normalize(z_pc, dim=-1)

    B = z_text.shape[0]
    random_chance = 100.0 / B

    diagnostics = {
        # Variance (should be > 0.3)
        'var_text': z_text.var(dim=0).mean().item(),
        'var_brep': z_brep.var(dim=0).mean().item(),
        'var_pc': z_pc.var(dim=0).mean().item(),

        # Batch R@1 (should be >> random chance)
        'r1_tb': compute_batch_r1(z_text, z_brep),
        'r1_tp': compute_batch_r1(z_text, z_pc),
        'r1_bp': compute_batch_r1(z_brep, z_pc),
        'random_chance': random_chance,
    }

    # Margins
    neg_mask = ~torch.eye(B, dtype=torch.bool, device=z_text.device)

    for name, (za, zb) in [('tb', (z_text, z_brep)), ('tp', (z_text, z_pc)), ('bp', (z_brep, z_pc))]:
        sim = za @ zb.T
        pos = sim.diag().mean().item()
        neg = sim[neg_mask].mean().item()
        diagnostics[f'margin_{name}'] = pos - neg
        diagnostics[f'pos_{name}'] = pos
        diagnostics[f'neg_{name}'] = neg

    return diagnostics


def compute_contrastive_quality(
    z_a: torch.Tensor,
    z_b: torch.Tensor
) -> Dict[str, float]:
    """
    Compute contrastive quality metrics.

    The key metric is MARGIN = pos_sim - neg_sim.
    - margin > 0.3 by epoch 10 is healthy
    - margin > 0.5 by epoch 20 is good
    - margin near 0 for >3 epochs = collapsed

    Args:
        z_a: First modality embeddings (B, d)
        z_b: Second modality embeddings (B, d)

    Returns:
        Dict with pos_sim, neg_sim, margin
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)

    B = z_a.shape[0]
    sim = z_a @ z_b.T

    pos_sim = sim.diag().mean().item()
    neg_mask = ~torch.eye(B, dtype=torch.bool, device=z_a.device)
    neg_sim = sim[neg_mask].mean().item()
    margin = pos_sim - neg_sim

    return {
        'pos_sim': pos_sim,
        'neg_sim': neg_sim,
        'margin': margin,
    }


def mine_hard_negatives_simple(
    model: nn.Module,
    dataloader,
    device: torch.device,
    top_k: int = 5,
    max_batches: int = 50,
    remap_fn=None,
) -> Dict[int, List[int]]:
    """
    Mine hard negatives based on embedding similarity.

    Hard negatives are samples that are very similar to each other
    but are not true pairs.

    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device
        top_k: Number of hard negatives per sample
        max_batches: Max batches to process
        remap_fn: Optional batch remapping function

    Returns:
        Dict mapping sample index to list of hard negative indices
    """
    print("Mining hard negatives by embedding similarity...")

    model.eval()
    all_z_brep = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if remap_fn is not None:
                batch = remap_fn(batch)

            outputs = model(batch, stage=1)
            z_brep = F.normalize(outputs['z_brep'], dim=-1).cpu()
            all_z_brep.append(z_brep)

            batch_size = z_brep.shape[0]
            start_idx = batch_idx * batch_size
            all_indices.extend(range(start_idx, start_idx + batch_size))

    z_brep = torch.cat(all_z_brep, dim=0)
    N = z_brep.shape[0]

    # Find hard negatives (high similarity, not true pair)
    sim = z_brep @ z_brep.T

    # Mask diagonal (true pairs)
    sim.fill_diagonal_(-float('inf'))

    # Get top-k most similar (hard negatives)
    _, hard_neg_indices = sim.topk(top_k, dim=1)

    hard_negatives = {}
    for i in range(N):
        hard_negatives[all_indices[i]] = hard_neg_indices[i].tolist()

    print(f"Mined hard negatives for {len(hard_negatives)} samples")

    model.train()
    return hard_negatives


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
            return step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def get_param_groups_with_lr(
    model,
    base_lr: float = 1e-4,
    text_lr_mult: float = 3.0,
    pool_lr_mult: float = 2.0,
    weight_decay: float = 0.01,
) -> List[Dict]:
    """
    Create parameter groups with different learning rates.

    The text encoder needs higher LR because:
    1. Text features (3072-dim) need more transformation
    2. Text modality starts further from geometry space

    Args:
        model: CLIP4CAD_GFA_v49 model
        base_lr: Base learning rate for BRep/PC encoders
        text_lr_mult: Multiplier for text encoder LR (default 3x)
        pool_lr_mult: Multiplier for attention pooling LR (default 2x)
        weight_decay: Weight decay

    Returns:
        List of parameter group dicts for optimizer
    """
    text_params = []
    pool_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'text_encoder' in name or 'text_proj' in name or 'text_pool' in name:
            text_params.append(param)
        elif 'pool' in name:
            pool_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': text_params, 'lr': base_lr * text_lr_mult, 'weight_decay': weight_decay},
        {'params': pool_params, 'lr': base_lr * pool_lr_mult, 'weight_decay': weight_decay},
    ]

    # Print summary
    print(f"Parameter groups:")
    print(f"  BRep/PC encoders: {len(other_params)} params, LR={base_lr}")
    print(f"  Text encoder: {len(text_params)} params, LR={base_lr * text_lr_mult}")
    print(f"  Attention pools: {len(pool_params)} params, LR={base_lr * pool_lr_mult}")

    return param_groups


__all__ = [
    'GFAv49Loss',
    'compute_retrieval_metrics',
    'compute_contrastive_quality',
    'compute_batch_r1',
    'diagnose_embeddings',
    'mine_hard_negatives_simple',
    'get_warmup_cosine_scheduler',
    'get_param_groups_with_lr',
]
