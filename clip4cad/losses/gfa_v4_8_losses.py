"""
GFA v4.8 Loss Functions - Gap-Closing with Shared Fusion

6 well-motivated loss terms:

1. L_contrastive: 3-way InfoNCE (preserves retrieval)
   - Standard contrastive loss for cross-modal alignment
   - Ensures relative ranking is preserved

2. L_ATP (Align True Pairs): CLOSES THE GAP!
   - ||z_brep - z_text||² + ||z_pc - z_text||²
   - Text is anchor modality - geometry pulled toward it
   - Applied to FINAL embeddings (after SharedFusionNetwork)
   - This is the KEY insight from the paper!

3. L_CU (Centroid Uniformity): Prevents collapse
   - log(Σ_{i≠j} exp(-2||μ_i - μ_j||²))
   - μ_i = (z_text_i + z_brep_i + z_pc_i) / 3
   - Pushes sample centroids apart

4. L_code: Code Activation Alignment
   - KL(w_brep || w_text) + KL(w_pc || w_text)
   - Matched samples should activate same codes

5. L_diversity: Codebook Utilization
   - -Entropy(avg_usage)
   - Encourages using all codes, prevents dead codes

6. L_hard_neg: Hard Negative Mining (Stage 2 only)
   - Boosts hard negatives in contrastive loss
   - Helps fine-grained discrimination
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GFAv48Loss(nn.Module):
    """
    Loss for GFA v4.8 - Gap-Closing Codebook Architecture.

    Unlike v4.4:
    - No self-grounding losses (embed, grounding, query losses)
    - Direct alignment loss instead
    - Simpler, more principled
    """

    def __init__(
        self,
        lambda_atp: float = 0.5,
        lambda_cu: float = 0.3,
        lambda_code: float = 0.3,
        lambda_diversity: float = 0.1,
        lambda_hard_neg: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            lambda_atp: Weight for ATP (Align True Pairs) loss - closes gap
            lambda_cu: Weight for CU (Centroid Uniformity) loss - prevents collapse
            lambda_code: Weight for code alignment loss
            lambda_diversity: Weight for diversity loss
            lambda_hard_neg: Weight for hard negative loss (Stage 2)
            label_smoothing: Label smoothing for contrastive loss
        """
        super().__init__()
        self.lambda_atp = lambda_atp
        self.lambda_cu = lambda_cu
        self.lambda_code = lambda_code
        self.lambda_diversity = lambda_diversity
        self.lambda_hard_neg = lambda_hard_neg
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
    ) -> tuple:
        """
        Compute total loss and individual components.

        Args:
            outputs: Model outputs dict with:
                - z_text, z_brep, z_pc: Final embeddings (after SharedFusionNetwork)
                - H_text, H_brep, H_pc: Per-code features
                - w_text, w_brep, w_pc: Code activation weights
                - tau: Temperature
            hard_negatives: List of hard negative indices per sample (Stage 2)

        Returns:
            total_loss: Scalar loss
            losses: Dict of individual loss components
        """
        losses = {}
        tau = outputs['tau']
        device = tau.device

        # Final embeddings (after SharedFusionNetwork)
        z_text = outputs['z_text']
        z_brep = outputs['z_brep']
        z_pc = outputs['z_pc']

        # Normalized embeddings for contrastive
        z_text_n = F.normalize(z_text, dim=-1)
        z_brep_n = F.normalize(z_brep, dim=-1)
        z_pc_n = F.normalize(z_pc, dim=-1)

        # Code activation weights
        w_text = outputs['w_text']
        w_brep = outputs['w_brep']
        w_pc = outputs['w_pc']

        # ─────────────────────────────────────────────────────────────────────
        # 1. CONTRASTIVE LOSS (preserves retrieval)
        # ─────────────────────────────────────────────────────────────────────
        losses['contrastive'] = self._infonce_3way(z_text_n, z_brep_n, z_pc_n, tau)

        # ─────────────────────────────────────────────────────────────────────
        # 2. ATP: ALIGN TRUE PAIRS (CLOSES THE GAP!)
        # Text is anchor - pull geometry toward it
        # Applied to FINAL embeddings (after SharedFusionNetwork)
        # ─────────────────────────────────────────────────────────────────────
        # Detach text so only geometry moves toward text
        atp_brep = (z_brep - z_text.detach()).pow(2).sum(dim=-1).mean()
        atp_pc = (z_pc - z_text.detach()).pow(2).sum(dim=-1).mean()
        losses['atp'] = (atp_brep + atp_pc) / 2

        # ─────────────────────────────────────────────────────────────────────
        # 3. CU: CENTROID UNIFORMITY (prevents collapse)
        # Push sample centroids apart
        # ─────────────────────────────────────────────────────────────────────
        centroids = (z_text + z_brep + z_pc) / 3  # (B, d)
        losses['cu'] = self._centroid_uniformity(centroids)

        # ─────────────────────────────────────────────────────────────────────
        # 4. CODE ACTIVATION ALIGNMENT
        # Matched samples should activate same codes
        # ─────────────────────────────────────────────────────────────────────
        # KL divergence: geometry codes should match text codes
        kl_brep = F.kl_div(
            (w_brep + 1e-8).log(),
            w_text.detach(),
            reduction='batchmean'
        )
        kl_pc = F.kl_div(
            (w_pc + 1e-8).log(),
            w_text.detach(),
            reduction='batchmean'
        )
        losses['code'] = (kl_brep + kl_pc) / 2

        # Clamp code loss to prevent explosion
        losses['code'] = losses['code'].clamp(max=10.0)

        # ─────────────────────────────────────────────────────────────────────
        # 5. CODEBOOK DIVERSITY (use all codes)
        # ─────────────────────────────────────────────────────────────────────
        # Average code usage across batch and modalities
        avg_usage = (w_text.mean(0) + w_brep.mean(0) + w_pc.mean(0)) / 3  # (M,)

        # Entropy of usage (higher = more uniform = better)
        entropy = -(avg_usage * (avg_usage + 1e-8).log()).sum()
        max_entropy = math.log(avg_usage.shape[0])

        # Diversity loss: penalize low entropy (concentrated usage)
        losses['diversity'] = 1 - (entropy / max_entropy)

        # ─────────────────────────────────────────────────────────────────────
        # 6. HARD NEGATIVE LOSS (Stage 2)
        # ─────────────────────────────────────────────────────────────────────
        if self.lambda_hard_neg > 0 and hard_negatives is not None:
            losses['hard_neg'] = self._hard_negative_loss(
                z_brep_n, z_text_n, hard_negatives, tau * 0.7
            )
        else:
            losses['hard_neg'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # TOTAL
        # ─────────────────────────────────────────────────────────────────────
        losses['total'] = (
            1.0 * losses['contrastive'] +
            self.lambda_atp * losses['atp'] +
            self.lambda_cu * losses['cu'] +
            self.lambda_code * losses['code'] +
            self.lambda_diversity * losses['diversity'] +
            self.lambda_hard_neg * losses['hard_neg']
        )

        return losses['total'], losses

    def _infonce_3way(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        z_c: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        3-way InfoNCE loss for (BRep, PC, Text) alignment.

        Computes symmetric loss for all three pairs:
        - Text <-> BRep
        - Text <-> PC
        - BRep <-> PC
        """
        B = z_a.shape[0]
        device = z_a.device
        labels = torch.arange(B, device=device)

        loss = 0.0
        for zi, zj in [(z_a, z_c), (z_b, z_c), (z_a, z_b)]:
            # Cosine similarity matrix scaled by temperature
            logits = (zi.float() @ zj.float().T) / tau.float()

            # Symmetric cross-entropy
            loss += (
                F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing) +
                F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
            ) / 2

        return loss / 3

    def _centroid_uniformity(self, centroids: torch.Tensor) -> torch.Tensor:
        """
        Centroid uniformity loss using RBF kernel.

        From the paper: log(Σ_{i≠j} exp(-2||μ_i - μ_j||²))

        This pushes sample centroids apart, preventing collapse
        where all samples map to the same point.

        Args:
            centroids: (B, d) - Per-sample centroids across modalities

        Returns:
            Uniformity loss (lower = more uniform distribution)
        """
        B = centroids.shape[0]

        if B <= 1:
            return torch.tensor(0.0, device=centroids.device)

        # Pairwise squared distances
        dists_sq = torch.cdist(centroids, centroids, p=2).pow(2)  # (B, B)

        # RBF kernel (exclude diagonal)
        mask = ~torch.eye(B, dtype=torch.bool, device=centroids.device)
        rbf = torch.exp(-2 * dists_sq)
        rbf_sum = rbf[mask].sum()

        # Normalize and return log
        return torch.log(rbf_sum / (B * (B - 1)) + 1e-8)

    def _hard_negative_loss(
        self,
        z_geo: torch.Tensor,
        z_text: torch.Tensor,
        hard_negatives: List,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Hard negative contrastive loss.

        Boosts importance of hard negatives in the contrastive denominator.

        Args:
            z_geo: (B, d) - Geometry embeddings
            z_text: (B, d) - Text embeddings
            hard_negatives: List of hard negative indices per sample
            tau: Temperature (typically lower than main loss)

        Returns:
            Hard negative loss
        """
        B = z_geo.shape[0]
        labels = torch.arange(B, device=z_geo.device)
        logits = z_geo @ z_text.T / tau

        # Boost hard negatives (2x multiplier)
        for i, negs in enumerate(hard_negatives):
            if negs is not None:
                for neg_idx in negs:
                    if neg_idx < B and neg_idx != i:
                        logits[i, neg_idx] *= 2.0

        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)


# =============================================================================
# Metric Functions
# =============================================================================

def compute_modality_gap(
    z_text: torch.Tensor,
    z_brep: torch.Tensor,
    z_pc: torch.Tensor
) -> tuple:
    """
    Compute modality gap (distance between modality centroids).

    The gap should approach 0 as training progresses with L_align.

    Args:
        z_text, z_brep, z_pc: (B, d) embeddings (raw, before projection)

    Returns:
        gap_brep: Distance from text centroid to brep centroid
        gap_pc: Distance from text centroid to pc centroid
    """
    with torch.no_grad():
        c_text = z_text.mean(0)
        c_brep = z_brep.mean(0)
        c_pc = z_pc.mean(0)

        gap_brep = (c_brep - c_text).norm().item()
        gap_pc = (c_pc - c_text).norm().item()

    return gap_brep, gap_pc


def compute_true_pair_cosine(
    z_text: torch.Tensor,
    z_brep: torch.Tensor,
    z_pc: torch.Tensor
) -> tuple:
    """
    Compute mean cosine similarity of matched pairs.

    Should approach 1.0 as training progresses.

    Args:
        z_text, z_brep, z_pc: (B, d) embeddings

    Returns:
        cos_brep: Mean cosine(z_text[i], z_brep[i])
        cos_pc: Mean cosine(z_text[i], z_pc[i])
    """
    with torch.no_grad():
        cos_brep = F.cosine_similarity(z_text, z_brep, dim=-1).mean().item()
        cos_pc = F.cosine_similarity(z_text, z_pc, dim=-1).mean().item()

    return cos_brep, cos_pc


def compute_code_diversity(w_text: torch.Tensor, w_brep: torch.Tensor, w_pc: torch.Tensor) -> float:
    """
    Compute code diversity (entropy of average usage).

    Higher = better (using more codes).

    Args:
        w_text, w_brep, w_pc: (B, M) code activation weights

    Returns:
        Diversity score (0 to 1, higher is better)
    """
    with torch.no_grad():
        avg_usage = (w_text.mean(0) + w_brep.mean(0) + w_pc.mean(0)) / 3
        entropy = -(avg_usage * (avg_usage + 1e-8).log()).sum()
        max_entropy = math.log(avg_usage.shape[0])
        diversity = (entropy / max_entropy).item()

    return diversity


def mine_hard_negatives_by_code(
    model,
    dataloader,
    device,
    top_k: int = 10,
    max_batches: int = 50
) -> Dict[int, List[int]]:
    """
    Mine hard negatives based on code activation similarity.

    Samples that activate similar codes but are different instances
    are good hard negatives (semantically similar but not the same).

    Args:
        model: GFA v4.8 model
        dataloader: DataLoader to process
        device: Device to use
        top_k: Number of hard negatives per sample
        max_batches: Maximum batches to process

    Returns:
        Dict mapping sample index -> list of hard negative indices
    """
    print("Mining hard negatives by code activation...")

    model.eval()
    all_w_brep = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            outputs = model(batch)
            w_brep = outputs['w_brep']  # (B, M)

            # Track global indices
            batch_size = w_brep.shape[0]
            start_idx = batch_idx * batch_size
            indices = list(range(start_idx, start_idx + batch_size))

            all_w_brep.append(w_brep.cpu())
            all_indices.extend(indices)

    # Concatenate all code weights
    W = torch.cat(all_w_brep, dim=0)  # (N, M)
    N = W.shape[0]

    # Compute pairwise code similarity (dot product of normalized weights)
    W_norm = F.normalize(W, dim=-1)
    sim_matrix = W_norm @ W_norm.T  # (N, N)

    # For each sample, find top-k most similar (excluding self)
    hard_negatives = {}
    for i in range(N):
        sim_i = sim_matrix[i].clone()
        sim_i[i] = -float('inf')  # Exclude self

        _, top_indices = sim_i.topk(top_k)
        hard_negatives[all_indices[i]] = top_indices.tolist()

    print(f"Mined hard negatives for {N} samples")
    model.train()

    return hard_negatives
