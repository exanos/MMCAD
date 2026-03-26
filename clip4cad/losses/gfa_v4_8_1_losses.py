"""
GFA v4.8.1 Loss Functions - Staged Training with Hierarchical Codebook

Three training stages:

Stage 0: Anchor BRep to PC
- L_contrastive: InfoNCE(z_brep, z_pc)
- L_align: MSE(z_brep, z_pc.detach())
- L_recon: MSE(recon, face_features)

Stage 1: Add Text + Codebook
- L_contrastive: 3-way InfoNCE
- L_code: Hierarchical KL divergence (per level)
- L_diversity: Entropy-based codebook utilization
- L_recon: MSE reconstruction (reduced weight)

Stage 2: Gap Closing + Hard Negatives
- All Stage 1 losses
- L_ATP: MSE alignment to text
- L_CU: Centroid uniformity
- L_hard_neg: Hard negative mining
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GFAv481Loss(nn.Module):
    """
    Staged loss for GFA v4.8.1.

    Stage 0: BRep-PC alignment + reconstruction (anchor phase)
    Stage 1: 3-way contrastive + code alignment (alignment phase)
    Stage 2: + gap closing + hard negatives (refinement phase)
    """

    def __init__(
        self,
        lambda_recon: float = 0.5,
        lambda_align: float = 0.5,
        lambda_uniform: float = 0.3,
        lambda_code: float = 0.3,
        lambda_diversity: float = 0.1,
        lambda_hard_neg: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            lambda_recon: Weight for reconstruction loss
            lambda_align: Weight for alignment loss (Stage 0: BRep-PC, Stage 2: ATP)
            lambda_uniform: Weight for centroid uniformity loss
            lambda_code: Weight for code alignment loss
            lambda_diversity: Weight for diversity loss
            lambda_hard_neg: Weight for hard negative loss (Stage 2)
            label_smoothing: Label smoothing for contrastive loss
        """
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_align = lambda_align
        self.lambda_uniform = lambda_uniform
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
        stage: int,
        hard_negatives: Optional[List] = None,
    ) -> tuple:
        """
        Compute loss based on training stage.

        Args:
            outputs: Model outputs dict
            stage: Training stage (0, 1, or 2)
            hard_negatives: List of hard negative indices (Stage 2)

        Returns:
            total_loss: Scalar loss
            losses: Dict of individual loss components
        """
        if stage == 0:
            return self._stage0_loss(outputs)
        elif stage == 1:
            return self._stage1_loss(outputs)
        else:
            return self._stage2_loss(outputs, hard_negatives)

    def _stage0_loss(self, outputs: Dict[str, torch.Tensor]) -> tuple:
        """
        Stage 0: Anchor BRep to PC.

        Goal: Make BRep encoder produce meaningful features by aligning to
        pre-trained PC encoder (ShapeLLM).

        Key: PC should be FROZEN, so we only train BRep to match PC.
        """
        losses = {}
        tau = outputs['tau']
        device = tau.device

        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)

        # ─────────────────────────────────────────────────────────────────────
        # CONTRASTIVE: BRep -> PC (one direction only since PC is frozen)
        # ─────────────────────────────────────────────────────────────────────
        B = z_brep.shape[0]
        labels = torch.arange(B, device=device)

        # Detach PC in contrastive loss (BRep learns to find its matching PC)
        logits = (z_brep.float() @ z_pc.detach().float().T) / tau.float()
        losses['contrastive'] = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        # ─────────────────────────────────────────────────────────────────────
        # ALIGNMENT: Pull BRep toward PC (MSE loss)
        # ─────────────────────────────────────────────────────────────────────
        z_brep_raw = outputs['z_brep_raw']
        z_pc_raw = outputs['z_pc_raw']

        # Detach PC so only BRep updates
        losses['align'] = (z_brep_raw - z_pc_raw.detach()).pow(2).sum(dim=-1).mean()

        # ─────────────────────────────────────────────────────────────────────
        # COSINE: Additional cosine similarity loss (encourages direction match)
        # ─────────────────────────────────────────────────────────────────────
        cosine_sim = F.cosine_similarity(z_brep_raw, z_pc_raw.detach(), dim=-1)
        losses['cosine'] = (1 - cosine_sim).mean()  # Minimize 1 - cos (maximize cos)

        # ─────────────────────────────────────────────────────────────────────
        # RECONSTRUCTION: Reconstruct face features
        # ─────────────────────────────────────────────────────────────────────
        recon = outputs['recon']
        target = outputs['recon_target']
        losses['recon'] = F.mse_loss(recon, target)

        # ─────────────────────────────────────────────────────────────────────
        # TOTAL (Stage 0: alignment and cosine are primary drivers)
        # ─────────────────────────────────────────────────────────────────────
        losses['total'] = (
            0.5 * losses['contrastive'] +  # Reduced weight
            self.lambda_align * losses['align'] +
            1.0 * losses['cosine'] +  # Strong cosine signal
            self.lambda_recon * losses['recon']
        )

        return losses['total'], losses

    def _stage1_loss(self, outputs: Dict[str, torch.Tensor]) -> tuple:
        """
        Stage 1: 3-way contrastive + soft code alignment.

        Goal: Learn codebook structure and establish relative alignment
        across all three modalities.

        Key insight: Contrastive loss is primary driver. Code alignment
        should be soft (cosine similarity) not hard (KL divergence).
        """
        losses = {}
        tau = outputs['tau']
        device = tau.device

        z_text = F.normalize(outputs['z_text'], dim=-1)
        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)

        # Also get raw (pre-projection) embeddings for direct alignment
        z_text_raw = F.normalize(outputs['z_text_raw'], dim=-1)
        z_brep_raw = F.normalize(outputs['z_brep_raw'], dim=-1)
        z_pc_raw = F.normalize(outputs['z_pc_raw'], dim=-1)

        # ─────────────────────────────────────────────────────────────────────
        # CONTRASTIVE: 3-way InfoNCE (PRIMARY LOSS)
        # ─────────────────────────────────────────────────────────────────────
        losses['contrastive'] = self._infonce_3way(z_text, z_brep, z_pc, tau)

        # ─────────────────────────────────────────────────────────────────────
        # DIRECT ALIGNMENT: Cosine similarity for true pairs
        # This is gentler than InfoNCE and helps with convergence
        # ─────────────────────────────────────────────────────────────────────
        cos_tb = (z_text_raw * z_brep_raw).sum(dim=-1).mean()
        cos_tp = (z_text_raw * z_pc_raw).sum(dim=-1).mean()
        cos_bp = (z_brep_raw * z_pc_raw).sum(dim=-1).mean()
        # Maximize cosine = minimize (1 - cosine)
        losses['align'] = (3 - cos_tb - cos_tp - cos_bp) / 3

        # ─────────────────────────────────────────────────────────────────────
        # CODE ALIGNMENT: Soft cosine similarity (not KL!)
        # ─────────────────────────────────────────────────────────────────────
        w_text = outputs['w_text']
        w_brep = outputs['w_brep']
        w_pc = outputs['w_pc']

        code_loss = 0.0
        for level in ['category', 'type']:  # Only coarse levels
            # Normalize code weights
            w_t = F.normalize(w_text[level], dim=-1)
            w_b = F.normalize(w_brep[level], dim=-1)
            w_p = F.normalize(w_pc[level], dim=-1)

            # Cosine similarity between matched pairs (maximize)
            cos_tb = (w_t * w_b).sum(dim=-1).mean()
            cos_tp = (w_t * w_p).sum(dim=-1).mean()
            cos_bp = (w_b * w_p).sum(dim=-1).mean()

            # Loss = 1 - average cosine
            code_loss += (3 - cos_tb - cos_tp - cos_bp) / 3

        losses['code'] = code_loss / 2  # Average over levels

        # ─────────────────────────────────────────────────────────────────────
        # DIVERSITY: Entropy-based codebook utilization
        # ─────────────────────────────────────────────────────────────────────
        avg_cat = (
            w_text['category'].mean(0) +
            w_brep['category'].mean(0) +
            w_pc['category'].mean(0)
        ) / 3
        entropy = -(avg_cat * (avg_cat + 1e-8).log()).sum()
        max_entropy = math.log(avg_cat.shape[0])
        losses['diversity'] = 1 - (entropy / max_entropy)

        # ─────────────────────────────────────────────────────────────────────
        # RECONSTRUCTION (reduced weight)
        # ─────────────────────────────────────────────────────────────────────
        losses['recon'] = F.mse_loss(outputs['recon'], outputs['recon_target'])

        # ─────────────────────────────────────────────────────────────────────
        # TOTAL: Contrastive + Alignment dominant, rest minor
        # ─────────────────────────────────────────────────────────────────────
        losses['total'] = (
            losses['contrastive'] +                    # ~3-6 typically
            self.lambda_align * losses['align'] +      # 0.5 * ~0.5 = 0.25
            self.lambda_code * losses['code'] +        # 0.3 * ~0.5 = 0.15
            self.lambda_diversity * losses['diversity'] +  # 0.1 * ~0.5 = 0.05
            self.lambda_recon * 0.1 * losses['recon']  # Very reduced
        )

        return losses['total'], losses

    def _stage2_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        hard_negatives: Optional[List] = None
    ) -> tuple:
        """
        Stage 2: Add gap closing + hard negatives.

        Goal: Close the absolute modality gap and improve fine-grained
        discrimination with hard negatives.
        """
        losses = {}
        tau = outputs['tau']
        device = tau.device

        z_text = F.normalize(outputs['z_text'], dim=-1)
        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)

        z_text_raw = outputs['z_text_raw']
        z_brep_raw = outputs['z_brep_raw']
        z_pc_raw = outputs['z_pc_raw']

        # ─────────────────────────────────────────────────────────────────────
        # CONTRASTIVE: 3-way InfoNCE
        # ─────────────────────────────────────────────────────────────────────
        losses['contrastive'] = self._infonce_3way(z_text, z_brep, z_pc, tau)

        # ─────────────────────────────────────────────────────────────────────
        # ATP: Align True Pairs (closes the gap!)
        # ─────────────────────────────────────────────────────────────────────
        align_brep = (z_brep_raw - z_text_raw.detach()).pow(2).sum(dim=-1).mean()
        align_pc = (z_pc_raw - z_text_raw.detach()).pow(2).sum(dim=-1).mean()
        losses['align'] = (align_brep + align_pc) / 2

        # ─────────────────────────────────────────────────────────────────────
        # CU: Centroid Uniformity (prevents collapse)
        # ─────────────────────────────────────────────────────────────────────
        centroids = (z_text_raw + z_brep_raw + z_pc_raw) / 3
        losses['uniform'] = self._centroid_uniformity(centroids)

        # ─────────────────────────────────────────────────────────────────────
        # CODE ALIGNMENT: Hierarchical KL divergence
        # ─────────────────────────────────────────────────────────────────────
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

        # ─────────────────────────────────────────────────────────────────────
        # DIVERSITY
        # ─────────────────────────────────────────────────────────────────────
        avg_cat = (
            w_text['category'].mean(0) +
            w_brep['category'].mean(0) +
            w_pc['category'].mean(0)
        ) / 3
        entropy = -(avg_cat * (avg_cat + 1e-8).log()).sum()
        max_entropy = math.log(avg_cat.shape[0])
        losses['diversity'] = 1 - (entropy / max_entropy)

        # ─────────────────────────────────────────────────────────────────────
        # HARD NEGATIVES
        # ─────────────────────────────────────────────────────────────────────
        if self.lambda_hard_neg > 0 and hard_negatives is not None:
            losses['hard_neg'] = self._hard_negative_loss(
                z_brep, z_text, hard_negatives, tau * 0.7
            )
        else:
            losses['hard_neg'] = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────────────
        # TOTAL
        # ─────────────────────────────────────────────────────────────────────
        losses['total'] = (
            losses['contrastive'] +
            self.lambda_align * losses['align'] +
            self.lambda_uniform * losses['uniform'] +
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
        3-way InfoNCE loss for (Text, BRep, PC) alignment.

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
            logits = (zi.float() @ zj.float().T) / tau.float()
            loss += (
                F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing) +
                F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
            ) / 2

        return loss / 3

    def _centroid_uniformity(self, centroids: torch.Tensor) -> torch.Tensor:
        """
        Centroid uniformity loss using RBF kernel.

        Pushes sample centroids apart, preventing collapse.
        """
        B = centroids.shape[0]

        if B <= 1:
            return torch.tensor(0.0, device=centroids.device)

        # Pairwise squared distances
        dists_sq = torch.cdist(centroids, centroids, p=2).pow(2)

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

    The gap should approach 0 as training progresses.
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
    """
    with torch.no_grad():
        cos_brep = F.cosine_similarity(z_text, z_brep, dim=-1).mean().item()
        cos_pc = F.cosine_similarity(z_text, z_pc, dim=-1).mean().item()

    return cos_brep, cos_pc


def compute_brep_pc_metrics(
    z_brep: torch.Tensor,
    z_pc: torch.Tensor
) -> tuple:
    """
    Compute BRep-PC alignment metrics (for Stage 0).

    Returns:
        gap: Distance between centroids
        cosine: Mean cosine similarity of matched pairs
    """
    with torch.no_grad():
        c_brep = z_brep.mean(0)
        c_pc = z_pc.mean(0)
        gap = (c_brep - c_pc).norm().item()
        cosine = F.cosine_similarity(z_brep, z_pc, dim=-1).mean().item()

    return gap, cosine


def compute_code_diversity(
    w_text: Dict[str, torch.Tensor],
    w_brep: Dict[str, torch.Tensor],
    w_pc: Dict[str, torch.Tensor],
    level: str = 'category'
) -> float:
    """
    Compute code diversity (entropy of average usage).

    Higher = better (using more codes).
    """
    with torch.no_grad():
        avg_usage = (
            w_text[level].mean(0) +
            w_brep[level].mean(0) +
            w_pc[level].mean(0)
        ) / 3

        entropy = -(avg_usage * (avg_usage + 1e-8).log()).sum()
        max_entropy = math.log(avg_usage.shape[0])
        diversity = (entropy / max_entropy).item()

    return diversity


def compute_active_codes(
    w_text: Dict[str, torch.Tensor],
    w_brep: Dict[str, torch.Tensor],
    w_pc: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute average number of active codes per modality and level.
    """
    with torch.no_grad():
        results = {}
        for level in ['category', 'type', 'variant', 'spatial']:
            n_text = (w_text[level] > 0).sum(dim=-1).float().mean().item()
            n_brep = (w_brep[level] > 0).sum(dim=-1).float().mean().item()
            n_pc = (w_pc[level] > 0).sum(dim=-1).float().mean().item()
            results[f'{level}_text'] = n_text
            results[f'{level}_brep'] = n_brep
            results[f'{level}_pc'] = n_pc
            results[f'{level}_avg'] = (n_text + n_brep + n_pc) / 3

    return results


def mine_hard_negatives_by_code(
    model,
    dataloader,
    device,
    top_k: int = 10,
    max_batches: int = 50,
    remap_fn=None
) -> Dict[int, List[int]]:
    """
    Mine hard negatives based on code activation similarity.

    Samples that activate similar codes but are different instances
    are good hard negatives (semantically similar but not the same).

    Args:
        remap_fn: Optional function to remap batch keys (e.g., remap_batch)
    """
    print("Mining hard negatives by code activation...")

    model.eval()
    all_w_cat = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if remap_fn is not None:
                batch = remap_fn(batch)

            outputs = model(batch, stage=1)
            w_cat = outputs['w_brep']['category']  # (B, n_cat)

            # Track global indices
            batch_size = w_cat.shape[0]
            start_idx = batch_idx * batch_size
            indices = list(range(start_idx, start_idx + batch_size))

            all_w_cat.append(w_cat.cpu())
            all_indices.extend(indices)

    # Concatenate all code weights (already on CPU)
    W = torch.cat(all_w_cat, dim=0)  # (N, n_cat)
    N = W.shape[0]

    # Clear GPU memory before CPU-intensive operations
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # Compute pairwise code similarity in chunks to save memory
    W_norm = F.normalize(W, dim=-1)

    hard_negatives = {}
    chunk_size = 1000  # Process in chunks to avoid OOM

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        # Compute similarity for this chunk against all samples
        sim_chunk = W_norm[i:end_i] @ W_norm.T  # (chunk, N)

        for j, global_j in enumerate(range(i, end_i)):
            sim_row = sim_chunk[j].clone()
            sim_row[global_j] = -float('inf')  # Exclude self
            _, top_indices = sim_row.topk(top_k)
            hard_negatives[all_indices[global_j]] = top_indices.tolist()

        del sim_chunk

    del W, W_norm, all_w_cat
    gc.collect()

    print(f"Mined hard negatives for {N} samples")
    model.train()

    # Final cleanup
    torch.cuda.empty_cache()

    return hard_negatives
