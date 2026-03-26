"""
GFA v4.8.3 Loss Functions - Fixed Cross-Modal Alignment

Key fixes over v4.8.2:
1. BIDIRECTIONAL alignment (both text→geo and geo→text get gradients)
2. Stronger text-centric contrastive loss (text is anchor, not just participant)
3. Proper gradient flow - no broken detach patterns
4. Simplified 2-stage training (BRep-PC alignment integrated, not separate)
5. Lower temperature (0.05) for sharper contrastive learning
6. Explicit cross-modal metrics for debugging

Two training stages:
- Stage 0: 3-way alignment from start (text, brep, pc together)
- Stage 1: Gap closing with hard negatives (refinement)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GFAv483Loss(nn.Module):
    """
    Fixed cross-modal alignment loss for GFA v4.8.3.

    Key differences from v4.8.2:
    1. All three modalities trained together from Stage 0
    2. Bidirectional alignment (no broken detach patterns)
    3. Text-anchor contrastive losses for stronger text alignment
    4. Lower temperature for sharper learning
    """

    def __init__(
        self,
        tau: float = 0.05,  # Lower than v4.8.2's 0.07
        lambda_align: float = 1.0,  # Stronger alignment
        lambda_code: float = 0.2,  # Reduced code loss
        lambda_diversity: float = 0.1,
        lambda_recon: float = 0.1,  # Reduced reconstruction
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.tau = tau
        self.lambda_align = lambda_align
        self.lambda_code = lambda_code
        self.lambda_diversity = lambda_diversity
        self.lambda_recon = lambda_recon
        self.label_smoothing = label_smoothing

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        stage: int = 0,
        epoch: int = 1,
        hard_negatives: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss based on training stage.

        Stage 0: Full 3-way alignment (all modalities together)
        Stage 1: Gap closing with hard negatives
        """
        if stage == 0:
            return self._stage0_loss(outputs, epoch)
        else:
            return self._stage1_loss(outputs, epoch, hard_negatives)

    def _stage0_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Stage 0: Full 3-way alignment from the start.

        All three modalities learn together, no pre-training.
        """
        losses = {}

        # Get normalized embeddings
        z_text = F.normalize(outputs['z_text'], dim=-1)
        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)

        B = z_text.shape[0]
        device = z_text.device
        labels = torch.arange(B, device=device)

        # =========================================
        # CORE: Text-Anchored Contrastive Losses
        # =========================================
        # Text is the semantic anchor - BRep and PC must align to it

        # Text -> BRep (text queries brep)
        logits_t2b = (z_text @ z_brep.T) / self.tau
        loss_t2b = F.cross_entropy(logits_t2b, labels, label_smoothing=self.label_smoothing)

        # BRep -> Text (brep queries text)
        loss_b2t = F.cross_entropy(logits_t2b.T, labels, label_smoothing=self.label_smoothing)

        # Text -> PC
        logits_t2p = (z_text @ z_pc.T) / self.tau
        loss_t2p = F.cross_entropy(logits_t2p, labels, label_smoothing=self.label_smoothing)

        # PC -> Text
        loss_p2t = F.cross_entropy(logits_t2p.T, labels, label_smoothing=self.label_smoothing)

        # BRep <-> PC (geometric consistency)
        logits_b2p = (z_brep @ z_pc.T) / self.tau
        loss_b2p = F.cross_entropy(logits_b2p, labels, label_smoothing=self.label_smoothing)
        loss_p2b = F.cross_entropy(logits_b2p.T, labels, label_smoothing=self.label_smoothing)

        # Weighted sum - text alignment gets 2x weight
        losses['contrastive'] = (
            2.0 * (loss_t2b + loss_b2t) / 2 +  # Text-BRep (2x)
            2.0 * (loss_t2p + loss_p2t) / 2 +  # Text-PC (2x)
            1.0 * (loss_b2p + loss_p2b) / 2    # BRep-PC (1x)
        ) / 5.0

        # =========================================
        # BIDIRECTIONAL Direct Alignment (MSE)
        # =========================================
        # Both directions get gradients - no detach!
        z_text_raw = outputs['z_text_raw']
        z_brep_raw = outputs['z_brep_raw']
        z_pc_raw = outputs['z_pc_raw']

        # Text-BRep alignment (bidirectional)
        align_tb = F.mse_loss(z_text_raw, z_brep_raw)

        # Text-PC alignment (bidirectional)
        align_tp = F.mse_loss(z_text_raw, z_pc_raw)

        # BRep-PC alignment (for geometric consistency)
        align_bp = F.mse_loss(z_brep_raw, z_pc_raw)

        losses['align'] = (align_tb + align_tp + align_bp) / 3

        # =========================================
        # Cosine Similarity (for monitoring)
        # =========================================
        with torch.no_grad():
            cos_tb = F.cosine_similarity(z_text_raw, z_brep_raw, dim=-1).mean()
            cos_tp = F.cosine_similarity(z_text_raw, z_pc_raw, dim=-1).mean()
            cos_bp = F.cosine_similarity(z_brep_raw, z_pc_raw, dim=-1).mean()
            losses['cos_text_brep'] = cos_tb
            losses['cos_text_pc'] = cos_tp
            losses['cos_brep_pc'] = cos_bp

        # =========================================
        # Code Alignment (simplified)
        # =========================================
        if 'w_text' in outputs:
            w_text = outputs['w_text']
            w_brep = outputs['w_brep']
            w_pc = outputs['w_pc']

            code_loss = 0.0
            for level in ['category', 'type']:
                # Cosine-based code alignment (soft, stable)
                w_t = F.normalize(w_text[level], dim=-1)
                w_b = F.normalize(w_brep[level], dim=-1)
                w_p = F.normalize(w_pc[level], dim=-1)

                cos_tb_code = (w_t * w_b).sum(dim=-1).mean()
                cos_tp_code = (w_t * w_p).sum(dim=-1).mean()
                cos_bp_code = (w_b * w_p).sum(dim=-1).mean()
                code_loss += (3 - cos_tb_code - cos_tp_code - cos_bp_code) / 3

            losses['code'] = code_loss / 2

            # Diversity
            avg_cat = (w_text['category'].mean(0) + w_brep['category'].mean(0) + w_pc['category'].mean(0)) / 3
            entropy = -(avg_cat * (avg_cat + 1e-8).log()).sum()
            max_entropy = math.log(avg_cat.shape[0])
            losses['diversity'] = 1 - (entropy / max_entropy)
        else:
            losses['code'] = torch.tensor(0.0, device=device)
            losses['diversity'] = torch.tensor(0.0, device=device)

        # =========================================
        # Reconstruction (minor)
        # =========================================
        if 'recon' in outputs and 'recon_target' in outputs:
            losses['recon'] = F.mse_loss(outputs['recon'], outputs['recon_target'])
        else:
            losses['recon'] = torch.tensor(0.0, device=device)

        # =========================================
        # Total Loss
        # =========================================
        losses['total'] = (
            1.0 * losses['contrastive'] +
            self.lambda_align * losses['align'] +
            self.lambda_code * losses['code'] +
            self.lambda_diversity * losses['diversity'] +
            self.lambda_recon * losses['recon']
        )

        return losses['total'], losses

    def _stage1_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        epoch: int,
        hard_negatives: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Stage 1: Gap closing with hard negatives.

        Same as Stage 0 but with hard negative mining.
        """
        # Get base Stage 0 loss
        total, losses = self._stage0_loss(outputs, epoch)

        if hard_negatives is not None and len(hard_negatives) > 0:
            z_text = F.normalize(outputs['z_text'], dim=-1)
            z_brep = F.normalize(outputs['z_brep'], dim=-1)

            hard_neg_loss = self._hard_negative_loss(z_text, z_brep, hard_negatives)
            losses['hard_neg'] = hard_neg_loss
            losses['total'] = total + 0.3 * hard_neg_loss
        else:
            losses['hard_neg'] = torch.tensor(0.0, device=outputs['z_text'].device)

        return losses['total'], losses

    def _hard_negative_loss(
        self,
        z_text: torch.Tensor,
        z_brep: torch.Tensor,
        hard_negatives: List,
    ) -> torch.Tensor:
        """Hard negative contrastive loss."""
        B = z_text.shape[0]
        device = z_text.device

        if not hard_negatives or len(hard_negatives) == 0:
            return torch.tensor(0.0, device=device)

        # Build hard negative indices
        hard_neg_mask = torch.zeros(B, B, device=device)
        for i, neg_indices in enumerate(hard_negatives[:B]):
            if neg_indices:
                for neg_idx in neg_indices[:5]:  # Top 5 hard negatives
                    if neg_idx < B:
                        hard_neg_mask[i, neg_idx] = 1.0

        if hard_neg_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Compute similarities
        logits = (z_text @ z_brep.T) / self.tau

        # Push down hard negatives
        hard_neg_logits = logits * hard_neg_mask
        loss = hard_neg_logits.sum() / hard_neg_mask.sum().clamp(min=1)

        return loss.clamp(min=0)


def compute_cross_modal_metrics(
    z_text: torch.Tensor,
    z_brep: torch.Tensor,
    z_pc: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute cross-modal alignment metrics for debugging.

    Returns cosine similarities between all pairs.
    """
    z_text = F.normalize(z_text, dim=-1)
    z_brep = F.normalize(z_brep, dim=-1)
    z_pc = F.normalize(z_pc, dim=-1)

    with torch.no_grad():
        # Pairwise cosine similarities (diagonal = matching pairs)
        cos_tb = (z_text * z_brep).sum(dim=-1).mean().item()
        cos_tp = (z_text * z_pc).sum(dim=-1).mean().item()
        cos_bp = (z_brep * z_pc).sum(dim=-1).mean().item()

        # Modality gap (L2 distance of mean embeddings)
        gap_tb = (z_text.mean(0) - z_brep.mean(0)).norm().item()
        gap_tp = (z_text.mean(0) - z_pc.mean(0)).norm().item()
        gap_bp = (z_brep.mean(0) - z_pc.mean(0)).norm().item()

    return {
        'cos_text_brep': cos_tb,
        'cos_text_pc': cos_tp,
        'cos_brep_pc': cos_bp,
        'gap_text_brep': gap_tb,
        'gap_text_pc': gap_tp,
        'gap_brep_pc': gap_bp,
    }


def compute_retrieval_metrics(
    z_query: torch.Tensor,
    z_gallery: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute retrieval metrics (R@k).

    Assumes z_query[i] should match z_gallery[i].
    """
    z_query = F.normalize(z_query, dim=-1)
    z_gallery = F.normalize(z_gallery, dim=-1)

    with torch.no_grad():
        # Similarity matrix
        sims = z_query @ z_gallery.T

        # Get ranking
        _, indices = sims.sort(dim=1, descending=True)

        # Ground truth: diagonal (i matches i)
        B = z_query.shape[0]
        gt = torch.arange(B, device=z_query.device)

        metrics = {}
        for k in k_values:
            # Check if GT is in top-k
            top_k = indices[:, :k]
            hits = (top_k == gt.unsqueeze(1)).any(dim=1).float()
            metrics[f'R@{k}'] = hits.mean().item() * 100

    return metrics


def diagnose_embeddings(
    z_text: torch.Tensor,
    z_brep: torch.Tensor,
    z_pc: torch.Tensor,
    num_samples: int = 5,
) -> Dict[str, float]:
    """
    Diagnose why retrieval might be failing.

    Returns statistics about embedding structure.
    """
    z_text = F.normalize(z_text, dim=-1)
    z_brep = F.normalize(z_brep, dim=-1)
    z_pc = F.normalize(z_pc, dim=-1)

    with torch.no_grad():
        # Similarity matrices
        sims_tb = z_text @ z_brep.T  # (N, N)
        sims_tp = z_text @ z_pc.T
        sims_bp = z_brep @ z_pc.T

        N = z_text.shape[0]

        # Diagonal (matching pairs) vs off-diagonal (non-matching)
        diag_tb = sims_tb.diag()
        diag_tp = sims_tp.diag()
        diag_bp = sims_bp.diag()

        # Off-diagonal mean (excluding diagonal)
        mask = ~torch.eye(N, dtype=torch.bool, device=z_text.device)
        offdiag_tb = sims_tb[mask].mean()
        offdiag_tp = sims_tp[mask].mean()
        offdiag_bp = sims_bp[mask].mean()

        # Within-modality similarity (collapse detection)
        sims_tt = z_text @ z_text.T
        sims_bb = z_brep @ z_brep.T
        within_text = sims_tt[mask].mean()
        within_brep = sims_bb[mask].mean()

        # Variance of embeddings
        var_text = z_text.var(dim=0).mean()
        var_brep = z_brep.var(dim=0).mean()

        results = {
            # Matching pair similarity (should be HIGH)
            'diag_text_brep': diag_tb.mean().item(),
            'diag_text_pc': diag_tp.mean().item(),
            'diag_brep_pc': diag_bp.mean().item(),
            # Non-matching similarity (should be LOW for good retrieval)
            'offdiag_text_brep': offdiag_tb.item(),
            'offdiag_text_pc': offdiag_tp.item(),
            'offdiag_brep_pc': offdiag_bp.item(),
            # Margin = diag - offdiag (should be POSITIVE and large)
            'margin_text_brep': (diag_tb.mean() - offdiag_tb).item(),
            'margin_text_pc': (diag_tp.mean() - offdiag_tp).item(),
            'margin_brep_pc': (diag_bp.mean() - offdiag_bp).item(),
            # Within-modality similarity (high = collapse)
            'within_text': within_text.item(),
            'within_brep': within_brep.item(),
            # Variance (low = collapse)
            'var_text': var_text.item(),
            'var_brep': var_brep.item(),
        }

        # Print diagnosis
        print("\n" + "="*60)
        print("EMBEDDING DIAGNOSIS")
        print("="*60)
        print(f"\nMatching pairs (diagonal) - should be HIGH:")
        print(f"  Text-BRep: {results['diag_text_brep']:.4f}")
        print(f"  Text-PC:   {results['diag_text_pc']:.4f}")
        print(f"  BRep-PC:   {results['diag_brep_pc']:.4f}")

        print(f"\nNon-matching pairs (off-diagonal) - should be LOW:")
        print(f"  Text-BRep: {results['offdiag_text_brep']:.4f}")
        print(f"  Text-PC:   {results['offdiag_text_pc']:.4f}")
        print(f"  BRep-PC:   {results['offdiag_brep_pc']:.4f}")

        print(f"\nMargin (diag - offdiag) - should be POSITIVE:")
        print(f"  Text-BRep: {results['margin_text_brep']:.4f}")
        print(f"  Text-PC:   {results['margin_text_pc']:.4f}")
        print(f"  BRep-PC:   {results['margin_brep_pc']:.4f}")

        print(f"\nWithin-modality similarity (HIGH = collapse):")
        print(f"  Text-Text: {results['within_text']:.4f}")
        print(f"  BRep-BRep: {results['within_brep']:.4f}")

        print(f"\nEmbedding variance (LOW = collapse):")
        print(f"  Text: {results['var_text']:.6f}")
        print(f"  BRep: {results['var_brep']:.6f}")

        if results['margin_text_brep'] < 0.01:
            print("\n⚠️  WARNING: Near-zero margin indicates EMBEDDING COLLAPSE!")
            print("    All embeddings are too similar - no discrimination between samples.")

        print("="*60)

    return results
