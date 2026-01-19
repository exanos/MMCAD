"""
GFA (Grounded Feature Alignment) Loss Functions

Novel loss components for CLIP4CAD-GFA:
1. Grounding Consistency Loss - Same text should ground similarly across modalities
2. Grounding Diversity Loss - Different slots should attend to different regions
3. Local Contrastive Loss - Per-slot contrastive with confidence weighting
4. Global Contrastive Loss - InfoNCE across all modality pairs
5. Confidence Regularization - Encourage confident slot predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class GFALoss(nn.Module):
    """
    Combined loss function for CLIP4CAD-GFA training.

    Components:
    1. Global contrastive: Align global embeddings across modality pairs
    2. Local contrastive: Align grounded features per feature slot
    3. Grounding consistency: Same text should ground similarly across modalities
    4. Grounding diversity: Different slots should attend to different regions
    5. Confidence regularization: Encourage confident slot predictions
    """

    def __init__(
        self,
        lambda_global: float = 1.0,
        lambda_local: float = 0.5,
        lambda_consist: float = 0.5,
        lambda_diverse: float = 0.2,
        lambda_conf_reg: float = 0.1,
        lambda_global_stage1: float = 0.2,
        confidence_threshold: float = 0.3,
    ):
        """
        Args:
            lambda_global: Weight for global contrastive loss
            lambda_local: Weight for local (per-slot) contrastive loss
            lambda_consist: Weight for grounding consistency loss
            lambda_diverse: Weight for grounding diversity loss
            lambda_conf_reg: Weight for confidence regularization
            lambda_global_stage1: Reduced global weight for stage 1
            confidence_threshold: Threshold for active feature slots
        """
        super().__init__()

        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.lambda_consist = lambda_consist
        self.lambda_diverse = lambda_diverse
        self.lambda_conf_reg = lambda_conf_reg
        self.lambda_global_stage1 = lambda_global_stage1
        self.confidence_threshold = confidence_threshold

    def get_active_slots(
        self,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify active feature slots based on confidence threshold.

        Args:
            confidence: Slot confidences, (B, K)

        Returns:
            active_mask: Boolean mask, (B, K), True = active
            num_active: Number of active slots per sample, (B,)
        """
        active_mask = confidence > self.confidence_threshold
        num_active = active_mask.sum(dim=-1).clamp(min=1)
        return active_mask, num_active

    def global_contrastive_loss(
        self,
        z_brep: Optional[torch.Tensor],
        z_pc: Optional[torch.Tensor],
        z_text: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE loss across all available modality pairs.

        Aligns: B-Rep<->Text, PC<->Text, B-Rep<->PC
        """
        device = z_text.device
        B = z_text.shape[0]

        # Normalize embeddings
        z_text = F.normalize(z_text, dim=-1)

        losses = []
        labels = torch.arange(B, device=device)

        # B-Rep <-> Text
        if z_brep is not None:
            z_brep = F.normalize(z_brep, dim=-1)
            sim_bt = z_brep @ z_text.T / tau
            loss_bt = (F.cross_entropy(sim_bt, labels) +
                       F.cross_entropy(sim_bt.T, labels)) / 2
            losses.append(loss_bt)

        # PC <-> Text
        if z_pc is not None:
            z_pc = F.normalize(z_pc, dim=-1)
            sim_pt = z_pc @ z_text.T / tau
            loss_pt = (F.cross_entropy(sim_pt, labels) +
                       F.cross_entropy(sim_pt.T, labels)) / 2
            losses.append(loss_pt)

        # B-Rep <-> PC
        if z_brep is not None and z_pc is not None:
            sim_bp = z_brep @ z_pc.T / tau
            loss_bp = (F.cross_entropy(sim_bp, labels) +
                       F.cross_entropy(sim_bp.T, labels)) / 2
            losses.append(loss_bp)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)

        return sum(losses) / len(losses)

    def local_contrastive_loss(
        self,
        F_brep: Optional[torch.Tensor],
        F_pc: Optional[torch.Tensor],
        confidence: torch.Tensor,
        tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-slot contrastive loss with confidence weighting.

        For each feature slot k, F_brep[:,k] should match F_pc[:,k] for the
        same sample and not match for different samples.
        """
        device = confidence.device

        if F_brep is None and F_pc is None:
            return torch.tensor(0.0, device=device)

        # Use whichever geometric modality is available with the other
        if F_brep is not None and F_pc is not None:
            B, K, d = F_brep.shape

            total_loss = 0.0
            total_weight = 0.0
            labels = torch.arange(B, device=device)

            for k in range(K):
                slot_weight = confidence[:, k].mean()

                if slot_weight < 0.1:
                    continue

                # Normalize features for slot k
                f_brep_k = F.normalize(F_brep[:, k], dim=-1)
                f_pc_k = F.normalize(F_pc[:, k], dim=-1)

                # Similarity matrix
                sim = f_brep_k @ f_pc_k.T / tau

                loss_k = (F.cross_entropy(sim, labels) +
                          F.cross_entropy(sim.T, labels)) / 2

                total_loss += slot_weight * loss_k
                total_weight += slot_weight

            if total_weight > 0:
                return total_loss / total_weight

        return torch.tensor(0.0, device=device)

    def grounding_consistency_loss(
        self,
        F_brep_aligned: Optional[torch.Tensor],
        F_pc_aligned: Optional[torch.Tensor],
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-modal grounding consistency in aligned space.

        For each active feature slot, the aligned B-Rep grounded features
        should match the aligned point cloud grounded features.
        """
        device = confidence.device

        if F_brep_aligned is None or F_pc_aligned is None:
            return torch.tensor(0.0, device=device)

        active_mask, _ = self.get_active_slots(confidence)

        # Normalize aligned features
        F_brep_norm = F.normalize(F_brep_aligned, dim=-1)
        F_pc_norm = F.normalize(F_pc_aligned, dim=-1)

        # Cosine similarity between corresponding slots
        similarity = (F_brep_norm * F_pc_norm).sum(dim=-1)  # (B, K)

        # Mask inactive slots and weight by confidence
        weighted_sim = similarity * confidence * active_mask.float()
        weight_sum = (confidence * active_mask.float()).sum(dim=-1).clamp(min=1e-8)

        # Mean similarity per sample
        mean_sim = weighted_sim.sum(dim=-1) / weight_sum

        # Loss: want similarity close to 1
        loss = (1 - mean_sim).mean()

        return loss

    def grounding_diversity_loss(
        self,
        G_brep: Optional[torch.Tensor],
        G_pc: Optional[torch.Tensor],
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage different feature slots to ground to different regions.

        Penalizes overlap between grounding distributions of different slots.
        """
        device = confidence.device
        active_mask, _ = self.get_active_slots(confidence)

        def compute_overlap_loss(G, conf, mask):
            if G is None:
                return torch.tensor(0.0, device=device)

            B, K, N = G.shape

            # Compute pairwise overlap: G[i] . G[j]
            overlap = torch.bmm(G, G.transpose(-2, -1))  # (B, K, K)

            # Weight by product of confidences
            conf_outer = conf.unsqueeze(-1) * conf.unsqueeze(-2)  # (B, K, K)

            # Mask for active pairs
            pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B, K, K)

            # Exclude diagonal
            diag_mask = 1 - torch.eye(K, device=device).unsqueeze(0)

            weighted_overlap = overlap * conf_outer * pair_mask.float() * diag_mask
            num_pairs = (pair_mask.float() * diag_mask).sum(dim=(-2, -1)).clamp(min=1)

            return weighted_overlap.sum(dim=(-2, -1)) / num_pairs

        losses = []

        if G_brep is not None:
            loss_brep = compute_overlap_loss(G_brep, confidence, active_mask).mean()
            losses.append(loss_brep)

        if G_pc is not None:
            loss_pc = compute_overlap_loss(G_pc, confidence, active_mask).mean()
            losses.append(loss_pc)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)

        return sum(losses) / len(losses)

    def confidence_regularization(
        self,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Entropy regularization to encourage confident predictions.

        Without this, the model might collapse all confidences to zero.
        """
        # Encourage confidence to be near 0 or 1, not 0.5
        entropy = -confidence * torch.log(confidence + 1e-8) - \
                  (1 - confidence) * torch.log(1 - confidence + 1e-8)

        return entropy.mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        stage: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute all losses.

        Args:
            outputs: Model outputs dictionary
            stage: Training stage (1 = grounding focus, 2 = full alignment)

        Returns:
            total_loss: Scalar loss
            loss_dict: Individual losses for logging
        """
        losses = {}
        device = outputs.get("z_text", outputs.get("confidence")).device

        confidence = outputs["confidence"]
        tau = outputs.get("temperature", torch.tensor(0.07, device=device))

        # Extract embeddings
        z_brep = outputs.get("z_brep")
        z_pc = outputs.get("z_pc")
        z_text = outputs.get("z_text")

        # Grounding matrices
        G_brep = outputs.get("G_brep")
        G_pc = outputs.get("G_pc")

        # Aligned features
        F_brep_aligned = outputs.get("F_brep_aligned")
        F_pc_aligned = outputs.get("F_pc_aligned")

        # Local features
        F_brep_local = outputs.get("F_brep_local")
        F_pc_local = outputs.get("F_pc_local")

        # Grounding losses (always active)
        losses["consistency"] = self.grounding_consistency_loss(
            F_brep_aligned, F_pc_aligned, confidence
        )

        losses["diversity"] = self.grounding_diversity_loss(
            G_brep, G_pc, confidence
        )

        # Local contrastive
        losses["local"] = self.local_contrastive_loss(
            F_brep_local, F_pc_local, confidence, tau
        )

        # Global contrastive
        losses["global"] = self.global_contrastive_loss(
            z_brep, z_pc, z_text, tau
        )

        # Confidence regularization
        losses["conf_reg"] = self.confidence_regularization(confidence)

        # Combine based on training stage
        if stage == 1:
            # Stage 1: Focus on grounding, reduced global loss
            total_loss = (
                self.lambda_global_stage1 * losses["global"] +
                self.lambda_local * losses["local"] +
                self.lambda_consist * losses["consistency"] +
                self.lambda_diverse * losses["diversity"] +
                self.lambda_conf_reg * losses["conf_reg"]
            )
        else:
            # Stage 2: Full training
            total_loss = (
                self.lambda_global * losses["global"] +
                self.lambda_local * losses["local"] +
                self.lambda_consist * losses["consistency"] +
                self.lambda_diverse * losses["diversity"] +
                self.lambda_conf_reg * losses["conf_reg"]
            )

        losses["total"] = total_loss

        # Convert to float for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in losses.items()}

        return total_loss, loss_dict

    def set_stage(self, stage: int):
        """Update loss weights based on training stage."""
        if stage == 1:
            self._current_stage = 1
        else:
            self._current_stage = 2


class GroundingConsistencyLoss(nn.Module):
    """
    Standalone grounding consistency loss.

    Enforces that the same text feature mention should ground to
    geometrically corresponding regions across different modalities.
    """

    def __init__(self, confidence_threshold: float = 0.3):
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        F_brep_aligned: torch.Tensor,
        F_pc_aligned: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        active_mask = confidence > self.confidence_threshold

        F_brep_norm = F.normalize(F_brep_aligned, dim=-1)
        F_pc_norm = F.normalize(F_pc_aligned, dim=-1)

        similarity = (F_brep_norm * F_pc_norm).sum(dim=-1)
        weighted_sim = similarity * confidence * active_mask.float()
        weight_sum = (confidence * active_mask.float()).sum(dim=-1).clamp(min=1e-8)

        mean_sim = weighted_sim.sum(dim=-1) / weight_sum
        return (1 - mean_sim).mean()


class GroundingDiversityLoss(nn.Module):
    """
    Standalone grounding diversity loss.

    Encourages different feature slots to attend to distinct geometric regions.
    """

    def __init__(self, confidence_threshold: float = 0.3):
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        G: torch.Tensor,
        confidence: torch.Tensor
    ) -> torch.Tensor:
        B, K, N = G.shape
        device = G.device

        active_mask = confidence > self.confidence_threshold

        # Pairwise overlap
        overlap = torch.bmm(G, G.transpose(-2, -1))

        # Weight by confidence product
        conf_outer = confidence.unsqueeze(-1) * confidence.unsqueeze(-2)
        pair_mask = active_mask.unsqueeze(-1) * active_mask.unsqueeze(-2)
        diag_mask = 1 - torch.eye(K, device=device).unsqueeze(0)

        weighted_overlap = overlap * conf_outer * pair_mask.float() * diag_mask
        num_pairs = (pair_mask.float() * diag_mask).sum(dim=(-2, -1)).clamp(min=1)

        return (weighted_overlap.sum(dim=(-2, -1)) / num_pairs).mean()


class SlotContrastiveLoss(nn.Module):
    """
    Per-slot contrastive loss with confidence weighting.
    """

    def __init__(self, min_confidence: float = 0.1):
        super().__init__()
        self.min_confidence = min_confidence

    def forward(
        self,
        F_a: torch.Tensor,
        F_b: torch.Tensor,
        confidence: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        B, K, d = F_a.shape
        device = F_a.device

        total_loss = 0.0
        total_weight = 0.0
        labels = torch.arange(B, device=device)

        for k in range(K):
            slot_weight = confidence[:, k].mean()

            if slot_weight < self.min_confidence:
                continue

            f_a_k = F.normalize(F_a[:, k], dim=-1)
            f_b_k = F.normalize(F_b[:, k], dim=-1)

            sim = f_a_k @ f_b_k.T / temperature

            loss_k = (F.cross_entropy(sim, labels) +
                      F.cross_entropy(sim.T, labels)) / 2

            total_loss += slot_weight * loss_k
            total_weight += slot_weight

        if total_weight > 0:
            return total_loss / total_weight

        return torch.tensor(0.0, device=device)
