"""
Local Matching Loss

Hungarian matching-based alignment for local features with confidence weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np


class LocalMatchingLoss(nn.Module):
    """
    Local contrastive loss with confidence-thresholded Hungarian matching.

    Aligns geometric detail features with text feature embeddings using
    optimal bipartite matching, weighted by confidence scores.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: Minimum confidence for text features to be considered
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        geom_features: torch.Tensor,
        text_features: torch.Tensor,
        text_confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute local matching loss.

        Args:
            geom_features: [B, N_d, D] geometric detail features
            text_features: [B, N_d, D] text local features
            text_confidence: [B, N_d] confidence scores for text features

        Returns:
            loss: Scalar loss value
            info: Dict with additional info (num_matches, etc.)
        """
        B, N_d, D = geom_features.shape
        device = geom_features.device

        total_loss = 0.0
        valid_samples = 0
        total_matches = 0

        for b in range(B):
            # Find active text features (confidence > threshold)
            active_mask = text_confidence[b] > self.confidence_threshold
            active_indices = active_mask.nonzero(as_tuple=True)[0]

            if len(active_indices) == 0:
                continue

            # Get active text features
            t_active = text_features[b, active_indices]  # [K, D]
            c_active = text_confidence[b, active_indices]  # [K]
            K = len(active_indices)

            # Normalize
            z_norm = F.normalize(geom_features[b], dim=-1)  # [N_d, D]
            t_norm = F.normalize(t_active, dim=-1)  # [K, D]

            # Cost matrix (1 - cosine similarity = cosine distance)
            cost = 1 - z_norm @ t_norm.T  # [N_d, K]

            # Hungarian matching
            cost_np = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            # Compute loss on matched pairs, weighted by confidence
            sample_loss = 0.0
            n_matches = 0

            for r, c in zip(row_ind, col_ind):
                if c < K:  # Valid assignment
                    loss = c_active[c] * cost[r, c]
                    sample_loss = sample_loss + loss
                    n_matches += 1

            if n_matches > 0:
                total_loss = total_loss + sample_loss / n_matches
                total_matches += n_matches
                valid_samples += 1

        if valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "num_valid_samples": 0,
                "avg_matches": 0,
            }

        return total_loss / valid_samples, {
            "num_valid_samples": valid_samples,
            "avg_matches": total_matches / valid_samples,
        }


class SoftLocalMatchingLoss(nn.Module):
    """
    Soft local matching loss without discrete Hungarian matching.

    Uses softmax attention to compute weighted alignment scores.
    More GPU-friendly than Hungarian matching.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        temperature: float = 0.1,
    ):
        """
        Args:
            confidence_threshold: Minimum confidence threshold
            temperature: Softmax temperature
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature

    def forward(
        self,
        geom_features: torch.Tensor,
        text_features: torch.Tensor,
        text_confidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft local matching loss.

        Args:
            geom_features: [B, N_d, D] geometric detail features
            text_features: [B, N_d, D] text local features
            text_confidence: [B, N_d] confidence scores

        Returns:
            Scalar loss value
        """
        B, N_d, D = geom_features.shape

        # Normalize features
        geom_norm = F.normalize(geom_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)

        # Similarity matrix
        sim = torch.bmm(geom_norm, text_norm.transpose(1, 2))  # [B, N_d, N_d]
        sim = sim / self.temperature

        # Create confidence mask
        conf_mask = text_confidence > self.confidence_threshold  # [B, N_d]

        # Compute soft assignment
        # Each geometric feature attends to text features
        attn_g2t = F.softmax(sim, dim=-1)  # [B, N_d, N_d]

        # Weight by text confidence
        weighted_sim = attn_g2t * text_confidence.unsqueeze(1)  # [B, N_d, N_d]

        # Mask out low-confidence text features
        conf_mask_expanded = conf_mask.unsqueeze(1).float()
        weighted_sim = weighted_sim * conf_mask_expanded

        # Loss: maximize similarity to best matches
        # -log of weighted similarity sum (acts like cross-entropy)
        match_scores = weighted_sim.sum(dim=-1)  # [B, N_d]
        loss = -torch.log(match_scores.clamp(min=1e-8)).mean()

        return loss
