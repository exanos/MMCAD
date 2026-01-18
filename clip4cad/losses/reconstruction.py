"""
Reconstruction Loss

Auxiliary loss for encouraging geometric grounding in the unified representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ReconstructionLoss(nn.Module):
    """
    L1 reconstruction loss for B-Rep geometry.

    Computes loss only on valid (non-padding) faces and edges.
    """

    def __init__(self, face_weight: float = 1.0, edge_weight: float = 1.0):
        """
        Args:
            face_weight: Weight for face reconstruction loss
            edge_weight: Weight for edge reconstruction loss
        """
        super().__init__()
        self.face_weight = face_weight
        self.edge_weight = edge_weight

    def forward(
        self,
        pred_faces: torch.Tensor,
        pred_edges: torch.Tensor,
        gt_faces: torch.Tensor,
        gt_edges: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.

        Args:
            pred_faces: [B, F, H, W, 3] predicted face grids
            pred_edges: [B, E, L, 3] predicted edge curves
            gt_faces: [B, F, H, W, 3] ground truth face grids
            gt_edges: [B, E, L, 3] ground truth edge curves
            face_mask: [B, F] face validity mask
            edge_mask: [B, E] edge validity mask

        Returns:
            Dictionary with total, face, and edge losses
        """
        # Face loss
        face_diff = (gt_faces - pred_faces).abs()
        face_diff = face_diff.mean(dim=(2, 3, 4))  # [B, F]
        face_loss = (face_diff * face_mask).sum() / (face_mask.sum() + 1e-8)

        # Edge loss
        edge_diff = (gt_edges - pred_edges).abs()
        edge_diff = edge_diff.mean(dim=(2, 3))  # [B, E]
        edge_loss = (edge_diff * edge_mask).sum() / (edge_mask.sum() + 1e-8)

        # Total
        total_loss = self.face_weight * face_loss + self.edge_weight * edge_loss

        return {
            "total": total_loss,
            "face": face_loss,
            "edge": edge_loss,
        }


class ValidityLoss(nn.Module):
    """
    Binary cross-entropy loss for predicting face/edge validity.
    """

    def forward(
        self,
        pred_face_validity: torch.Tensor,
        pred_edge_validity: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute validity prediction loss.

        Args:
            pred_face_validity: [B, F] predicted face validity
            pred_edge_validity: [B, E] predicted edge validity
            face_mask: [B, F] ground truth face mask
            edge_mask: [B, E] ground truth edge mask

        Returns:
            Dictionary with total, face, and edge losses
        """
        face_loss = F.binary_cross_entropy(pred_face_validity, face_mask.float())
        edge_loss = F.binary_cross_entropy(pred_edge_validity, edge_mask.float())

        total_loss = face_loss + edge_loss

        return {
            "total": total_loss,
            "face": face_loss,
            "edge": edge_loss,
        }
