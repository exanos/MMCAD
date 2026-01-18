"""
Combined Loss for CLIP4CAD Training

Manages multiple loss components:
1. Global contrastive (InfoNCE across modalities)
2. Local contrastive (Hungarian matching with confidence)
3. Reconstruction (auxiliary L1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .infonce import MultiModalInfoNCE
from .local_matching import LocalMatchingLoss, SoftLocalMatchingLoss
from .reconstruction import ReconstructionLoss


class CLIP4CADLoss(nn.Module):
    """
    Combined loss manager for CLIP4CAD training.

    Handles:
    - Missing modalities
    - Staged training (global only vs global + local)
    - Loss weighting
    """

    def __init__(
        self,
        lambda_global: float = 1.0,
        lambda_local: float = 0.5,
        lambda_recon: float = 0.3,
        confidence_threshold: float = 0.5,
        use_soft_local: bool = False,
    ):
        """
        Args:
            lambda_global: Weight for global contrastive loss
            lambda_local: Weight for local contrastive loss
            lambda_recon: Weight for reconstruction loss
            confidence_threshold: Threshold for local matching
            use_soft_local: Use soft matching instead of Hungarian
        """
        super().__init__()

        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.lambda_recon = lambda_recon

        # Loss components
        self.global_loss = MultiModalInfoNCE()

        if use_soft_local:
            self.local_loss = SoftLocalMatchingLoss(confidence_threshold)
        else:
            self.local_loss = LocalMatchingLoss(confidence_threshold)

        self.recon_loss = ReconstructionLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.

        Args:
            outputs: Model outputs containing embeddings
            batch: Input batch with ground truth

        Returns:
            total_loss: Scalar
            loss_dict: Individual losses for logging
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self._get_device(outputs))

        temp = outputs.get("temperature", 0.07)

        # Global contrastive loss
        if self.lambda_global > 0:
            global_loss = self._compute_global_loss(outputs, temp)
            total_loss = total_loss + self.lambda_global * global_loss
            loss_dict["global_contrastive"] = global_loss.item()

        # Local contrastive loss
        if self.lambda_local > 0 and "text" in outputs:
            local_loss_val = self._compute_local_loss(outputs)
            if local_loss_val is not None:
                total_loss = total_loss + self.lambda_local * local_loss_val
                loss_dict["local_contrastive"] = local_loss_val.item()

        # Reconstruction loss
        if self.lambda_recon > 0 and "reconstruction" in outputs:
            recon_losses = self._compute_reconstruction_loss(outputs, batch)
            total_loss = total_loss + self.lambda_recon * recon_losses["total"]
            loss_dict["reconstruction"] = recon_losses["total"].item()
            loss_dict["recon_face"] = recon_losses["face"].item()
            loss_dict["recon_edge"] = recon_losses["edge"].item()

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict

    def _get_device(self, outputs: Dict) -> torch.device:
        """Get device from outputs."""
        for key in ["brep", "pointcloud", "text"]:
            if key in outputs:
                for k, v in outputs[key].items():
                    if isinstance(v, torch.Tensor):
                        return v.device
        return torch.device("cpu")

    def _compute_global_loss(
        self,
        outputs: Dict,
        temperature: float,
    ) -> torch.Tensor:
        """Compute global InfoNCE loss across modality pairs."""
        # Collect global projections
        embeds = {}
        if "brep" in outputs and "z_global_proj" in outputs["brep"]:
            embeds["brep"] = F.normalize(outputs["brep"]["z_global_proj"], dim=-1)
        if "pointcloud" in outputs and "z_global_proj" in outputs["pointcloud"]:
            embeds["pointcloud"] = F.normalize(outputs["pointcloud"]["z_global_proj"], dim=-1)
        if "text" in outputs and "t_global_proj" in outputs["text"]:
            embeds["text"] = F.normalize(outputs["text"]["t_global_proj"], dim=-1)

        if len(embeds) < 2:
            device = list(embeds.values())[0].device if embeds else torch.device("cpu")
            return torch.tensor(0.0, device=device)

        return self.global_loss(embeds, temperature)

    def _compute_local_loss(self, outputs: Dict) -> Optional[torch.Tensor]:
        """Compute local matching loss."""
        # Get geometric local features
        z_local = None
        if "brep" in outputs and "z_local_proj" in outputs["brep"]:
            z_local = outputs["brep"]["z_local_proj"]
        elif "pointcloud" in outputs and "z_local_proj" in outputs["pointcloud"]:
            z_local = outputs["pointcloud"]["z_local_proj"]

        if z_local is None:
            return None

        # Get text local features and confidence
        if "text" not in outputs:
            return None

        t_local = outputs["text"]["t_local_proj"]
        t_conf = outputs["text"]["t_local_confidence"]

        if isinstance(self.local_loss, LocalMatchingLoss):
            loss, _ = self.local_loss(z_local, t_local, t_conf)
        else:
            loss = self.local_loss(z_local, t_local, t_conf)

        return loss

    def _compute_reconstruction_loss(
        self,
        outputs: Dict,
        batch: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction loss."""
        recon = outputs["reconstruction"]
        device = recon["face_grids"].device

        # Check if we have B-Rep ground truth
        if "brep_faces" not in batch:
            return {
                "total": torch.tensor(0.0, device=device),
                "face": torch.tensor(0.0, device=device),
                "edge": torch.tensor(0.0, device=device),
            }

        gt_faces = batch["brep_faces"].to(device)
        gt_edges = batch["brep_edges"].to(device)
        face_mask = batch["brep_face_mask"].to(device)
        edge_mask = batch["brep_edge_mask"].to(device)

        return self.recon_loss(
            pred_faces=recon["face_grids"],
            pred_edges=recon["edge_curves"],
            gt_faces=gt_faces,
            gt_edges=gt_edges,
            face_mask=face_mask,
            edge_mask=edge_mask,
        )

    def set_local_weight(self, weight: float):
        """Update local loss weight (for staged training)."""
        self.lambda_local = weight

    def set_recon_weight(self, weight: float):
        """Update reconstruction loss weight."""
        self.lambda_recon = weight
