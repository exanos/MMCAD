"""
B-Rep Encoder based on AutoBrep's FSQ VAE architecture.

Encodes:
- Face point grids [B, F, H, W, 3] -> [B, F, face_dim]
- Edge point curves [B, E, L, 3] -> [B, E, edge_dim]

Architecture adapted from AutoBrep's SurfaceFSQVAE and EdgeFSQVAE encoders.
For CLIP4CAD, we extract continuous features (pre-FSQ) for contrastive learning.

Reference:
- AutoBrep: https://github.com/AutodeskAILab/AutoBrep
- Pretrained weights: https://huggingface.co/SamGiantEagle/AutoBrep
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from ..fsq import SurfaceFSQEncoder, EdgeFSQEncoder


class BRepEncoder(nn.Module):
    """
    Combined B-Rep encoder for faces and edges.

    Uses AutoBrep-style FSQ VAE encoders to extract geometric features.
    Supports loading pretrained weights from AutoBrep checkpoints.
    """

    def __init__(
        self,
        # Output dimensions (after projection for CLIP4CAD)
        face_dim: int = 48,  # 3 * 16 (matches AutoBrep XAEncoder surfz)
        edge_dim: int = 12,  # 3 * 4 (matches AutoBrep XAEncoder edgez)
        # Input sizes
        face_grid_size: int = 32,
        edge_curve_size: int = 32,
        # Architecture params (matching AutoBrep defaults)
        face_base_channels: int = 64,
        face_channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        face_latent_channels: int = 16,
        edge_base_channels: int = 64,
        edge_channel_mult: Tuple[int, ...] = (1, 2, 4),
        edge_latent_channels: int = 4,
        # FSQ settings (from AutoBrep config: levels [8,5,5,5] = 1000 codes)
        fsq_levels: Tuple[int, ...] = (8, 5, 5, 5),
        use_fsq: bool = False,  # We don't use FSQ for continuous features
        # Pretrained weights
        surface_checkpoint: Optional[str] = None,
        edge_checkpoint: Optional[str] = None,
        freeze: bool = False,
    ):
        """
        Args:
            face_dim: Output dimension for face embeddings
            edge_dim: Output dimension for edge embeddings
            face_grid_size: Size of face UV grid (H=W)
            edge_curve_size: Number of points in edge curve
            face_base_channels: Base channels for face encoder
            face_channel_mult: Channel multipliers for face encoder
            face_latent_channels: Latent channels before projection
            edge_base_channels: Base channels for edge encoder
            edge_channel_mult: Channel multipliers for edge encoder
            edge_latent_channels: Latent channels before projection
            fsq_levels: FSQ quantization levels (not used for features)
            use_fsq: Whether to apply FSQ (typically False for CLIP4CAD)
            surface_checkpoint: Path to AutoBrep surface FSQ VAE checkpoint
            edge_checkpoint: Path to AutoBrep edge FSQ VAE checkpoint
            freeze: Whether to freeze encoder weights
        """
        super().__init__()

        self.face_dim = face_dim
        self.edge_dim = edge_dim

        # Surface/Face encoder
        self.face_encoder = SurfaceFSQEncoder(
            grid_size=face_grid_size,
            base_channels=face_base_channels,
            channel_mult=face_channel_mult,
            latent_channels=face_latent_channels,
            fsq_levels=list(fsq_levels),
            output_dim=face_dim,
            use_fsq=use_fsq,
        )

        # Edge encoder
        self.edge_encoder = EdgeFSQEncoder(
            curve_length=edge_curve_size,
            base_channels=edge_base_channels,
            channel_mult=edge_channel_mult,
            latent_channels=edge_latent_channels,
            fsq_levels=list(fsq_levels),
            output_dim=edge_dim,
            use_fsq=use_fsq,
        )

        # Load pretrained weights if provided
        if surface_checkpoint:
            self.load_surface_checkpoint(surface_checkpoint)
        if edge_checkpoint:
            self.load_edge_checkpoint(edge_checkpoint)

        # Freeze if requested
        if freeze:
            self.freeze()

    def load_surface_checkpoint(self, path: str) -> bool:
        """
        Load pretrained surface FSQ VAE weights from AutoBrep.

        AutoBrep checkpoint structure:
        - 'state_dict' or direct state dict
        - Keys like 'encoder.conv_in.weight', 'encoder.down_blocks.0.conv1.weight', etc.
        """
        path = Path(path)
        if not path.exists():
            print(f"Warning: Surface checkpoint not found: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                print(f"Warning: Unexpected checkpoint format: {type(checkpoint)}")
                return False

            # Map AutoBrep keys to our encoder
            mapped_state = self._map_autobrep_surface_keys(state_dict)

            # Load with flexibility for architecture differences
            missing, unexpected = self.face_encoder.encoder.load_state_dict(
                mapped_state, strict=False
            )

            if missing:
                print(f"Surface encoder - missing keys: {len(missing)}")
            if unexpected:
                print(f"Surface encoder - unexpected keys: {len(unexpected)}")

            print(f"Loaded surface encoder weights from {path}")
            return True

        except Exception as e:
            print(f"Warning: Failed to load surface checkpoint: {e}")
            return False

    def load_edge_checkpoint(self, path: str) -> bool:
        """Load pretrained edge FSQ VAE weights from AutoBrep."""
        path = Path(path)
        if not path.exists():
            print(f"Warning: Edge checkpoint not found: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                print(f"Warning: Unexpected checkpoint format: {type(checkpoint)}")
                return False

            mapped_state = self._map_autobrep_edge_keys(state_dict)

            missing, unexpected = self.edge_encoder.encoder.load_state_dict(
                mapped_state, strict=False
            )

            if missing:
                print(f"Edge encoder - missing keys: {len(missing)}")
            if unexpected:
                print(f"Edge encoder - unexpected keys: {len(unexpected)}")

            print(f"Loaded edge encoder weights from {path}")
            return True

        except Exception as e:
            print(f"Warning: Failed to load edge checkpoint: {e}")
            return False

    def _map_autobrep_surface_keys(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map AutoBrep SurfaceFSQVAE keys to our Encoder2D structure.

        AutoBrep uses:
        - encoder.conv_in, encoder.down_blocks, encoder.mid_block, etc.
        - We need to extract just the encoder part
        """
        mapped = {}

        for key, value in state_dict.items():
            # Only take encoder keys, skip decoder
            if key.startswith("encoder."):
                new_key = key[len("encoder."):]  # Remove "encoder." prefix
                mapped[new_key] = value
            elif not any(key.startswith(p) for p in ["decoder.", "fsq.", "quantizer."]):
                # Direct key without prefix
                mapped[key] = value

        return mapped

    def _map_autobrep_edge_keys(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Map AutoBrep EdgeFSQVAE keys to our Encoder1D structure."""
        mapped = {}

        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = key[len("encoder."):]
                mapped[new_key] = value
            elif not any(key.startswith(p) for p in ["decoder.", "fsq.", "quantizer."]):
                mapped[key] = value

        return mapped

    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
        print("B-Rep encoder frozen")

    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("B-Rep encoder unfrozen")

    def forward(
        self,
        face_grids: torch.Tensor,
        edge_curves: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode B-Rep faces and edges.

        Args:
            face_grids: [B, F, H, W, 3] face point grids
            edge_curves: [B, E, L, 3] edge point curves
            face_mask: [B, F] validity mask (1=valid, 0=padding)
            edge_mask: [B, E] validity mask

        Returns:
            face_tokens: [B, F, face_dim]
            edge_tokens: [B, E, edge_dim]
        """
        B, F, H, W, _ = face_grids.shape
        E, L = edge_curves.shape[1], edge_curves.shape[2]

        # Encode faces: reshape to [B*F, H, W, 3], encode, reshape back
        face_flat = face_grids.view(B * F, H, W, 3)
        face_tokens = self.face_encoder(face_flat)  # [B*F, face_dim]
        face_tokens = face_tokens.view(B, F, -1)  # [B, F, face_dim]

        # Encode edges: reshape to [B*E, L, 3], encode, reshape back
        edge_flat = edge_curves.view(B * E, L, 3)
        edge_tokens = self.edge_encoder(edge_flat)  # [B*E, edge_dim]
        edge_tokens = edge_tokens.view(B, E, -1)  # [B, E, edge_dim]

        # Zero out padding tokens
        face_tokens = face_tokens * face_mask.unsqueeze(-1)
        edge_tokens = edge_tokens * edge_mask.unsqueeze(-1)

        return face_tokens, edge_tokens

    def encode_faces(
        self,
        face_grids: torch.Tensor,
        face_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode faces only.

        Args:
            face_grids: [B, F, H, W, 3] face point grids
            face_mask: Optional [B, F] validity mask

        Returns:
            face_tokens: [B, F, face_dim]
        """
        B, F, H, W, _ = face_grids.shape
        face_flat = face_grids.view(B * F, H, W, 3)
        face_tokens = self.face_encoder(face_flat)
        face_tokens = face_tokens.view(B, F, -1)

        if face_mask is not None:
            face_tokens = face_tokens * face_mask.unsqueeze(-1)

        return face_tokens

    def encode_edges(
        self,
        edge_curves: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode edges only.

        Args:
            edge_curves: [B, E, L, 3] edge point curves
            edge_mask: Optional [B, E] validity mask

        Returns:
            edge_tokens: [B, E, edge_dim]
        """
        B, E, L, _ = edge_curves.shape
        edge_flat = edge_curves.view(B * E, L, 3)
        edge_tokens = self.edge_encoder(edge_flat)
        edge_tokens = edge_tokens.view(B, E, -1)

        if edge_mask is not None:
            edge_tokens = edge_tokens * edge_mask.unsqueeze(-1)

        return edge_tokens


def load_autobrep_weights(
    brep_encoder: BRepEncoder,
    weights_dir: str,
    surface_filename: str = "surface_fsq_vae.pt",
    edge_filename: str = "edge_fsq_vae.pt",
) -> Tuple[bool, bool]:
    """
    Convenience function to load AutoBrep pretrained weights.

    Args:
        brep_encoder: BRepEncoder instance
        weights_dir: Directory containing AutoBrep checkpoints
        surface_filename: Surface VAE checkpoint filename
        edge_filename: Edge VAE checkpoint filename

    Returns:
        Tuple of (surface_loaded, edge_loaded) booleans
    """
    weights_dir = Path(weights_dir)

    surface_path = weights_dir / surface_filename
    edge_path = weights_dir / edge_filename

    surface_loaded = brep_encoder.load_surface_checkpoint(str(surface_path))
    edge_loaded = brep_encoder.load_edge_checkpoint(str(edge_path))

    return surface_loaded, edge_loaded
