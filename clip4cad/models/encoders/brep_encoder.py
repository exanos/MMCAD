"""
B-Rep Encoder based on AutoBrep's Deep Compression architecture.

Encodes:
- Face point grids [B, F, H, W, 3] -> [B, F, face_dim]
- Edge point curves [B, E, L, 3] -> [B, E, edge_dim]

Architecture adapted from AutoBrep VAE encoders, outputting continuous
features suitable for contrastive learning (no FSQ quantization).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class DownBlock2D(nn.Module):
    """2D downsampling block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)

        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

        # Residual projection if channels change
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        # Add residual before downsampling
        x = x + residual
        x = self.act(x)

        # Downsample
        x = self.downsample(x)
        return x


class DownBlock1D(nn.Module):
    """1D downsampling block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)

        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.act(x)

        x = self.downsample(x)
        return x


class FaceEncoder(nn.Module):
    """
    Encodes face point grids using 2D convolutions.

    Input: [B, H, W, 3] (per face) where H=W=32 typically
    Output: [B, face_dim] face embedding
    """

    def __init__(
        self,
        input_size: int = 32,
        output_dim: int = 32,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()

        self.output_dim = output_dim

        # Initial projection
        self.input_proj = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)

        # Downsampling blocks
        blocks = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            blocks.append(DownBlock2D(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Calculate final spatial size after downsampling
        # 32 -> 16 -> 8 -> 4 (3 downsamples for 4 channel stages)
        num_downsamples = len(channels) - 1
        final_size = input_size // (2**num_downsamples)
        final_flat_dim = channels[-1] * final_size * final_size

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_flat_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, 3] face point grid

        Returns:
            [B, face_dim] face embedding
        """
        # Rearrange to [B, 3, H, W]
        x = x.permute(0, 3, 1, 2)

        # Encode
        x = self.input_proj(x)
        x = self.blocks(x)

        # Project to output dim
        x = self.output_proj(x)

        return x


class EdgeEncoder(nn.Module):
    """
    Encodes edge point curves using 1D convolutions.

    Input: [B, L, 3] (per edge) where L=32 typically
    Output: [B, edge_dim] edge embedding
    """

    def __init__(
        self,
        input_size: int = 32,
        output_dim: int = 16,
        channels: Tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()

        self.output_dim = output_dim

        # Initial projection
        self.input_proj = nn.Conv1d(3, channels[0], kernel_size=3, padding=1)

        # Downsampling blocks
        blocks = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            blocks.append(DownBlock1D(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Calculate final size after downsampling
        num_downsamples = len(channels) - 1
        final_size = input_size // (2**num_downsamples)
        final_flat_dim = channels[-1] * final_size

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_flat_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, 3] edge point curve

        Returns:
            [B, edge_dim] edge embedding
        """
        # Rearrange to [B, 3, L]
        x = x.permute(0, 2, 1)

        # Encode
        x = self.input_proj(x)
        x = self.blocks(x)

        # Project to output dim
        x = self.output_proj(x)

        return x


class BRepEncoder(nn.Module):
    """
    Combined B-Rep encoder for faces and edges.

    Encodes variable numbers of faces and edges using shared encoders,
    handling padding via masks.
    """

    def __init__(
        self,
        face_dim: int = 32,
        edge_dim: int = 16,
        face_grid_size: int = 32,
        edge_curve_size: int = 32,
        face_channels: Tuple[int, ...] = (32, 64, 128, 256),
        edge_channels: Tuple[int, ...] = (32, 64, 128),
        checkpoint_path: Optional[str] = None,
        freeze: bool = False,
    ):
        """
        Args:
            face_dim: Output dimension for face embeddings
            edge_dim: Output dimension for edge embeddings
            face_grid_size: Size of face UV grid (H=W)
            edge_curve_size: Number of points in edge curve
            face_channels: Channel progression for face encoder
            edge_channels: Channel progression for edge encoder
            checkpoint_path: Path to pretrained weights
            freeze: Whether to freeze encoder weights
        """
        super().__init__()

        self.face_encoder = FaceEncoder(
            input_size=face_grid_size,
            output_dim=face_dim,
            channels=face_channels,
        )
        self.edge_encoder = EdgeEncoder(
            input_size=edge_curve_size,
            output_dim=edge_dim,
            channels=edge_channels,
        )

        self.face_dim = face_dim
        self.edge_dim = edge_dim

        # Load pretrained if available
        if checkpoint_path:
            self._load_pretrained(checkpoint_path)

        # Freeze if requested
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def _load_pretrained(self, path: str):
        """Load pretrained encoder weights."""
        try:
            checkpoint = torch.load(path, map_location="cpu")

            # Handle different checkpoint formats
            if "face_encoder" in checkpoint:
                self.face_encoder.load_state_dict(checkpoint["face_encoder"], strict=False)
            if "edge_encoder" in checkpoint:
                self.edge_encoder.load_state_dict(checkpoint["edge_encoder"], strict=False)
            if "state_dict" in checkpoint:
                self.load_state_dict(checkpoint["state_dict"], strict=False)
            if "model" in checkpoint:
                self.load_state_dict(checkpoint["model"], strict=False)

            print(f"Loaded B-Rep encoder weights from {path}")
        except Exception as e:
            print(f"Warning: Could not load B-Rep encoder weights: {e}")
            print("Using random initialization.")

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

    def encode_faces(self, face_grids: torch.Tensor) -> torch.Tensor:
        """Encode faces only (for inference)."""
        B, F, H, W, _ = face_grids.shape
        face_flat = face_grids.view(B * F, H, W, 3)
        face_tokens = self.face_encoder(face_flat)
        return face_tokens.view(B, F, -1)

    def encode_edges(self, edge_curves: torch.Tensor) -> torch.Tensor:
        """Encode edges only (for inference)."""
        B, E, L, _ = edge_curves.shape
        edge_flat = edge_curves.view(B * E, L, 3)
        edge_tokens = self.edge_encoder(edge_flat)
        return edge_tokens.view(B, E, -1)
