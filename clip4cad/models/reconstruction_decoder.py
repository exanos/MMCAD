"""
Reconstruction Decoder (Auxiliary Regularization)

Purpose: Encourage unified representation to encode actual geometry.
NOT for high-fidelity reconstruction - the compression ratio is extreme.

Decodes unified features -> face point grids + edge curves
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class UpBlock2D(nn.Module):
    """2D upsampling block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)

        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv(x)
        x = self.norm2(x)
        x = self.act(x)

        return x


class UpBlock1D(nn.Module):
    """1D upsampling block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)

        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv(x)
        x = self.norm2(x)
        x = self.act(x)

        return x


class FaceDecoder(nn.Module):
    """Decode face latent to point grid using transposed convolutions."""

    def __init__(
        self,
        input_dim: int = 32,
        output_size: int = 32,
        channels: Tuple[int, ...] = (256, 128, 64, 32),
    ):
        super().__init__()

        # Compute initial spatial size
        num_upsamples = len(channels)
        initial_size = output_size // (2**num_upsamples)  # e.g., 32 // 16 = 2

        # Project input to initial feature map
        self.initial_proj = nn.Linear(input_dim, channels[0] * initial_size * initial_size)
        self.initial_size = initial_size
        self.initial_channels = channels[0]

        # Upsampling blocks
        blocks = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            blocks.append(UpBlock2D(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Final projection to 3D coordinates
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels[-1], 3, kernel_size=3, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] face latent

        Returns:
            [B, H, W, 3] face point grid
        """
        B = x.shape[0]

        # Project and reshape
        x = self.initial_proj(x)
        x = x.view(B, self.initial_channels, self.initial_size, self.initial_size)

        # Upsample
        x = self.blocks(x)

        # Output
        x = self.output_proj(x)  # [B, 3, H, W]

        # Rearrange to [B, H, W, 3]
        x = x.permute(0, 2, 3, 1)

        return x


class EdgeDecoder(nn.Module):
    """Decode edge latent to point curve."""

    def __init__(
        self,
        input_dim: int = 16,
        output_size: int = 32,
        channels: Tuple[int, ...] = (128, 64, 32),
    ):
        super().__init__()

        # Compute initial size
        num_upsamples = len(channels)
        initial_size = output_size // (2**num_upsamples)

        # Project input
        self.initial_proj = nn.Linear(input_dim, channels[0] * initial_size)
        self.initial_size = initial_size
        self.initial_channels = channels[0]

        # Upsampling blocks
        blocks = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            blocks.append(UpBlock1D(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(channels[-1], 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] edge latent

        Returns:
            [B, L, 3] edge curve
        """
        B = x.shape[0]

        # Project and reshape
        x = self.initial_proj(x)
        x = x.view(B, self.initial_channels, self.initial_size)

        # Upsample
        x = self.blocks(x)

        # Output
        x = self.output_proj(x)  # [B, 3, L]

        # Rearrange to [B, L, 3]
        x = x.permute(0, 2, 1)

        return x


class ReconstructionDecoder(nn.Module):
    """
    Full reconstruction decoder.

    Takes unified representation and predicts:
    - Face point grids
    - Edge curves
    - Validity scores
    """

    def __init__(
        self,
        d_unified: int = 256,
        n_unified_tokens: int = 16,
        max_faces: int = 192,
        max_edges: int = 512,
        d_face: int = 32,
        d_edge: int = 16,
        face_grid_size: int = 32,
        edge_curve_size: int = 32,
    ):
        """
        Args:
            d_unified: Unified representation dimension
            n_unified_tokens: Number of tokens in unified representation
            max_faces: Maximum number of faces to reconstruct
            max_edges: Maximum number of edges to reconstruct
            d_face: Face latent dimension
            d_edge: Edge latent dimension
            face_grid_size: Size of face UV grid
            edge_curve_size: Length of edge curve
        """
        super().__init__()

        self.max_faces = max_faces
        self.max_edges = max_edges
        self.d_face = d_face
        self.d_edge = d_edge

        unified_flat_dim = n_unified_tokens * d_unified

        # Predict face latents from unified
        self.face_predictor = nn.Sequential(
            nn.Linear(unified_flat_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, max_faces * d_face),
        )

        # Predict edge latents from unified
        self.edge_predictor = nn.Sequential(
            nn.Linear(unified_flat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, max_edges * d_edge),
        )

        # Decoders
        self.face_decoder = FaceDecoder(
            input_dim=d_face,
            output_size=face_grid_size,
        )
        self.edge_decoder = EdgeDecoder(
            input_dim=d_edge,
            output_size=edge_curve_size,
        )

        # Face/edge validity predictors
        self.face_validity = nn.Sequential(
            nn.Linear(d_face, d_face // 2),
            nn.GELU(),
            nn.Linear(d_face // 2, 1),
            nn.Sigmoid(),
        )
        self.edge_validity = nn.Sequential(
            nn.Linear(d_edge, d_edge // 2),
            nn.GELU(),
            nn.Linear(d_edge // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, unified: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            unified: [B, N, D] unified representation

        Returns:
            face_grids: [B, max_F, H, W, 3]
            edge_curves: [B, max_E, L, 3]
            face_validity: [B, max_F]
            edge_validity: [B, max_E]
        """
        B = unified.size(0)

        # Flatten unified
        unified_flat = unified.view(B, -1)

        # Predict face latents
        face_latents = self.face_predictor(unified_flat)
        face_latents = face_latents.view(B, self.max_faces, self.d_face)

        # Predict edge latents
        edge_latents = self.edge_predictor(unified_flat)
        edge_latents = edge_latents.view(B, self.max_edges, self.d_edge)

        # Decode faces
        face_flat = face_latents.view(B * self.max_faces, self.d_face)
        face_grids = self.face_decoder(face_flat)
        face_grids = face_grids.view(B, self.max_faces, *face_grids.shape[1:])

        # Decode edges
        edge_flat = edge_latents.view(B * self.max_edges, self.d_edge)
        edge_curves = self.edge_decoder(edge_flat)
        edge_curves = edge_curves.view(B, self.max_edges, *edge_curves.shape[1:])

        # Predict validity
        face_valid = self.face_validity(face_latents).squeeze(-1)
        edge_valid = self.edge_validity(edge_latents).squeeze(-1)

        return {
            "face_grids": face_grids,
            "edge_curves": edge_curves,
            "face_validity": face_valid,
            "edge_validity": edge_valid,
            "face_latents": face_latents,
            "edge_latents": edge_latents,
        }
