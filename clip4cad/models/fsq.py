"""
Finite Scalar Quantization (FSQ) implementation.

Based on AutoBrep's FSQ module for discretizing latent representations.
For CLIP4CAD, we use the encoder side to extract continuous features
before quantization for contrastive learning.

Reference: https://github.com/AutodeskAILab/AutoBrep
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from einops import rearrange


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator for gradient flow."""
    return z + (z.round() - z).detach()


def bound(z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Bound values to [-1+eps, 1-eps] range using tanh."""
    half_l = (1 - eps) / 2
    return torch.tanh(z) * half_l + half_l


class FSQ(nn.Module):
    """
    Finite Scalar Quantization module.

    Quantizes continuous vectors to discrete codes using implicit codebooks.
    Each dimension is quantized to one of L possible values.

    Args:
        levels: List of quantization levels per dimension, e.g., [8, 5, 5, 5]
        dim: Input feature dimension (will be projected to match levels)
        num_codebooks: Number of independent codebooks
        eps: Epsilon for numerical stability in bounding
    """

    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        eps: float = 1e-3,
    ):
        super().__init__()

        self._levels = torch.tensor(levels, dtype=torch.int32)
        self.num_codebooks = num_codebooks
        self.eps = eps

        # Calculate codebook size as product of levels
        self.codebook_size = self._levels.prod().item()
        self.codebook_dim = len(levels)
        self.effective_codebook_dim = self.codebook_dim * num_codebooks

        # Input projection if dim specified
        self.project_in = (
            nn.Linear(dim, self.effective_codebook_dim)
            if dim is not None else nn.Identity()
        )
        self.project_out = (
            nn.Linear(self.effective_codebook_dim, dim)
            if dim is not None else nn.Identity()
        )

        # Precompute basis for index conversion
        self.register_buffer("_basis", self._compute_basis())

        # Store half-levels for quantization
        self.register_buffer(
            "_half_levels",
            (self._levels.float() - 1) / 2
        )

    def _compute_basis(self) -> torch.Tensor:
        """Compute basis for converting codes to indices."""
        basis = torch.cumprod(
            torch.cat([
                torch.ones(1, dtype=torch.int64),
                self._levels[:-1].long()
            ]),
            dim=0
        )
        return basis

    @property
    def levels(self) -> torch.Tensor:
        return self._levels

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantize continuous values to discrete levels.

        Args:
            z: [..., codebook_dim] continuous values

        Returns:
            [..., codebook_dim] quantized values in [-1, 1] range
        """
        # Bound to valid range
        z_bounded = bound(z, self.eps)

        # Scale to [0, L-1] range and round
        half_levels = self._half_levels.to(z.device)
        z_scaled = z_bounded * half_levels
        z_quantized = round_ste(z_scaled)

        # Scale back to [-1, 1]
        z_out = z_quantized / half_levels

        return z_out

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert quantized codes to integer indices.

        Args:
            codes: [..., codebook_dim] codes in [-1, 1] range

        Returns:
            [...] integer indices in [0, codebook_size)
        """
        half_levels = self._half_levels.to(codes.device)
        basis = self._basis.to(codes.device)

        # Convert from [-1, 1] to [0, L-1]
        codes_scaled = (codes * half_levels + half_levels).long()

        # Compute indices
        indices = (codes_scaled * basis).sum(dim=-1)

        return indices

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert integer indices back to codes.

        Args:
            indices: [...] integer indices

        Returns:
            [..., codebook_dim] codes in [-1, 1] range
        """
        levels = self._levels.to(indices.device)
        half_levels = self._half_levels.to(indices.device)

        codes = []
        for i in range(self.codebook_dim):
            codes.append(indices % levels[i])
            indices = indices // levels[i]

        codes = torch.stack(codes, dim=-1).float()
        codes = (codes - half_levels) / half_levels

        return codes

    def forward(
        self,
        z: torch.Tensor,
        return_indices: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through FSQ.

        Args:
            z: Input tensor of shape [..., dim] or [..., codebook_dim]
            return_indices: Whether to also return discrete indices

        Returns:
            z_q: Quantized tensor (same shape as input)
            indices: Optional discrete indices if return_indices=True
        """
        # Project if needed
        z = self.project_in(z)

        # Handle multiple codebooks
        if self.num_codebooks > 1:
            z = rearrange(z, "... (c d) -> ... c d", c=self.num_codebooks)

        # Quantize
        z_q = self.quantize(z)

        # Get indices if requested
        indices = None
        if return_indices:
            indices = self.codes_to_indices(z_q)

        # Reshape back
        if self.num_codebooks > 1:
            z_q = rearrange(z_q, "... c d -> ... (c d)")

        # Project back
        z_q = self.project_out(z_q)

        return z_q, indices


class ResBlock2D(nn.Module):
    """Residual block for 2D convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size,
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size,
            padding=kernel_size // 2
        )
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual


class ResBlock1D(nn.Module):
    """Residual block for 1D convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2
        )
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual


class Encoder2D(nn.Module):
    """
    2D Encoder for surface grids, based on AutoBrep SurfaceFSQVAE.

    Progressively downsamples spatial dimensions while increasing channels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        latent_channels: int = 16,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = base_channels

        for mult in channel_mult:
            out_ch = base_channels * mult

            # Residual blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock2D(ch))
                ch = ch  # ResBlocks keep channels same

            # Channel change + downsample
            self.down_blocks.append(
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1)
            )
            ch = out_ch

        # Final blocks
        self.mid_block = ResBlock2D(ch)
        self.norm_out = nn.GroupNorm(min(32, ch), ch)
        self.conv_out = nn.Conv2d(ch, latent_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] surface point grid

        Returns:
            z: [B, latent_channels, H', W'] latent features
        """
        x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x)
            if isinstance(block, ResBlock2D):
                x = self.act(x)

        x = self.mid_block(x)
        x = self.norm_out(x)
        x = self.act(x)
        x = self.conv_out(x)

        return x


class Encoder1D(nn.Module):
    """
    1D Encoder for edge curves, based on AutoBrep EdgeFSQVAE.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Initial conv
        self.conv_in = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = base_channels

        for mult in channel_mult:
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock1D(ch))

            self.down_blocks.append(
                nn.Conv1d(ch, out_ch, 4, stride=2, padding=1)
            )
            ch = out_ch

        # Final blocks
        self.mid_block = ResBlock1D(ch)
        self.norm_out = nn.GroupNorm(min(32, ch), ch)
        self.conv_out = nn.Conv1d(ch, latent_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, L] edge point curve

        Returns:
            z: [B, latent_channels, L'] latent features
        """
        x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x)
            if isinstance(block, ResBlock1D):
                x = self.act(x)

        x = self.mid_block(x)
        x = self.norm_out(x)
        x = self.act(x)
        x = self.conv_out(x)

        return x


class SurfaceFSQEncoder(nn.Module):
    """
    Surface encoder with optional FSQ, matching AutoBrep's SurfaceFSQVAE encoder.

    For CLIP4CAD, we extract features before quantization for continuous embeddings.
    """

    def __init__(
        self,
        grid_size: int = 32,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        latent_channels: int = 16,
        fsq_levels: List[int] = [8, 5, 5, 5],
        output_dim: int = 48,  # 3 * 16 as in XAEncoder surfz
        use_fsq: bool = False,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.output_dim = output_dim
        self.use_fsq = use_fsq

        # Encoder
        self.encoder = Encoder2D(
            in_channels=3,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )

        # Calculate spatial size after encoding
        num_downsamples = len(channel_mult)
        self.latent_size = grid_size // (2 ** num_downsamples)
        self.latent_flat_dim = latent_channels * self.latent_size * self.latent_size

        # FSQ if used
        if use_fsq:
            self.fsq = FSQ(levels=fsq_levels, dim=latent_channels)

        # Output projection to unified dimension
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.latent_flat_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, 3] surface point grid
            return_latent: Whether to return pre-projection latent

        Returns:
            z: [B, output_dim] surface embedding
        """
        # Rearrange to [B, 3, H, W]
        x = x.permute(0, 3, 1, 2)

        # Encode
        z = self.encoder(x)  # [B, latent_ch, H', W']

        # Optional FSQ (for visualization/reconstruction)
        if self.use_fsq:
            z = rearrange(z, "b c h w -> b h w c")
            z_q, _ = self.fsq(z)
            z = rearrange(z_q, "b h w c -> b c h w")

        # Project to output
        out = self.output_proj(z)

        if return_latent:
            return out, z
        return out


class EdgeFSQEncoder(nn.Module):
    """
    Edge encoder with optional FSQ, matching AutoBrep's EdgeFSQVAE encoder.
    """

    def __init__(
        self,
        curve_length: int = 32,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
        fsq_levels: List[int] = [8, 5, 5, 5],
        output_dim: int = 12,  # 3 * 4 as in XAEncoder edgez
        use_fsq: bool = False,
    ):
        super().__init__()

        self.curve_length = curve_length
        self.output_dim = output_dim
        self.use_fsq = use_fsq

        # Encoder
        self.encoder = Encoder1D(
            in_channels=3,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            latent_channels=latent_channels,
        )

        # Calculate length after encoding
        num_downsamples = len(channel_mult)
        self.latent_length = curve_length // (2 ** num_downsamples)
        self.latent_flat_dim = latent_channels * self.latent_length

        # FSQ if used
        if use_fsq:
            self.fsq = FSQ(levels=fsq_levels, dim=latent_channels)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.latent_flat_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, 3] edge point curve
            return_latent: Whether to return pre-projection latent

        Returns:
            z: [B, output_dim] edge embedding
        """
        # Rearrange to [B, 3, L]
        x = x.permute(0, 2, 1)

        # Encode
        z = self.encoder(x)  # [B, latent_ch, L']

        # Optional FSQ
        if self.use_fsq:
            z = rearrange(z, "b c l -> b l c")
            z_q, _ = self.fsq(z)
            z = rearrange(z_q, "b l c -> b c l")

        # Project to output
        out = self.output_proj(z)

        if return_latent:
            return out, z
        return out
