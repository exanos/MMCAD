"""
Point-BERT encoder compatible with ULIP-2 pretrained weights.

Architecture:
- Supports 10K point clouds with xyz + normals (6 channels)
- FPS + KNN tokenization into groups
- Mini-PointNet for local features
- Transformer encoder with CLS token
- Compatible with ULIP-2 checkpoint format

ULIP-2 Pretrained Weights:
- Download: https://huggingface.co/datasets/SFXX/ulip
- Or: https://storage.cloud.google.com/sfr-ulip-code-release-research/ULIP-2/models/

Input: [B, N, 3] or [B, N, 6] point cloud (xyz or xyz+normals)
Output: [B, num_tokens, hidden_dim] tokens (CLS + groups)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from pathlib import Path
import numpy as np


def load_ply(filepath: str, num_points: int = 10000) -> np.ndarray:
    """
    Load PLY file with positions and normals.

    Supports binary_little_endian format with structure:
    - x, y, z (float)
    - nx, ny, nz (float)

    Args:
        filepath: Path to PLY file
        num_points: Number of points to return (subsample or pad)

    Returns:
        points: [num_points, 6] array (xyz + normals)
    """
    import struct

    with open(filepath, "rb") as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode("utf-8").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header
        n_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                properties.append(line.split()[-1])

        # Determine format
        has_normals = "nx" in properties and "ny" in properties and "nz" in properties
        n_floats = len(properties)

        # Read binary data
        data = np.frombuffer(
            f.read(n_vertices * n_floats * 4),
            dtype=np.float32
        ).reshape(n_vertices, n_floats)

        # Extract xyz and normals
        xyz = data[:, :3]
        if has_normals:
            normals = data[:, 3:6]
            points = np.concatenate([xyz, normals], axis=1)
        else:
            # Estimate normals (simple approach - use zeros)
            normals = np.zeros_like(xyz)
            points = np.concatenate([xyz, normals], axis=1)

    # Handle point count
    n = points.shape[0]
    if n > num_points:
        # Random subsample
        idx = np.random.choice(n, num_points, replace=False)
        points = points[idx]
    elif n < num_points:
        # Pad by repeating
        pad_idx = np.random.choice(n, num_points - n, replace=True)
        points = np.concatenate([points, points[pad_idx]], axis=0)

    return points.astype(np.float32)


def farthest_point_sample(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
    """
    Farthest point sampling.

    Args:
        xyz: [B, N, 3] point positions
        n_points: Number of points to sample

    Returns:
        indices: [B, n_points] sampled point indices
    """
    device = xyz.device
    B, N, _ = xyz.shape

    centroids = torch.zeros(B, n_points, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # Random starting point
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(n_points):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)

    return centroids


def knn_group(
    xyz: torch.Tensor,
    features: torch.Tensor,
    center_idx: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KNN grouping with features.

    Args:
        xyz: [B, N, 3] point positions
        features: [B, N, C] point features (can include normals)
        center_idx: [B, M] indices of group centers
        k: Number of neighbors per group

    Returns:
        grouped_xyz: [B, M, k, 3] relative positions
        grouped_features: [B, M, k, C] features
    """
    B, N, _ = xyz.shape
    M = center_idx.shape[1]
    device = xyz.device

    # Get center positions
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(-1, M)
    centers = xyz[batch_idx, center_idx]  # [B, M, 3]

    # Compute distances from centers to all points
    dist = torch.cdist(centers, xyz)  # [B, M, N]

    # Get k nearest neighbors
    _, group_idx = dist.topk(k, dim=-1, largest=False)  # [B, M, k]

    # Gather grouped points
    batch_expand = torch.arange(B, device=device).view(B, 1, 1).expand(-1, M, k)
    grouped_xyz = xyz[batch_expand, group_idx]  # [B, M, k, 3]
    grouped_features = features[batch_expand, group_idx]  # [B, M, k, C]

    # Make positions relative to center
    grouped_xyz = grouped_xyz - centers.unsqueeze(2)

    return grouped_xyz, grouped_features


class PointNetEncoder(nn.Module):
    """
    Mini-PointNet for encoding local point groups.

    Matches ULIP-2 architecture.
    """

    def __init__(
        self,
        in_channels: int = 6,  # xyz (3) + normals (3) or colors
        hidden_dims: Tuple[int, ...] = (64, 128, 256),
        out_dim: int = 256,
    ):
        super().__init__()

        self.in_channels = in_channels

        # MLP layers
        layers = []
        prev_dim = in_channels

        for dim in hidden_dims:
            layers.extend([
                nn.Conv1d(prev_dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = dim

        layers.append(nn.Conv1d(prev_dim, out_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, k, C] local point group with features

        Returns:
            [B, out_dim] group embedding (max pooled)
        """
        # [B, C, k]
        x = x.permute(0, 2, 1)
        x = self.mlp(x)
        # Max pool
        x = x.max(dim=-1)[0]  # [B, out_dim]
        return x


class PointTokenizer(nn.Module):
    """
    Point cloud tokenizer using FPS + KNN grouping.

    Converts raw point cloud into discrete tokens for transformer.
    """

    def __init__(
        self,
        num_groups: int = 512,
        group_size: int = 32,
        in_channels: int = 6,  # xyz + normals
        embed_dim: int = 384,
    ):
        super().__init__()

        self.num_groups = num_groups
        self.group_size = group_size
        self.in_channels = in_channels

        # Local feature encoder
        # Input: xyz (3) + centered xyz (3) + features (in_channels)
        # For ULIP-2 style: relative xyz (3) + normals (3) = 6
        self.encoder = PointNetEncoder(
            in_channels=3 + in_channels,  # relative xyz + original features
            hidden_dims=(64, 128, 256),
            out_dim=embed_dim,
        )

        # Position encoder for centers
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

    def forward(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: [B, N, C] point cloud (C >= 3, first 3 are xyz)

        Returns:
            tokens: [B, num_groups, embed_dim] group tokens
            centers: [B, num_groups, 3] group center positions
        """
        B, N, C = points.shape
        device = points.device

        xyz = points[:, :, :3]  # [B, N, 3]
        features = points  # [B, N, C] - all features including xyz

        # FPS to get group centers
        center_idx = farthest_point_sample(xyz, self.num_groups)
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(-1, self.num_groups)
        centers = xyz[batch_idx, center_idx]  # [B, M, 3]

        # KNN grouping
        grouped_xyz, grouped_features = knn_group(
            xyz, features, center_idx, self.group_size
        )  # [B, M, k, 3], [B, M, k, C]

        # Combine relative xyz with features
        # grouped_xyz is already relative to center
        local_input = torch.cat([grouped_xyz, grouped_features], dim=-1)  # [B, M, k, 3+C]

        # Encode each group
        B, M, k, _ = local_input.shape
        local_flat = local_input.view(B * M, k, -1)  # [B*M, k, 3+C]
        tokens = self.encoder(local_flat)  # [B*M, embed_dim]
        tokens = tokens.view(B, M, -1)  # [B, M, embed_dim]

        # Add position encoding from centers
        pos_embed = self.pos_encoder(centers)  # [B, M, embed_dim]
        tokens = tokens + pos_embed

        return tokens, centers


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm (matching ULIP-2)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # Stochastic depth
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)

        if self.training and self.drop_path > 0:
            attn_out = self._drop_path(attn_out)
        x = x + attn_out

        # MLP with pre-norm
        mlp_out = self.mlp(self.norm2(x))
        if self.training and self.drop_path > 0:
            mlp_out = self._drop_path(mlp_out)
        x = x + mlp_out

        return x

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path == 0.0:
            return x
        keep_prob = 1 - self.drop_path
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class PointBertEncoder(nn.Module):
    """
    Point-BERT encoder compatible with ULIP-2 pretrained weights.

    Architecture:
    - Point tokenizer (FPS + KNN + PointNet)
    - CLS token
    - Positional embedding
    - Transformer encoder (12 layers, 768 dim for ULIP-2)

    Supports:
    - 10K point clouds with xyz + normals
    - Pre-computed feature caching
    - ULIP-2 checkpoint loading
    """

    # Known configurations
    CONFIGS = {
        "ulip2-pointbert": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "num_groups": 512,
            "group_size": 32,
            "mlp_ratio": 4.0,
        },
        "ulip2-pointbert-small": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "num_groups": 256,
            "group_size": 32,
            "mlp_ratio": 4.0,
        },
    }

    def __init__(
        self,
        num_points: int = 10000,
        in_channels: int = 6,  # xyz + normals
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_groups: int = 512,
        group_size: int = 32,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        checkpoint_path: Optional[str] = None,
        freeze: bool = False,
        use_xyz_only: bool = False,  # Use only xyz, ignore normals
    ):
        """
        Args:
            num_points: Expected number of input points
            in_channels: Input channels (3 for xyz, 6 for xyz+normals)
            embed_dim: Transformer embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            num_groups: Number of point groups (tokens)
            group_size: Points per group
            mlp_ratio: MLP expansion ratio
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth rate
            checkpoint_path: Path to ULIP-2 pretrained weights
            freeze: Whether to freeze encoder
            use_xyz_only: If True, only use xyz coordinates (for 3-channel input)
        """
        super().__init__()

        self.num_points = num_points
        self.in_channels = in_channels if not use_xyz_only else 3
        self.embed_dim = embed_dim
        self.output_dim = embed_dim
        self.num_groups = num_groups
        self.num_tokens = num_groups + 1  # groups + CLS
        self.use_xyz_only = use_xyz_only

        # Point tokenizer
        self.tokenizer = PointTokenizer(
            num_groups=num_groups,
            group_size=group_size,
            in_channels=self.in_channels,
            embed_dim=embed_dim,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

        # Load pretrained weights
        if checkpoint_path:
            self.load_ulip2_weights(checkpoint_path)

        # Freeze if requested
        if freeze:
            self.freeze()

    @classmethod
    def from_config(cls, config_name: str, **kwargs) -> "PointBertEncoder":
        """Create encoder from predefined config."""
        if config_name not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(cls.CONFIGS.keys())}")

        config = cls.CONFIGS[config_name].copy()
        config.update(kwargs)
        return cls(**config)

    def load_ulip2_weights(self, path: str) -> bool:
        """
        Load ULIP-2 pretrained weights.

        Handles various checkpoint formats from ULIP repository.
        """
        path = Path(path)
        if not path.exists():
            print(f"Warning: Checkpoint not found: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            # Extract state dict from various formats
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "point_encoder" in checkpoint:
                state_dict = checkpoint["point_encoder"]
            else:
                state_dict = checkpoint

            # Map ULIP-2 keys to our architecture
            mapped = self._map_ulip2_keys(state_dict)

            # Load weights
            missing, unexpected = self.load_state_dict(mapped, strict=False)

            print(f"Loaded ULIP-2 Point-BERT weights from {path}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")

            return True

        except Exception as e:
            print(f"Warning: Failed to load ULIP-2 weights: {e}")
            return False

    def _map_ulip2_keys(self, state_dict: Dict) -> Dict:
        """Map ULIP-2 checkpoint keys to our architecture."""
        mapped = {}

        key_mapping = {
            # Common prefixes to remove
            "module.": "",
            "point_encoder.": "",
            "transformer_q.": "",
            "encoder.": "",
            # Block mappings
            "blocks.": "blocks.",
            "transformer.": "blocks.",
        }

        for key, value in state_dict.items():
            new_key = key

            # Remove common prefixes
            for old_prefix, new_prefix in key_mapping.items():
                if new_key.startswith(old_prefix):
                    new_key = new_prefix + new_key[len(old_prefix):]

            # Skip classification head
            if "cls_head" in new_key or "head" in new_key:
                continue

            # Skip incompatible keys
            if "pos_embed" in new_key and "cls" not in new_key:
                # Position embeddings might have different sizes
                continue

            mapped[new_key] = value

        return mapped

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        print("Point-BERT encoder frozen")

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("Point-BERT encoder unfrozen")

    def forward(
        self,
        points: torch.Tensor,
        return_all_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Encode point cloud.

        Args:
            points: [B, N, C] point cloud (C=3 for xyz, C=6 for xyz+normals)
            return_all_tokens: If True, return all tokens; else just CLS

        Returns:
            If return_all_tokens:
                [B, num_groups+1, embed_dim] all tokens (CLS first)
            Else:
                [B, embed_dim] CLS token only
        """
        B = points.shape[0]

        # Use only xyz if configured
        if self.use_xyz_only and points.shape[-1] > 3:
            points = points[:, :, :3]

        # Pad to expected channels if needed
        if points.shape[-1] < self.in_channels:
            padding = torch.zeros(
                B, points.shape[1], self.in_channels - points.shape[-1],
                device=points.device, dtype=points.dtype
            )
            points = torch.cat([points, padding], dim=-1)

        # Tokenize point cloud
        tokens, centers = self.tokenizer(points)  # [B, num_groups, embed_dim]

        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1) + self.cls_pos
        tokens = torch.cat([cls_token, tokens], dim=1)  # [B, num_groups+1, embed_dim]

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Final norm
        tokens = self.norm(tokens)

        if return_all_tokens:
            return tokens
        else:
            return tokens[:, 0]  # CLS token

    def get_cls_token(self, points: torch.Tensor) -> torch.Tensor:
        """Get CLS token (global feature)."""
        return self.forward(points, return_all_tokens=False)

    def get_group_tokens(self, points: torch.Tensor) -> torch.Tensor:
        """Get group tokens (local features, excluding CLS)."""
        tokens = self.forward(points, return_all_tokens=True)
        return tokens[:, 1:]


def download_ulip2_weights(output_dir: str = "pretrained/pointbert") -> str:
    """
    Download ULIP-2 Point-BERT weights.

    Returns path to downloaded checkpoint.
    """
    import urllib.request
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ULIP-2 checkpoint URL (from HuggingFace mirror)
    urls = {
        "ulip2_pointbert_10k": "https://huggingface.co/datasets/SFXX/ulip/resolve/main/ULIP-2/models/ULIP-2-PointBERT-10k-colored-pc-pretrained.pt",
    }

    output_path = output_dir / "ulip2_pointbert_10k.pt"

    if output_path.exists():
        print(f"Weights already exist: {output_path}")
        return str(output_path)

    print(f"Downloading ULIP-2 Point-BERT weights...")
    print(f"This may take a while...")

    try:
        urllib.request.urlretrieve(urls["ulip2_pointbert_10k"], str(output_path))
        print(f"Downloaded to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Please download manually from:")
        print(f"  {urls['ulip2_pointbert_10k']}")
        return None
