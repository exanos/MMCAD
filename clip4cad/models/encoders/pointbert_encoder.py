"""
Point-BERT encoder wrapper.

Architecture based on Point-BERT with support for ULIP-2 pretrained weights.
Processes point clouds through:
1. FPS + KNN tokenization
2. Mini-PointNet per group
3. Transformer encoder

Input: [B, N, 3] point cloud
Output: [B, num_tokens, hidden_dim] tokens (groups + CLS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


def farthest_point_sample(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
    """
    Farthest point sampling on point cloud.

    Args:
        xyz: [B, N, 3] input point cloud
        n_points: Number of points to sample

    Returns:
        [B, n_points] indices of sampled points
    """
    device = xyz.device
    B, N, _ = xyz.shape

    centroids = torch.zeros(B, n_points, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # Start from random point
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(n_points):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)  # [B, 1, 3]
        dist = ((xyz - centroid) ** 2).sum(dim=-1)  # [B, N]
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)

    return centroids


def knn_group(
    xyz: torch.Tensor,
    centers: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KNN grouping of points around centers.

    Args:
        xyz: [B, N, 3] all points
        centers: [B, M, 3] center points
        k: Number of neighbors

    Returns:
        grouped_xyz: [B, M, k, 3] grouped points (relative to center)
        group_idx: [B, M, k] indices of grouped points
    """
    B, N, _ = xyz.shape
    M = centers.shape[1]

    # Compute distances
    dist = torch.cdist(centers, xyz)  # [B, M, N]

    # Get k nearest
    _, group_idx = dist.topk(k, dim=-1, largest=False)  # [B, M, k]

    # Gather points
    batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(-1, M, k)
    grouped_xyz = xyz[batch_idx, group_idx]  # [B, M, k, 3]

    # Make relative to center
    grouped_xyz = grouped_xyz - centers.unsqueeze(2)

    return grouped_xyz, group_idx


class MiniPointNet(nn.Module):
    """Mini-PointNet for encoding local point groups."""

    def __init__(self, in_dim: int = 3, out_dim: int = 384):
        super().__init__()

        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, k, 3] local point group

        Returns:
            [B, out_dim] group feature
        """
        # [B, 3, k]
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Max pool over points
        x = x.max(dim=-1)[0]  # [B, out_dim]

        return x


class PointTokenizer(nn.Module):
    """Point cloud tokenizer using FPS + KNN + Mini-PointNet."""

    def __init__(
        self,
        num_groups: int = 512,
        group_size: int = 32,
        hidden_dim: int = 384,
    ):
        super().__init__()

        self.num_groups = num_groups
        self.group_size = group_size

        self.group_encoder = MiniPointNet(in_dim=3, out_dim=hidden_dim)

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: [B, N, 3] input point cloud

        Returns:
            tokens: [B, num_groups, hidden_dim] group tokens
            centers: [B, num_groups, 3] group centers
        """
        B, N, _ = points.shape
        device = points.device

        # FPS to get group centers
        center_idx = farthest_point_sample(points, self.num_groups)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, self.num_groups)
        centers = points[batch_idx, center_idx]  # [B, num_groups, 3]

        # KNN grouping
        grouped_xyz, _ = knn_group(points, centers, self.group_size)  # [B, M, k, 3]

        # Encode each group
        B, M, k, _ = grouped_xyz.shape
        grouped_flat = grouped_xyz.view(B * M, k, 3)
        tokens = self.group_encoder(grouped_flat)  # [B*M, hidden_dim]
        tokens = tokens.view(B, M, -1)  # [B, M, hidden_dim]

        return tokens, centers


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))

        return x


class PointBertEncoder(nn.Module):
    """
    Point-BERT encoder with optional pretrained weights.

    Architecture:
    - Point tokenizer (FPS + KNN + Mini-PointNet)
    - CLS token
    - Positional embedding
    - Transformer encoder
    """

    def __init__(
        self,
        num_points: int = 2048,
        num_groups: int = 512,
        group_size: int = 32,
        hidden_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        checkpoint_path: Optional[str] = None,
        freeze: bool = False,
    ):
        """
        Args:
            num_points: Expected number of input points
            num_groups: Number of point groups (tokens)
            group_size: Points per group
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            checkpoint_path: Path to pretrained weights
            freeze: Whether to freeze encoder
        """
        super().__init__()

        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.num_tokens = num_groups + 1  # groups + CLS

        # Point tokenizer
        self.tokenizer = PointTokenizer(
            num_groups=num_groups,
            group_size=group_size,
            hidden_dim=hidden_dim,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_groups + 1, hidden_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Load pretrained
        if checkpoint_path:
            self._load_pretrained(checkpoint_path)

        # Freeze
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def _load_pretrained(self, path: str):
        """Load pretrained Point-BERT / ULIP-2 weights."""
        try:
            checkpoint = torch.load(path, map_location="cpu")

            # Handle different checkpoint formats
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Filter compatible keys
            model_dict = self.state_dict()
            pretrained_dict = {}
            for k, v in state_dict.items():
                # Remove "module." prefix if present
                k = k.replace("module.", "")
                # Map common key name variations
                k = k.replace("transformer.", "")
                k = k.replace("encoder.", "")

                if k in model_dict and v.shape == model_dict[k].shape:
                    pretrained_dict[k] = v

            if pretrained_dict:
                self.load_state_dict(pretrained_dict, strict=False)
                print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} Point-BERT params from {path}")
            else:
                print(f"Warning: No compatible weights found in {path}")

        except Exception as e:
            print(f"Warning: Could not load Point-BERT weights: {e}")
            print("Using random initialization.")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Encode point cloud.

        Args:
            points: [B, N, 3] input point cloud

        Returns:
            tokens: [B, num_groups+1, hidden_dim] output tokens (CLS + groups)
        """
        B = points.shape[0]

        # Tokenize
        tokens, _ = self.tokenizer(points)  # [B, num_groups, hidden_dim]

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # [B, num_groups+1, hidden_dim]

        # Add position embedding
        tokens = tokens + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Final norm
        tokens = self.norm(tokens)

        return tokens

    def get_cls_token(self, points: torch.Tensor) -> torch.Tensor:
        """Get CLS token embedding (global feature)."""
        tokens = self.forward(points)
        return tokens[:, 0]  # [B, hidden_dim]

    def get_group_tokens(self, points: torch.Tensor) -> torch.Tensor:
        """Get group tokens (local features)."""
        tokens = self.forward(points)
        return tokens[:, 1:]  # [B, num_groups, hidden_dim]
