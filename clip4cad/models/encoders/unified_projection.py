"""
Unified Input Projection

Projects tokens from different encoders to a common dimension,
adding learnable modality embeddings.
"""

import torch
import torch.nn as nn
from typing import Tuple


class UnifiedInputProjection(nn.Module):
    """
    Projects tokens from different encoders to unified dimension.
    Adds learnable modality embeddings to distinguish token sources.
    """

    def __init__(
        self,
        d_unified: int = 256,
        d_brep_face: int = 32,
        d_brep_edge: int = 16,
        d_pointbert: int = 384,
    ):
        """
        Args:
            d_unified: Unified output dimension
            d_brep_face: B-Rep face encoder output dimension
            d_brep_edge: B-Rep edge encoder output dimension
            d_pointbert: Point-BERT output dimension
        """
        super().__init__()

        self.d_unified = d_unified

        # B-Rep projections
        self.face_proj = nn.Sequential(
            nn.Linear(d_brep_face, d_unified),
            nn.LayerNorm(d_unified),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(d_brep_edge, d_unified),
            nn.LayerNorm(d_unified),
        )

        # Point cloud projection
        self.pc_proj = nn.Sequential(
            nn.Linear(d_pointbert, d_unified),
            nn.LayerNorm(d_unified),
        )

        # Learnable modality embeddings (distinguish token sources)
        self.face_embed = nn.Parameter(torch.zeros(1, 1, d_unified))
        self.edge_embed = nn.Parameter(torch.zeros(1, 1, d_unified))
        self.pc_embed = nn.Parameter(torch.zeros(1, 1, d_unified))

        # Initialize
        nn.init.trunc_normal_(self.face_embed, std=0.02)
        nn.init.trunc_normal_(self.edge_embed, std=0.02)
        nn.init.trunc_normal_(self.pc_embed, std=0.02)

    def project_brep(
        self,
        face_tokens: torch.Tensor,
        edge_tokens: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project B-Rep tokens to unified space.

        Args:
            face_tokens: [B, F, face_dim]
            edge_tokens: [B, E, edge_dim]
            face_mask: [B, F] validity mask
            edge_mask: [B, E] validity mask

        Returns:
            tokens: [B, F+E, d_unified] projected tokens
            mask: [B, F+E] combined mask
        """
        # Project and add modality embeddings
        face_proj = self.face_proj(face_tokens) + self.face_embed
        edge_proj = self.edge_proj(edge_tokens) + self.edge_embed

        # Concatenate
        tokens = torch.cat([face_proj, edge_proj], dim=1)
        mask = torch.cat([face_mask, edge_mask], dim=1)

        # Zero out padding
        tokens = tokens * mask.unsqueeze(-1)

        return tokens, mask

    def project_pointcloud(self, pc_tokens: torch.Tensor) -> torch.Tensor:
        """
        Project point cloud tokens to unified space.

        Args:
            pc_tokens: [B, num_tokens, pc_dim]

        Returns:
            tokens: [B, num_tokens, d_unified]
        """
        tokens = self.pc_proj(pc_tokens) + self.pc_embed
        return tokens

    def project_faces_only(
        self,
        face_tokens: torch.Tensor,
        face_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project face tokens only."""
        face_proj = self.face_proj(face_tokens) + self.face_embed
        face_proj = face_proj * face_mask.unsqueeze(-1)
        return face_proj, face_mask

    def project_edges_only(
        self,
        edge_tokens: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project edge tokens only."""
        edge_proj = self.edge_proj(edge_tokens) + self.edge_embed
        edge_proj = edge_proj * edge_mask.unsqueeze(-1)
        return edge_proj, edge_mask
