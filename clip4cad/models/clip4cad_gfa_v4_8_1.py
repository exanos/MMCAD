"""
CLIP4CAD GFA v4.8.1 - Anchored Staged Learning with Hierarchical Codebook

Key Innovations:
1. Three-stage training (anchor -> align -> close gap)
2. Hierarchical codebook (category -> type -> variant + spatial)
3. Sparse code selection (variable-length representation)
4. Position-aware aggregation (captures spatial nuances)
5. BRep reconstruction auxiliary loss

Architecture:
- Stage 0: Anchor BRep to PC (no codebook, direct projection)
- Stage 1: Add text + hierarchical codebook
- Stage 2: Gap closing + hard negative mining

The key insight is that we need to establish ANCHORS first:
- PC encoder (ShapeLLM) is pre-trained and produces meaningful features
- BRep encoder is random -> anchor it to PC first
- Then add text and codebook when BRep is meaningful
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GFAv481Config:
    """Configuration for GFA v4.8.1 model."""

    # Feature dimensions (input)
    d_face: int = 48       # AutoBrep face FSQ features
    d_edge: int = 12       # AutoBrep edge FSQ features
    d_pc: int = 1024       # ShapeLLM point cloud features
    d_text: int = 3072     # Phi-4-mini text features

    # Model dimensions
    d: int = 256           # Internal unified dimension
    d_proj: int = 128      # Final projection output

    # Hierarchical Codebook
    n_category: int = 16           # Level 0: coarse categories
    n_type_per_cat: int = 8        # Level 1: 16 * 8 = 128 types
    n_variant_per_type: int = 4    # Level 2: 128 * 4 = 512 variants
    n_spatial: int = 16            # Spatial position codes
    code_sparsity: float = 0.1     # Activation threshold

    # Architecture
    num_heads: int = 8
    dropout: float = 0.1

    # BRep Encoder
    num_msg_layers: int = 3        # Topology message passing layers
    num_brep_tf_layers: int = 4    # BRep transformer layers
    max_bfs_levels: int = 32

    # Text Encoder
    num_text_tf_layers: int = 2

    # PC Encoder
    num_pc_tokens: int = 48        # Expected local PC tokens

    @property
    def total_codes(self) -> int:
        """Total number of codes across all levels."""
        return (
            self.n_category +
            self.n_category * self.n_type_per_cat +
            self.n_category * self.n_type_per_cat * self.n_variant_per_type +
            self.n_spatial
        )


# =============================================================================
# Hierarchical Codebook
# =============================================================================

class HierarchicalCodebook(nn.Module):
    """
    Three-level semantic codebook + spatial codes.

    Hierarchy:
    - Level 0: Category (16 codes) - coarse semantic type
    - Level 1: Type (128 codes) - specific feature type
    - Level 2: Variant (512 codes) - fine-grained variant
    - Spatial: Position (16 codes) - where in the model

    Total: 672 codes, but sparse selection typically uses ~10-50 per sample.
    """

    def __init__(self, config: GFAv481Config):
        super().__init__()
        self.config = config
        d = config.d

        # Level 0: Category codes
        self.category_codes = nn.Parameter(
            torch.randn(config.n_category, d) * 0.02
        )

        # Level 1: Type codes (hierarchical - each category has types)
        self.type_codes = nn.Parameter(
            torch.randn(config.n_category, config.n_type_per_cat, d) * 0.02
        )

        # Level 2: Variant codes (hierarchical)
        self.variant_codes = nn.Parameter(
            torch.randn(
                config.n_category,
                config.n_type_per_cat,
                config.n_variant_per_type,
                d
            ) * 0.02
        )

        # Spatial codes (independent of hierarchy)
        self.spatial_codes = nn.Parameter(
            torch.randn(config.n_spatial, d) * 0.02
        )

        # Temperature
        self.log_tau = nn.Parameter(torch.zeros(1))

        # Projections for hierarchical attention
        self.category_proj = nn.Linear(d, d)
        self.type_proj = nn.Linear(d, d)
        self.variant_proj = nn.Linear(d, d)
        self.spatial_proj = nn.Linear(d, d)

    @property
    def tau(self):
        return (self.log_tau.exp() + 0.1).clamp(0.1, 2.0)

    @property
    def total_codes(self):
        c = self.config
        return (
            c.n_category +
            c.n_category * c.n_type_per_cat +
            c.n_category * c.n_type_per_cat * c.n_variant_per_type +
            c.n_spatial
        )

    def get_flat_codes(self) -> torch.Tensor:
        """Get all codes as flat tensor for analysis."""
        codes = [
            self.category_codes,  # (16, d)
            self.type_codes.view(-1, self.config.d),  # (128, d)
            self.variant_codes.view(-1, self.config.d),  # (512, d)
            self.spatial_codes,  # (16, d)
        ]
        return torch.cat(codes, dim=0)  # (672, d)

    def initialize_from_text(self, text_features: torch.Tensor):
        """
        Initialize category codes from text encoder outputs using K-means.

        Args:
            text_features: (N, d) - Pooled text features
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("sklearn not available, using random initialization")
            return

        with torch.no_grad():
            feats_np = text_features.cpu().numpy()

            # Subsample if too many
            if len(feats_np) > 50000:
                indices = torch.randperm(len(feats_np))[:50000].numpy()
                feats_np = feats_np[indices]

            # Initialize category codes
            kmeans_cat = KMeans(n_clusters=self.config.n_category, random_state=42, n_init=10)
            kmeans_cat.fit(feats_np)
            self.category_codes.data = torch.tensor(
                kmeans_cat.cluster_centers_,
                dtype=self.category_codes.dtype,
                device=self.category_codes.device
            )

            # Initialize spatial codes with different clustering
            kmeans_spatial = KMeans(n_clusters=self.config.n_spatial, random_state=123, n_init=10)
            kmeans_spatial.fit(feats_np)
            self.spatial_codes.data = torch.tensor(
                kmeans_spatial.cluster_centers_,
                dtype=self.spatial_codes.dtype,
                device=self.spatial_codes.device
            )

            print(f"Initialized {self.config.n_category} category codes and {self.config.n_spatial} spatial codes")


# =============================================================================
# Hierarchical Codebook Grounding
# =============================================================================

class HierarchicalCodebookGrounding(nn.Module):
    """
    Ground tokens to hierarchical codebook with sparse selection.

    Process:
    1. Attend to category codes -> select top categories
    2. For selected categories, attend to type codes (gated by category)
    3. For selected types, attend to variant codes (gated by type)
    4. Attend to spatial codes (independent)
    5. Combine levels with learned weights + optional position gating
    """

    def __init__(self, config: GFAv481Config):
        super().__init__()
        self.config = config
        d = config.d

        # Key projection (applied to tokens)
        self.k_proj = nn.Linear(d, d)

        # Position-aware aggregation
        self.position_gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.Sigmoid()
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d)
        )

        # Level combination weights (learnable)
        self.level_weights = nn.Parameter(torch.ones(4) / 4)

    def forward(
        self,
        X: torch.Tensor,
        codebook: HierarchicalCodebook,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            X: (B, N, d) tokens
            codebook: Hierarchical codebook
            positions: (B, N, d) position encodings for position-aware aggregation
            mask: (B, N) valid token mask (True = valid)

        Returns:
            Dict with z, code activations at each level, grounding matrices
        """
        B, N, d = X.shape
        tau = codebook.tau
        threshold = self.config.code_sparsity
        device = X.device

        # FP16-safe mask value (use tensor to ensure correct dtype)
        mask_value = torch.finfo(X.dtype).min / 2  # Safe for any dtype

        # Project tokens once
        K = self.k_proj(X)  # (B, N, d)

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 0: Category attention
        # ─────────────────────────────────────────────────────────────────────
        Q_cat = codebook.category_proj(codebook.category_codes)  # (n_cat, d)

        attn_cat = torch.einsum('md,bnd->bmn', Q_cat, K) / math.sqrt(d)
        if mask is not None:
            attn_cat = attn_cat.masked_fill(~mask.bool().unsqueeze(1), mask_value)
        attn_cat = attn_cat.clamp(-50, 50)
        G_cat = F.softmax(attn_cat, dim=-1)  # (B, n_cat, N)
        G_cat = torch.nan_to_num(G_cat, nan=0.0)

        # Aggregate per category
        H_cat = torch.einsum('bmn,bnd->bmd', G_cat, X)  # (B, n_cat, d)

        # Category activation (sparse)
        w_cat_raw = H_cat.norm(dim=-1) / tau  # (B, n_cat)
        w_cat = F.softmax(w_cat_raw, dim=-1)
        w_cat_sparse = torch.where(w_cat > threshold, w_cat, torch.zeros_like(w_cat))
        w_cat_sparse = w_cat_sparse / (w_cat_sparse.sum(dim=-1, keepdim=True) + 1e-8)

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 1: Type attention (conditioned on active categories)
        # ─────────────────────────────────────────────────────────────────────
        n_type_total = self.config.n_category * self.config.n_type_per_cat
        type_codes_flat = codebook.type_codes.view(-1, d)  # (128, d)
        Q_type = codebook.type_proj(type_codes_flat)

        attn_type = torch.einsum('md,bnd->bmn', Q_type, K) / math.sqrt(d)
        if mask is not None:
            attn_type = attn_type.masked_fill(~mask.bool().unsqueeze(1), mask_value)
        attn_type = attn_type.clamp(-50, 50)
        G_type = F.softmax(attn_type, dim=-1)  # (B, n_type_total, N)
        G_type = torch.nan_to_num(G_type, nan=0.0)

        H_type = torch.einsum('bmn,bnd->bmd', G_type, X)  # (B, n_type_total, d)

        # Type activation (sparse, modulated by category)
        w_type_raw = H_type.norm(dim=-1) / tau  # (B, n_type_total)
        # Reshape to (B, n_cat, n_type) and modulate by category weights
        w_type_reshaped = w_type_raw.view(B, self.config.n_category, self.config.n_type_per_cat)
        w_type_reshaped = w_type_reshaped * w_cat_sparse.unsqueeze(-1)  # Category gating
        w_type_raw_gated = w_type_reshaped.view(B, -1)
        w_type = F.softmax(w_type_raw_gated, dim=-1)
        w_type_sparse = torch.where(w_type > threshold / 2, w_type, torch.zeros_like(w_type))
        w_type_sparse = w_type_sparse / (w_type_sparse.sum(dim=-1, keepdim=True) + 1e-8)

        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 2: Variant attention (conditioned on active types)
        # ─────────────────────────────────────────────────────────────────────
        n_var_total = n_type_total * self.config.n_variant_per_type
        variant_codes_flat = codebook.variant_codes.view(-1, d)  # (512, d)
        Q_var = codebook.variant_proj(variant_codes_flat)

        attn_var = torch.einsum('md,bnd->bmn', Q_var, K) / math.sqrt(d)
        if mask is not None:
            attn_var = attn_var.masked_fill(~mask.bool().unsqueeze(1), mask_value)
        attn_var = attn_var.clamp(-50, 50)
        G_var = F.softmax(attn_var, dim=-1)
        G_var = torch.nan_to_num(G_var, nan=0.0)

        H_var = torch.einsum('bmn,bnd->bmd', G_var, X)  # (B, n_var_total, d)

        # Variant activation (sparse, modulated by type)
        w_var_raw = H_var.norm(dim=-1) / tau
        # Modulate by type weights (expanded)
        w_type_expanded = w_type_sparse.view(B, -1, 1).expand(-1, -1, self.config.n_variant_per_type)
        w_type_expanded = w_type_expanded.reshape(B, n_var_total)
        w_var_raw_gated = w_var_raw * w_type_expanded
        w_var = F.softmax(w_var_raw_gated, dim=-1)
        w_var_sparse = torch.where(w_var > threshold / 4, w_var, torch.zeros_like(w_var))
        w_var_sparse = w_var_sparse / (w_var_sparse.sum(dim=-1, keepdim=True) + 1e-8)

        # ─────────────────────────────────────────────────────────────────────
        # SPATIAL: Independent spatial attention
        # ─────────────────────────────────────────────────────────────────────
        Q_spatial = codebook.spatial_proj(codebook.spatial_codes)

        attn_spatial = torch.einsum('md,bnd->bmn', Q_spatial, K) / math.sqrt(d)
        if mask is not None:
            attn_spatial = attn_spatial.masked_fill(~mask.bool().unsqueeze(1), mask_value)
        attn_spatial = attn_spatial.clamp(-50, 50)
        G_spatial = F.softmax(attn_spatial, dim=-1)
        G_spatial = torch.nan_to_num(G_spatial, nan=0.0)

        H_spatial = torch.einsum('bmn,bnd->bmd', G_spatial, X)  # (B, n_spatial, d)

        w_spatial_raw = H_spatial.norm(dim=-1) / tau
        w_spatial = F.softmax(w_spatial_raw, dim=-1)
        w_spatial_sparse = torch.where(w_spatial > threshold, w_spatial, torch.zeros_like(w_spatial))
        w_spatial_sparse = w_spatial_sparse / (w_spatial_sparse.sum(dim=-1, keepdim=True) + 1e-8)

        # ─────────────────────────────────────────────────────────────────────
        # AGGREGATE with position awareness
        # ─────────────────────────────────────────────────────────────────────

        # Level contributions
        z_cat = torch.einsum('bm,bmd->bd', w_cat_sparse, H_cat)
        z_type = torch.einsum('bm,bmd->bd', w_type_sparse, H_type)
        z_var = torch.einsum('bm,bmd->bd', w_var_sparse, H_var)
        z_spatial = torch.einsum('bm,bmd->bd', w_spatial_sparse, H_spatial)

        # Position-aware gating if positions provided
        if positions is not None:
            # Pool position info
            if mask is not None:
                mask_float = mask.float().unsqueeze(-1)
                pos_pooled = (positions * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
            else:
                pos_pooled = positions.mean(dim=1)

            # Gate semantic levels by position
            gate_input = torch.cat([z_cat, pos_pooled], dim=-1)
            pos_gate = self.position_gate(gate_input)
            z_cat = z_cat * pos_gate
            z_type = z_type * pos_gate
            z_var = z_var * pos_gate

        # Combine levels with learned weights
        level_w = F.softmax(self.level_weights, dim=0)
        z = level_w[0] * z_cat + level_w[1] * z_type + level_w[2] * z_var + level_w[3] * z_spatial

        z = self.out_proj(z)
        z = torch.nan_to_num(z, nan=0.0)

        # Count active codes
        n_active = (
            (w_cat_sparse > 0).sum(dim=-1).float().mean() +
            (w_type_sparse > 0).sum(dim=-1).float().mean() +
            (w_var_sparse > 0).sum(dim=-1).float().mean() +
            (w_spatial_sparse > 0).sum(dim=-1).float().mean()
        )

        return {
            'z': z,
            'w_category': w_cat_sparse,
            'w_type': w_type_sparse,
            'w_variant': w_var_sparse,
            'w_spatial': w_spatial_sparse,
            'G_category': G_cat,
            'G_spatial': G_spatial,
            'n_active_codes': n_active,
            'level_weights': level_w,
        }


# =============================================================================
# Edge Message Layer (with gated residuals)
# =============================================================================

class EdgeMessageLayer(nn.Module):
    """Message passing between faces and edges through topology with gated residuals."""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()

        # Face -> Edge message
        self.face_to_edge = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d)
        )

        # Edge -> Face message
        self.edge_to_face = nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d)
        )

        # Layer norms
        self.norm_f = nn.LayerNorm(d)
        self.norm_e = nn.LayerNorm(d)

        # Gating for residual (prevents gradient explosion)
        self.gate_f = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.gate_e = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())

    def forward(
        self,
        F: torch.Tensor,
        E: torch.Tensor,
        edge_to_faces: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            F: (B, N_f, d) - Face features
            E: (B, N_e, d) - Edge features
            edge_to_faces: (B, N_e, 2) - For each edge, indices of connected faces
            face_mask: (B, N_f) - Valid face mask
            edge_mask: (B, N_e) - Valid edge mask

        Returns:
            F_new, E_new: Updated features
        """
        B, N_e, d = E.shape
        N_f = F.shape[1]

        # Clamp indices to valid range
        f1_idx = edge_to_faces[:, :, 0].clamp(0, N_f - 1).long()
        f2_idx = edge_to_faces[:, :, 1].clamp(0, N_f - 1).long()

        # Valid edge mask
        valid_edge = edge_mask.bool() & (edge_to_faces[:, :, 0] >= 0) & (edge_to_faces[:, :, 1] >= 0)

        # ─────────────────────────────────────────────────────────────────────
        # FACE -> EDGE
        # ─────────────────────────────────────────────────────────────────────
        f1 = torch.gather(F, 1, f1_idx.unsqueeze(-1).expand(-1, -1, d))
        f2 = torch.gather(F, 1, f2_idx.unsqueeze(-1).expand(-1, -1, d))

        msg_e = self.face_to_edge(torch.cat([E, f1, f2], dim=-1))
        msg_e = msg_e * valid_edge.unsqueeze(-1).float()
        gate_e = self.gate_e(torch.cat([E, msg_e], dim=-1))
        E_new = self.norm_e(E + gate_e * msg_e)

        # ─────────────────────────────────────────────────────────────────────
        # EDGE -> FACE
        # ─────────────────────────────────────────────────────────────────────
        face_msg = torch.zeros_like(F)
        face_count = torch.zeros(B, N_f, 1, device=F.device)

        edge_contrib = E_new * valid_edge.unsqueeze(-1).float()
        count_contrib = valid_edge.unsqueeze(-1).float()

        face_msg.scatter_add_(1, f1_idx.unsqueeze(-1).expand(-1, -1, d), edge_contrib)
        face_count.scatter_add_(1, f1_idx.unsqueeze(-1), count_contrib)
        face_msg.scatter_add_(1, f2_idx.unsqueeze(-1).expand(-1, -1, d), edge_contrib)
        face_count.scatter_add_(1, f2_idx.unsqueeze(-1), count_contrib)

        face_msg = face_msg / (face_count + 1e-8)

        msg_f = self.edge_to_face(torch.cat([F, face_msg], dim=-1))
        gate_f = self.gate_f(torch.cat([F, msg_f], dim=-1))
        F_new = self.norm_f(F + gate_f * msg_f * face_mask.unsqueeze(-1).float())

        return F_new, E_new


# =============================================================================
# Topology-Aware BRep Encoder (with position output)
# =============================================================================

class TopologyBRepEncoder(nn.Module):
    """
    Topology-aware BRep encoder with message passing.

    Returns both encoded tokens AND position encodings for position-aware grounding.
    """

    def __init__(self, config: GFAv481Config):
        super().__init__()
        self.config = config
        d = config.d

        # Geometry projections
        self.face_proj = nn.Sequential(
            nn.Linear(config.d_face, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(config.d_edge, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # Spatial embeddings
        self.centroid_proj = nn.Linear(3, d)
        self.normal_proj = nn.Linear(3, d)
        self.area_proj = nn.Linear(1, d)
        self.level_emb = nn.Embedding(config.max_bfs_levels, d)

        # Edge spatial
        self.edge_midpoint_proj = nn.Linear(3, d)
        self.edge_length_proj = nn.Linear(1, d)

        # Type embeddings
        self.face_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Message passing
        self.msg_layers = nn.ModuleList([
            EdgeMessageLayer(d, config.dropout)
            for _ in range(config.num_msg_layers)
        ])

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=config.num_heads,
                dim_feedforward=d * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=config.num_brep_tf_layers
        )

        self.norm = nn.LayerNorm(d)

        # Position encoding output (for position-aware grounding)
        self.position_encoder = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d)
        )

    def forward(
        self,
        face_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_to_faces: torch.Tensor,
        face_centroids: torch.Tensor,
        face_normals: torch.Tensor,
        face_areas: torch.Tensor,
        bfs_level: torch.Tensor,
        edge_midpoints: Optional[torch.Tensor] = None,
        edge_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            face_feats: (B, N_f, d_face)
            edge_feats: (B, N_e, d_edge)
            face_mask, edge_mask: (B, N_f), (B, N_e)
            edge_to_faces: (B, N_e, 2)
            face_centroids: (B, N_f, 3)
            face_normals: (B, N_f, 3)
            face_areas: (B, N_f)
            bfs_level: (B, N_f)
            edge_midpoints: (B, N_e, 3) optional
            edge_lengths: (B, N_e) optional

        Returns:
            X: (B, N_f + N_e, d) - Encoded tokens
            mask: (B, N_f + N_e) - Combined mask
            positions: (B, N_f + N_e, d) - Position encodings
        """
        B = face_feats.shape[0]
        N_f = face_feats.shape[1]
        N_e = edge_feats.shape[1]
        d = self.config.d
        device = face_feats.device

        # ─────────────────────────────────────────────────────────────────────
        # SANITIZE SPATIAL FIELDS
        # ─────────────────────────────────────────────────────────────────────
        face_centroids = torch.nan_to_num(face_centroids, nan=0.0).clamp(-1e4, 1e4)
        face_normals = torch.nan_to_num(face_normals, nan=0.0).clamp(-1.0, 1.0)
        face_areas = torch.nan_to_num(face_areas, nan=0.0).clamp(0, 1e6)

        # ─────────────────────────────────────────────────────────────────────
        # PROJECT AND EMBED FACES
        # ─────────────────────────────────────────────────────────────────────
        F = self.face_proj(face_feats.float())
        F = F + self.face_type

        # Spatial embeddings
        spatial_f = (
            self.centroid_proj(face_centroids.float()) +
            self.normal_proj(face_normals.float()) +
            self.area_proj(face_areas.float().unsqueeze(-1)) +
            self.level_emb(bfs_level.clamp(0, self.config.max_bfs_levels - 1).long())
        )
        F = F + spatial_f

        # ─────────────────────────────────────────────────────────────────────
        # PROJECT AND EMBED EDGES
        # ─────────────────────────────────────────────────────────────────────
        E = self.edge_proj(edge_feats.float())
        E = E + self.edge_type

        if edge_midpoints is not None:
            edge_midpoints = torch.nan_to_num(edge_midpoints, nan=0.0).clamp(-1e4, 1e4)
            E = E + self.edge_midpoint_proj(edge_midpoints.float())

        if edge_lengths is not None:
            edge_lengths = torch.nan_to_num(edge_lengths, nan=0.0).clamp(0, 1e6)
            E = E + self.edge_length_proj(edge_lengths.float().unsqueeze(-1))

        # ─────────────────────────────────────────────────────────────────────
        # TOPOLOGY MESSAGE PASSING
        # ─────────────────────────────────────────────────────────────────────
        for layer in self.msg_layers:
            F, E = layer(F, E, edge_to_faces, face_mask, edge_mask)

        # ─────────────────────────────────────────────────────────────────────
        # TRANSFORMER ENCODING
        # ─────────────────────────────────────────────────────────────────────
        X = torch.cat([F, E], dim=1)
        mask = torch.cat([face_mask, edge_mask], dim=1).bool()

        X = self.transformer(X, src_key_padding_mask=~mask)
        X = self.norm(X)
        X = torch.nan_to_num(X, nan=0.0)

        # ─────────────────────────────────────────────────────────────────────
        # POSITION ENCODINGS (for position-aware grounding)
        # ─────────────────────────────────────────────────────────────────────
        # Face positions from spatial embeddings
        positions_f = self.position_encoder(spatial_f)

        # Edge positions (interpolate from connected faces or use zeros)
        positions_e = torch.zeros(B, N_e, d, device=device)

        positions = torch.cat([positions_f, positions_e], dim=1)
        positions = torch.nan_to_num(positions, nan=0.0)

        return X, mask, positions


# =============================================================================
# Text Encoder
# =============================================================================

class TextEncoder(nn.Module):
    """Text encoder with projection and transformer."""

    def __init__(self, config: GFAv481Config):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(config.d_text, config.d),
            nn.LayerNorm(config.d)
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d,
                nhead=config.num_heads,
                dim_feedforward=config.d * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=config.num_text_tf_layers
        )

        self.norm = nn.LayerNorm(config.d)

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            X: (B, T, d_text) - Phi-4 text embeddings
            mask: (B, T) - Valid token mask

        Returns:
            X_text: (B, T, d) - Projected text tokens
            mask: Unchanged
        """
        X = self.proj(X.float())

        if mask is not None:
            X = self.encoder(X, src_key_padding_mask=~mask.bool())
        else:
            X = self.encoder(X)

        X = self.norm(X)
        X = torch.nan_to_num(X, nan=0.0)

        return X, mask


# =============================================================================
# Point Cloud Encoder
# =============================================================================

class PCEncoder(nn.Module):
    """Point cloud encoder (simple projection)."""

    def __init__(self, config: GFAv481Config):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(config.d_pc, config.d),
            nn.LayerNorm(config.d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d, config.d),
            nn.LayerNorm(config.d)
        )

    def forward(
        self,
        pc_local: torch.Tensor,
        pc_global: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pc_local: (B, N, d_pc) - Local PC features
            pc_global: (B, d_pc) - Global PC feature

        Returns:
            X_pc: (B, N+1, d) - Projected PC tokens
        """
        # Add global as extra token
        pc_global = pc_global.unsqueeze(1)  # (B, 1, d_pc)
        X = torch.cat([pc_local, pc_global], dim=1)  # (B, N+1, d_pc)

        X = self.proj(X.float())
        X = torch.nan_to_num(X, nan=0.0)

        return X


# =============================================================================
# BRep Decoder (for reconstruction auxiliary loss)
# =============================================================================

class BRepDecoder(nn.Module):
    """
    Lightweight decoder for BRep reconstruction.
    Reconstructs pooled face features from global embedding.
    """

    def __init__(self, d: int, d_face: int = 48):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d_face)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d) - Global embedding

        Returns:
            recon: (B, d_face) - Reconstructed pooled face features
        """
        return self.decoder(z)


# =============================================================================
# Main Model: CLIP4CAD_GFA_v481
# =============================================================================

class CLIP4CAD_GFA_v481(nn.Module):
    """
    GFA v4.8.1: Anchored Staged Learning with Hierarchical Codebook

    Key features:
    1. Three-stage training (anchor -> align -> close gap)
    2. Hierarchical codebook (category -> type -> variant + spatial)
    3. Sparse code selection (variable-length representation)
    4. Position-aware aggregation (captures spatial nuances)
    5. BRep reconstruction auxiliary loss
    """

    def __init__(self, config: GFAv481Config):
        super().__init__()
        self.config = config
        d = config.d

        # ─────────────────────────────────────────────────────────────────────
        # ENCODERS
        # ─────────────────────────────────────────────────────────────────────
        self.text_encoder = TextEncoder(config)
        self.brep_encoder = TopologyBRepEncoder(config)
        self.pc_encoder = PCEncoder(config)

        # ─────────────────────────────────────────────────────────────────────
        # HIERARCHICAL CODEBOOK
        # ─────────────────────────────────────────────────────────────────────
        self.codebook = HierarchicalCodebook(config)

        # ─────────────────────────────────────────────────────────────────────
        # CODEBOOK GROUNDING (one per modality)
        # ─────────────────────────────────────────────────────────────────────
        self.text_grounding = HierarchicalCodebookGrounding(config)
        self.brep_grounding = HierarchicalCodebookGrounding(config)
        self.pc_grounding = HierarchicalCodebookGrounding(config)

        # ─────────────────────────────────────────────────────────────────────
        # BREP RECONSTRUCTION (auxiliary)
        # ─────────────────────────────────────────────────────────────────────
        self.brep_decoder = BRepDecoder(d, config.d_face)

        # ─────────────────────────────────────────────────────────────────────
        # DIRECT PROJECTION (for Stage 0: BRep-PC anchoring)
        # ─────────────────────────────────────────────────────────────────────
        self.brep_direct_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d)
        )
        self.pc_direct_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d)
        )

        # ─────────────────────────────────────────────────────────────────────
        # OUTPUT PROJECTION
        # ─────────────────────────────────────────────────────────────────────
        self.proj_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.d_proj)
        )

        # Contrastive temperature
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))

    @property
    def tau(self):
        return self.log_tau.exp().clamp(0.01, 1.0)

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_pc_encoder(self):
        """Freeze PC encoder for Stage 0 (use as anchor)."""
        for param in self.pc_encoder.parameters():
            param.requires_grad = False
        for param in self.pc_direct_proj.parameters():
            param.requires_grad = False
        print("PC encoder frozen (anchor mode)")

    def unfreeze_pc_encoder(self):
        """Unfreeze PC encoder for Stage 1+."""
        for param in self.pc_encoder.parameters():
            param.requires_grad = True
        for param in self.pc_direct_proj.parameters():
            param.requires_grad = True
        print("PC encoder unfrozen")

    def forward_stage0(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Stage 0: Anchor BRep to PC.
        Only uses BRep and PC encoders, NO codebook.
        """
        device = next(self.parameters()).device

        # Encode BRep
        X_brep, brep_mask, _ = self.brep_encoder(
            face_feats=batch['face_features'].to(device),
            edge_feats=batch['edge_features'].to(device),
            face_mask=batch['face_mask'].to(device),
            edge_mask=batch['edge_mask'].to(device),
            edge_to_faces=batch['edge_to_faces'].to(device).long(),
            face_centroids=batch['face_centroids'].to(device),
            face_normals=batch['face_normals'].to(device),
            face_areas=batch['face_areas'].to(device),
            bfs_level=batch['bfs_level'].to(device).long(),
            edge_midpoints=batch.get('edge_midpoints', torch.zeros(batch['edge_features'].shape[0], batch['edge_features'].shape[1], 3)).to(device),
            edge_lengths=batch.get('edge_lengths', torch.zeros(batch['edge_features'].shape[0], batch['edge_features'].shape[1])).to(device),
        )

        # Encode PC
        X_pc = self.pc_encoder(
            batch['pc_local_features'].to(device),
            batch['pc_global_features'].to(device),
        )

        # Pool to global
        mask_float = brep_mask.float().unsqueeze(-1)
        z_brep_pooled = (X_brep * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
        z_pc_pooled = X_pc.mean(dim=1)

        # Direct projection (bypass codebook)
        z_brep_raw = self.brep_direct_proj(z_brep_pooled)
        z_pc_raw = self.pc_direct_proj(z_pc_pooled)

        # Project to output space
        z_brep = self.proj_head(z_brep_raw)
        z_pc = self.proj_head(z_pc_raw)

        # Reconstruction target
        face_mask = batch['face_mask'].to(device)
        face_feats = batch['face_features'].to(device).float()
        face_pooled = (face_feats * face_mask.float().unsqueeze(-1)).sum(1) / face_mask.float().sum(1, keepdim=True).clamp(min=1)
        recon = self.brep_decoder(z_brep_raw)

        return {
            'z_brep': z_brep,
            'z_pc': z_pc,
            'z_brep_raw': z_brep_raw,
            'z_pc_raw': z_pc_raw,
            'recon': recon,
            'recon_target': face_pooled,
            'tau': self.tau,
        }

    def forward(self, batch: Dict[str, torch.Tensor], stage: int = 1) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with hierarchical codebook.

        Args:
            batch: Input batch
            stage: 0 = anchor (BRep-PC only), 1 = align (3-way), 2 = close gap
        """
        if stage == 0:
            return self.forward_stage0(batch)

        device = next(self.parameters()).device

        # ─────────────────────────────────────────────────────────────────────
        # ENCODE MODALITIES
        # ─────────────────────────────────────────────────────────────────────

        # Text
        text_features = batch['text_features'].to(device)
        text_mask = batch.get('text_mask')
        if text_mask is not None:
            text_mask = text_mask.to(device)
        X_text, text_mask = self.text_encoder(text_features, text_mask)

        # BRep (with positions)
        X_brep, brep_mask, brep_positions = self.brep_encoder(
            face_feats=batch['face_features'].to(device),
            edge_feats=batch['edge_features'].to(device),
            face_mask=batch['face_mask'].to(device),
            edge_mask=batch['edge_mask'].to(device),
            edge_to_faces=batch['edge_to_faces'].to(device).long(),
            face_centroids=batch['face_centroids'].to(device),
            face_normals=batch['face_normals'].to(device),
            face_areas=batch['face_areas'].to(device),
            bfs_level=batch['bfs_level'].to(device).long(),
            edge_midpoints=batch.get('edge_midpoints', torch.zeros(batch['edge_features'].shape[0], batch['edge_features'].shape[1], 3)).to(device),
            edge_lengths=batch.get('edge_lengths', torch.zeros(batch['edge_features'].shape[0], batch['edge_features'].shape[1])).to(device),
        )

        # PC
        X_pc = self.pc_encoder(
            batch['pc_local_features'].to(device),
            batch['pc_global_features'].to(device),
        )

        # ─────────────────────────────────────────────────────────────────────
        # GROUND TO HIERARCHICAL CODEBOOK
        # ─────────────────────────────────────────────────────────────────────
        text_out = self.text_grounding(X_text, self.codebook, None, text_mask)
        brep_out = self.brep_grounding(X_brep, self.codebook, brep_positions, brep_mask)
        pc_out = self.pc_grounding(X_pc, self.codebook, None, None)

        z_text_raw = text_out['z']
        z_brep_raw = brep_out['z']
        z_pc_raw = pc_out['z']

        # ─────────────────────────────────────────────────────────────────────
        # PROJECT TO OUTPUT SPACE
        # ─────────────────────────────────────────────────────────────────────
        z_text = self.proj_head(z_text_raw)
        z_brep = self.proj_head(z_brep_raw)
        z_pc = self.proj_head(z_pc_raw)

        # ─────────────────────────────────────────────────────────────────────
        # RECONSTRUCTION (auxiliary)
        # ─────────────────────────────────────────────────────────────────────
        face_mask = batch['face_mask'].to(device)
        face_feats = batch['face_features'].to(device).float()
        face_pooled = (face_feats * face_mask.float().unsqueeze(-1)).sum(1) / face_mask.float().sum(1, keepdim=True).clamp(min=1)
        recon = self.brep_decoder(z_brep_raw)

        return {
            # Final embeddings
            'z_text': z_text,
            'z_brep': z_brep,
            'z_pc': z_pc,

            # Raw embeddings (before proj_head)
            'z_text_raw': z_text_raw,
            'z_brep_raw': z_brep_raw,
            'z_pc_raw': z_pc_raw,

            # Code activations (hierarchical)
            'w_text': {
                'category': text_out['w_category'],
                'type': text_out['w_type'],
                'variant': text_out['w_variant'],
                'spatial': text_out['w_spatial'],
            },
            'w_brep': {
                'category': brep_out['w_category'],
                'type': brep_out['w_type'],
                'variant': brep_out['w_variant'],
                'spatial': brep_out['w_spatial'],
            },
            'w_pc': {
                'category': pc_out['w_category'],
                'type': pc_out['w_type'],
                'variant': pc_out['w_variant'],
                'spatial': pc_out['w_spatial'],
            },

            # Grounding matrices
            'G_text': text_out['G_category'],
            'G_brep': brep_out['G_category'],
            'G_pc': pc_out['G_category'],

            # Active code counts
            'n_active': {
                'text': text_out['n_active_codes'],
                'brep': brep_out['n_active_codes'],
                'pc': pc_out['n_active_codes'],
            },

            # Level weights
            'level_weights': brep_out['level_weights'],

            # Reconstruction
            'recon': recon,
            'recon_target': face_pooled,

            'tau': self.tau,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CONVENIENCE METHODS FOR INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def encode_text(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode text only (for retrieval)."""
        device = next(self.parameters()).device

        text_features = batch['text_features'].to(device).float()
        text_mask = batch.get('text_mask')
        if text_mask is not None:
            text_mask = text_mask.to(device)

        X_text, text_mask = self.text_encoder(text_features, text_mask)
        text_out = self.text_grounding(X_text, self.codebook, None, text_mask)
        z_text = self.proj_head(text_out['z'])

        return z_text

    def encode_brep(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode BRep only (for geometry-only retrieval)."""
        device = next(self.parameters()).device

        X_brep, brep_mask, brep_positions = self.brep_encoder(
            face_feats=batch['face_features'].to(device).float(),
            edge_feats=batch['edge_features'].to(device).float(),
            face_mask=batch['face_mask'].to(device),
            edge_mask=batch['edge_mask'].to(device),
            edge_to_faces=batch['edge_to_faces'].to(device).long(),
            face_centroids=batch['face_centroids'].to(device).float(),
            face_normals=batch['face_normals'].to(device).float(),
            face_areas=batch['face_areas'].to(device).float(),
            bfs_level=batch['bfs_level'].to(device).long(),
            edge_midpoints=batch.get('edge_midpoints', torch.zeros(batch['edge_features'].shape[0], batch['edge_features'].shape[1], 3)).to(device).float(),
            edge_lengths=batch.get('edge_lengths', torch.zeros(batch['edge_features'].shape[0], batch['edge_features'].shape[1])).to(device).float(),
        )

        brep_out = self.brep_grounding(X_brep, self.codebook, brep_positions, brep_mask)
        z_brep = self.proj_head(brep_out['z'])

        return z_brep

    def encode_pc(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode point cloud only."""
        device = next(self.parameters()).device

        X_pc = self.pc_encoder(
            batch['pc_local_features'].to(device).float(),
            batch['pc_global_features'].to(device).float()
        )

        pc_out = self.pc_grounding(X_pc, self.codebook, None, None)
        z_pc = self.proj_head(pc_out['z'])

        return z_pc

    def initialize_codebook(
        self,
        dataloader,
        device,
        remap_fn=None,
        max_samples: int = 10000,
        max_batches: int = 50
    ):
        """
        Initialize codebook from text encoder outputs using K-means.

        Args:
            dataloader: DataLoader with text features
            device: Device to use
            remap_fn: Optional function to remap batch keys
            max_samples: Maximum samples to use
            max_batches: Maximum batches to process
        """
        print("Initializing hierarchical codebook from text encoder outputs...")

        all_features = []
        self.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                if remap_fn is not None:
                    batch = remap_fn(batch)

                # Cast to float32 for stability
                text_features = batch['text_features'].to(device).float()
                text_mask = batch.get('text_mask')
                if text_mask is not None:
                    text_mask = text_mask.to(device)

                X_text, mask = self.text_encoder(text_features, text_mask)

                # Pool text features
                if mask is not None:
                    mask_float = mask.float().unsqueeze(-1)
                    pooled = (X_text * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
                else:
                    pooled = X_text.mean(dim=1)

                all_features.append(pooled.cpu())

                if len(all_features) * pooled.shape[0] >= max_samples:
                    break

        all_features = torch.cat(all_features, dim=0)
        self.codebook.initialize_from_text(all_features)
        print(f"Initialized codebook with {self.codebook.total_codes} total codes from {len(all_features)} samples")

        self.train()
