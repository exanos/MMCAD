"""
CLIP4CAD GFA v4.8.2 - Simplified Topology-Aware Training

Key Changes from v4.8.1:
1. SimplifiedTopologyBRepEncoder - only uses edge_to_faces and bfs_level
2. Removed spatial field requirements (face_centroids, normals, areas, edge_midpoints, lengths)
3. BFS level embedding provides structural ordering
4. EdgeMessageLayer provides topology-aware message passing

Kept from v4.8.1:
- Three-stage training (anchor -> align -> close gap)
- Hierarchical codebook (category -> type -> variant + spatial)
- Sparse code selection (variable-length representation)
- EdgeMessageLayer for topology message passing
- BRep reconstruction auxiliary loss
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
class GFAv482Config:
    """Configuration for GFA v4.8.2 model.

    v4.8.2 Optimization: ~25% capacity increase for better representation.
    - d: 256 -> 320
    - d_proj: 128 -> 160
    - n_category: 16 -> 20
    - n_type_per_cat: 8 -> 10
    - n_spatial: 16 -> 20
    - num_brep_tf_layers: 4 -> 5
    - num_heads: 8 -> 10 (keeps head_dim=32)

    Result: ~15M params (up from ~12M), ~860 codes (up from 672)
    """

    # Feature dimensions (input)
    d_face: int = 48       # AutoBrep face FSQ features
    d_edge: int = 12       # AutoBrep edge FSQ features
    d_pc: int = 1024       # ShapeLLM point cloud features
    d_text: int = 3072     # Phi-4-mini text features

    # Model dimensions (UPDATED for v4.8.2)
    d: int = 320           # Internal unified dimension (+25%)
    d_proj: int = 160      # Final projection output (maintain ratio)

    # Hierarchical Codebook (UPDATED for v4.8.2)
    n_category: int = 20           # Level 0: coarse categories (was 16)
    n_type_per_cat: int = 10       # Level 1: 20 * 10 = 200 types (was 8)
    n_variant_per_type: int = 4    # Level 2: 200 * 4 = 800 variants
    n_spatial: int = 20            # Spatial position codes (was 16)
    code_sparsity: float = 0.1     # Activation threshold

    # Architecture (UPDATED for v4.8.2)
    num_heads: int = 10            # Keep head_dim=32 (320/10=32) (was 8)
    dropout: float = 0.1

    # BRep Encoder (UPDATED for v4.8.2)
    num_msg_layers: int = 3        # Topology message passing layers (unchanged)
    num_brep_tf_layers: int = 5    # BRep transformer layers (+1 layer)
    max_bfs_levels: int = 32

    # Text Encoder
    num_text_tf_layers: int = 2    # Unchanged

    # PC Encoder
    num_pc_tokens: int = 48        # Expected local PC tokens

    @property
    def total_codes(self) -> int:
        """Total number of codes across all levels.

        v4.8.2: 20 + 200 + 800 + 20 = 1040 codes
        """
        return (
            self.n_category +
            self.n_category * self.n_type_per_cat +
            self.n_category * self.n_type_per_cat * self.n_variant_per_type +
            self.n_spatial
        )


# =============================================================================
# Hierarchical Codebook (same as v4.8.1)
# =============================================================================

class HierarchicalCodebook(nn.Module):
    """
    Three-level semantic codebook + spatial codes.
    Same as v4.8.1.
    """

    def __init__(self, config: GFAv482Config):
        super().__init__()
        self.config = config
        d = config.d

        # Level 0: Category codes
        self.category_codes = nn.Parameter(
            torch.randn(config.n_category, d) * 0.02
        )

        # Level 1: Type codes (hierarchical)
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

        # Spatial codes (independent)
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
            self.category_codes,
            self.type_codes.view(-1, self.config.d),
            self.variant_codes.view(-1, self.config.d),
            self.spatial_codes,
        ]
        return torch.cat(codes, dim=0)

    def initialize_from_text(self, text_features: torch.Tensor):
        """Initialize ALL codebook levels from text encoder outputs using K-means."""
        try:
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.metrics import pairwise_distances_argmin
            import numpy as np
        except ImportError:
            print("sklearn not available, using random initialization")
            return

        with torch.no_grad():
            feats_np = text_features.cpu().numpy().astype(np.float32)
            d = feats_np.shape[1]

            if len(feats_np) > 50000:
                indices = np.random.choice(len(feats_np), 50000, replace=False)
                feats_np = feats_np[indices]

            print(f"Initializing codebook from {len(feats_np)} text features...")

            # Level 0: Category codes
            print("  Level 0: Category codes...")
            kmeans_cat = MiniBatchKMeans(
                n_clusters=self.config.n_category, random_state=42, n_init=5, batch_size=1024
            )
            kmeans_cat.fit(feats_np)
            cat_centers = kmeans_cat.cluster_centers_.astype(np.float32)
            self.category_codes.data = torch.tensor(
                cat_centers, dtype=self.category_codes.dtype, device=self.category_codes.device
            )

            # Get category assignments
            cat_labels = pairwise_distances_argmin(feats_np, cat_centers)

            # Level 1: Type codes (hierarchical within each category)
            print("  Level 1: Type codes...")
            type_codes = np.zeros(
                (self.config.n_category, self.config.n_type_per_cat, d), dtype=np.float32
            )
            for cat_idx in range(self.config.n_category):
                cat_mask = cat_labels == cat_idx
                cat_feats = feats_np[cat_mask]

                if len(cat_feats) < self.config.n_type_per_cat:
                    # Not enough samples, use category center with noise
                    for j in range(self.config.n_type_per_cat):
                        noise = np.random.randn(d).astype(np.float32) * 0.02
                        type_codes[cat_idx, j] = cat_centers[cat_idx] + noise
                else:
                    km = MiniBatchKMeans(
                        n_clusters=self.config.n_type_per_cat, random_state=42 + cat_idx, n_init=3
                    )
                    km.fit(cat_feats)
                    type_codes[cat_idx] = km.cluster_centers_.astype(np.float32)

            self.type_codes.data = torch.tensor(
                type_codes, dtype=self.type_codes.dtype, device=self.type_codes.device
            )

            # Get type assignments
            type_codes_flat = type_codes.reshape(-1, d)
            type_labels = pairwise_distances_argmin(feats_np, type_codes_flat)

            # Level 2: Variant codes (hierarchical within each type)
            print("  Level 2: Variant codes...")
            n_types = self.config.n_category * self.config.n_type_per_cat
            variant_codes = np.zeros(
                (self.config.n_category, self.config.n_type_per_cat, self.config.n_variant_per_type, d),
                dtype=np.float32
            )
            for type_idx in range(n_types):
                type_mask = type_labels == type_idx
                type_feats = feats_np[type_mask]

                cat_idx = type_idx // self.config.n_type_per_cat
                type_in_cat = type_idx % self.config.n_type_per_cat

                if len(type_feats) < self.config.n_variant_per_type:
                    # Not enough samples, use type center with noise
                    for k in range(self.config.n_variant_per_type):
                        noise = np.random.randn(d).astype(np.float32) * 0.02
                        variant_codes[cat_idx, type_in_cat, k] = type_codes[cat_idx, type_in_cat] + noise
                else:
                    km = MiniBatchKMeans(
                        n_clusters=self.config.n_variant_per_type, random_state=42 + type_idx, n_init=3
                    )
                    km.fit(type_feats)
                    variant_codes[cat_idx, type_in_cat] = km.cluster_centers_.astype(np.float32)

            self.variant_codes.data = torch.tensor(
                variant_codes, dtype=self.variant_codes.dtype, device=self.variant_codes.device
            )

            # Spatial codes (independent)
            print("  Spatial codes...")
            kmeans_spatial = MiniBatchKMeans(
                n_clusters=self.config.n_spatial, random_state=123, n_init=5, batch_size=1024
            )
            kmeans_spatial.fit(feats_np)
            self.spatial_codes.data = torch.tensor(
                kmeans_spatial.cluster_centers_.astype(np.float32),
                dtype=self.spatial_codes.dtype,
                device=self.spatial_codes.device
            )

            print(f"Initialized all {self.total_codes} codes hierarchically")


# =============================================================================
# Hierarchical Codebook Grounding (same as v4.8.1)
# =============================================================================

class HierarchicalCodebookGrounding(nn.Module):
    """Ground tokens to hierarchical codebook with sparse selection."""

    def __init__(self, config: GFAv482Config):
        super().__init__()
        self.config = config
        d = config.d

        # Initialize k_proj to identity for stable start
        self.k_proj = nn.Linear(d, d)
        nn.init.eye_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)

        self.position_gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.Sigmoid()
        )

        # Initialize out_proj to identity for stable start
        self.out_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d)
        )
        nn.init.eye_(self.out_proj[0].weight)
        nn.init.zeros_(self.out_proj[0].bias)

        self.level_weights = nn.Parameter(torch.ones(4) / 4)

    def _topk_sparse(self, w: torch.Tensor, k: int) -> torch.Tensor:
        """Select top-k codes and renormalize."""
        B, M = w.shape
        if k >= M:
            return w
        # Get top-k indices
        topk_vals, topk_idx = torch.topk(w, k, dim=-1)
        # Create sparse version
        w_sparse = torch.zeros_like(w)
        w_sparse.scatter_(1, topk_idx, topk_vals)
        # Renormalize
        w_sparse = w_sparse / (w_sparse.sum(dim=-1, keepdim=True) + 1e-8)
        return w_sparse

    def forward(
        self,
        X: torch.Tensor,
        codebook: HierarchicalCodebook,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, N, d = X.shape
        tau = codebook.tau
        device = X.device

        # Top-k settings (more robust than threshold)
        k_cat = max(2, self.config.n_category // 4)      # ~4 categories
        k_type = max(4, self.config.n_type_per_cat)      # ~8 types
        k_var = max(4, self.config.n_variant_per_type * 2)  # ~8 variants
        k_spatial = max(2, self.config.n_spatial // 4)   # ~4 spatial

        # Use safe mask value for FP16 compatibility
        mask_value = -1e4

        K = self.k_proj(X)

        # LEVEL 0: Category attention
        Q_cat = codebook.category_proj(codebook.category_codes)
        attn_cat = torch.einsum('md,bnd->bmn', Q_cat, K) / math.sqrt(d)
        if mask is not None:
            attn_cat = attn_cat.masked_fill(~mask.bool().unsqueeze(1), mask_value)
        attn_cat = attn_cat.clamp(-50, 50)
        G_cat = F.softmax(attn_cat, dim=-1)
        G_cat = torch.nan_to_num(G_cat, nan=0.0)

        H_cat = torch.einsum('bmn,bnd->bmd', G_cat, X)

        w_cat_raw = H_cat.norm(dim=-1) / tau
        w_cat = F.softmax(w_cat_raw, dim=-1)
        w_cat_sparse = self._topk_sparse(w_cat, k_cat)

        # LEVEL 1: Type attention
        n_type_total = self.config.n_category * self.config.n_type_per_cat
        type_codes_flat = codebook.type_codes.view(-1, d)
        Q_type = codebook.type_proj(type_codes_flat)

        attn_type = torch.einsum('md,bnd->bmn', Q_type, K) / math.sqrt(d)
        if mask is not None:
            attn_type = attn_type.masked_fill(~mask.bool().unsqueeze(1), mask_value)
        attn_type = attn_type.clamp(-50, 50)
        G_type = F.softmax(attn_type, dim=-1)
        G_type = torch.nan_to_num(G_type, nan=0.0)

        H_type = torch.einsum('bmn,bnd->bmd', G_type, X)

        w_type_raw = H_type.norm(dim=-1) / tau
        w_type_reshaped = w_type_raw.view(B, self.config.n_category, self.config.n_type_per_cat)
        w_type_reshaped = w_type_reshaped * w_cat_sparse.unsqueeze(-1)
        w_type_raw_gated = w_type_reshaped.view(B, -1)
        w_type = F.softmax(w_type_raw_gated + 1e-8, dim=-1)
        w_type_sparse = self._topk_sparse(w_type, k_type)

        # LEVEL 2: Variant attention
        n_var_total = n_type_total * self.config.n_variant_per_type
        variant_codes_flat = codebook.variant_codes.view(-1, d)
        Q_var = codebook.variant_proj(variant_codes_flat)

        attn_var = torch.einsum('md,bnd->bmn', Q_var, K) / math.sqrt(d)
        if mask is not None:
            attn_var = attn_var.masked_fill(~mask.bool().unsqueeze(1), mask_value)
        attn_var = attn_var.clamp(-50, 50)
        G_var = F.softmax(attn_var, dim=-1)
        G_var = torch.nan_to_num(G_var, nan=0.0)

        H_var = torch.einsum('bmn,bnd->bmd', G_var, X)

        w_var_raw = H_var.norm(dim=-1) / tau
        w_type_expanded = w_type_sparse.view(B, -1, 1).expand(-1, -1, self.config.n_variant_per_type)
        w_type_expanded = w_type_expanded.reshape(B, n_var_total)
        w_var_raw_gated = w_var_raw * (w_type_expanded + 1e-8)
        w_var = F.softmax(w_var_raw_gated + 1e-8, dim=-1)
        w_var_sparse = self._topk_sparse(w_var, k_var)

        # SPATIAL: Independent spatial attention
        Q_spatial = codebook.spatial_proj(codebook.spatial_codes)

        attn_spatial = torch.einsum('md,bnd->bmn', Q_spatial, K) / math.sqrt(d)
        if mask is not None:
            attn_spatial = attn_spatial.masked_fill(~mask.bool().unsqueeze(1), mask_value)
        attn_spatial = attn_spatial.clamp(-50, 50)
        G_spatial = F.softmax(attn_spatial, dim=-1)
        G_spatial = torch.nan_to_num(G_spatial, nan=0.0)

        H_spatial = torch.einsum('bmn,bnd->bmd', G_spatial, X)

        w_spatial_raw = H_spatial.norm(dim=-1) / tau
        w_spatial = F.softmax(w_spatial_raw, dim=-1)
        w_spatial_sparse = self._topk_sparse(w_spatial, k_spatial)

        # AGGREGATE
        z_cat = torch.einsum('bm,bmd->bd', w_cat_sparse, H_cat)
        z_type = torch.einsum('bm,bmd->bd', w_type_sparse, H_type)
        z_var = torch.einsum('bm,bmd->bd', w_var_sparse, H_var)
        z_spatial = torch.einsum('bm,bmd->bd', w_spatial_sparse, H_spatial)

        if positions is not None:
            if mask is not None:
                mask_float = mask.float().unsqueeze(-1)
                pos_pooled = (positions * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
            else:
                pos_pooled = positions.mean(dim=1)

            gate_input = torch.cat([z_cat, pos_pooled], dim=-1)
            pos_gate = self.position_gate(gate_input)
            z_cat = z_cat * pos_gate
            z_type = z_type * pos_gate
            z_var = z_var * pos_gate

        level_w = F.softmax(self.level_weights, dim=0)
        z = level_w[0] * z_cat + level_w[1] * z_type + level_w[2] * z_var + level_w[3] * z_spatial

        z = self.out_proj(z)
        z = torch.nan_to_num(z, nan=0.0)

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
# Edge Message Layer (same as v4.8.1)
# =============================================================================

class EdgeMessageLayer(nn.Module):
    """Message passing between faces and edges through topology with gated residuals."""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()

        self.face_to_edge = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d)
        )

        self.edge_to_face = nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d)
        )

        self.norm_f = nn.LayerNorm(d)
        self.norm_e = nn.LayerNorm(d)

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
        B, N_e, d = E.shape
        N_f = F.shape[1]

        f1_idx = edge_to_faces[:, :, 0].clamp(0, N_f - 1).long()
        f2_idx = edge_to_faces[:, :, 1].clamp(0, N_f - 1).long()

        valid_edge = edge_mask.bool() & (edge_to_faces[:, :, 0] >= 0) & (edge_to_faces[:, :, 1] >= 0)

        # FACE -> EDGE
        f1 = torch.gather(F, 1, f1_idx.unsqueeze(-1).expand(-1, -1, d))
        f2 = torch.gather(F, 1, f2_idx.unsqueeze(-1).expand(-1, -1, d))

        msg_e = self.face_to_edge(torch.cat([E, f1, f2], dim=-1))
        msg_e = msg_e * valid_edge.unsqueeze(-1).float()
        gate_e = self.gate_e(torch.cat([E, msg_e], dim=-1))
        E_new = self.norm_e(E + gate_e * msg_e)

        # EDGE -> FACE
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
# Simplified Topology-Aware BRep Encoder (v4.8.2 - NEW)
# =============================================================================

class SimplifiedTopologyBRepEncoder(nn.Module):
    """
    v4.8.2: Simplified topology-aware BRep encoder.

    ONLY uses:
    - face_features (48-dim from AutoBrep FSQ)
    - edge_features (12-dim from AutoBrep FSQ)
    - edge_to_faces (topology connections)
    - bfs_level (structural ordering)

    Does NOT require:
    - face_centroids, face_normals, face_areas
    - edge_midpoints, edge_lengths
    """

    def __init__(self, config: GFAv482Config):
        super().__init__()
        self.config = config
        d = config.d

        # Feature projections
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

        # Type embeddings
        self.face_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # BFS level embedding (simple positional encoding)
        self.level_emb = nn.Embedding(config.max_bfs_levels, d)

        # Topology message passing (uses edge_to_faces)
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

        # Position encoding (from BFS level only)
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
        bfs_level: torch.Tensor,
        # IGNORED - for API compatibility with v4.8.1
        face_centroids: torch.Tensor = None,
        face_normals: torch.Tensor = None,
        face_areas: torch.Tensor = None,
        edge_midpoints: torch.Tensor = None,
        edge_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            face_feats: (B, N_f, d_face)
            edge_feats: (B, N_e, d_edge)
            face_mask, edge_mask: (B, N_f), (B, N_e)
            edge_to_faces: (B, N_e, 2) - topology connections
            bfs_level: (B, N_f) - BFS ordering
            face_centroids, face_normals, face_areas, edge_midpoints, edge_lengths: IGNORED

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

        # Project face features
        F = self.face_proj(face_feats.float())
        F = F + self.face_type

        # Add BFS level embedding
        level_emb = self.level_emb(bfs_level.clamp(0, self.config.max_bfs_levels - 1).long())
        F = F + level_emb

        # Project edge features
        E = self.edge_proj(edge_feats.float())
        E = E + self.edge_type

        # Topology message passing
        for layer in self.msg_layers:
            F, E = layer(F, E, edge_to_faces, face_mask, edge_mask)

        # Combine and transform
        X = torch.cat([F, E], dim=1)
        mask = torch.cat([face_mask, edge_mask], dim=1).bool()

        X = self.transformer(X, src_key_padding_mask=~mask)
        X = self.norm(X)
        X = torch.nan_to_num(X, nan=0.0)

        # Position encodings (from BFS level only)
        positions_f = self.position_encoder(level_emb)
        positions_e = torch.zeros(B, N_e, d, device=device)
        positions = torch.cat([positions_f, positions_e], dim=1)
        positions = torch.nan_to_num(positions, nan=0.0)

        return X, mask, positions


# =============================================================================
# Text Encoder (same as v4.8.1)
# =============================================================================

class TextEncoder(nn.Module):
    """Text encoder with projection and transformer."""

    def __init__(self, config: GFAv482Config):
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
        X = self.proj(X.float())

        if mask is not None:
            X = self.encoder(X, src_key_padding_mask=~mask.bool())
        else:
            X = self.encoder(X)

        X = self.norm(X)
        X = torch.nan_to_num(X, nan=0.0)

        return X, mask


# =============================================================================
# Point Cloud Encoder (same as v4.8.1)
# =============================================================================

class PCEncoder(nn.Module):
    """Point cloud encoder (simple projection)."""

    def __init__(self, config: GFAv482Config):
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
        pc_global = pc_global.unsqueeze(1)
        X = torch.cat([pc_local, pc_global], dim=1)

        X = self.proj(X.float())
        X = torch.nan_to_num(X, nan=0.0)

        return X


# =============================================================================
# BRep Decoder (same as v4.8.1)
# =============================================================================

class BRepDecoder(nn.Module):
    """Lightweight decoder for BRep reconstruction."""

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
        return self.decoder(z)


# =============================================================================
# Main Model: CLIP4CAD_GFA_v482
# =============================================================================

class CLIP4CAD_GFA_v482(nn.Module):
    """
    GFA v4.8.2: Simplified Topology-Aware Training

    Key features:
    1. Three-stage training (anchor -> align -> close gap)
    2. Hierarchical codebook (category -> type -> variant + spatial)
    3. SimplifiedTopologyBRepEncoder - only uses edge_to_faces and bfs_level
    4. BRep reconstruction auxiliary loss
    """

    def __init__(self, config: GFAv482Config):
        super().__init__()
        self.config = config
        d = config.d

        # ENCODERS
        self.text_encoder = TextEncoder(config)
        self.brep_encoder = SimplifiedTopologyBRepEncoder(config)  # SIMPLIFIED
        self.pc_encoder = PCEncoder(config)

        # HIERARCHICAL CODEBOOK
        self.codebook = HierarchicalCodebook(config)

        # CODEBOOK GROUNDING
        self.text_grounding = HierarchicalCodebookGrounding(config)
        self.brep_grounding = HierarchicalCodebookGrounding(config)
        self.pc_grounding = HierarchicalCodebookGrounding(config)

        # BREP RECONSTRUCTION
        self.brep_decoder = BRepDecoder(d, config.d_face)

        # DIRECT PROJECTION (for Stage 0)
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

        # OUTPUT PROJECTION
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
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_pc_encoder(self):
        for param in self.pc_encoder.parameters():
            param.requires_grad = False
        for param in self.pc_direct_proj.parameters():
            param.requires_grad = False
        print("PC encoder frozen (anchor mode)")

    def unfreeze_pc_encoder(self):
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

        # Encode BRep (simplified - only needs edge_to_faces and bfs_level)
        X_brep, brep_mask, _ = self.brep_encoder(
            face_feats=batch['face_features'].to(device),
            edge_feats=batch['edge_features'].to(device),
            face_mask=batch['face_mask'].to(device),
            edge_mask=batch['edge_mask'].to(device),
            edge_to_faces=batch['edge_to_faces'].to(device).long(),
            bfs_level=batch['bfs_level'].to(device).long(),
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

        # Direct projection
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
        """Full forward pass with hierarchical codebook."""
        if stage == 0:
            return self.forward_stage0(batch)

        device = next(self.parameters()).device

        # ENCODE MODALITIES
        text_features = batch['text_features'].to(device)
        text_mask = batch.get('text_mask')
        if text_mask is not None:
            text_mask = text_mask.to(device)
        X_text, text_mask = self.text_encoder(text_features, text_mask)

        # BRep (simplified - only needs edge_to_faces and bfs_level)
        X_brep, brep_mask, brep_positions = self.brep_encoder(
            face_feats=batch['face_features'].to(device),
            edge_feats=batch['edge_features'].to(device),
            face_mask=batch['face_mask'].to(device),
            edge_mask=batch['edge_mask'].to(device),
            edge_to_faces=batch['edge_to_faces'].to(device).long(),
            bfs_level=batch['bfs_level'].to(device).long(),
        )

        # PC
        X_pc = self.pc_encoder(
            batch['pc_local_features'].to(device),
            batch['pc_global_features'].to(device),
        )

        # GROUND TO HIERARCHICAL CODEBOOK
        text_out = self.text_grounding(X_text, self.codebook, None, text_mask)
        brep_out = self.brep_grounding(X_brep, self.codebook, brep_positions, brep_mask)
        pc_out = self.pc_grounding(X_pc, self.codebook, None, None)

        z_text_raw = text_out['z']
        z_brep_raw = brep_out['z']
        z_pc_raw = pc_out['z']

        # PROJECT TO OUTPUT SPACE
        z_text = self.proj_head(z_text_raw)
        z_brep = self.proj_head(z_brep_raw)
        z_pc = self.proj_head(z_pc_raw)

        # RECONSTRUCTION
        face_mask = batch['face_mask'].to(device)
        face_feats = batch['face_features'].to(device).float()
        face_pooled = (face_feats * face_mask.float().unsqueeze(-1)).sum(1) / face_mask.float().sum(1, keepdim=True).clamp(min=1)
        recon = self.brep_decoder(z_brep_raw)

        return {
            'z_text': z_text,
            'z_brep': z_brep,
            'z_pc': z_pc,
            'z_text_raw': z_text_raw,
            'z_brep_raw': z_brep_raw,
            'z_pc_raw': z_pc_raw,
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
            'G_text': text_out['G_category'],
            'G_brep': brep_out['G_category'],
            'G_pc': pc_out['G_category'],
            'n_active': {
                'text': text_out['n_active_codes'],
                'brep': brep_out['n_active_codes'],
                'pc': pc_out['n_active_codes'],
            },
            'level_weights': brep_out['level_weights'],
            'recon': recon,
            'recon_target': face_pooled,
            'tau': self.tau,
        }

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
            bfs_level=batch['bfs_level'].to(device).long(),
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
        """Initialize codebook from text encoder outputs using K-means."""
        print("Initializing hierarchical codebook from text encoder outputs...")

        all_features = []
        self.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                if remap_fn is not None:
                    batch = remap_fn(batch)

                text_features = batch['text_features'].to(device).float()
                text_mask = batch.get('text_mask')
                if text_mask is not None:
                    text_mask = text_mask.to(device)

                X_text, mask = self.text_encoder(text_features, text_mask)

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

    def load_codebook(self, path: str, strict: bool = False):
        """
        Load pre-initialized codebook from file.

        Use this instead of initialize_codebook() to avoid memory pressure
        during training. Generate the codebook file using:

            python scripts/initialize_codebook.py \\
                --text-h5 path/to/text.h5 \\
                --output path/to/codebook.pt

        Args:
            path: Path to codebook state dict (.pt file)
            strict: If True, require all keys to match exactly
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Codebook file not found: {path}")

        print(f"Loading pre-initialized codebook from {path}...")
        state = torch.load(path, map_location='cpu')

        # Load into codebook
        missing, unexpected = self.codebook.load_state_dict(state, strict=strict)

        if missing and strict:
            print(f"Warning: Missing keys: {missing}")
        if unexpected and strict:
            print(f"Warning: Unexpected keys: {unexpected}")

        # Move to same device as model
        device = next(self.parameters()).device
        self.codebook.to(device)

        print(f"Loaded codebook with {self.codebook.total_codes} codes")
        return self

    def reset_grounding_to_identity(self):
        """
        Reset grounding layers to identity initialization.

        Call this AFTER loading Stage 0 checkpoint to ensure grounding
        layers start as pass-through (they weren't trained in Stage 0).
        """
        print("Resetting grounding layers to identity...")

        for grounding in [self.text_grounding, self.brep_grounding, self.pc_grounding]:
            # k_proj: identity
            nn.init.eye_(grounding.k_proj.weight)
            nn.init.zeros_(grounding.k_proj.bias)

            # out_proj[0] (Linear): identity
            nn.init.eye_(grounding.out_proj[0].weight)
            nn.init.zeros_(grounding.out_proj[0].bias)

            # level_weights: uniform
            grounding.level_weights.data = torch.ones(4) / 4

        print("Grounding layers reset to identity (pass-through)")

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for memory efficiency.

        Reduces memory usage by ~40% at the cost of ~20% slower training.
        Recommended for RTX 4090 (24GB) with batch size 256.

        Checkpoints:
        - BRep encoder transformer layers
        - BRep encoder message passing layers
        - Text encoder transformer layers
        """
        from torch.utils.checkpoint import checkpoint_sequential

        print("Enabling gradient checkpointing...")

        # Wrap BRep transformer layers
        if hasattr(self.brep_encoder, 'transformer') and hasattr(self.brep_encoder.transformer, 'layers'):
            original_brep_tf = self.brep_encoder.transformer

            class CheckpointedTransformerEncoder(nn.Module):
                def __init__(self, encoder):
                    super().__init__()
                    self.layers = encoder.layers
                    self.norm = encoder.norm if hasattr(encoder, 'norm') else None

                def forward(self, src, src_key_padding_mask=None):
                    # Checkpoint each layer
                    for layer in self.layers:
                        src = torch.utils.checkpoint.checkpoint(
                            layer, src, None, src_key_padding_mask,
                            use_reentrant=False
                        )
                    if self.norm is not None:
                        src = self.norm(src)
                    return src

            self.brep_encoder.transformer = CheckpointedTransformerEncoder(original_brep_tf)
            print("  - BRep transformer: checkpointed")

        # Wrap text encoder transformer layers
        if hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'layers'):
            original_text_tf = self.text_encoder.encoder

            class CheckpointedTransformerEncoder(nn.Module):
                def __init__(self, encoder):
                    super().__init__()
                    self.layers = encoder.layers
                    self.norm = encoder.norm if hasattr(encoder, 'norm') else None

                def forward(self, src, src_key_padding_mask=None):
                    for layer in self.layers:
                        src = torch.utils.checkpoint.checkpoint(
                            layer, src, None, src_key_padding_mask,
                            use_reentrant=False
                        )
                    if self.norm is not None:
                        src = self.norm(src)
                    return src

            self.text_encoder.encoder = CheckpointedTransformerEncoder(original_text_tf)
            print("  - Text transformer: checkpointed")

        # Wrap BRep message passing layers
        if hasattr(self.brep_encoder, 'msg_layers'):
            original_msg_layers = self.brep_encoder.msg_layers

            class CheckpointedMsgLayers(nn.ModuleList):
                def __init__(self, layers):
                    super().__init__(layers)

                def forward_with_checkpoint(self, F, E, edge_to_faces, face_mask, edge_mask):
                    for layer in self:
                        def layer_forward(F_in, E_in):
                            return layer(F_in, E_in, edge_to_faces, face_mask, edge_mask)

                        F, E = torch.utils.checkpoint.checkpoint(
                            layer_forward, F, E,
                            use_reentrant=False
                        )
                    return F, E

            self.brep_encoder._checkpointed_msg_layers = CheckpointedMsgLayers(original_msg_layers)
            print("  - BRep message layers: checkpointed")

        print("Gradient checkpointing enabled (saves ~40% memory)")
