"""
Topology-Aware BRep Encoder for GFA v4.4

Key components:
1. TopologyMessageLayer - Face↔Edge message passing via edge_to_faces graph
2. TopologyAwareBRepEncoder - Full encoder with spatial embeddings and message passing
3. HierarchicalPatternAggregator - BFS-level grouping for semantic patterns
4. SemanticQueryGenerator - Multi-source attention with curriculum conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TopologyMessageLayer(nn.Module):
    """
    Message passing between faces and edges using topology graph.

    Information flow:
    1. Edges collect information from their two connected faces
    2. Faces collect information from their incident edges

    This explicitly encodes "F1 and F2 share edge E0" relationships.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d

        # Face → Edge: Aggregate info from connected faces
        self.face_to_edge_msg = nn.Sequential(
            nn.Linear(d * 3, d * 2),  # [edge, face1, face2] → message
            nn.GELU(),
            nn.Linear(d * 2, d)
        )

        # Edge → Face: Aggregate info from incident edges
        self.edge_to_face_msg = nn.Sequential(
            nn.Linear(d * 2, d),  # [face, edge_agg] → message
            nn.GELU(),
            nn.Linear(d, d)
        )

        # Gated updates
        self.edge_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())
        self.face_gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())

        self.norm_f = nn.LayerNorm(d)
        self.norm_e = nn.LayerNorm(d)

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
            edge_to_faces: (B, N_e, 2) - For each edge, indices of two connected faces
            face_mask: (B, N_f) - Valid face mask
            edge_mask: (B, N_e) - Valid edge mask

        Returns:
            F_new: (B, N_f, d) - Updated face features
            E_new: (B, N_e, d) - Updated edge features
        """
        B, N_e, d = E.shape
        N_f = F.shape[1]

        # Clamp indices for safety (invalid edges have -1)
        # Cast to long for gather/scatter operations
        f1_idx = edge_to_faces[:, :, 0].clamp(0, N_f - 1).long()  # (B, N_e)
        f2_idx = edge_to_faces[:, :, 1].clamp(0, N_f - 1).long()

        # Valid edge mask (both faces must be valid and indices >= 0)
        # Convert to bool for logical AND operations
        valid_edge = edge_mask.bool() & (edge_to_faces[:, :, 0] >= 0) & (edge_to_faces[:, :, 1] >= 0)

        # ─────────────────────────────────────────────────────────────────────
        # FACE → EDGE
        # Each edge gathers features from its two connected faces
        # ─────────────────────────────────────────────────────────────────────

        # Gather face features for each edge's connected faces
        f1_feats = torch.gather(F, 1, f1_idx.unsqueeze(-1).expand(-1, -1, d))  # (B, N_e, d)
        f2_feats = torch.gather(F, 1, f2_idx.unsqueeze(-1).expand(-1, -1, d))

        # Compute edge message from connected faces
        edge_input = torch.cat([E, f1_feats, f2_feats], dim=-1)  # (B, N_e, 3d)
        edge_msg = self.face_to_edge_msg(edge_input)
        edge_msg = edge_msg * valid_edge.unsqueeze(-1).float()

        # Gated update
        gate_e = self.edge_gate(torch.cat([E, edge_msg], dim=-1))
        E_new = self.norm_e(E + gate_e * edge_msg)

        # ─────────────────────────────────────────────────────────────────────
        # EDGE → FACE
        # Each face aggregates features from its incident edges
        # ─────────────────────────────────────────────────────────────────────

        # Scatter edge features to connected faces
        face_msg = torch.zeros_like(F)  # (B, N_f, d)
        face_count = torch.zeros(B, N_f, 1, device=F.device)

        # Edge contribution (masked by validity)
        edge_contrib = E_new * valid_edge.unsqueeze(-1).float()
        count_contrib = valid_edge.unsqueeze(-1).float()

        # Scatter add (edge → face1)
        face_msg.scatter_add_(
            1,
            f1_idx.unsqueeze(-1).expand(-1, -1, d),
            edge_contrib
        )
        face_count.scatter_add_(
            1,
            f1_idx.unsqueeze(-1),
            count_contrib
        )

        # Scatter add (edge → face2)
        face_msg.scatter_add_(
            1,
            f2_idx.unsqueeze(-1).expand(-1, -1, d),
            edge_contrib
        )
        face_count.scatter_add_(
            1,
            f2_idx.unsqueeze(-1),
            count_contrib
        )

        # Average
        face_msg = face_msg / (face_count + 1e-8)

        # Compute face update
        face_input = torch.cat([F, face_msg], dim=-1)
        face_update = self.edge_to_face_msg(face_input)

        # Gated update with face mask
        gate_f = self.face_gate(torch.cat([F, face_update], dim=-1))
        F_new = self.norm_f(F + gate_f * face_update * face_mask.unsqueeze(-1).float())

        return F_new, E_new


class TopologyAwareBRepEncoder(nn.Module):
    """
    Stage 1: Encode BRep with explicit topology.

    Key insight: Use edge_to_faces to pass messages between connected faces.
    This makes spatial patterns explicit rather than implicit.
    """

    def __init__(
        self,
        d: int,
        d_face: int = 48,
        d_edge: int = 12,
        num_msg_layers: int = 3,
        num_tf_layers: int = 4,
        num_heads: int = 8,
        max_levels: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d = d

        # ─────────────────────────────────────────────────────────────────────
        # GEOMETRY PROJECTIONS
        # ─────────────────────────────────────────────────────────────────────
        self.face_proj = nn.Sequential(
            nn.Linear(d_face, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d)
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(d_edge, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d)
        )

        # ─────────────────────────────────────────────────────────────────────
        # SPATIAL EMBEDDINGS
        # ─────────────────────────────────────────────────────────────────────
        self.centroid_proj = nn.Linear(3, d)
        self.normal_proj = nn.Linear(3, d)
        self.area_proj = nn.Linear(1, d)
        self.level_emb = nn.Embedding(max_levels, d)

        # Edge spatial
        self.edge_midpoint_proj = nn.Linear(3, d)
        self.edge_length_proj = nn.Linear(1, d)

        # Type embeddings
        self.face_type_emb = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type_emb = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # ─────────────────────────────────────────────────────────────────────
        # TOPOLOGY MESSAGE PASSING (Key innovation!)
        # ─────────────────────────────────────────────────────────────────────
        self.msg_layers = nn.ModuleList([
            TopologyMessageLayer(d) for _ in range(num_msg_layers)
        ])

        # ─────────────────────────────────────────────────────────────────────
        # TRANSFORMER
        # ─────────────────────────────────────────────────────────────────────
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d, nhead=num_heads, dim_feedforward=d*4,
                dropout=dropout, activation='gelu', batch_first=True
            ),
            num_layers=num_tf_layers
        )

        self.norm = nn.LayerNorm(d)

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
        edge_midpoints: torch.Tensor,
        edge_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            face_feats: (B, N_f, 48) - AutoBrep face FSQ latents
            edge_feats: (B, N_e, 12) - AutoBrep edge FSQ latents
            face_mask: (B, N_f) - Valid face mask
            edge_mask: (B, N_e) - Valid edge mask
            edge_to_faces: (B, N_e, 2) - Edge topology
            face_centroids: (B, N_f, 3) - Face centers
            face_normals: (B, N_f, 3) - Face normals
            face_areas: (B, N_f) - Face areas
            bfs_level: (B, N_f) - BFS level for each face
            edge_midpoints: (B, N_e, 3) - Edge midpoints
            edge_lengths: (B, N_e) - Edge lengths

        Returns:
            X_brep: (B, N_f + N_e, d) - Topology-enriched tokens
            face_tokens: (B, N_f, d) - Just face tokens (for level grouping)
        """
        # ─────────────────────────────────────────────────────────────────────
        # PROJECT AND EMBED FACES
        # Sanitize spatial fields to avoid NaN from bad geometry
        # ─────────────────────────────────────────────────────────────────────
        face_centroids = torch.nan_to_num(face_centroids, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        face_normals = torch.nan_to_num(face_normals, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        face_areas = torch.nan_to_num(face_areas, nan=0.0, posinf=1e6, neginf=0.0).clamp(0, 1e6)

        F = self.face_proj(face_feats)
        F = F + self.face_type_emb
        F = F + self.centroid_proj(face_centroids)
        F = F + self.normal_proj(face_normals)
        F = F + self.area_proj(face_areas.unsqueeze(-1))
        F = F + self.level_emb(bfs_level.clamp(0, 31))

        # ─────────────────────────────────────────────────────────────────────
        # PROJECT AND EMBED EDGES
        # Sanitize spatial fields to avoid NaN from bad geometry
        # ─────────────────────────────────────────────────────────────────────
        edge_midpoints = torch.nan_to_num(edge_midpoints, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        edge_lengths = torch.nan_to_num(edge_lengths, nan=0.0, posinf=1e6, neginf=0.0).clamp(0, 1e6)

        E = self.edge_proj(edge_feats)
        E = E + self.edge_type_emb
        E = E + self.edge_midpoint_proj(edge_midpoints)
        E = E + self.edge_length_proj(edge_lengths.unsqueeze(-1))

        # ─────────────────────────────────────────────────────────────────────
        # TOPOLOGY MESSAGE PASSING
        # Faces and edges exchange information through connectivity
        # ─────────────────────────────────────────────────────────────────────
        for msg_layer in self.msg_layers:
            F, E = msg_layer(F, E, edge_to_faces, face_mask, edge_mask)

        # ─────────────────────────────────────────────────────────────────────
        # TRANSFORMER ENCODING
        # ─────────────────────────────────────────────────────────────────────
        X = torch.cat([F, E], dim=1)
        mask = torch.cat([face_mask, edge_mask], dim=1).bool()

        X = self.transformer(X, src_key_padding_mask=~mask)
        X = self.norm(X)

        # Sanitize output (transformer can produce NaN with all-masked positions)
        X = torch.nan_to_num(X, nan=0.0)

        # Split back
        N_f = F.shape[1]
        face_tokens = X[:, :N_f]

        return X, face_tokens


class HierarchicalPatternAggregator(nn.Module):
    """
    Stage 2: Aggregate patterns at multiple scales using BFS hierarchy.

    Key insight: BFS levels naturally encode semantic hierarchy:
    - Level 0: Core shape (center face)
    - Level 1-2: Primary features (bore, shaft, main surfaces)
    - Level 3+: Details (teeth, fillets, chamfers)

    This matches how text describes CAD: "A gear (L0) with central bore (L1) and 32 teeth (L2+)"
    """

    def __init__(self, d: int, max_levels: int = 10, num_heads: int = 8):
        super().__init__()
        self.d = d
        self.max_levels = max_levels

        # Level-specific processing
        self.level_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, num_heads, d*2, 0.1, 'gelu', batch_first=True),
                num_layers=1
            ) for _ in range(max_levels)
        ])

        # Cross-level attention (parent level informs child level)
        self.cross_level_attn = nn.ModuleList([
            nn.MultiheadAttention(d, num_heads, batch_first=True)
            for _ in range(max_levels - 1)
        ])

        # Level pooling (attention-based)
        self.level_pool_query = nn.Parameter(torch.randn(max_levels, 1, d) * 0.02)
        self.level_pool_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)

        # Global summary
        self.global_pool = nn.Sequential(
            nn.Linear(d * max_levels, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d)
        )

    def forward(
        self,
        face_tokens: torch.Tensor,
        bfs_level: torch.Tensor,
        face_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            face_tokens: (B, N_f, d) - Topology-enriched face features
            bfs_level: (B, N_f) - BFS level for each face
            face_mask: (B, N_f) - Valid face mask

        Returns:
            level_summaries: (B, max_levels, d) - Summary per BFS level
            global_summary: (B, d) - Global shape summary
        """
        B, N_f, d = face_tokens.shape
        device = face_tokens.device

        level_summaries = []
        prev_level_summary = None

        # ─────────────────────────────────────────────────────────────────────
        # PROCESS EACH LEVEL
        # ─────────────────────────────────────────────────────────────────────
        for level in range(self.max_levels):
            # Get faces at this level
            level_mask = (bfs_level == level) & face_mask.bool()  # (B, N_f)

            # Check if any faces at this level
            has_faces = level_mask.any(dim=1).any()

            if not has_faces:
                # No faces at this level, use zeros
                level_summary = torch.zeros(B, d, device=device)
            else:
                # Extract faces at this level (padded)
                max_at_level = max(level_mask.sum(dim=1).max().item(), 1)

                level_feat = torch.zeros(B, max_at_level, d, device=device)
                level_feat_mask = torch.zeros(B, max_at_level, dtype=torch.bool, device=device)

                for b in range(B):
                    indices = level_mask[b].nonzero(as_tuple=True)[0]
                    n = len(indices)
                    if n > 0:
                        level_feat[b, :n] = face_tokens[b, indices]
                        level_feat_mask[b, :n] = True

                # Encode this level
                if level_feat_mask.any():
                    level_feat = self.level_encoders[level](
                        level_feat,
                        src_key_padding_mask=~level_feat_mask
                    )

                # Cross-level attention: inform by parent level
                if prev_level_summary is not None and level > 0:
                    # Previous level summary conditions current level
                    prev_expanded = prev_level_summary.unsqueeze(1)  # (B, 1, d)
                    level_feat_attended, _ = self.cross_level_attn[level-1](
                        level_feat, prev_expanded, prev_expanded,
                        key_padding_mask=None
                    )
                    level_feat = level_feat + level_feat_attended

                # Pool this level to single summary
                query = self.level_pool_query[level].expand(B, -1, -1)  # (B, 1, d)
                level_summary, _ = self.level_pool_attn(
                    query, level_feat, level_feat,
                    key_padding_mask=~level_feat_mask if level_feat_mask.any() else None
                )
                level_summary = level_summary.squeeze(1)  # (B, d)

            level_summaries.append(level_summary)
            prev_level_summary = level_summary

        # Stack level summaries
        level_summaries = torch.stack(level_summaries, dim=1)  # (B, max_levels, d)
        level_summaries = torch.nan_to_num(level_summaries, nan=0.0)

        # ─────────────────────────────────────────────────────────────────────
        # GLOBAL SUMMARY
        # ─────────────────────────────────────────────────────────────────────
        global_summary = self.global_pool(level_summaries.flatten(1))  # (B, d)
        global_summary = torch.nan_to_num(global_summary, nan=0.0)

        return level_summaries, global_summary


class SemanticQueryGenerator(nn.Module):
    """
    Stage 3: Generate semantic queries from topology-enriched features.

    Key insight: Queries should attend to BOTH:
    1. Individual tokens (for specific face/edge features)
    2. Hierarchical summaries (for global patterns)

    This allows the model to encode both:
    - "cylindrical bore" (specific geometry)
    - "32 teeth arranged radially" (pattern across level)
    """

    def __init__(
        self,
        d: int,
        K: int,
        num_layers: int = 4,
        num_heads: int = 8,
        max_levels: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.K = K
        self.d = d
        self.max_levels = max_levels

        # Learnable base queries
        self.base_queries = nn.Parameter(torch.randn(K, d) * 0.02)

        # Conditioning projection (for curriculum learning)
        self.cond_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d)
        )
        self.cond_scale = nn.Parameter(torch.ones(1) * 0.5)

        # ─────────────────────────────────────────────────────────────────────
        # MULTI-SOURCE DECODER
        # Queries attend to: tokens, level summaries, global summary
        # ─────────────────────────────────────────────────────────────────────

        # Cross-attention to BRep tokens
        self.token_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Cross-attention to level summaries
        self.level_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Self-attention among queries
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])

        # FFN
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d * 4, d),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # Layer norms
        self.norm_self = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        self.norm_token = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        self.norm_level = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d)
        )

        # Confidence prediction
        self.conf_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1)
        )

        # Curriculum dropout
        self.cond_drop_rate = 0.2

    def forward(
        self,
        X_brep: torch.Tensor,
        level_summaries: torch.Tensor,
        global_summary: torch.Tensor,
        brep_mask: torch.Tensor,
        T_feat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X_brep: (B, N, d) - Topology-enriched BRep tokens
            level_summaries: (B, max_levels, d) - Per-level summaries
            global_summary: (B, d) - Global shape summary
            brep_mask: (B, N) - Valid token mask
            T_feat: (B, K, d) - Text features for conditioning (training only)

        Returns:
            Q_self: (B, K, d) - Generated queries
            confidence: (B, K) - Confidence per slot
        """
        B = X_brep.shape[0]

        # Initialize queries
        Q = self.base_queries.unsqueeze(0).expand(B, -1, -1).clone()

        # ─────────────────────────────────────────────────────────────────────
        # CONDITIONING (curriculum learning)
        # ─────────────────────────────────────────────────────────────────────
        if self.training and T_feat is not None:
            keep_cond = (torch.rand(B, device=Q.device) > self.cond_drop_rate)
            keep_cond = keep_cond.view(B, 1, 1).float()

            cond = self.cond_proj(T_feat.detach())
            scaled_cond = cond * self.cond_scale.sigmoid()
            Q = Q + scaled_cond * keep_cond

        # Add global context to queries
        Q = Q + global_summary.unsqueeze(1) * 0.1

        # ─────────────────────────────────────────────────────────────────────
        # DECODER LAYERS
        # ─────────────────────────────────────────────────────────────────────
        for i in range(len(self.self_attn)):
            # Self-attention among queries
            Q2, _ = self.self_attn[i](Q, Q, Q)
            Q = self.norm_self[i](Q + Q2)

            # Cross-attention to BRep tokens (local features)
            Q2, _ = self.token_cross_attn[i](
                Q, X_brep, X_brep,
                key_padding_mask=~brep_mask.bool()
            )
            Q = self.norm_token[i](Q + Q2)

            # Cross-attention to level summaries (pattern features)
            Q2, _ = self.level_cross_attn[i](
                Q, level_summaries, level_summaries
            )
            Q = self.norm_level[i](Q + Q2)

            # FFN
            Q = self.norm_ffn[i](Q + self.ffn[i](Q))

        # ─────────────────────────────────────────────────────────────────────
        # OUTPUT
        # ─────────────────────────────────────────────────────────────────────
        Q_self = self.output_proj(Q)
        Q_self = torch.nan_to_num(Q_self, nan=0.0)

        # Confidence
        conf_logits = self.conf_head(Q).squeeze(-1)
        confidence = torch.sigmoid(conf_logits.clamp(-5, 5))
        confidence = torch.nan_to_num(confidence, nan=0.5)  # Default to 0.5 confidence

        return Q_self, confidence

    def set_cond_dropout(self, rate: float):
        """Set conditioning dropout rate for curriculum learning."""
        self.cond_drop_rate = float(rate)
