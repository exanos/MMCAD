"""
CLIP4CAD GFA v4.8 - Gap-Closing Codebook Architecture

Key Insight:
    InfoNCE only optimizes relative ranking, leaving a modality gap.
    v4.8 directly closes this gap with L_align, making self-grounding unnecessary.

Architecture:
    1. Shared Semantic Codebook (256 codes)
    2. Codebook Grounding (ground all modalities to shared codes)
    3. Topology-aware BRep Encoder (message passing + transformer)
    4. Simple Text/PC Encoders

Losses (5 terms, well-motivated):
    - L_contrastive: 3-way InfoNCE (preserves retrieval)
    - L_align: MSE alignment (CLOSES THE GAP!)
    - L_uniform: Centroid uniformity (prevents collapse)
    - L_code: Code activation alignment
    - L_diversity: Codebook utilization
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
class GFAv48Config:
    """Configuration for GFA v4.8 model."""

    # Feature dimensions (input)
    d_face: int = 48       # AutoBrep face FSQ features
    d_edge: int = 12       # AutoBrep edge FSQ features
    d_pc: int = 1024       # ShapeLLM point cloud features
    d_text: int = 3072     # Phi-4-mini text features

    # Model dimensions
    d_unified: int = 256   # Internal unified dimension
    d_proj: int = 128      # Final projection output

    # Codebook
    num_codes: int = 256   # Number of semantic codes

    # Architecture
    num_heads: int = 8
    dropout: float = 0.1

    # BRep Encoder
    num_msg_layers: int = 2     # Topology message passing layers
    num_brep_tf_layers: int = 3 # BRep transformer layers
    max_bfs_levels: int = 32

    # Text Encoder
    num_text_tf_layers: int = 2

    # PC Encoder (simple projection)
    num_pc_tokens: int = 48     # Expected local PC tokens


# =============================================================================
# Semantic Codebook
# =============================================================================

class SemanticCodebook(nn.Module):
    """
    Shared semantic codebook for all modalities.

    All embeddings are grounded through this codebook, ensuring
    structural alignment across text, BRep, and point cloud.

    Codes learn semantic primitives like:
    [cylindrical] [planar] [teeth] [hole] [fillet] [thread] [boss] ...
    """

    def __init__(self, num_codes: int = 256, d: int = 256):
        super().__init__()
        self.num_codes = num_codes
        self.d = d

        # Learnable code vectors
        self.codes = nn.Parameter(torch.randn(num_codes, d) * 0.02)

        # Temperature for code attention
        self.log_tau = nn.Parameter(torch.zeros(1))

    @property
    def tau(self):
        return (self.log_tau.exp() + 0.1).clamp(0.1, 2.0)

    def initialize_from_data(self, features: torch.Tensor):
        """
        Initialize codes via K-means on encoder outputs.
        Call this before training with text encoder outputs.

        Args:
            features: (N, d) - Pooled features from text encoder
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("sklearn not available, using random initialization")
            return

        with torch.no_grad():
            feats_np = features.cpu().numpy()

            # Subsample if too many features
            if len(feats_np) > 50000:
                indices = torch.randperm(len(feats_np))[:50000].numpy()
                feats_np = feats_np[indices]

            kmeans = KMeans(n_clusters=self.num_codes, random_state=42, n_init=10)
            kmeans.fit(feats_np)

            self.codes.data = torch.tensor(
                kmeans.cluster_centers_,
                dtype=self.codes.dtype,
                device=self.codes.device
            )


# =============================================================================
# Codebook Grounding
# =============================================================================

class CodebookGrounding(nn.Module):
    """
    Ground tokens to shared codebook.

    For each input modality:
    1. Compute code-token attention (which tokens does each code attend to?)
    2. Aggregate tokens per code
    3. Compute code activation weights

    Returns:
        H: (B, M, d) - Aggregated features per code
        w: (B, M) - Code activation weights (for code alignment loss)
        G: (B, M, N) - Grounding matrix (for interpretability)
    """

    def __init__(self, d: int, num_heads: int = 8):
        super().__init__()
        self.d = d

        # Learnable projections for attention computation
        self.token_proj = nn.Linear(d, d)
        self.code_proj = nn.Linear(d, d)

        # LayerNorm for aggregated features
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        X: torch.Tensor,
        codebook: SemanticCodebook,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X: (B, N, d) - Tokens from any modality
            codebook: Shared semantic codebook
            mask: (B, N) - Valid token mask (bool, True = valid)

        Returns:
            H: (B, M, d) - Aggregated features per code
            w: (B, M) - Code activation weights
            G: (B, M, N) - Grounding matrix (soft attention)
        """
        B, N, d = X.shape
        M = codebook.num_codes
        C = codebook.codes  # (M, d)
        tau = codebook.tau

        # Project for attention
        X_proj = self.token_proj(X)     # (B, N, d) - tokens as keys
        C_proj = self.code_proj(C)      # (M, d) - codes as queries

        # Code-token attention: which tokens does each code attend to?
        attn_logits = torch.einsum('md,bnd->bmn', C_proj, X_proj) / math.sqrt(d)

        # Mask invalid tokens
        if mask is not None:
            # mask is (B, N) with True for valid tokens
            # Use dtype-appropriate min value (FP16 max is ~65504)
            mask_value = -65000.0 if attn_logits.dtype == torch.float16 else -1e9
            attn_logits = attn_logits.masked_fill(~mask.bool().unsqueeze(1), mask_value)

        # Clamp for numerical stability
        attn_logits = attn_logits.clamp(-50, 50)

        # Grounding matrix (soft attention)
        G = F.softmax(attn_logits, dim=-1)  # (B, M, N)
        G = torch.nan_to_num(G, nan=0.0)

        # Aggregate tokens per code
        H = torch.einsum('bmn,bnd->bmd', G, X)  # (B, M, d)
        H = self.norm(H)
        H = torch.nan_to_num(H, nan=0.0)

        # Code activation weights based on information captured
        code_energy = H.norm(dim=-1)  # (B, M)
        w = F.softmax(code_energy / tau, dim=-1)  # (B, M)
        w = torch.nan_to_num(w, nan=1.0 / M)

        return H, w, G


# =============================================================================
# Shared Fusion Network
# =============================================================================

class SharedFusionNetwork(nn.Module):
    """
    SHARED network that converts code representations to final embeddings.

    CRITICAL: Same weights for all modalities!
    This ensures that if code activations (w) are aligned,
    the output embeddings (z) will also be aligned.
    """

    def __init__(self, d: int, d_out: int, num_heads: int = 8):
        super().__init__()
        self.d = d
        self.d_out = d_out

        # Attention-based aggregation over codes
        self.agg_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.agg_attn = nn.MultiheadAttention(d, num_heads=num_heads, batch_first=True)

        # Final projection
        self.proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d_out),
        )

    def forward(self, H: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (B, M, d) - Aggregated features per code
            w: (B, M) - Code activation weights

        Returns:
            z: (B, d_out) - Final embedding
        """
        B = H.shape[0]

        # Method 1: Weighted sum (simple)
        z_weighted = torch.einsum('bm,bmd->bd', w, H)  # (B, d)

        # Method 2: Attention-based (captures code interactions)
        query = self.agg_query.expand(B, -1, -1)  # (B, 1, d)
        z_attn, _ = self.agg_attn(query, H, H)    # (B, 1, d)
        z_attn = z_attn.squeeze(1)                # (B, d)

        # Combine
        z = z_weighted + z_attn

        # Project to output space
        z = self.proj(z)
        z = torch.nan_to_num(z, nan=0.0)

        return z


# =============================================================================
# Topology Message Layer (Face <-> Edge)
# =============================================================================

class TopologyMessageLayer(nn.Module):
    """
    Message passing between faces and edges using edge_to_faces topology.

    Each layer performs:
    1. Face -> Edge: Edges gather info from connected faces
    2. Edge -> Face: Faces aggregate info from incident edges
    """

    def __init__(self, d: int):
        super().__init__()

        # Face -> Edge message
        self.face_to_edge = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d)
        )

        # Edge -> Face message
        self.edge_to_face = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, d)
        )

        # Layer norms
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
            edge_to_faces: (B, N_e, 2) - For each edge, indices of connected faces
            face_mask: (B, N_f) - Valid face mask
            edge_mask: (B, N_e) - Valid edge mask

        Returns:
            F_new: (B, N_f, d) - Updated face features
            E_new: (B, N_e, d) - Updated edge features
        """
        B, N_e, d = E.shape
        N_f = F.shape[1]

        # Clamp indices (invalid edges have -1)
        f1_idx = edge_to_faces[:, :, 0].clamp(0, N_f - 1).long()
        f2_idx = edge_to_faces[:, :, 1].clamp(0, N_f - 1).long()

        # Valid edge mask
        valid_edge = edge_mask.bool() & (edge_to_faces[:, :, 0] >= 0) & (edge_to_faces[:, :, 1] >= 0)

        # ─────────────────────────────────────────────────────────────────────
        # FACE -> EDGE
        # ─────────────────────────────────────────────────────────────────────
        f1 = torch.gather(F, 1, f1_idx.unsqueeze(-1).expand(-1, -1, d))
        f2 = torch.gather(F, 1, f2_idx.unsqueeze(-1).expand(-1, -1, d))

        edge_msg = self.face_to_edge(torch.cat([E, f1, f2], dim=-1))
        edge_msg = edge_msg * valid_edge.unsqueeze(-1).float()
        E_new = self.norm_e(E + edge_msg)

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
        face_update = self.edge_to_face(torch.cat([F, face_msg], dim=-1))
        F_new = self.norm_f(F + face_update * face_mask.unsqueeze(-1).float())

        return F_new, E_new


# =============================================================================
# Topology-Aware BRep Encoder
# =============================================================================

class TopologyBRepEncoder(nn.Module):
    """
    Topology-aware BRep encoder with message passing.

    Architecture:
    1. Project face/edge features to unified space
    2. Add spatial embeddings (centroids, normals, BFS level)
    3. Message passing through face-edge topology
    4. Transformer for global context
    """

    def __init__(
        self,
        d: int,
        d_face: int = 48,
        d_edge: int = 12,
        num_msg_layers: int = 2,
        num_tf_layers: int = 3,
        num_heads: int = 8,
        max_levels: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d = d

        # Geometry projections
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

        # Spatial embeddings
        self.centroid_proj = nn.Linear(3, d)
        self.normal_proj = nn.Linear(3, d)
        self.area_proj = nn.Linear(1, d)
        self.level_emb = nn.Embedding(max_levels, d)

        # Edge spatial
        self.edge_midpoint_proj = nn.Linear(3, d)
        self.edge_length_proj = nn.Linear(1, d)

        # Type embeddings
        self.face_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Message passing
        self.msg_layers = nn.ModuleList([
            TopologyMessageLayer(d) for _ in range(num_msg_layers)
        ])

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=num_heads,
                dim_feedforward=d * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
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
        edge_midpoints: Optional[torch.Tensor] = None,
        edge_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            face_feats: (B, N_f, 48) - AutoBrep face FSQ features
            edge_feats: (B, N_e, 12) - AutoBrep edge FSQ features
            face_mask: (B, N_f) - Valid face mask
            edge_mask: (B, N_e) - Valid edge mask
            edge_to_faces: (B, N_e, 2) - Edge topology
            face_centroids: (B, N_f, 3) - Face centers
            face_normals: (B, N_f, 3) - Face normals
            face_areas: (B, N_f) - Face areas
            bfs_level: (B, N_f) - BFS level per face
            edge_midpoints: (B, N_e, 3) - Optional edge midpoints
            edge_lengths: (B, N_e) - Optional edge lengths

        Returns:
            X_brep: (B, N_f + N_e, d) - Topology-enriched tokens
            brep_mask: (B, N_f + N_e) - Combined mask
        """
        B = face_feats.shape[0]
        N_f = face_feats.shape[1]
        N_e = edge_feats.shape[1]
        device = face_feats.device

        # ─────────────────────────────────────────────────────────────────────
        # SANITIZE SPATIAL FIELDS (prevent NaN from bad geometry)
        # ─────────────────────────────────────────────────────────────────────
        face_centroids = torch.nan_to_num(face_centroids, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        face_normals = torch.nan_to_num(face_normals, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        face_areas = torch.nan_to_num(face_areas, nan=0.0, posinf=1e6, neginf=0.0).clamp(0, 1e6)

        # ─────────────────────────────────────────────────────────────────────
        # PROJECT AND EMBED FACES
        # ─────────────────────────────────────────────────────────────────────
        F = self.face_proj(face_feats)
        F = F + self.face_type
        F = F + self.centroid_proj(face_centroids)
        F = F + self.normal_proj(face_normals)
        F = F + self.area_proj(face_areas.unsqueeze(-1))
        F = F + self.level_emb(bfs_level.clamp(0, 31).long())

        # ─────────────────────────────────────────────────────────────────────
        # PROJECT AND EMBED EDGES
        # ─────────────────────────────────────────────────────────────────────
        E = self.edge_proj(edge_feats)
        E = E + self.edge_type

        if edge_midpoints is not None:
            edge_midpoints = torch.nan_to_num(edge_midpoints, nan=0.0).clamp(-1e4, 1e4)
            E = E + self.edge_midpoint_proj(edge_midpoints)

        if edge_lengths is not None:
            edge_lengths = torch.nan_to_num(edge_lengths, nan=0.0).clamp(0, 1e6)
            E = E + self.edge_length_proj(edge_lengths.unsqueeze(-1))

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

        # Sanitize output
        X = torch.nan_to_num(X, nan=0.0)

        return X, mask


# =============================================================================
# Text Encoder
# =============================================================================

class TextEncoder(nn.Module):
    """
    Text encoder with projection and transformer.

    Processes Phi-4 embeddings (3072d) into unified space (256d).
    """

    def __init__(
        self,
        d_in: int = 3072,
        d: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_in, d),
            nn.LayerNorm(d)
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=num_heads,
                dim_feedforward=d * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            X: (B, T, 3072) - Phi-4 text embeddings
            mask: (B, T) - Valid token mask (bool, True = valid)

        Returns:
            X_text: (B, T, d) - Projected text tokens
            mask: (B, T) - Unchanged mask
        """
        X = self.proj(X)

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
    """
    Point cloud encoder (simple projection).

    ShapeLLM features are already well-structured, so we just project.
    """

    def __init__(self, d_in: int = 1024, d: int = 256, dropout: float = 0.1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_in, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d)
        )

    def forward(
        self,
        pc_local: torch.Tensor,
        pc_global: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pc_local: (B, N, 1024) - Local PC features
            pc_global: (B, 1024) - Global PC feature

        Returns:
            X_pc: (B, N+1, d) - Projected PC tokens
        """
        # Add global as extra token
        pc_global = pc_global.unsqueeze(1)  # (B, 1, 1024)
        X = torch.cat([pc_local, pc_global], dim=1)  # (B, N+1, 1024)

        X = self.proj(X)
        X = torch.nan_to_num(X, nan=0.0)

        return X


# =============================================================================
# Main Model: CLIP4CAD_GFA_v48
# =============================================================================

class CLIP4CAD_GFA_v48(nn.Module):
    """
    GFA v4.8: Gap-Closing Codebook Architecture

    Key Principles:
    1. Shared codebook for structural alignment across modalities
    2. Direct gap-closing loss (L_align pulls geometry toward text)
    3. No self-grounding path (gap closing makes it unnecessary)
    4. Topology-aware BRep encoding (spatial reasoning)
    5. Code-level supervision (matched pairs activate same codes)

    Unlike v4.4:
    - No curriculum learning
    - No hints/conditioning
    - Single path (no guided vs self)
    - 5 losses instead of 7
    """

    def __init__(self, config: GFAv48Config):
        super().__init__()
        self.config = config
        d = config.d_unified

        # ─────────────────────────────────────────────────────────────────────
        # SHARED SEMANTIC CODEBOOK
        # ─────────────────────────────────────────────────────────────────────
        self.codebook = SemanticCodebook(config.num_codes, d)

        # ─────────────────────────────────────────────────────────────────────
        # MODALITY ENCODERS
        # ─────────────────────────────────────────────────────────────────────
        self.text_encoder = TextEncoder(
            d_in=config.d_text,
            d=d,
            num_layers=config.num_text_tf_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

        self.brep_encoder = TopologyBRepEncoder(
            d=d,
            d_face=config.d_face,
            d_edge=config.d_edge,
            num_msg_layers=config.num_msg_layers,
            num_tf_layers=config.num_brep_tf_layers,
            num_heads=config.num_heads,
            max_levels=config.max_bfs_levels,
            dropout=config.dropout
        )

        self.pc_encoder = PCEncoder(
            d_in=config.d_pc,
            d=d,
            dropout=config.dropout
        )

        # ─────────────────────────────────────────────────────────────────────
        # CODEBOOK GROUNDING (one per modality, shared codebook)
        # ─────────────────────────────────────────────────────────────────────
        self.text_grounding = CodebookGrounding(d, config.num_heads)
        self.brep_grounding = CodebookGrounding(d, config.num_heads)
        self.pc_grounding = CodebookGrounding(d, config.num_heads)

        # ─────────────────────────────────────────────────────────────────────
        # SHARED FUSION NETWORK (CRITICAL: Same weights for all modalities!)
        # ─────────────────────────────────────────────────────────────────────
        self.shared_fusion = SharedFusionNetwork(d, config.d_proj, config.num_heads)

        # ─────────────────────────────────────────────────────────────────────
        # CONTRASTIVE TEMPERATURE
        # ─────────────────────────────────────────────────────────────────────
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))

    @property
    def tau(self):
        return self.log_tau.exp().clamp(0.01, 1.0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GFA v4.8.

        Args:
            batch: Dict with keys:
                - text_features: (B, T, 3072)
                - text_mask: (B, T) optional
                - face_features: (B, N_f, 48)
                - edge_features: (B, N_e, 12)
                - face_mask: (B, N_f)
                - edge_mask: (B, N_e)
                - edge_to_faces: (B, N_e, 2)
                - face_centroids: (B, N_f, 3)
                - face_normals: (B, N_f, 3)
                - face_areas: (B, N_f)
                - bfs_level: (B, N_f)
                - pc_local_features: (B, N_p, 1024)
                - pc_global_features: (B, 1024)

        Returns:
            Dict with:
                - z_text, z_brep, z_pc: Final embeddings (B, d_proj)
                - H_text, H_brep, H_pc: Per-code features (B, M, d)
                - w_text, w_brep, w_pc: Code activation weights (B, num_codes)
                - G_text, G_brep, G_pc: Grounding matrices
                - tau: Temperature
        """
        device = next(self.parameters()).device

        # ─────────────────────────────────────────────────────────────────────
        # ENCODE TEXT
        # ─────────────────────────────────────────────────────────────────────
        text_features = batch['text_features'].to(device)
        text_mask = batch.get('text_mask')
        if text_mask is not None:
            text_mask = text_mask.to(device)

        X_text, text_mask = self.text_encoder(text_features, text_mask)

        # ─────────────────────────────────────────────────────────────────────
        # ENCODE BREP (Topology-Aware)
        # ─────────────────────────────────────────────────────────────────────
        X_brep, brep_mask = self.brep_encoder(
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

        # ─────────────────────────────────────────────────────────────────────
        # ENCODE POINT CLOUD
        # ─────────────────────────────────────────────────────────────────────
        X_pc = self.pc_encoder(
            batch['pc_local_features'].to(device),
            batch['pc_global_features'].to(device)
        )

        # ─────────────────────────────────────────────────────────────────────
        # GROUND TO SHARED CODEBOOK
        # ─────────────────────────────────────────────────────────────────────
        H_text, w_text, G_text = self.text_grounding(X_text, self.codebook, text_mask)
        H_brep, w_brep, G_brep = self.brep_grounding(X_brep, self.codebook, brep_mask)
        H_pc, w_pc, G_pc = self.pc_grounding(X_pc, self.codebook, None)

        # ─────────────────────────────────────────────────────────────────────
        # SHARED FUSION → FINAL EMBEDDINGS
        # ─────────────────────────────────────────────────────────────────────
        z_text = self.shared_fusion(H_text, w_text)
        z_brep = self.shared_fusion(H_brep, w_brep)
        z_pc = self.shared_fusion(H_pc, w_pc)

        return {
            # Final embeddings (for contrastive and ATP loss)
            'z_text': z_text,
            'z_brep': z_brep,
            'z_pc': z_pc,

            # Per-code features (for interpretability)
            'H_text': H_text,
            'H_brep': H_brep,
            'H_pc': H_pc,

            # Code activation weights (for code alignment loss)
            'w_text': w_text,
            'w_brep': w_brep,
            'w_pc': w_pc,

            # Grounding matrices (for interpretability)
            'G_text': G_text,
            'G_brep': G_brep,
            'G_pc': G_pc,

            # Temperature
            'tau': self.tau,
        }

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

        Call this before training for better initialization.

        Args:
            dataloader: DataLoader with text features
            device: Device to use
            remap_fn: Optional function to remap batch keys
            max_samples: Maximum samples to use for K-means
            max_batches: Maximum batches to process
        """
        print("Initializing codebook from text encoder outputs...")

        all_features = []
        self.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                # Apply remap function if provided
                if remap_fn is not None:
                    batch = remap_fn(batch)

                # Cast to float32 to match model weights (avoid FP16 mismatch)
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
        self.codebook.initialize_from_data(all_features)
        print(f"Initialized {self.codebook.num_codes} codes from {len(all_features)} samples")

        self.train()

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ─────────────────────────────────────────────────────────────────────────
    # CONVENIENCE METHODS FOR INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def encode_text(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode text only (for retrieval)."""
        device = next(self.parameters()).device

        # Cast to float32 to match model weights (avoid FP16 mismatch during inference)
        text_features = batch['text_features'].to(device).float()
        text_mask = batch.get('text_mask')
        if text_mask is not None:
            text_mask = text_mask.to(device)

        X_text, text_mask = self.text_encoder(text_features, text_mask)
        H_text, w_text, _ = self.text_grounding(X_text, self.codebook, text_mask)
        z_text = self.shared_fusion(H_text, w_text)

        return z_text

    def encode_brep(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode BRep only (for geometry-only retrieval)."""
        device = next(self.parameters()).device

        # Cast to float32 to match model weights (avoid FP16 mismatch during inference)
        X_brep, brep_mask = self.brep_encoder(
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

        H_brep, w_brep, _ = self.brep_grounding(X_brep, self.codebook, brep_mask)
        z_brep = self.shared_fusion(H_brep, w_brep)

        return z_brep

    def encode_pc(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode point cloud only."""
        device = next(self.parameters()).device

        # Cast to float32 to match model weights (avoid FP16 mismatch during inference)
        X_pc = self.pc_encoder(
            batch['pc_local_features'].to(device).float(),
            batch['pc_global_features'].to(device).float()
        )

        H_pc, w_pc, _ = self.pc_grounding(X_pc, self.codebook, None)
        z_pc = self.shared_fusion(H_pc, w_pc)

        return z_pc
