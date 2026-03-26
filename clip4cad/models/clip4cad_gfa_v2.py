"""
CLIP4CAD-GFA v2: Grounded Feature Alignment for CAD Multimodal Learning

Key innovations from GFA v1:
1. Modality-Specific Grounding: Separate projections for B-Rep and PC
2. Joint Self-Grounding Training: Self-path learns via direct contrastive loss
3. Hierarchical Aggregation: Global → Detail conditioning for fine-grained discrimination
4. Simplified Loss: 4 terms (guided + self + distill + detail) instead of 8

Based on CLIP4CAD_GFA_v2_Architecture.md
"""

import math
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GFAv2Config:
    """Configuration for CLIP4CAD-GFA v2."""
    # Model dimensions (FSQ-encoded AutoBrep features)
    d_face: int = 48          # AutoBrep face features (FSQ latent: rich surface info)
    d_edge: int = 12          # AutoBrep edge features (FSQ latent: edge geometry)
    d_pc: int = 1024          # ShapeLLM features
    d_text: int = 3072        # Phi-4-mini features
    d_unified: int = 256      # Internal dimension
    d_proj: int = 128         # Projection head output
    d_ground: int = 128       # Grounding projection dimension

    # Architecture
    num_slots: int = 12                 # Feature parsing slots (K)
    num_detail_queries: int = 8         # Hierarchical detail queries
    num_heads: int = 8                  # Attention heads
    num_parser_layers: int = 2          # Text parser transformer layers
    num_self_ground_layers: int = 2     # Self-grounding adapter layers
    dropout: float = 0.1

    # Max sequence lengths
    max_faces: int = 192
    max_edges: int = 512
    max_pc_tokens: int = 33   # 32 local + 1 global
    max_text_tokens: int = 512

    # Temperature initialization
    tau_init: float = 0.07
    tau_ground_init: float = 0.1

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "GFAv2Config":
        """Create config from dictionary (for OmegaConf compatibility)."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class BRepProjection(nn.Module):
    """
    Project B-Rep face and edge features to unified space with type embeddings.

    B-Rep tokens are discrete semantic units (faces = surfaces, edges = boundaries).
    Type embeddings help the model distinguish between them.

    Enhanced to better utilize FSQ-encoded features:
    - Deeper projection for faces (48-dim FSQ latent has rich surface information)
    - Intermediate bottleneck to extract geometric primitives
    - Residual connection to preserve FSQ structure
    """

    def __init__(self, d_face: int, d_edge: int, d: int, dropout: float = 0.1):
        super().__init__()

        # Face projection: deeper network to extract FSQ information
        # FSQ encodes surface type, curvature, orientation, etc.
        d_face_hidden = max(d_face * 2, d)  # Expand first to capture interactions
        self.proj_face = nn.Sequential(
            nn.Linear(d_face, d_face_hidden),
            nn.GELU(),
            nn.LayerNorm(d_face_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_face_hidden, d),
            nn.LayerNorm(d),
            nn.Dropout(dropout)
        )

        # Edge projection: similar treatment for edge FSQ features
        # Edges encode boundary curves, connectivity
        d_edge_hidden = max(d_edge * 2, d // 2)
        self.proj_edge = nn.Sequential(
            nn.Linear(d_edge, d_edge_hidden),
            nn.GELU(),
            nn.LayerNorm(d_edge_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_edge_hidden, d),
            nn.LayerNorm(d),
            nn.Dropout(dropout)
        )

        # Learnable type embeddings to distinguish faces from edges
        self.face_type_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(
        self,
        Z_face: torch.Tensor,
        Z_edge: torch.Tensor,
        face_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z_face: (B, F, d_face) - Face features from AutoBrep
            Z_edge: (B, E, d_edge) - Edge features from AutoBrep
            face_mask: (B, F) - Valid faces
            edge_mask: (B, E) - Valid edges

        Returns:
            X_brep: (B, F+E, d) - Unified B-Rep tokens
            brep_mask: (B, F+E) - Combined mask
        """
        X_face = self.proj_face(Z_face) + self.face_type_embed
        X_edge = self.proj_edge(Z_edge) + self.edge_type_embed

        X_brep = torch.cat([X_face, X_edge], dim=1)

        # Combine masks
        if face_mask is not None and edge_mask is not None:
            brep_mask = torch.cat([face_mask, edge_mask], dim=1)
            # Zero out padding
            X_brep = X_brep * brep_mask.unsqueeze(-1).float()
        else:
            brep_mask = None

        return X_brep, brep_mask


class PCProjection(nn.Module):
    """
    Project point cloud features (local patches + global token) to unified space.

    ShapeLLM provides 32 local patches + 1 global token.
    """

    def __init__(self, d_pc: int, d: int, dropout: float = 0.1):
        super().__init__()

        self.proj_local = nn.Sequential(
            nn.Linear(d_pc, d),
            nn.LayerNorm(d),
            nn.Dropout(dropout)
        )
        self.proj_global = nn.Sequential(
            nn.Linear(d_pc, d),
            nn.LayerNorm(d)
        )

    def forward(
        self,
        Z_local: torch.Tensor,
        Z_global: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            Z_local: (B, L, d_pc) - Local patch features (32 patches)
            Z_global: (B, 1, d_pc) - Global token from ShapeLLM

        Returns:
            X_pc: (B, L+1, d) - All PC tokens
        """
        X_local = self.proj_local(Z_local)
        X_global = self.proj_global(Z_global)
        X_pc = torch.cat([X_local, X_global], dim=1)
        return X_pc


class TextFeatureParser(nn.Module):
    """
    Parse text into K feature slots using cross-attention.

    Each slot represents a potential geometric feature mentioned in the text
    (e.g., "serrated teeth", "through-hole", "chamfered edge").
    """

    def __init__(self, d: int, K: int = 12, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.K = K

        # Learnable feature queries
        self.feature_queries = nn.Parameter(torch.randn(K, d) * 0.02)

        # Cross-attention: queries attend to text tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.parser = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Predict which slots are active (confidence)
        self.confidence_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1)
        )

    def forward(
        self,
        X_text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X_text: (B, T, d) - Projected text tokens
            text_mask: (B, T) - Valid token mask (True = valid)

        Returns:
            T_feat: (B, K, d) - K feature slot embeddings
            confidence: (B, K) - Importance weight per slot
        """
        B = X_text.shape[0]

        # Expand queries for batch
        queries = self.feature_queries.unsqueeze(0).expand(B, -1, -1)

        # Convert mask: our True=valid, PyTorch expects True=ignore
        memory_key_padding_mask = None
        if text_mask is not None:
            memory_key_padding_mask = ~text_mask.bool()

        # Cross-attend to text
        T_feat = self.parser(
            queries, X_text,
            memory_key_padding_mask=memory_key_padding_mask
        )
        T_feat = torch.nan_to_num(T_feat, nan=0.0)  # Handle all-masked cases

        # Predict confidence (which slots are active) with FP16-safe clamping
        conf_logits = self.confidence_head(T_feat).squeeze(-1)
        conf_logits = conf_logits.clamp(-5, 5)  # Prevent extreme values in FP16
        confidence = torch.sigmoid(conf_logits)

        return T_feat, confidence


class ModalitySpecificGrounding(nn.Module):
    """
    Modality-specific grounding projections (KEY FIX #1).

    B-Rep tokens are discrete semantic units (faces, edges), while PC tokens
    are spatial patches. They need different projections to map to the same
    grounding space.
    """

    def __init__(self, d: int, d_ground: int = 128):
        super().__init__()

        # Text projection (shared anchor)
        self.proj_text = nn.Linear(d, d_ground)

        # B-Rep projection (deeper - needs to learn CAD semantics)
        self.proj_brep = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d_ground)
        )

        # PC projection (deeper - needs to learn spatial semantics)
        self.proj_pc = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d_ground)
        )

        # Learnable temperature per modality
        self.log_tau_brep = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_tau_pc = nn.Parameter(torch.log(torch.tensor(0.1)))

        self.d_ground = d_ground

    def compute_grounding(
        self,
        T_feat: torch.Tensor,
        X_geo: torch.Tensor,
        modality: str,
        geo_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute grounding matrix: where does each text slot attend in geometry?

        Args:
            T_feat: (B, K, d) - Text feature slots
            X_geo: (B, N, d) - Geometry tokens
            modality: 'brep' or 'pc'
            geo_mask: (B, N) - Valid geometry tokens (True = valid)

        Returns:
            G: (B, K, N) - Grounding matrix (soft attention)
        """
        # Project to grounding space
        T_g = self.proj_text(T_feat)  # (B, K, d_g)

        if modality == 'brep':
            X_g = self.proj_brep(X_geo)  # (B, N, d_g)
            tau = self.log_tau_brep.exp().clamp(0.01, 1.0)
        else:
            X_g = self.proj_pc(X_geo)
            tau = self.log_tau_pc.exp().clamp(0.01, 1.0)

        # Scaled dot-product attention
        scores = torch.bmm(T_g, X_g.transpose(-2, -1))  # (B, K, N)
        scores = scores / (self.d_ground ** 0.5 * tau)

        # Mask invalid positions
        if geo_mask is not None:
            scores = scores.masked_fill(~geo_mask.bool().unsqueeze(1), float('-inf'))

        # Softmax to get attention weights
        G = F.softmax(scores, dim=-1)
        G = torch.nan_to_num(G, nan=0.0)  # Handle all-masked rows

        return G


class HierarchicalAggregator(nn.Module):
    """
    Two-level feature extraction: Global → Detail.

    Global: Overall shape identity ("This is a gear")
    Detail: Fine-grained features ("It has 32 teeth")

    Global CONDITIONS detail: "I know it's a gear, so look for teeth"
    """

    def __init__(self, d: int, num_detail_queries: int = 8, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Global extraction: single query for overall shape
        self.global_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.global_attn = nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=dropout)
        self.global_norm = nn.LayerNorm(d)
        self.global_ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )

        # Detail extraction: multiple queries for fine features
        self.detail_queries = nn.Parameter(torch.randn(1, num_detail_queries, d) * 0.02)
        self.detail_attn = nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=dropout)
        self.detail_norm = nn.LayerNorm(d)
        self.detail_ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )

        # Global-to-detail conditioning
        self.global_to_detail = nn.Linear(d, d)

        # Learned fusion weights
        self.fusion_gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, 2),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        X_geo: torch.Tensor,
        G: torch.Tensor,
        confidence: torch.Tensor,
        geo_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X_geo: (B, N, d) - Geometry tokens
            G: (B, K, N) - Grounding matrix (THIS IS KEY - differentiates guided vs self)
            confidence: (B, K) - Slot confidence
            geo_mask: (B, N) - Valid token mask

        Returns:
            z_global: (B, d) - Global shape embedding
            z_detail: (B, d) - Detail features embedding
            z_unified: (B, d) - Fused embedding
        """
        B = X_geo.shape[0]

        # ─────────────────────────────────────────────────────────────────
        # USE GROUNDING MATRIX G to aggregate X_geo into slot features
        # This is what differentiates guided vs self paths!
        # G: (B, K, N), X_geo: (B, N, d) -> X_slots: (B, K, d)
        # ─────────────────────────────────────────────────────────────────

        X_slots = torch.bmm(G, X_geo)  # (B, K, d) - grounding-weighted features

        # Weight by confidence to get global embedding
        # confidence: (B, K) -> (B, K, 1)
        conf_weights = confidence.unsqueeze(-1)
        conf_sum = conf_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        conf_norm = conf_weights / conf_sum

        # Global: confidence-weighted sum of grounded slots
        z_global = (X_slots * conf_norm).sum(dim=1)  # (B, d)
        z_global = self.global_norm(z_global)
        z_global = z_global + self.global_ffn(z_global)

        # ─────────────────────────────────────────────────────────────────
        # LEVEL 2: Detail Extraction from grounded slots
        # ─────────────────────────────────────────────────────────────────

        detail_q = self.detail_queries.expand(B, -1, -1)

        # Global context guides detail queries
        global_cond = self.global_to_detail(z_global).unsqueeze(1)
        detail_q = detail_q + global_cond

        # Attend to the GROUNDED slots (not raw X_geo)
        z_detail, _ = self.detail_attn(
            detail_q, X_slots, X_slots,
            key_padding_mask=None  # All K slots are valid
        )
        z_detail = torch.nan_to_num(z_detail, nan=0.0)
        z_detail = self.detail_norm(z_detail + detail_q)
        z_detail = z_detail + self.detail_ffn(z_detail)
        z_detail = z_detail.mean(dim=1)  # (B, d)

        # ─────────────────────────────────────────────────────────────────
        # FUSION
        # ─────────────────────────────────────────────────────────────────

        concat = torch.cat([z_global, z_detail], dim=-1)
        gate = self.fusion_gate(concat)  # (B, 2)
        z_unified = gate[:, 0:1] * z_global + gate[:, 1:2] * z_detail

        return z_global, z_detail, z_unified


class JointSelfGrounding(nn.Module):
    """
    Self-grounding for text-free inference (KEY FIX #2).

    KEY INSIGHT: Train jointly with contrastive loss, not just distillation!

    The queries adapt to each geometry input, learning to identify
    important regions WITHOUT text guidance.
    """

    def __init__(self, d: int, num_slots: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_slots = num_slots

        # Learnable base queries
        self.base_queries = nn.Parameter(torch.randn(num_slots, d) * 0.02)

        # Geometry-adaptive: queries attend to geometry to understand it
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.query_adapter = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Modality-specific grounding (mirrors text-guided)
        self.proj_brep = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 128)
        )
        self.proj_pc = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 128)
        )
        self.proj_query = nn.Linear(d, 128)

        # Confidence prediction (which slots matter)
        self.confidence_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1)
        )

        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.1)))

    def forward(
        self,
        X_geo: torch.Tensor,
        modality: str,
        geo_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute self-grounding WITHOUT text input.

        Args:
            X_geo: (B, N, d) - Geometry tokens
            modality: 'brep' or 'pc'
            geo_mask: (B, N) - Valid token mask

        Returns:
            G_self: (B, K, N) - Self-grounding matrix
            confidence: (B, K) - Learned slot confidence
            Q: (B, K, d) - Adapted queries (for hierarchical agg)
        """
        B = X_geo.shape[0]

        # Expand base queries
        Q = self.base_queries.unsqueeze(0).expand(B, -1, -1)

        # Convert mask for attention
        memory_key_padding_mask = None
        if geo_mask is not None:
            memory_key_padding_mask = ~geo_mask.bool()

        # Adapt queries to THIS specific geometry
        Q = self.query_adapter(
            Q, X_geo,
            memory_key_padding_mask=memory_key_padding_mask
        )
        Q = torch.nan_to_num(Q, nan=0.0)  # Handle all-masked cases

        # Predict confidence per slot with FP16-safe clamping
        conf_logits = self.confidence_head(Q).squeeze(-1)
        conf_logits = conf_logits.clamp(-5, 5)
        confidence = torch.sigmoid(conf_logits)

        # Compute grounding matrix
        Q_g = self.proj_query(Q)

        if modality == 'brep':
            X_g = self.proj_brep(X_geo)
        else:
            X_g = self.proj_pc(X_geo)

        tau = self.log_tau.exp().clamp(0.01, 1.0)
        scores = torch.bmm(Q_g, X_g.transpose(-2, -1)) / (128 ** 0.5 * tau)

        if geo_mask is not None:
            scores = scores.masked_fill(~geo_mask.bool().unsqueeze(1), float('-inf'))

        G_self = F.softmax(scores, dim=-1)
        G_self = torch.nan_to_num(G_self, nan=0.0)

        return G_self, confidence, Q


class CLIP4CAD_GFA_v2(nn.Module):
    """
    CLIP4CAD-GFA v2: Grounded Feature Alignment for CAD Multimodal Learning

    Key innovations:
    1. Modality-specific grounding projections
    2. Joint self-grounding training (not post-hoc!)
    3. Hierarchical global→detail extraction
    4. Unified embedding for retrieval + generation
    """

    def __init__(self, config: GFAv2Config):
        super().__init__()
        self.config = config

        d = config.d_unified
        K = config.num_slots

        # ─────────────────────────────────────────────────────────────────
        # Input Projections
        # ─────────────────────────────────────────────────────────────────

        self.brep_proj = BRepProjection(
            config.d_face, config.d_edge, d, config.dropout
        )
        self.pc_proj = PCProjection(config.d_pc, d, config.dropout)
        self.text_proj = nn.Sequential(
            nn.Linear(config.d_text, d),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout)
        )

        # ─────────────────────────────────────────────────────────────────
        # Text Parsing
        # ─────────────────────────────────────────────────────────────────

        self.text_parser = TextFeatureParser(
            d, K, config.num_parser_layers, config.num_heads, config.dropout
        )

        # ─────────────────────────────────────────────────────────────────
        # Grounding
        # ─────────────────────────────────────────────────────────────────

        self.grounding = ModalitySpecificGrounding(d, d_ground=config.d_ground)
        self.self_grounding = JointSelfGrounding(
            d, K, config.num_self_ground_layers, config.num_heads, config.dropout
        )

        # ─────────────────────────────────────────────────────────────────
        # Hierarchical Aggregation
        # ─────────────────────────────────────────────────────────────────

        self.hierarchical_agg = HierarchicalAggregator(
            d, config.num_detail_queries, config.num_heads, config.dropout
        )

        # ─────────────────────────────────────────────────────────────────
        # Projection Head (Shared)
        # ─────────────────────────────────────────────────────────────────

        self.proj_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.d_proj)
        )

        # Separate detail projection head
        self.detail_proj_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.d_proj)
        )

        # Learnable temperature
        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.tau_init)))

    @property
    def tau(self) -> torch.Tensor:
        """Contrastive temperature, clamped to valid range."""
        return self.log_tau.exp().clamp(0.01, 1.0)

    def encode_text(
        self,
        H_text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode text into feature slots and global embedding."""
        X_text = self.text_proj(H_text)
        T_feat, confidence = self.text_parser(X_text, text_mask)

        # Global text embedding (confidence-weighted)
        conf_sum = confidence.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        conf_norm = confidence / conf_sum
        z_text = (T_feat * conf_norm.unsqueeze(-1)).sum(dim=1)

        return {
            'T_feat': T_feat,
            'confidence': confidence,
            'z_text': z_text,
            'X_text': X_text
        }

    def encode_geometry_guided(
        self,
        X_geo: torch.Tensor,
        T_feat: torch.Tensor,
        confidence: torch.Tensor,
        modality: str,
        geo_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Text-guided geometry encoding (training path)."""

        # Compute grounding matrix
        G = self.grounding.compute_grounding(T_feat, X_geo, modality, geo_mask)

        # Hierarchical aggregation
        z_global, z_detail, z_unified = self.hierarchical_agg(
            X_geo, G, confidence, geo_mask
        )

        return {
            'z_unified': z_unified,
            'z_global': z_global,
            'z_detail': z_detail,
            'G': G
        }

    def encode_geometry_self(
        self,
        X_geo: torch.Tensor,
        modality: str,
        geo_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Self-grounding geometry encoding (inference path)."""

        # Self-grounding
        G_self, confidence, Q = self.self_grounding(X_geo, modality, geo_mask)

        # Same aggregation pipeline
        z_global, z_detail, z_unified = self.hierarchical_agg(
            X_geo, G_self, confidence, geo_mask
        )

        return {
            'z_unified': z_unified,
            'z_global': z_global,
            'z_detail': z_detail,
            'G': G_self,
            'confidence': confidence
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Full forward pass with both encoding paths.

        Args:
            batch: Dictionary containing:
                - brep_face_features: (B, F, d_face)
                - brep_edge_features: (B, E, d_edge)
                - brep_face_mask: (B, F)
                - brep_edge_mask: (B, E)
                - pc_local_features: (B, L, d_pc)
                - pc_global_features: (B, 1, d_pc)
                - desc_embedding: (B, T, d_text)
                - desc_mask: (B, T)

        Returns:
            Dictionary with all embeddings and intermediate outputs
        """
        device = next(self.parameters()).device

        # ─────────────────────────────────────────────────────────────────
        # Project inputs
        # ─────────────────────────────────────────────────────────────────

        # B-Rep
        face_features = batch.get("brep_face_features")
        edge_features = batch.get("brep_edge_features")
        face_mask = batch.get("brep_face_mask")
        edge_mask = batch.get("brep_edge_mask")

        X_brep, brep_mask = None, None
        if face_features is not None and edge_features is not None:
            face_features = face_features.to(device)
            edge_features = edge_features.to(device)
            face_mask = face_mask.to(device) if face_mask is not None else None
            edge_mask = edge_mask.to(device) if edge_mask is not None else None
            X_brep, brep_mask = self.brep_proj(face_features, edge_features, face_mask, edge_mask)

        # Point Cloud
        pc_local = batch.get("pc_local_features")
        pc_global = batch.get("pc_global_features")

        # Backward compatibility: check for old key names
        if pc_local is None:
            pc_features = batch.get("pc_features")
            if pc_features is not None:
                pc_features = pc_features.to(device)
                # Split into local and global (assuming last token is global)
                pc_local = pc_features[:, :-1]
                pc_global = pc_features[:, -1:]

        X_pc = None
        if pc_local is not None and pc_global is not None:
            pc_local = pc_local.to(device)
            pc_global = pc_global.to(device)
            X_pc = self.pc_proj(pc_local, pc_global)

        # Text
        text_features = batch.get("desc_embedding")
        text_mask = batch.get("desc_mask")

        if text_features is not None:
            text_features = text_features.to(device)
            text_mask = text_mask.to(device) if text_mask is not None else None

        # ─────────────────────────────────────────────────────────────────
        # Encode text
        # ─────────────────────────────────────────────────────────────────

        text_out = self.encode_text(text_features, text_mask)
        z_text = self.proj_head(text_out['z_text'])

        # ─────────────────────────────────────────────────────────────────
        # TEXT-GUIDED ENCODING (Training path)
        # ─────────────────────────────────────────────────────────────────

        z_brep_guided, z_brep_detail = None, None
        G_brep_guided = None
        if X_brep is not None:
            brep_guided = self.encode_geometry_guided(
                X_brep, text_out['T_feat'], text_out['confidence'], 'brep', brep_mask
            )
            z_brep_guided = self.proj_head(brep_guided['z_unified'])
            z_brep_detail = self.detail_proj_head(brep_guided['z_detail'])
            G_brep_guided = brep_guided['G']

        z_pc_guided, z_pc_detail = None, None
        G_pc_guided = None
        if X_pc is not None:
            pc_guided = self.encode_geometry_guided(
                X_pc, text_out['T_feat'], text_out['confidence'], 'pc', None
            )
            z_pc_guided = self.proj_head(pc_guided['z_unified'])
            z_pc_detail = self.detail_proj_head(pc_guided['z_detail'])
            G_pc_guided = pc_guided['G']

        # ─────────────────────────────────────────────────────────────────
        # SELF ENCODING (Inference path)
        # ─────────────────────────────────────────────────────────────────

        z_brep_self, conf_brep_self = None, None
        G_brep_self = None
        if X_brep is not None:
            brep_self = self.encode_geometry_self(X_brep, 'brep', brep_mask)
            z_brep_self = self.proj_head(brep_self['z_unified'])
            conf_brep_self = brep_self['confidence']
            G_brep_self = brep_self['G']

        z_pc_self, conf_pc_self = None, None
        G_pc_self = None
        if X_pc is not None:
            pc_self = self.encode_geometry_self(X_pc, 'pc', None)
            z_pc_self = self.proj_head(pc_self['z_unified'])
            conf_pc_self = pc_self['confidence']
            G_pc_self = pc_self['G']

        return {
            # Primary embeddings (text-guided)
            'z_brep': z_brep_guided,
            'z_pc': z_pc_guided,
            'z_text': z_text,

            # Self-encoded (for inference)
            'z_brep_self': z_brep_self,
            'z_pc_self': z_pc_self,

            # Detail level (for hard negatives)
            'z_brep_detail': z_brep_detail,
            'z_pc_detail': z_pc_detail,

            # Grounding matrices (for distillation + visualization)
            'G_brep_guided': G_brep_guided,
            'G_pc_guided': G_pc_guided,
            'G_brep_self': G_brep_self,
            'G_pc_self': G_pc_self,

            # Confidence
            'confidence': text_out['confidence'],
            'confidence_brep_self': conf_brep_self,
            'confidence_pc_self': conf_pc_self,

            # Temperature
            'tau': self.tau,
        }

    @torch.no_grad()
    def encode_for_retrieval(
        self,
        batch: Dict[str, torch.Tensor],
        use_self_grounding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode for retrieval (inference mode).

        Args:
            batch: Input batch
            use_self_grounding: If True, use self-grounding (no text needed)

        Returns:
            Normalized embeddings for retrieval
        """
        self.eval()
        outputs = self.forward(batch)

        result = {}

        if use_self_grounding:
            if outputs['z_brep_self'] is not None:
                result['z_brep'] = F.normalize(outputs['z_brep_self'], dim=-1)
            if outputs['z_pc_self'] is not None:
                result['z_pc'] = F.normalize(outputs['z_pc_self'], dim=-1)
        else:
            if outputs['z_brep'] is not None:
                result['z_brep'] = F.normalize(outputs['z_brep'], dim=-1)
            if outputs['z_pc'] is not None:
                result['z_pc'] = F.normalize(outputs['z_pc'], dim=-1)

        if outputs['z_text'] is not None:
            result['z_text'] = F.normalize(outputs['z_text'], dim=-1)

        return result

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config_path: str) -> "CLIP4CAD_GFA_v2":
        """Create model from config file."""
        from omegaconf import OmegaConf
        config_dict = OmegaConf.load(config_path)
        config = GFAv2Config.from_dict(OmegaConf.to_container(config_dict, resolve=True))
        return cls(config)
