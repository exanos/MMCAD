"""
CLIP4CAD-GFA v2.4: Shared Encoder Architecture

Key fix from v2: Share grounding and aggregation between guided and self paths.
Only query generation is separate.

PROBLEM WITH v2:
- Self-grounding had too much capacity (separate encoder)
- It learned a different embedding space that minimizes loss but doesn't align with guided
- Embedding distillation (cosine) wasn't strong enough

SOLUTION (v2.4):
- SHARED grounding mechanism (same query projection for T_feat and Q_self)
- SHARED hierarchical aggregation
- Only SelfQueryGenerator is separate (minimal capacity)
- MSE distillation (stronger gradient than cosine)
- Confidence alignment loss

Result: Both paths guaranteed to produce embeddings in the SAME space.
"""

import math
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GFAv2_4Config:
    """Configuration for CLIP4CAD-GFA v2.4."""
    # Model dimensions
    d_face: int = 48          # AutoBrep face features
    d_edge: int = 12          # AutoBrep edge features
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
    num_self_query_layers: int = 2      # Self query adapter layers
    dropout: float = 0.1

    # Max sequence lengths
    max_faces: int = 192
    max_edges: int = 512
    max_pc_tokens: int = 33

    # Temperature initialization
    tau_init: float = 0.07

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "GFAv2_4Config":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Input Projections (unchanged from v2)
# =============================================================================

class BRepProjection(nn.Module):
    """Project B-Rep face and edge features to unified space."""

    def __init__(self, d_face: int, d_edge: int, d: int, dropout: float = 0.1):
        super().__init__()

        d_face_hidden = max(d_face * 2, d)
        self.proj_face = nn.Sequential(
            nn.Linear(d_face, d_face_hidden),
            nn.GELU(),
            nn.LayerNorm(d_face_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_face_hidden, d),
            nn.LayerNorm(d),
            nn.Dropout(dropout)
        )

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

        self.face_type_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(
        self,
        Z_face: torch.Tensor,
        Z_edge: torch.Tensor,
        face_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_face = self.proj_face(Z_face) + self.face_type_embed
        X_edge = self.proj_edge(Z_edge) + self.edge_type_embed
        X_brep = torch.cat([X_face, X_edge], dim=1)

        if face_mask is not None and edge_mask is not None:
            brep_mask = torch.cat([face_mask, edge_mask], dim=1)
            X_brep = X_brep * brep_mask.unsqueeze(-1).float()
        else:
            brep_mask = None

        return X_brep, brep_mask


class PCProjection(nn.Module):
    """Project point cloud features to unified space."""

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

    def forward(self, Z_local: torch.Tensor, Z_global: torch.Tensor) -> torch.Tensor:
        X_local = self.proj_local(Z_local)
        X_global = self.proj_global(Z_global)
        return torch.cat([X_local, X_global], dim=1)


# =============================================================================
# Text Feature Parser (unchanged from v2)
# =============================================================================

class TextFeatureParser(nn.Module):
    """Parse text into K feature slots using cross-attention."""

    def __init__(self, d: int, K: int = 12, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.K = K

        self.feature_queries = nn.Parameter(torch.randn(K, d) * 0.02)

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
        B = X_text.shape[0]
        queries = self.feature_queries.unsqueeze(0).expand(B, -1, -1)

        memory_key_padding_mask = None
        if text_mask is not None:
            memory_key_padding_mask = ~text_mask.bool()

        T_feat = self.parser(queries, X_text, memory_key_padding_mask=memory_key_padding_mask)
        T_feat = torch.nan_to_num(T_feat, nan=0.0)

        conf_logits = self.confidence_head(T_feat).squeeze(-1).clamp(-5, 5)
        confidence = torch.sigmoid(conf_logits)

        return T_feat, confidence


# =============================================================================
# Self Query Generator (NEW - minimal capacity)
# =============================================================================

class SelfQueryGenerator(nn.Module):
    """
    Generate queries for self-grounding from geometry.

    This is the ONLY separate component between guided and self paths.
    Intentionally minimal capacity to prevent learning a different embedding space.
    """

    def __init__(self, d: int, num_slots: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Learnable base queries
        self.base_queries = nn.Parameter(torch.randn(num_slots, d) * 0.02)

        # Lightweight adapter (smaller FFN than text parser: d*2 instead of d*4)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=d * 2,  # SMALLER than text parser!
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.query_adapter = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1)
        )

    def forward(
        self,
        X_geo: torch.Tensor,
        geo_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate queries and confidence from geometry.

        Args:
            X_geo: (B, N, d) - Geometry tokens
            geo_mask: (B, N) - Valid token mask

        Returns:
            Q_self: (B, K, d) - Self-generated queries
            confidence: (B, K) - Slot confidence
        """
        B = X_geo.shape[0]

        Q = self.base_queries.unsqueeze(0).expand(B, -1, -1)

        memory_key_padding_mask = None
        if geo_mask is not None:
            memory_key_padding_mask = ~geo_mask.bool()

        Q = self.query_adapter(Q, X_geo, memory_key_padding_mask=memory_key_padding_mask)
        Q = torch.nan_to_num(Q, nan=0.0)

        conf_logits = self.confidence_head(Q).squeeze(-1).clamp(-5, 5)
        confidence = torch.sigmoid(conf_logits)

        return Q, confidence


# =============================================================================
# Unified Grounding (NEW - SHARED between paths)
# =============================================================================

class UnifiedGrounding(nn.Module):
    """
    SHARED grounding mechanism for both text-guided and self paths.

    KEY CHANGE: Same query projection for T_feat (text) and Q_self (self-generated).
    This forces both paths to use the same grounding space.
    """

    def __init__(self, d: int, d_ground: int = 128):
        super().__init__()

        # SHARED query projection (works for both T_feat and Q_self!)
        self.proj_query = nn.Linear(d, d_ground)

        # Modality-specific geometry projections
        self.proj_brep = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d_ground)
        )
        self.proj_pc = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d_ground)
        )

        # Learnable temperatures per modality
        self.log_tau_brep = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_tau_pc = nn.Parameter(torch.log(torch.tensor(0.1)))

        self.d_ground = d_ground

    def compute_grounding(
        self,
        queries: torch.Tensor,
        X_geo: torch.Tensor,
        modality: str,
        geo_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute grounding matrix.

        Args:
            queries: (B, K, d) - Either T_feat (from text) or Q_self (self-generated)
            X_geo: (B, N, d) - Geometry tokens
            modality: 'brep' or 'pc'
            geo_mask: (B, N) - Valid token mask

        Returns:
            G: (B, K, N) - Grounding matrix (soft attention)
        """
        # SHARED query projection (same for both text and self queries!)
        Q_g = self.proj_query(queries)

        # Modality-specific geometry projection
        if modality == 'brep':
            X_g = self.proj_brep(X_geo)
            tau = self.log_tau_brep.exp().clamp(0.01, 1.0)
        else:
            X_g = self.proj_pc(X_geo)
            tau = self.log_tau_pc.exp().clamp(0.01, 1.0)

        # Scaled dot-product attention
        scores = torch.bmm(Q_g, X_g.transpose(-2, -1))
        scores = scores / (self.d_ground ** 0.5 * tau)

        # Mask invalid positions
        if geo_mask is not None:
            scores = scores.masked_fill(~geo_mask.bool().unsqueeze(1), float('-inf'))

        G = F.softmax(scores, dim=-1)
        return torch.nan_to_num(G, nan=0.0)


# =============================================================================
# Hierarchical Aggregator (unchanged from v2)
# =============================================================================

class HierarchicalAggregator(nn.Module):
    """Two-level feature extraction: Global → Detail."""

    def __init__(self, d: int, num_detail_queries: int = 8, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.global_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.global_attn = nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=dropout)
        self.global_norm = nn.LayerNorm(d)
        self.global_ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )

        self.detail_queries = nn.Parameter(torch.randn(1, num_detail_queries, d) * 0.02)
        self.detail_attn = nn.MultiheadAttention(d, num_heads, batch_first=True, dropout=dropout)
        self.detail_norm = nn.LayerNorm(d)
        self.detail_ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )

        self.global_to_detail = nn.Linear(d, d)

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
        B = X_geo.shape[0]

        # Use grounding matrix G to aggregate X_geo into slot features
        X_slots = torch.bmm(G, X_geo)  # (B, K, d)

        # Confidence-weighted global embedding
        conf_weights = confidence.unsqueeze(-1)
        conf_sum = conf_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        conf_norm = conf_weights / conf_sum

        z_global = (X_slots * conf_norm).sum(dim=1)
        z_global = self.global_norm(z_global)
        z_global = z_global + self.global_ffn(z_global)

        # Detail extraction
        detail_q = self.detail_queries.expand(B, -1, -1)
        global_cond = self.global_to_detail(z_global).unsqueeze(1)
        detail_q = detail_q + global_cond

        z_detail, _ = self.detail_attn(detail_q, X_slots, X_slots)
        z_detail = torch.nan_to_num(z_detail, nan=0.0)
        z_detail = self.detail_norm(z_detail + detail_q)
        z_detail = z_detail + self.detail_ffn(z_detail)
        z_detail = z_detail.mean(dim=1)

        # Fusion
        concat = torch.cat([z_global, z_detail], dim=-1)
        gate = self.fusion_gate(concat)
        z_unified = gate[:, 0:1] * z_global + gate[:, 1:2] * z_detail

        return z_global, z_detail, z_unified


# =============================================================================
# Main Model
# =============================================================================

class CLIP4CAD_GFA_v2_4(nn.Module):
    """
    CLIP4CAD-GFA v2.4: Shared Encoder Architecture

    KEY CHANGE: Grounding and aggregation are SHARED between guided and self paths.
    Only query generation is separate (SelfQueryGenerator).

    This guarantees both paths produce embeddings in the SAME space.
    """

    def __init__(self, config: GFAv2_4Config):
        super().__init__()
        self.config = config

        d = config.d_unified
        K = config.num_slots

        # ─────────────────────────────────────────────────────────────────────
        # Input Projections
        # ─────────────────────────────────────────────────────────────────────
        self.brep_proj = BRepProjection(
            config.d_face, config.d_edge, d, config.dropout
        )
        self.pc_proj = PCProjection(config.d_pc, d, config.dropout)
        self.text_proj = nn.Sequential(
            nn.Linear(config.d_text, d),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout)
        )

        # ─────────────────────────────────────────────────────────────────────
        # Text Feature Parser (for guided path)
        # ─────────────────────────────────────────────────────────────────────
        self.text_parser = TextFeatureParser(
            d, K, config.num_parser_layers, config.num_heads, config.dropout
        )

        # ─────────────────────────────────────────────────────────────────────
        # Self Query Generator (for self path) - ONLY SEPARATE COMPONENT
        # ─────────────────────────────────────────────────────────────────────
        self.self_query_gen = SelfQueryGenerator(
            d, K, config.num_self_query_layers, config.num_heads, config.dropout
        )

        # ─────────────────────────────────────────────────────────────────────
        # SHARED: Unified Grounding
        # ─────────────────────────────────────────────────────────────────────
        self.grounding = UnifiedGrounding(d, config.d_ground)

        # ─────────────────────────────────────────────────────────────────────
        # SHARED: Hierarchical Aggregation
        # ─────────────────────────────────────────────────────────────────────
        self.hierarchical_agg = HierarchicalAggregator(
            d, config.num_detail_queries, config.num_heads, config.dropout
        )

        # ─────────────────────────────────────────────────────────────────────
        # SHARED: Projection Heads
        # ─────────────────────────────────────────────────────────────────────
        self.proj_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.d_proj)
        )
        self.detail_proj_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.d_proj)
        )

        # Learnable temperature
        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.tau_init)))

    def encode_text(
        self,
        H_text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Parse text into feature slots."""
        X_text = self.text_proj(H_text)
        T_feat, confidence = self.text_parser(X_text, text_mask)

        # Global text embedding (confidence-weighted)
        conf_weights = confidence.unsqueeze(-1)
        conf_sum = conf_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        z_text = (T_feat * conf_weights).sum(dim=1) / conf_sum.squeeze(-1)

        return {
            'T_feat': T_feat,
            'confidence': confidence,
            'z_text': z_text
        }

    def encode_geometry(
        self,
        X_geo: torch.Tensor,
        queries: torch.Tensor,
        confidence: torch.Tensor,
        modality: str,
        geo_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        SHARED encoding function for both guided and self paths.

        Args:
            X_geo: (B, N, d) - Geometry tokens
            queries: (B, K, d) - Either T_feat (text) or Q_self (self-generated)
            confidence: (B, K) - Slot confidence
            modality: 'brep' or 'pc'
            geo_mask: (B, N) - Valid token mask
        """
        # SHARED grounding computation
        G = self.grounding.compute_grounding(queries, X_geo, modality, geo_mask)

        # SHARED hierarchical aggregation
        z_global, z_detail, z_unified = self.hierarchical_agg(X_geo, G, confidence, geo_mask)

        return {
            'z_unified': z_unified,
            'z_global': z_global,
            'z_detail': z_detail,
            'G': G
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Full forward pass with both encoding paths.

        Both paths use the SAME grounding and aggregation - only queries differ.
        """
        device = next(self.parameters()).device

        # ─────────────────────────────────────────────────────────────────────
        # Extract and move inputs to device
        # ─────────────────────────────────────────────────────────────────────
        Z_face = batch['brep_face_features'].to(device)
        Z_edge = batch['brep_edge_features'].to(device)
        face_mask = batch.get('brep_face_mask')
        edge_mask = batch.get('brep_edge_mask')

        if face_mask is not None:
            face_mask = face_mask.to(device)
        if edge_mask is not None:
            edge_mask = edge_mask.to(device)

        pc_features = batch['pc_features'].to(device)
        Z_pc_local = pc_features[:, :32, :]
        Z_pc_global = pc_features[:, 32:33, :]

        H_text = batch['desc_embedding'].to(device)
        if H_text.dim() == 2:
            H_text = H_text.unsqueeze(1)

        # ─────────────────────────────────────────────────────────────────────
        # Project inputs
        # ─────────────────────────────────────────────────────────────────────
        X_brep, brep_mask = self.brep_proj(Z_face, Z_edge, face_mask, edge_mask)
        X_pc = self.pc_proj(Z_pc_local, Z_pc_global)

        # ─────────────────────────────────────────────────────────────────────
        # TEXT PATH: Parse text into queries
        # ─────────────────────────────────────────────────────────────────────
        text_out = self.encode_text(H_text)
        z_text = self.proj_head(text_out['z_text'])

        # ─────────────────────────────────────────────────────────────────────
        # GUIDED PATH: Use text features as queries (SHARED encoding!)
        # ─────────────────────────────────────────────────────────────────────
        brep_guided = self.encode_geometry(
            X_brep,
            queries=text_out['T_feat'],
            confidence=text_out['confidence'],
            modality='brep',
            geo_mask=brep_mask
        )
        pc_guided = self.encode_geometry(
            X_pc,
            queries=text_out['T_feat'],
            confidence=text_out['confidence'],
            modality='pc',
            geo_mask=None
        )

        z_brep_guided = self.proj_head(brep_guided['z_unified'])
        z_pc_guided = self.proj_head(pc_guided['z_unified'])
        z_brep_detail = self.detail_proj_head(brep_guided['z_detail'])
        z_pc_detail = self.detail_proj_head(pc_guided['z_detail'])

        # ─────────────────────────────────────────────────────────────────────
        # SELF PATH: Generate queries from geometry (SHARED encoding!)
        # ─────────────────────────────────────────────────────────────────────
        Q_brep_self, conf_brep_self = self.self_query_gen(X_brep, brep_mask)
        Q_pc_self, conf_pc_self = self.self_query_gen(X_pc, None)

        brep_self = self.encode_geometry(
            X_brep,
            queries=Q_brep_self,
            confidence=conf_brep_self,
            modality='brep',
            geo_mask=brep_mask
        )
        pc_self = self.encode_geometry(
            X_pc,
            queries=Q_pc_self,
            confidence=conf_pc_self,
            modality='pc',
            geo_mask=None
        )

        z_brep_self = self.proj_head(brep_self['z_unified'])
        z_pc_self = self.proj_head(pc_self['z_unified'])

        return {
            # Primary embeddings (text-guided)
            'z_brep': z_brep_guided,
            'z_pc': z_pc_guided,
            'z_text': z_text,

            # Self embeddings (for inference)
            'z_brep_self': z_brep_self,
            'z_pc_self': z_pc_self,

            # Detail level (for hard negatives)
            'z_brep_detail': z_brep_detail,
            'z_pc_detail': z_pc_detail,

            # Grounding matrices (for distillation + visualization)
            'G_brep_guided': brep_guided['G'],
            'G_pc_guided': pc_guided['G'],
            'G_brep_self': brep_self['G'],
            'G_pc_self': pc_self['G'],

            # Confidence (for alignment loss)
            'confidence': text_out['confidence'],
            'confidence_brep_self': conf_brep_self,
            'confidence_pc_self': conf_pc_self,

            # Temperature
            'tau': self.log_tau.exp().clamp(0.01, 1.0),
        }

    def encode_for_retrieval(
        self,
        batch: Dict[str, torch.Tensor],
        use_self: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode geometry for retrieval (inference mode).

        Args:
            batch: Input batch
            use_self: If True, use self-grounding (no text needed)
        """
        outputs = self.forward(batch)

        if use_self:
            return {
                'z_brep': outputs['z_brep_self'],
                'z_pc': outputs['z_pc_self'],
            }
        else:
            return {
                'z_brep': outputs['z_brep'],
                'z_pc': outputs['z_pc'],
            }

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
