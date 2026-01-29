"""
CLIP4CAD-GFA v4.2: Conditional Self-Query Generation with Curriculum Learning

Key insight: The decoder doesn't know what T_feat "looks like".
Solution: During training, occasionally SHOW the model T_feat as a hint,
then gradually remove the hints via curriculum learning.

Training curriculum:
- Early: Heavy conditioning (90% samples get T_feat hints)
- Middle: Partial conditioning (50% samples get hints)
- Late: Minimal conditioning (30% samples get hints)
- Stage 2: No conditioning (fully independent)

This teaches the model what kind of features to produce, then forces independence.
"""

import math
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GFAv4_2Config:
    """Configuration for CLIP4CAD-GFA v4.2."""
    # Model dimensions
    d_face: int = 48
    d_edge: int = 12
    d_pc: int = 1024
    d_text: int = 3072
    d_unified: int = 256
    d_proj: int = 128
    d_ground: int = 128

    # Architecture
    num_slots: int = 12
    num_detail_queries: int = 8
    num_heads: int = 8
    num_parser_layers: int = 2
    dropout: float = 0.1

    # Conditional Self-Query Generator (NEW)
    brep_encoder_layers: int = 4      # Deeper for BRep (no multimodal alignment)
    brep_decoder_layers: int = 4
    pc_encoder_layers: int = 2        # Lighter for PC (ShapeLLM already aligned)
    pc_decoder_layers: int = 2

    # Max sequence lengths
    max_faces: int = 192
    max_edges: int = 512
    max_pc_tokens: int = 33

    # Temperature
    tau_init: float = 0.07

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "GFAv4_2Config":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Input Projections (same as v4)
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
# Text Feature Parser (same as v4)
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

        conf_logits = self.confidence_head(T_feat).squeeze(-1).clamp(-5, 5)
        confidence = torch.sigmoid(conf_logits)

        return T_feat, confidence


# =============================================================================
# Conditional Self-Query Generator (NEW in v4.2)
# =============================================================================

class ConditionalSelfQueryGenerator(nn.Module):
    """
    Self-query generator with optional text conditioning (curriculum learning).

    Key insight: During training, occasionally show the model what T_feat looks like.
    This teaches it what kind of features to produce.

    Training curriculum:
    - Early: Heavy conditioning (learn output distribution)
    - Middle: Partial conditioning (learn to fill in gaps)
    - Late: No conditioning (fully independent)

    Inference: Always independent (no text available)
    """

    def __init__(
        self,
        d: int,
        K: int,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d = d
        self.K = K

        # ─────────────────────────────────────────────────────────────────────
        # GEOMETRY ENCODER: Understand the geometry deeply
        # ─────────────────────────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.geo_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # ─────────────────────────────────────────────────────────────────────
        # LEARNABLE QUERIES
        # ─────────────────────────────────────────────────────────────────────
        self.base_queries = nn.Parameter(torch.randn(K, d) * 0.02)

        # ─────────────────────────────────────────────────────────────────────
        # CONDITIONING: Project T_feat to add to queries
        # ─────────────────────────────────────────────────────────────────────
        self.cond_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d)
        )

        # ─────────────────────────────────────────────────────────────────────
        # DECODER: Cross-attend to encoded geometry
        # ─────────────────────────────────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # ─────────────────────────────────────────────────────────────────────
        # OUTPUT PROJECTION
        # ─────────────────────────────────────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d)
        )

        # Confidence prediction
        self.conf_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1)
        )

        # ─────────────────────────────────────────────────────────────────────
        # CONDITIONING STATE (curriculum learning)
        # ─────────────────────────────────────────────────────────────────────
        self.cond_drop_rate = 0.1  # Start with 10% dropout (90% get hints)

    def forward(
        self,
        X_geo: torch.Tensor,
        geo_mask: Optional[torch.Tensor] = None,
        T_feat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate queries from geometry with optional text conditioning.

        Args:
            X_geo: (B, N, d) - Geometry tokens
            geo_mask: (B, N) - Valid token mask (True = valid)
            T_feat: (B, K, d) - Text features (only during training!)

        Returns:
            Q_self: (B, K, d) - Self-generated queries
            confidence: (B, K) - Query confidence
        """
        B = X_geo.shape[0]
        device = X_geo.device

        # Prepare mask for transformer (True = ignore)
        src_key_padding_mask = None
        if geo_mask is not None:
            src_key_padding_mask = ~geo_mask.bool()

        # Encode geometry
        Z_geo = self.geo_encoder(X_geo, src_key_padding_mask=src_key_padding_mask)

        # Start with base queries
        queries = self.base_queries.unsqueeze(0).expand(B, -1, -1).clone()

        # ─────────────────────────────────────────────────────────────────────
        # CONDITIONAL HINT (training only, with curriculum dropout)
        # ─────────────────────────────────────────────────────────────────────
        if self.training and T_feat is not None:
            # Per-sample dropout: some samples get hints, others don't
            keep_cond = (torch.rand(B, device=device) > self.cond_drop_rate)
            keep_cond = keep_cond.view(B, 1, 1).float()

            # Project and add T_feat as hint (DETACH to not backprop to text parser)
            cond = self.cond_proj(T_feat.detach())
            queries = queries + cond * keep_cond

        # Decode: queries cross-attend to encoded geometry
        Q = self.decoder(queries, Z_geo, memory_key_padding_mask=src_key_padding_mask)

        # Output projection
        Q = self.output_proj(Q)

        # Confidence prediction
        conf_logits = self.conf_head(Q).squeeze(-1)
        conf = torch.sigmoid(conf_logits.clamp(-5, 5))

        return Q, conf

    def set_cond_dropout(self, rate: float):
        """Set conditioning dropout rate. 0=always hint, 1=never hint."""
        self.cond_drop_rate = float(rate)


# =============================================================================
# Unified Grounding (same as v4)
# =============================================================================

class UnifiedGrounding(nn.Module):
    """Shared grounding mechanism for both text-guided and self paths."""

    def __init__(self, d: int, d_ground: int = 128):
        super().__init__()

        self.proj_query = nn.Linear(d, d_ground)

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
        Q_g = self.proj_query(queries)

        if modality == 'brep':
            X_g = self.proj_brep(X_geo)
            tau = self.log_tau_brep.exp().clamp(0.01, 1.0)
        else:
            X_g = self.proj_pc(X_geo)
            tau = self.log_tau_pc.exp().clamp(0.01, 1.0)

        scores = torch.bmm(Q_g, X_g.transpose(-2, -1))
        scores = scores / (self.d_ground ** 0.5 * tau)

        if geo_mask is not None:
            scores = scores.masked_fill(~geo_mask.bool().unsqueeze(1), -1e4)

        G = F.softmax(scores, dim=-1)
        return G


# =============================================================================
# Hierarchical Aggregator (same as v4)
# =============================================================================

class HierarchicalAggregator(nn.Module):
    """Two-level feature extraction: Global -> Detail."""

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

        # Use grounding matrix to aggregate geometry into slots
        X_slots = torch.bmm(G, X_geo)

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
        z_detail = self.detail_norm(z_detail + detail_q)
        z_detail = z_detail + self.detail_ffn(z_detail)
        z_detail = z_detail.mean(dim=1)

        # Learned fusion
        concat = torch.cat([z_global, z_detail], dim=-1)
        gate = self.fusion_gate(concat)
        z_unified = gate[:, 0:1] * z_global + gate[:, 1:2] * z_detail

        return z_global, z_detail, z_unified


# =============================================================================
# Main Model
# =============================================================================

class CLIP4CAD_GFA_v4_2(nn.Module):
    """
    CLIP4CAD-GFA v4.2: Conditional Self-Query Generation with Curriculum Learning

    Key changes from v4:
    1. Self-query generators take T_feat as optional conditioning
    2. Curriculum learning: reduce conditioning over training (90% -> 0%)
    3. BRep gets deeper encoder (4 layers) since it needs more help
    4. Distribution matching loss (in loss function)
    """

    def __init__(self, config: GFAv4_2Config):
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
        # Text Feature Parser
        # ─────────────────────────────────────────────────────────────────────
        self.text_parser = TextFeatureParser(
            d, K, config.num_parser_layers, config.num_heads, config.dropout
        )

        # ─────────────────────────────────────────────────────────────────────
        # CONDITIONAL SELF QUERY GENERATORS (NEW in v4.2)
        # BRep: Deeper (needs more capacity to learn geometry->text mapping)
        # PC: Lighter (ShapeLLM already multimodal)
        # ─────────────────────────────────────────────────────────────────────
        self.brep_self_gen = ConditionalSelfQueryGenerator(
            d=d,
            K=K,
            num_encoder_layers=config.brep_encoder_layers,
            num_decoder_layers=config.brep_decoder_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

        self.pc_self_gen = ConditionalSelfQueryGenerator(
            d=d,
            K=K,
            num_encoder_layers=config.pc_encoder_layers,
            num_decoder_layers=config.pc_decoder_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

        # ─────────────────────────────────────────────────────────────────────
        # SHARED: Grounding and Aggregation
        # ─────────────────────────────────────────────────────────────────────
        self.grounding = UnifiedGrounding(d, config.d_ground)
        self.hierarchical_agg = HierarchicalAggregator(
            d, config.num_detail_queries, config.num_heads, config.dropout
        )

        # ─────────────────────────────────────────────────────────────────────
        # Projection Heads
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

        # Temperature
        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.tau_init)))

    def encode_text(
        self,
        H_text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Parse text into feature slots."""
        X_text = self.text_proj(H_text)
        T_feat, confidence = self.text_parser(X_text, text_mask)

        # Global text embedding
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
        """Shared encoding for both guided and self paths."""
        G = self.grounding.compute_grounding(queries, X_geo, modality, geo_mask)
        z_global, z_detail, z_unified = self.hierarchical_agg(X_geo, G, confidence, geo_mask)

        return {
            'z_unified': z_unified,
            'z_global': z_global,
            'z_detail': z_detail,
            'G': G
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Full forward pass with both encoding paths."""
        device = next(self.parameters()).device

        # ─────────────────────────────────────────────────────────────────────
        # Extract inputs
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
        # TEXT PATH
        # ─────────────────────────────────────────────────────────────────────
        text_out = self.encode_text(H_text)
        z_text = self.proj_head(text_out['z_text'])

        # ─────────────────────────────────────────────────────────────────────
        # GUIDED PATH (same as v4)
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
        # SELF PATH (with conditioning! - T_feat passed for curriculum learning)
        # ─────────────────────────────────────────────────────────────────────
        Q_brep_self, conf_brep = self.brep_self_gen(
            X_brep, brep_mask, T_feat=text_out['T_feat']
        )
        Q_pc_self, conf_pc = self.pc_self_gen(
            X_pc, None, T_feat=text_out['T_feat']
        )

        brep_self = self.encode_geometry(
            X_brep,
            queries=Q_brep_self,
            confidence=conf_brep,
            modality='brep',
            geo_mask=brep_mask
        )
        pc_self = self.encode_geometry(
            X_pc,
            queries=Q_pc_self,
            confidence=conf_pc,
            modality='pc',
            geo_mask=None
        )

        z_brep_self = self.proj_head(brep_self['z_unified'])
        z_pc_self = self.proj_head(pc_self['z_unified'])

        return {
            # Primary embeddings (guided)
            'z_brep': z_brep_guided,
            'z_pc': z_pc_guided,
            'z_text': z_text,

            # Self embeddings
            'z_brep_self': z_brep_self,
            'z_pc_self': z_pc_self,

            # Detail embeddings
            'z_brep_detail': z_brep_detail,
            'z_pc_detail': z_pc_detail,

            # Grounding matrices
            'G_brep_guided': brep_guided['G'],
            'G_pc_guided': pc_guided['G'],
            'G_brep_self': brep_self['G'],
            'G_pc_self': pc_self['G'],

            # Queries (for distillation)
            'T_feat': text_out['T_feat'],
            'Q_brep_self': Q_brep_self,
            'Q_pc_self': Q_pc_self,

            # Confidence
            'confidence': text_out['confidence'],
            'conf_brep_self': conf_brep,
            'conf_pc_self': conf_pc,

            # Temperature
            'tau': self.log_tau.exp().clamp(0.01, 1.0),
        }

    def set_cond_dropout(self, rate: float):
        """Set conditioning dropout rate for both modalities."""
        self.brep_self_gen.set_cond_dropout(rate)
        self.pc_self_gen.set_cond_dropout(rate)

    def encode_for_retrieval(
        self,
        batch: Dict[str, torch.Tensor],
        use_self: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Encode geometry for retrieval (inference mode)."""
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
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Curriculum Schedule Helper
# =============================================================================

def get_cond_dropout(epoch: int, stage: int) -> float:
    """
    Curriculum schedule for conditioning dropout.

    Stage 1 (epochs 1-15):
      - Epoch 1-3:   0.1 (90% samples get hints) - learn what T_feat looks like
      - Epoch 4-7:   0.3 (70% samples get hints) - start learning independence
      - Epoch 8-11:  0.5 (50% samples get hints) - balanced
      - Epoch 12-15: 0.7 (30% samples get hints) - mostly independent

    Stage 2 (epochs 16+):
      - Always 1.0 (0% hints) - fully independent
    """
    if stage == 2:
        return 1.0  # No hints in stage 2

    if epoch <= 3:
        return 0.1
    elif epoch <= 7:
        return 0.3
    elif epoch <= 11:
        return 0.5
    else:
        return 0.7
