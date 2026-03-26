"""
CLIP4CAD-GFA v4.4: Topology-Aware Multimodal Alignment

Key innovations from v4:
1. Explicit topology via edge_to_faces message passing
2. BFS-level hierarchical pattern aggregation
3. Multi-source query generation (tokens + levels + global)
4. Modality-specific query generators (heavy for BRep, light for PC)

This architecture addresses the core problem: recognizing semantic features
(teeth, bores, fillets) that were hidden in the bag-of-tokens representation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology_encoder import (
    TopologyAwareBRepEncoder,
    HierarchicalPatternAggregator,
    SemanticQueryGenerator
)


@dataclass
class GFAv44Config:
    """Configuration for CLIP4CAD-GFA v4.4."""
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

    # Topology encoder (Stage 1)
    num_msg_layers: int = 3             # Message passing layers
    num_brep_tf_layers: int = 4         # Transformer layers
    max_bfs_levels: int = 32            # Max BFS level for embedding

    # Pattern aggregator (Stage 2)
    num_pattern_levels: int = 10        # BFS levels to aggregate

    # Query generator (Stage 3)
    num_query_layers: int = 4           # Query decoder layers

    # PC encoder (lighter - ShapeLLM already multimodal)
    num_pc_query_layers: int = 2

    # Text parser
    num_parser_layers: int = 2

    # General
    dropout: float = 0.1

    # Temperature initialization
    tau_init: float = 0.07

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "GFAv44Config":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Input Projections
# =============================================================================

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
        X_local = self.proj_local(Z_local)  # (B, N, d)
        X_global = self.proj_global(Z_global)  # (B, d)
        # Add sequence dimension to global token for concatenation
        X_global = X_global.unsqueeze(1)  # (B, 1, d)
        return torch.cat([X_local, X_global], dim=1)  # (B, N+1, d)


# =============================================================================
# Text Feature Parser
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
# Unified Grounding (SHARED between text-guided and self paths)
# =============================================================================

class UnifiedGrounding(nn.Module):
    """
    SHARED grounding mechanism for both text-guided and self paths.

    KEY: Same query projection for T_feat (text) and Q_self (generated).
    This forces both paths to use the same grounding space.
    """

    def __init__(self, d: int, d_ground: int = 128):
        super().__init__()

        # SHARED query projection
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

        # Learnable temperatures
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
            queries: (B, K, d) - Either T_feat or Q_self
            X_geo: (B, N, d) - Geometry tokens
            modality: 'brep' or 'pc'
            geo_mask: (B, N) - Valid token mask

        Returns:
            G: (B, K, N) - Grounding matrix (soft attention)
        """
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

        scores = scores.clamp(-50, 50)
        G = F.softmax(scores, dim=-1)
        return torch.nan_to_num(G, nan=0.0)


# =============================================================================
# Hierarchical Aggregator
# =============================================================================

class HierarchicalAggregator(nn.Module):
    """Two-level feature extraction: Global + Detail."""

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = X_geo.shape[0]

        # Use grounding matrix to aggregate geometry into slots
        X_slots = torch.bmm(G, X_geo)  # (B, K, d)
        X_slots = torch.nan_to_num(X_slots, nan=0.0)  # Protect against NaN in X_geo

        # Confidence-weighted global embedding
        conf_weights = confidence.unsqueeze(-1)
        conf_sum = conf_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        conf_norm = conf_weights / conf_sum

        z_global = (X_slots * conf_norm).sum(dim=1)
        z_global = torch.nan_to_num(z_global, nan=0.0)
        z_global = self.global_norm(z_global)
        z_global = z_global + self.global_ffn(z_global)

        # Detail extraction conditioned on global
        detail_q = self.detail_queries.expand(B, -1, -1)
        global_cond = self.global_to_detail(z_global).unsqueeze(1)
        detail_q = detail_q + global_cond

        z_detail, _ = self.detail_attn(detail_q, X_slots, X_slots)
        z_detail = torch.nan_to_num(z_detail, nan=0.0)
        z_detail = self.detail_norm(z_detail + detail_q)
        z_detail = z_detail + self.detail_ffn(z_detail)
        z_detail = z_detail.mean(dim=1)

        # Learned fusion
        concat = torch.cat([z_global, z_detail], dim=-1)
        gate = self.fusion_gate(concat)
        z_unified = gate[:, 0:1] * z_global + gate[:, 1:2] * z_detail
        z_unified = torch.nan_to_num(z_unified, nan=0.0)

        return z_global, z_detail, z_unified, X_slots, gate


# =============================================================================
# Main Model: CLIP4CAD_GFA_v44
# =============================================================================

class CLIP4CAD_GFA_v44(nn.Module):
    """
    GFA v4.4: Topology-Aware Multimodal Alignment

    Key innovations:
    1. Explicit topology via edge_to_faces message passing
    2. BFS-level hierarchical pattern aggregation
    3. Multi-source query generation (tokens + levels + global)
    4. Modality-specific query generators (heavy for BRep, light for PC)
    """

    def __init__(self, config: GFAv44Config):
        super().__init__()
        self.config = config
        d = config.d_unified
        K = config.num_slots

        # ─────────────────────────────────────────────────────────────────────
        # TEXT PATH
        # ─────────────────────────────────────────────────────────────────────
        self.text_proj = nn.Sequential(
            nn.Linear(config.d_text, d),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout)
        )
        self.text_parser = TextFeatureParser(d, K, config.num_parser_layers)

        # ─────────────────────────────────────────────────────────────────────
        # BREP PATH (Topology-Aware) - NEW!
        # ─────────────────────────────────────────────────────────────────────
        self.brep_encoder = TopologyAwareBRepEncoder(
            d=d,
            d_face=config.d_face,
            d_edge=config.d_edge,
            num_msg_layers=config.num_msg_layers,
            num_tf_layers=config.num_brep_tf_layers,
            num_heads=config.num_heads,
            max_levels=config.max_bfs_levels,
            dropout=config.dropout
        )

        self.pattern_agg = HierarchicalPatternAggregator(
            d=d,
            max_levels=config.num_pattern_levels,
            num_heads=config.num_heads
        )

        self.brep_query_gen = SemanticQueryGenerator(
            d=d,
            K=K,
            num_layers=config.num_query_layers,
            num_heads=config.num_heads,
            max_levels=config.num_pattern_levels,
            dropout=config.dropout
        )

        # ─────────────────────────────────────────────────────────────────────
        # PC PATH (Lighter - ShapeLLM already multimodal)
        # ─────────────────────────────────────────────────────────────────────
        self.pc_proj = PCProjection(config.d_pc, d, config.dropout)

        self.pc_queries = nn.Parameter(torch.randn(K, d) * 0.02)
        self.pc_query_gen = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d, config.num_heads, d*4, config.dropout, 'gelu', batch_first=True
            ),
            num_layers=config.num_pc_query_layers
        )
        self.pc_cond_proj = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.pc_conf_head = nn.Sequential(
            nn.Linear(d, d//4), nn.GELU(), nn.Linear(d//4, 1)
        )
        self.pc_cond_drop_rate = 0.3

        # ─────────────────────────────────────────────────────────────────────
        # SHARED: Grounding and Aggregation
        # ─────────────────────────────────────────────────────────────────────
        self.grounding = UnifiedGrounding(d, d_ground=config.d_ground)
        self.hier_agg = HierarchicalAggregator(d, num_detail_queries=config.num_detail_queries)

        # Projection head
        self.proj_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.d_proj)
        )

        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.tau_init)))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = self.device

        # ─────────────────────────────────────────────────────────────────────
        # UNPACK BATCH
        # ─────────────────────────────────────────────────────────────────────
        face_feats = batch['face_features'].to(device)
        edge_feats = batch['edge_features'].to(device)
        face_mask = batch['face_mask'].to(device)
        edge_mask = batch['edge_mask'].to(device)

        # Topology fields (required for v4.4)
        # Cast indices to long for gather/scatter operations
        edge_to_faces = batch['edge_to_faces'].to(device).long()
        face_centroids = batch['face_centroids'].to(device)
        bfs_level = batch['bfs_level'].to(device).long()

        # Optional spatial fields (use zeros if not available)
        B, N_f = face_feats.shape[:2]
        N_e = edge_feats.shape[1]

        if 'face_normals' in batch:
            face_normals = batch['face_normals'].to(device)
        else:
            face_normals = torch.zeros(B, N_f, 3, device=device)

        if 'face_areas' in batch:
            face_areas = batch['face_areas'].to(device)
        else:
            face_areas = torch.zeros(B, N_f, device=device)

        if 'edge_midpoints' in batch:
            edge_midpoints = batch['edge_midpoints'].to(device)
        else:
            edge_midpoints = torch.zeros(B, N_e, 3, device=device)

        if 'edge_lengths' in batch:
            edge_lengths = batch['edge_lengths'].to(device)
        else:
            edge_lengths = torch.zeros(B, N_e, device=device)

        pc_local = batch['pc_local_features'].to(device)
        pc_global = batch['pc_global_features'].to(device)
        H_text = batch['text_features'].to(device)
        text_mask = batch.get('text_mask')
        if text_mask is not None:
            text_mask = text_mask.to(device)

        # ─────────────────────────────────────────────────────────────────────
        # TEXT ENCODING
        # ─────────────────────────────────────────────────────────────────────
        text_out = self._encode_text(H_text, text_mask)
        z_text = self.proj_head(text_out['z_text'])

        # ─────────────────────────────────────────────────────────────────────
        # BREP ENCODING (Topology-Aware)
        # ─────────────────────────────────────────────────────────────────────
        X_brep, face_tokens = self.brep_encoder(
            face_feats, edge_feats, face_mask, edge_mask,
            edge_to_faces, face_centroids, face_normals, face_areas,
            bfs_level, edge_midpoints, edge_lengths
        )
        brep_mask = torch.cat([face_mask, edge_mask], dim=1).bool()

        # Hierarchical pattern aggregation
        level_summaries, global_summary = self.pattern_agg(
            face_tokens, bfs_level, face_mask
        )

        # ─────────────────────────────────────────────────────────────────────
        # PC ENCODING
        # ─────────────────────────────────────────────────────────────────────
        X_pc = self.pc_proj(pc_local, pc_global)

        # ─────────────────────────────────────────────────────────────────────
        # GUIDED PATH
        # ─────────────────────────────────────────────────────────────────────
        brep_guided = self._encode_geometry(
            X_brep, text_out['T_feat'], text_out['confidence'], 'brep', brep_mask
        )
        pc_guided = self._encode_geometry(
            X_pc, text_out['T_feat'], text_out['confidence'], 'pc', None
        )

        z_brep_guided = self.proj_head(brep_guided['z_unified'])
        z_pc_guided = self.proj_head(pc_guided['z_unified'])
        z_brep_detail = self.proj_head(brep_guided['z_detail'])
        z_pc_detail = self.proj_head(pc_guided['z_detail'])

        # ─────────────────────────────────────────────────────────────────────
        # SELF PATH: Generate queries from geometry
        # ─────────────────────────────────────────────────────────────────────

        # BRep: Multi-source query generation (key innovation!)
        Q_brep_self, conf_brep = self.brep_query_gen(
            X_brep, level_summaries, global_summary, brep_mask,
            T_feat=text_out['T_feat']  # For curriculum conditioning
        )

        # PC: Simpler query generation
        Q_pc_self, conf_pc = self._generate_pc_queries(X_pc, text_out['T_feat'])

        # Encode with self queries
        brep_self = self._encode_geometry(
            X_brep, Q_brep_self, conf_brep, 'brep', brep_mask
        )
        pc_self = self._encode_geometry(
            X_pc, Q_pc_self, conf_pc, 'pc', None
        )

        z_brep_self = self.proj_head(brep_self['z_unified'])
        z_pc_self = self.proj_head(pc_self['z_unified'])

        # ─────────────────────────────────────────────────────────────────────
        # OUTPUTS
        # ─────────────────────────────────────────────────────────────────────
        return {
            # Primary embeddings
            'z_brep': z_brep_guided,
            'z_pc': z_pc_guided,
            'z_text': z_text,

            # Self embeddings
            'z_brep_self': z_brep_self,
            'z_pc_self': z_pc_self,

            # Detail embeddings (for hard negative loss)
            'z_brep_detail': z_brep_detail,
            'z_pc_detail': z_pc_detail,

            # Queries (for query losses)
            'T_feat': text_out['T_feat'],
            'Q_brep_self': Q_brep_self,
            'Q_pc_self': Q_pc_self,

            # Confidence
            'confidence': text_out['confidence'],
            'conf_brep_self': conf_brep,
            'conf_pc_self': conf_pc,

            # Grounding matrices (for grounding loss)
            'G_brep_guided': brep_guided['G'],
            'G_pc_guided': pc_guided['G'],
            'G_brep_self': brep_self['G'],
            'G_pc_self': pc_self['G'],

            # Hierarchical features (for analysis)
            'level_summaries': level_summaries,
            'global_summary': global_summary,

            'tau': self.log_tau.exp().clamp(0.01, 1.0),
        }

    def _encode_text(self, H_text: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> Dict:
        X_text = self.text_proj(H_text)
        T_feat, confidence = self.text_parser(X_text, text_mask)
        z_text = (T_feat * confidence.unsqueeze(-1)).sum(dim=1)
        z_text = z_text / (confidence.sum(dim=1, keepdim=True) + 1e-8)
        return {'T_feat': T_feat, 'confidence': confidence, 'z_text': z_text}

    def _encode_geometry(
        self,
        X_geo: torch.Tensor,
        queries: torch.Tensor,
        confidence: torch.Tensor,
        modality: str,
        geo_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        G = self.grounding.compute_grounding(queries, X_geo, modality, geo_mask)
        z_global, z_detail, z_unified, X_slots, gate = self.hier_agg(X_geo, G, confidence, geo_mask)
        return {'z_unified': z_unified, 'z_detail': z_detail, 'G': G, 'z_global': z_global}

    def _generate_pc_queries(
        self,
        X_pc: torch.Tensor,
        T_feat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = X_pc.shape[0]
        queries = self.pc_queries.unsqueeze(0).expand(B, -1, -1).clone()

        if self.training and T_feat is not None:
            keep = (torch.rand(B, device=queries.device) > self.pc_cond_drop_rate)
            keep = keep.view(B, 1, 1).float()
            cond = self.pc_cond_proj(T_feat.detach()) * 0.3
            queries = queries + cond * keep

        Q = self.pc_query_gen(queries, X_pc)
        conf = torch.sigmoid(self.pc_conf_head(Q).squeeze(-1).clamp(-5, 5))
        return Q, conf

    def set_cond_dropout(self, brep_rate: float, pc_rate: Optional[float] = None):
        """Set conditioning dropout for curriculum learning."""
        self.brep_query_gen.set_cond_dropout(brep_rate)
        if pc_rate is not None:
            self.pc_cond_drop_rate = pc_rate

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
