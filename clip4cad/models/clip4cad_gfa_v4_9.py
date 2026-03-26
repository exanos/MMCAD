"""
CLIP4CAD GFA v4.9 - Direct Contrastive Alignment (No Codebook)

Key Design Principles:
1. NO CODEBOOK - The codebook in v4.8.x acts as an information bottleneck
   causing all samples to produce nearly identical representations
2. Attention pooling replaces codebook grounding - K learnable queries
   cross-attend to tokens, preserving instance-level information
3. Single clear objective - InfoNCE only, no competing losses
4. Separate projection heads per modality (proven in CLIP, ULIP)

Architecture:
    Encoder -> AttentionPooling -> Projection -> InfoNCE

This is exactly what CLIP, ULIP, and successful contrastive methods use.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GFAv49Config:
    """Configuration for GFA v4.9 model.

    v4.9: Direct contrastive alignment without codebook.
    Simpler architecture, fewer parameters, no information bottleneck.

    v4.9.1 Updates (capacity increase for text alignment):
    - d: 256 -> 384 (50% increase)
    - d_proj: 128 -> 256 (better discrimination)
    - num_pool_queries: 8 -> 16 (capture more aspects)
    - num_text_tf_layers: 2 -> 4 (deeper text processing)
    - num_brep_tf_layers: 4 -> 6 (match text depth)
    - Added d_text_hidden for gradual text projection
    """

    # Input dimensions
    d_face: int = 48       # AutoBrep face FSQ features
    d_edge: int = 12       # AutoBrep edge FSQ features
    d_pc: int = 1024       # ShapeLLM point cloud features
    d_text: int = 3072     # Phi-4-mini text features

    # Model dimensions
    d: int = 384           # Internal unified dimension (was 256)
    d_proj: int = 256      # Final projection output dimension (was 128)
    d_text_hidden: int = 768  # Intermediate text dimension (gradual reduction)

    # BRep encoder
    num_msg_layers: int = 3        # Topology message passing layers
    num_brep_tf_layers: int = 6    # BRep transformer layers (was 4)
    num_heads: int = 8
    dropout: float = 0.1
    max_bfs_levels: int = 32

    # Text encoder
    num_text_tf_layers: int = 4    # (was 2)

    # Attention pooling (replaces codebook)
    num_pool_queries: int = 16     # K learnable queries (was 8)

    # Contrastive temperature
    init_tau: float = 0.07


# =============================================================================
# Attention Pooling (replaces codebook) - Enhanced version
# =============================================================================

class AttentionPooling(nn.Module):
    """
    Enhanced attention pooling with multiple cross-attention layers.

    Key improvements over basic version:
    1. Multiple cross-attention layers (2 instead of 1)
    2. Self-attention between queries before pooling
    3. Weighted pooling instead of mean pooling
    """

    def __init__(self, d: int, num_queries: int = 16, num_heads: int = 8, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.d = d
        self.num_queries = num_queries
        self.num_layers = num_layers

        # Learnable query tokens (initialized more carefully)
        self.queries = nn.Parameter(torch.randn(num_queries, d) * 0.02)

        # Multiple cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # FFN after each cross-attention
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d * 4, d)
            )
            for _ in range(num_layers)
        ])

        # Norms
        self.norms1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])

        # Self-attention for query interaction
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(d)

        # Weighted pooling
        self.pool_weights = nn.Linear(d, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            X: Input tokens (B, N, d)
            mask: Padding mask (B, N) - True for valid positions

        Returns:
            Pooled representation (B, d)
        """
        B = X.shape[0]

        # Expand queries for batch
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)

        # Convert mask to key_padding_mask format (True = ignore)
        if mask is not None:
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None

        # Multiple cross-attention layers
        for i in range(self.num_layers):
            attn_out, _ = self.cross_attn_layers[i](
                query=Q,
                key=X,
                value=X,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            Q = self.norms1[i](Q + self.dropout(attn_out))
            Q = self.norms2[i](Q + self.dropout(self.ffn_layers[i](Q)))

        # Self-attention for query interaction
        self_attn_out, _ = self.self_attn(Q, Q, Q, need_weights=False)
        Q = self.self_attn_norm(Q + self.dropout(self_attn_out))

        # Weighted pooling (learn importance of each query)
        weights = torch.softmax(self.pool_weights(Q).squeeze(-1), dim=-1)  # (B, K)
        pooled = torch.einsum('bk,bkd->bd', weights, Q)  # (B, d)

        return pooled


# =============================================================================
# Edge Message Layer (from v4.8.x)
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
# Topology-Aware BRep Encoder
# =============================================================================

class TopologyBRepEncoder(nn.Module):
    """
    Topology-aware BRep encoder using EdgeMessageLayer for face-edge interaction.

    Uses:
    - face_features (48-dim from AutoBrep FSQ)
    - edge_features (12-dim from AutoBrep FSQ)
    - edge_to_faces (topology connections)
    - bfs_level (structural ordering)
    """

    def __init__(self, config: GFAv49Config):
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

        # BFS level embedding
        self.level_emb = nn.Embedding(config.max_bfs_levels, d)

        # Topology message passing
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

    def forward(
        self,
        face_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_to_faces: torch.Tensor,
        bfs_level: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            face_feats: (B, N_f, d_face)
            edge_feats: (B, N_e, d_edge)
            face_mask, edge_mask: (B, N_f), (B, N_e)
            edge_to_faces: (B, N_e, 2) - topology connections
            bfs_level: (B, N_f) - BFS ordering

        Returns:
            X: (B, N_f + N_e, d) - Encoded tokens
            mask: (B, N_f + N_e) - Combined mask
        """
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

        return X, mask


# =============================================================================
# Text Encoder (Enhanced for better text-geometry alignment)
# =============================================================================

class TextEncoder(nn.Module):
    """
    Enhanced text encoder with gradual dimension reduction.

    Key improvements:
    1. Gradual projection: 3072 -> d_text_hidden -> d (not direct 3072 -> d)
    2. Deeper transformer (4 layers instead of 2)
    3. Pre-norm architecture for stability
    """

    def __init__(self, config: GFAv49Config):
        super().__init__()
        d = config.d
        d_hidden = getattr(config, 'd_text_hidden', config.d)

        # Gradual projection: 3072 -> d_hidden -> d
        self.proj_stage1 = nn.Sequential(
            nn.Linear(config.d_text, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.proj_stage2 = nn.Sequential(
            nn.Linear(d_hidden, d),
            nn.LayerNorm(d),
        )

        # Deeper transformer
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=config.num_heads,
                dim_feedforward=d * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=config.num_text_tf_layers
        )

        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Gradual projection
        X = self.proj_stage1(X.float())
        X = self.proj_stage2(X)

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
    """Point cloud encoder (MLP projection)."""

    def __init__(self, config: GFAv49Config):
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
            pc_local: (B, N, d_pc) local tokens
            pc_global: (B, d_pc) global token

        Returns:
            X: (B, N+1, d) encoded tokens
        """
        pc_global = pc_global.unsqueeze(1)
        X = torch.cat([pc_local, pc_global], dim=1)

        X = self.proj(X.float())
        X = torch.nan_to_num(X, nan=0.0)

        return X


# =============================================================================
# Projection Head
# =============================================================================

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =============================================================================
# Main Model: CLIP4CAD_GFA_v49
# =============================================================================

class CLIP4CAD_GFA_v49(nn.Module):
    """
    GFA v4.9: Direct Contrastive Alignment (No Codebook)

    Key features:
    1. NO codebook - attention pooling instead
    2. Separate projection heads per modality
    3. Single InfoNCE objective
    4. ~10-12M parameters (vs 15M in v4.8.2)

    Training stages:
    - Stage 0: BRep ↔ PC symmetric InfoNCE (anchor BRep to PC)
    - Stage 1: 3-way InfoNCE (Text ↔ BRep ↔ PC)
    - Stage 2: (Optional) Hard negative boosting
    """

    def __init__(self, config: GFAv49Config):
        super().__init__()
        self.config = config
        d = config.d

        # ENCODERS
        self.text_encoder = TextEncoder(config)
        self.brep_encoder = TopologyBRepEncoder(config)
        self.pc_encoder = PCEncoder(config)

        # ATTENTION POOLING (replaces codebook)
        self.text_pool = AttentionPooling(d, config.num_pool_queries, num_heads=4, dropout=config.dropout)
        self.brep_pool = AttentionPooling(d, config.num_pool_queries, num_heads=4, dropout=config.dropout)
        self.pc_pool = AttentionPooling(d, config.num_pool_queries, num_heads=4, dropout=config.dropout)

        # SEPARATE PROJECTION HEADS (per modality)
        self.text_proj = ProjectionHead(d, config.d_proj, config.dropout)
        self.brep_proj = ProjectionHead(d, config.d_proj, config.dropout)
        self.pc_proj = ProjectionHead(d, config.d_proj, config.dropout)

        # Learnable temperature
        self.log_tau = nn.Parameter(torch.log(torch.tensor(config.init_tau)))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(0.01, 1.0)

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_pc_encoder(self):
        """Freeze PC encoder for Stage 0 anchoring."""
        for param in self.pc_encoder.parameters():
            param.requires_grad = False
        for param in self.pc_pool.parameters():
            param.requires_grad = False
        for param in self.pc_proj.parameters():
            param.requires_grad = False
        print("PC encoder frozen (anchor mode)")

    def unfreeze_pc_encoder(self):
        """Unfreeze PC encoder for full training."""
        for param in self.pc_encoder.parameters():
            param.requires_grad = True
        for param in self.pc_pool.parameters():
            param.requires_grad = True
        for param in self.pc_proj.parameters():
            param.requires_grad = True
        print("PC encoder unfrozen")

    def forward_stage0(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Stage 0: BRep ↔ PC symmetric InfoNCE.
        Anchor BRep encoder to pre-trained PC encoder.
        """
        device = next(self.parameters()).device

        # Encode BRep
        X_brep, brep_mask = self.brep_encoder(
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

        # Attention pooling
        z_brep_pooled = self.brep_pool(X_brep, brep_mask)
        z_pc_pooled = self.pc_pool(X_pc)  # No mask needed for PC

        # Project
        z_brep = self.brep_proj(z_brep_pooled)
        z_pc = self.pc_proj(z_pc_pooled)

        return {
            'z_brep': z_brep,
            'z_pc': z_pc,
            'z_brep_pooled': z_brep_pooled,
            'z_pc_pooled': z_pc_pooled,
            'tau': self.tau,
        }

    def forward(self, batch: Dict[str, torch.Tensor], stage: int = 1) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with 3-way contrastive.

        Args:
            batch: Input batch
            stage: Training stage (0 = BRep-PC only, 1+ = full 3-way)
        """
        if stage == 0:
            return self.forward_stage0(batch)

        device = next(self.parameters()).device

        # ENCODE TEXT
        text_features = batch['text_features'].to(device)
        text_mask = batch.get('text_mask')
        if text_mask is not None:
            text_mask = text_mask.to(device)
        X_text, text_mask = self.text_encoder(text_features, text_mask)

        # ENCODE BREP
        X_brep, brep_mask = self.brep_encoder(
            face_feats=batch['face_features'].to(device),
            edge_feats=batch['edge_features'].to(device),
            face_mask=batch['face_mask'].to(device),
            edge_mask=batch['edge_mask'].to(device),
            edge_to_faces=batch['edge_to_faces'].to(device).long(),
            bfs_level=batch['bfs_level'].to(device).long(),
        )

        # ENCODE PC
        X_pc = self.pc_encoder(
            batch['pc_local_features'].to(device),
            batch['pc_global_features'].to(device),
        )

        # ATTENTION POOLING
        z_text_pooled = self.text_pool(X_text, text_mask)
        z_brep_pooled = self.brep_pool(X_brep, brep_mask)
        z_pc_pooled = self.pc_pool(X_pc)

        # PROJECT
        z_text = self.text_proj(z_text_pooled)
        z_brep = self.brep_proj(z_brep_pooled)
        z_pc = self.pc_proj(z_pc_pooled)

        return {
            'z_text': z_text,
            'z_brep': z_brep,
            'z_pc': z_pc,
            'z_text_pooled': z_text_pooled,
            'z_brep_pooled': z_brep_pooled,
            'z_pc_pooled': z_pc_pooled,
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
        z_text_pooled = self.text_pool(X_text, text_mask)
        z_text = self.text_proj(z_text_pooled)

        return z_text

    def encode_brep(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode BRep only (for retrieval)."""
        device = next(self.parameters()).device

        X_brep, brep_mask = self.brep_encoder(
            face_feats=batch['face_features'].to(device).float(),
            edge_feats=batch['edge_features'].to(device).float(),
            face_mask=batch['face_mask'].to(device),
            edge_mask=batch['edge_mask'].to(device),
            edge_to_faces=batch['edge_to_faces'].to(device).long(),
            bfs_level=batch['bfs_level'].to(device).long(),
        )

        z_brep_pooled = self.brep_pool(X_brep, brep_mask)
        z_brep = self.brep_proj(z_brep_pooled)

        return z_brep

    def encode_pc(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode point cloud only (for retrieval)."""
        device = next(self.parameters()).device

        X_pc = self.pc_encoder(
            batch['pc_local_features'].to(device).float(),
            batch['pc_global_features'].to(device).float()
        )

        z_pc_pooled = self.pc_pool(X_pc)
        z_pc = self.pc_proj(z_pc_pooled)

        return z_pc

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
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
                    for layer in self.layers:
                        src = gradient_checkpoint(
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
                        src = gradient_checkpoint(
                            layer, src, None, src_key_padding_mask,
                            use_reentrant=False
                        )
                    if self.norm is not None:
                        src = self.norm(src)
                    return src

            self.text_encoder.encoder = CheckpointedTransformerEncoder(original_text_tf)
            print("  - Text transformer: checkpointed")

        print("Gradient checkpointing enabled")


__all__ = [
    'GFAv49Config',
    'CLIP4CAD_GFA_v49',
    'AttentionPooling',
    'TopologyBRepEncoder',
    'TextEncoder',
    'PCEncoder',
    'ProjectionHead',
]
