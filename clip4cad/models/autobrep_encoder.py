"""
AutoBrep-Style Encoder for CLIP4CAD

This module implements an encoder following AutoBrep's design philosophy but adapted
for embedding extraction (not generation). The key insight is that AutoBrep proves
transformer-only approaches work on the same FSQ latent representation.

Key design principles from AutoBrep paper:
- Unified transformer on token sequence (no message passing)
- BFS-ordered sequence with faces then edges
- Type embeddings (face vs edge)
- BFS level embedding (encodes hierarchy position)
- Spatial position from centroids

Key adaptations for embedding:
- Bidirectional attention (not causal)
- Mean pooling to fixed embedding
- Contrastive loss training

Architecture (scaled for encoding, not generation):
- 6 transformer layers (vs 16 in AutoBrep)
- 512 hidden dim (vs 2048)
- 8 attention heads (vs 32)
- Pre-norm transformer blocks
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AutoBrepEncoderConfig:
    """Configuration for AutoBrep-style encoder."""

    # Input dimensions (from FSQ latents)
    d_face: int = 48      # FSQ face latent dimension (4 tokens * 12 dim each)
    d_edge: int = 12      # FSQ edge latent dimension (2 tokens * 6 dim each)

    # Model dimensions
    d: int = 512          # Hidden dimension (smaller than AutoBrep's 2048)
    d_proj: int = 128     # Output projection dimension

    # Transformer architecture (scaled from AutoBrep)
    num_layers: int = 6   # 6 layers (vs 16 in AutoBrep - encoder needs less)
    num_heads: int = 8    # 8 heads (vs 32)
    ff_mult: int = 4      # FFN expansion factor (same as AutoBrep)
    dropout: float = 0.1  # Same as AutoBrep

    # Sequence limits
    max_faces: int = 192
    max_edges: int = 512
    max_bfs_levels: int = 16

    # Spatial encoding
    use_centroids: bool = True      # Use face centroids for spatial position
    use_bfs_levels: bool = True     # Use BFS level embeddings
    use_normals: bool = True        # Use face normals
    use_areas: bool = True          # Use face areas

    # Architecture choices
    norm_first: bool = True         # Pre-norm (like AutoBrep with RMSNorm)
    activation: str = 'gelu'        # Activation function


# =============================================================================
# Position Encoding
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (for sequence position)."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (B, T, d)"""
        return x + self.pe[:x.size(1)]


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm (following AutoBrep)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        activation: str = 'gelu'
    ):
        super().__init__()

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU() if activation == 'gelu' else nn.SiLU(),  # SiLU = Swish
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: (B, T) - True = valid, False = padding

        Returns:
            (B, T, d_model)
        """
        # Self-attention with pre-norm
        normed = self.norm1(x)

        # Create attention mask (True = ignore for nn.MultiheadAttention)
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.bool()  # Invert: True means ignore

        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=attn_mask)
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.ff(self.norm2(x))

        return x


# =============================================================================
# Masked Mean Pooling
# =============================================================================

class MaskedMeanPool(nn.Module):
    """Masked mean pooling over sequence dimension."""

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d)
            mask: (B, T) - 1 = valid, 0 = padding

        Returns:
            (B, d) - Pooled embedding
        """
        mask = mask.float().unsqueeze(-1)  # (B, T, 1)
        x_masked = x * mask
        pooled = x_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled


# =============================================================================
# AutoBrep-Style Encoder
# =============================================================================

class AutoBrepStyleEncoder(nn.Module):
    """
    AutoBrep-style BRep encoder for embedding extraction.

    Following AutoBrep's design but for embedding (not generation):
    - NO message passing (AutoBrep doesn't use it)
    - NO hierarchical codebook (not needed for encoding)
    - Single unified transformer on token sequence
    - BFS-ordered sequence with topology encoded in position

    Token sequence:
        [Face_1, Face_2, ..., Face_N, Edge_1, Edge_2, ..., Edge_M]

    Each token gets:
        - Projected FSQ features
        - Type embedding (face=0, edge=1)
        - Position encoding (sinusoidal for sequence order)
        - BFS level embedding (for faces only)
        - Spatial encoding from centroids (for faces only)
    """

    def __init__(self, config: AutoBrepEncoderConfig):
        super().__init__()
        self.config = config
        d = config.d

        # ─────────────────────────────────────────────────────────────────────
        # Token projections (like AutoBrep's embedding layer)
        # ─────────────────────────────────────────────────────────────────────
        self.face_proj = nn.Sequential(
            nn.Linear(config.d_face, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.edge_proj = nn.Sequential(
            nn.Linear(config.d_edge, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # ─────────────────────────────────────────────────────────────────────
        # Spatial encoding (replaces AutoBrep's bbox tokens C)
        # ─────────────────────────────────────────────────────────────────────
        if config.use_centroids:
            self.centroid_proj = nn.Linear(3, d)

        if config.use_normals:
            self.normal_proj = nn.Linear(3, d)

        if config.use_areas:
            self.area_proj = nn.Linear(1, d)

        # ─────────────────────────────────────────────────────────────────────
        # BFS level embedding (like AutoBrep's level structure)
        # ─────────────────────────────────────────────────────────────────────
        if config.use_bfs_levels:
            self.level_emb = nn.Embedding(config.max_bfs_levels, d)

        # ─────────────────────────────────────────────────────────────────────
        # Type embedding (face vs edge)
        # ─────────────────────────────────────────────────────────────────────
        self.type_emb = nn.Embedding(2, d)  # 0 = face, 1 = edge

        # ─────────────────────────────────────────────────────────────────────
        # Positional encoding (sequence position)
        # ─────────────────────────────────────────────────────────────────────
        self.pos_enc = SinusoidalPositionalEncoding(
            d, max_len=config.max_faces + config.max_edges
        )

        # ─────────────────────────────────────────────────────────────────────
        # Transformer encoder (following AutoBrep specs, scaled down)
        # ─────────────────────────────────────────────────────────────────────
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model=d,
                num_heads=config.num_heads,
                dropout=config.dropout,
                ff_mult=config.ff_mult,
                activation=config.activation,
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(d)

        # ─────────────────────────────────────────────────────────────────────
        # Pooling + projection
        # ─────────────────────────────────────────────────────────────────────
        self.pool = MaskedMeanPool()

        self.proj_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.d_proj),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        face_features: torch.Tensor,
        edge_features: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        face_centroids: Optional[torch.Tensor] = None,
        face_normals: Optional[torch.Tensor] = None,
        face_areas: Optional[torch.Tensor] = None,
        bfs_level: Optional[torch.Tensor] = None,
        **kwargs  # Ignore other spatial fields
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            face_features: (B, N_f, d_face) - FSQ face latents
            edge_features: (B, N_e, d_edge) - FSQ edge latents
            face_mask: (B, N_f) - 1 = valid, 0 = padding
            edge_mask: (B, N_e) - 1 = valid, 0 = padding
            face_centroids: (B, N_f, 3) - Face centroid positions
            face_normals: (B, N_f, 3) - Face normal vectors
            face_areas: (B, N_f) - Face areas
            bfs_level: (B, N_f) - BFS traversal level
            **kwargs: Ignored (edge_to_faces, etc.)

        Returns:
            Dict with:
                z_brep: (B, d_proj) - Projected embedding
                z_brep_raw: (B, d) - Raw pooled embedding
        """
        B = face_features.size(0)
        N_f = face_features.size(1)
        N_e = edge_features.size(1)
        device = face_features.device
        d = self.config.d

        # ─────────────────────────────────────────────────────────────────────
        # 1. Project features to model dimension
        # ─────────────────────────────────────────────────────────────────────
        F = self.face_proj(face_features.float())  # (B, N_f, d)
        E = self.edge_proj(edge_features.float())  # (B, N_e, d)

        # ─────────────────────────────────────────────────────────────────────
        # 2. Add spatial encoding from centroids (faces only)
        # ─────────────────────────────────────────────────────────────────────
        if self.config.use_centroids and face_centroids is not None:
            centroids = torch.nan_to_num(face_centroids.float(), nan=0.0).clamp(-1e4, 1e4)
            F = F + self.centroid_proj(centroids)

        if self.config.use_normals and face_normals is not None:
            normals = torch.nan_to_num(face_normals.float(), nan=0.0).clamp(-1.0, 1.0)
            F = F + self.normal_proj(normals)

        if self.config.use_areas and face_areas is not None:
            areas = torch.nan_to_num(face_areas.float(), nan=0.0).clamp(0, 1e6)
            F = F + self.area_proj(areas.unsqueeze(-1))

        # ─────────────────────────────────────────────────────────────────────
        # 3. Add BFS level embedding (faces only)
        # ─────────────────────────────────────────────────────────────────────
        if self.config.use_bfs_levels and bfs_level is not None:
            level = bfs_level.long().clamp(0, self.config.max_bfs_levels - 1)
            F = F + self.level_emb(level)

        # ─────────────────────────────────────────────────────────────────────
        # 4. Add type embeddings
        # ─────────────────────────────────────────────────────────────────────
        face_type = torch.zeros(B, N_f, dtype=torch.long, device=device)  # type 0 = face
        edge_type = torch.ones(B, N_e, dtype=torch.long, device=device)   # type 1 = edge

        F = F + self.type_emb(face_type)
        E = E + self.type_emb(edge_type)

        # ─────────────────────────────────────────────────────────────────────
        # 5. Build sequence (faces then edges)
        # ─────────────────────────────────────────────────────────────────────
        # AutoBrep interleaves edges with faces, but for bidirectional encoder
        # we can concatenate and let attention figure it out
        X = torch.cat([F, E], dim=1)  # (B, N_f + N_e, d)
        mask = torch.cat([face_mask.float(), edge_mask.float()], dim=1)  # (B, N_f + N_e)

        # ─────────────────────────────────────────────────────────────────────
        # 6. Add positional encoding
        # ─────────────────────────────────────────────────────────────────────
        X = self.pos_enc(X)

        # ─────────────────────────────────────────────────────────────────────
        # 7. Transformer encoding
        # ─────────────────────────────────────────────────────────────────────
        for layer in self.transformer_layers:
            X = layer(X, mask=mask)

        X = self.final_norm(X)
        X = torch.nan_to_num(X, nan=0.0)

        # ─────────────────────────────────────────────────────────────────────
        # 8. Pool to embedding
        # ─────────────────────────────────────────────────────────────────────
        z_raw = self.pool(X, mask)
        z_proj = self.proj_head(z_raw)

        return {
            'z_brep': z_proj,
            'z_brep_raw': z_raw,
        }

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Full CLIP4CAD Model with AutoBrep-style Encoder
# =============================================================================

class CLIP4CAD_AutoBrep(nn.Module):
    """
    CLIP4CAD model using AutoBrep-style BRep encoder.

    Simple contrastive model for BRep-PC alignment.
    For Stage 0 training: anchor BRep encoder to pre-trained PC encoder.
    """

    def __init__(
        self,
        config: Optional[AutoBrepEncoderConfig] = None,
        d_pc: int = 1024,  # ShapeLLM feature dimension
    ):
        super().__init__()

        if config is None:
            config = AutoBrepEncoderConfig()

        self.config = config
        d = config.d

        # BRep encoder (AutoBrep-style)
        self.brep_encoder = AutoBrepStyleEncoder(config)

        # PC encoder (simple projection from ShapeLLM features)
        self.pc_encoder = nn.Sequential(
            nn.Linear(d_pc, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # PC projection head
        self.pc_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.d_proj),
        )

        # Learnable temperature
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))

    @property
    def tau(self):
        return self.log_tau.exp().clamp(0.01, 1.0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for BRep-PC alignment.

        Args:
            batch: Dictionary with BRep and PC features

        Returns:
            Dictionary with embeddings and temperature
        """
        device = next(self.parameters()).device

        # Support both key naming conventions
        face_features = batch.get('face_features', batch.get('brep_face_features'))
        edge_features = batch.get('edge_features', batch.get('brep_edge_features'))
        face_mask = batch.get('face_mask', batch.get('brep_face_mask'))
        edge_mask = batch.get('edge_mask', batch.get('brep_edge_mask'))

        # Spatial fields (optional)
        face_centroids = batch.get('face_centroids')
        face_normals = batch.get('face_normals')
        face_areas = batch.get('face_areas')
        bfs_level = batch.get('bfs_level')

        # Encode BRep
        brep_out = self.brep_encoder(
            face_features=face_features.to(device),
            edge_features=edge_features.to(device),
            face_mask=face_mask.to(device),
            edge_mask=edge_mask.to(device),
            face_centroids=face_centroids.to(device) if face_centroids is not None else None,
            face_normals=face_normals.to(device) if face_normals is not None else None,
            face_areas=face_areas.to(device) if face_areas is not None else None,
            bfs_level=bfs_level.to(device) if bfs_level is not None else None,
        )

        z_brep = brep_out['z_brep']
        z_brep_raw = brep_out['z_brep_raw']

        # Encode PC - handle both split and combined formats
        if 'pc_local_features' in batch:
            pc_local = batch['pc_local_features'].to(device)  # (B, N, 1024)
            pc_global = batch['pc_global_features'].to(device)  # (B, 1024)
            pc_all = torch.cat([pc_local, pc_global.unsqueeze(1)], dim=1)  # (B, N+1, 1024)
        elif 'pc_features' in batch:
            # Combined format: last token is global
            pc_all = batch['pc_features'].to(device)  # (B, N+1, 1024)
        else:
            raise KeyError("Batch must contain 'pc_features' or 'pc_local_features'/'pc_global_features'")

        X_pc = self.pc_encoder(pc_all.float())  # (B, N+1, d)
        z_pc_raw = X_pc.mean(dim=1)  # (B, d)
        z_pc = self.pc_proj(z_pc_raw)  # (B, d_proj)

        return {
            'z_brep': z_brep,
            'z_pc': z_pc,
            'z_brep_raw': z_brep_raw,
            'z_pc_raw': z_pc_raw,
            'tau': self.tau,
        }

    def forward_stage0(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Alias for forward() - compatibility with staged training code."""
        return self.forward(batch)

    def encode_brep(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode BRep only (for retrieval)."""
        device = next(self.parameters()).device

        face_features = batch.get('face_features', batch.get('brep_face_features'))
        edge_features = batch.get('edge_features', batch.get('brep_edge_features'))
        face_mask = batch.get('face_mask', batch.get('brep_face_mask'))
        edge_mask = batch.get('edge_mask', batch.get('brep_edge_mask'))

        brep_out = self.brep_encoder(
            face_features=face_features.to(device),
            edge_features=edge_features.to(device),
            face_mask=face_mask.to(device),
            edge_mask=edge_mask.to(device),
            face_centroids=batch.get('face_centroids', torch.zeros_like(face_features[..., :3])).to(device),
            face_normals=batch.get('face_normals'),
            face_areas=batch.get('face_areas'),
            bfs_level=batch.get('bfs_level'),
        )

        return brep_out['z_brep']

    def encode_pc(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode point cloud only."""
        device = next(self.parameters()).device

        if 'pc_local_features' in batch:
            pc_local = batch['pc_local_features'].to(device)
            pc_global = batch['pc_global_features'].to(device)
            pc_all = torch.cat([pc_local, pc_global.unsqueeze(1)], dim=1)
        else:
            pc_all = batch['pc_features'].to(device)

        X_pc = self.pc_encoder(pc_all.float())
        z_pc_raw = X_pc.mean(dim=1)
        z_pc = self.pc_proj(z_pc_raw)

        return z_pc

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Contrastive Loss
# =============================================================================

class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss with label smoothing."""

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        stage: int = 0,  # Ignored - for compatibility
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute InfoNCE loss.

        Args:
            outputs: Model outputs with z_brep, z_pc, tau
            stage: Training stage (ignored, for compatibility)

        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        z_brep = outputs['z_brep']
        z_pc = outputs['z_pc']
        tau = outputs['tau']

        B = z_brep.shape[0]
        device = z_brep.device

        # Normalize
        z_brep = F.normalize(z_brep.float(), dim=-1)
        z_pc = F.normalize(z_pc.float(), dim=-1)

        # Similarity matrix
        sim = z_brep @ z_pc.T / tau
        labels = torch.arange(B, device=device)

        # InfoNCE both directions
        loss_b2p = F.cross_entropy(sim, labels, label_smoothing=self.label_smoothing)
        loss_p2b = F.cross_entropy(sim.T, labels, label_smoothing=self.label_smoothing)
        loss = (loss_b2p + loss_p2b) / 2

        # Metrics
        with torch.no_grad():
            z_brep_raw = outputs['z_brep_raw']
            z_pc_raw = outputs['z_pc_raw']

            # L2 gap
            gap = (z_brep_raw - z_pc_raw).pow(2).sum(-1).sqrt().mean()

            # True-pair cosine
            cos = F.cosine_similarity(z_brep_raw, z_pc_raw, dim=-1).mean()

            # R@1
            pred = sim.argmax(dim=1)
            r_at_1 = (pred == labels).float().mean()

        return loss, {
            'total': loss,
            'contrastive': loss,
            'gap': gap,
            'cosine': cos,
            'r_at_1': r_at_1,
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing AutoBrep-style encoder...")
    print("=" * 60)

    # Create config and model
    config = AutoBrepEncoderConfig(
        d=512,
        num_layers=6,
        num_heads=8,
    )

    encoder = AutoBrepStyleEncoder(config)
    print(f"Encoder parameters: {encoder.count_parameters():,}")

    # Test encoder
    B = 4
    batch = {
        'face_features': torch.randn(B, 192, 48),
        'edge_features': torch.randn(B, 512, 12),
        'face_mask': torch.ones(B, 192),
        'edge_mask': torch.ones(B, 512),
        'face_centroids': torch.randn(B, 192, 3),
        'face_normals': torch.randn(B, 192, 3),
        'face_areas': torch.rand(B, 192),
        'bfs_level': torch.randint(0, 16, (B, 192)),
    }

    # Set some padding
    batch['face_mask'][:, 50:] = 0
    batch['edge_mask'][:, 200:] = 0

    outputs = encoder(**batch)
    print(f"z_brep_raw: {outputs['z_brep_raw'].shape}")
    print(f"z_brep: {outputs['z_brep'].shape}")

    # Test full model
    print("\n" + "=" * 60)
    model = CLIP4CAD_AutoBrep(config)
    print(f"Full model parameters: {model.count_parameters():,}")

    batch['pc_local_features'] = torch.randn(B, 48, 1024)
    batch['pc_global_features'] = torch.randn(B, 1024)

    outputs = model(batch)
    print(f"z_brep: {outputs['z_brep'].shape}")
    print(f"z_pc: {outputs['z_pc'].shape}")

    # Test loss
    print("\n" + "=" * 60)
    criterion = ContrastiveLoss()
    loss, metrics = criterion(outputs)
    print(f"Loss: {loss.item():.4f}")
    print(f"Gap: {metrics['gap'].item():.4f}")
    print(f"Cosine: {metrics['cosine'].item():.4f}")
    print(f"R@1: {metrics['r_at_1'].item()*100:.1f}%")

    # Test backward
    loss.backward()
    print("\nBackward OK")

    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"Total gradient norm: {total_grad:.4f}")
