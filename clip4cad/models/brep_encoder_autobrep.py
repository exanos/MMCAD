"""
AutoBrep-style BRep Encoder for CLIP4CAD

This encoder follows AutoBrep's design philosophy:
1. Topology is encoded through SEQUENCE ORDER (faces sorted by XYZ centroid)
2. NO explicit edge-to-face connectivity required (broken in our dataset)
3. Transformer attention learns relationships between faces/edges
4. Simple and effective - works with pre-computed FSQ latents

Key differences from v4.8.1's TopologyBRepEncoder:
- Does NOT use edge_to_faces (which is 100% -1 / broken)
- Does NOT use bfs_level (which is 100% 0 / broken)
- DOES sort faces by centroid XYZ order (like AutoBrep's BFS ordering)
- DOES use transformer self-attention to learn topology implicitly
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AutoBrepEncoderConfig:
    """Configuration for AutoBrep-style BRep encoder."""
    # Input dimensions (from FSQ latents)
    d_face: int = 48  # FSQ face latent dimension
    d_edge: int = 12  # FSQ edge latent dimension

    # Model dimensions
    d_model: int = 256  # Transformer hidden dimension
    d_proj: int = 128   # Output projection dimension

    # Transformer architecture
    num_layers: int = 2      # Number of transformer layers (reduced for memory)
    num_heads: int = 8       # Number of attention heads
    dropout: float = 0.1     # Dropout rate
    ff_mult: int = 2         # FFN expansion factor (reduced for memory)

    # Sequence handling
    max_faces: int = 192     # Max faces per sample (for positional encoding)
    max_edges: int = 512     # Max edges per sample

    # Centroid-based sorting
    use_centroid_sorting: bool = True  # Sort faces by XYZ centroid order

    # Memory efficiency
    use_separate_encoders: bool = True  # Encode faces and edges separately (saves memory)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (B, T, d)"""
        return x + self.pe[:x.size(1)]


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: (B, T) - True = valid, False = padding
        """
        # Self-attention with pre-norm
        normed = self.norm1(x)

        # Create attention mask (True = ignore)
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Invert: True means ignore

        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=attn_mask)
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.ff(self.norm2(x))
        return x


class AutoBrepEncoder(nn.Module):
    """
    AutoBrep-style BRep encoder using centroid-ordered sequences and transformer.

    This encoder:
    1. Projects face and edge features to hidden dimension
    2. Sorts faces by XYZ centroid order (like AutoBrep's BFS ordering)
    3. Creates a sequence: [face_1, face_2, ..., face_N, edge_1, ..., edge_M]
    4. Applies transformer self-attention to learn topology relationships
    5. Pools to get global BRep embedding

    Key: Does NOT use broken edge_to_faces or bfs_level fields!
    """

    def __init__(self, config: AutoBrepEncoderConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # Face and edge projections
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

        # Type embeddings (distinguish faces from edges)
        self.type_embedding = nn.Embedding(2, d)  # 0 = face, 1 = edge

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d, config.max_faces + config.max_edges)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d, config.num_heads, config.dropout, config.ff_mult)
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(d)

        # Output projection head
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

    def _sort_faces_by_centroid(
        self,
        face_feats: torch.Tensor,
        face_mask: torch.Tensor,
        face_centroids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sort faces by XYZ centroid order (like AutoBrep's BFS ordering).

        Args:
            face_feats: (B, N_f, d_face)
            face_mask: (B, N_f)
            face_centroids: (B, N_f, 3)

        Returns:
            Sorted versions of (face_feats, face_mask, sort_indices)
        """
        B, N_f = face_feats.shape[:2]
        device = face_feats.device

        # Create large penalty for invalid faces (push to end)
        large_val = 1e6
        centroids = face_centroids.clone()
        invalid_mask = ~face_mask.bool()
        centroids[invalid_mask] = large_val

        # Sort by (x, y, z) lexicographically
        # PyTorch doesn't have lexsort, so we create a composite key
        # Scale factors to ensure proper ordering
        x, y, z = centroids[..., 0], centroids[..., 1], centroids[..., 2]

        # Normalize to [0, 1] range for valid faces
        valid_min = centroids[face_mask.bool()].min() if face_mask.any() else 0
        valid_max = centroids[face_mask.bool()].max() if face_mask.any() else 1
        scale = valid_max - valid_min + 1e-6

        # Composite sort key: x * 1e6 + y * 1e3 + z
        sort_key = (x - valid_min) / scale * 1e6 + (y - valid_min) / scale * 1e3 + (z - valid_min) / scale

        # Sort indices
        sort_indices = sort_key.argsort(dim=1)

        # Gather sorted tensors
        sorted_face_feats = torch.gather(
            face_feats, 1,
            sort_indices.unsqueeze(-1).expand(-1, -1, face_feats.size(-1))
        )
        sorted_face_mask = torch.gather(face_mask, 1, sort_indices)

        return sorted_face_feats, sorted_face_mask, sort_indices

    def forward(
        self,
        face_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        face_centroids: Optional[torch.Tensor] = None,
        **kwargs  # Ignore unused topology args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            face_feats: (B, N_f, d_face) - FSQ face latents
            edge_feats: (B, N_e, d_edge) - FSQ edge latents
            face_mask: (B, N_f) - 1 = valid, 0 = padding
            edge_mask: (B, N_e) - 1 = valid, 0 = padding
            face_centroids: (B, N_f, 3) - Face centroid positions (optional)
            **kwargs: Ignored (edge_to_faces, bfs_level, etc.)

        Returns:
            z_raw: (B, d_model) - Raw pooled embedding (before projection)
            z_proj: (B, d_proj) - Projected embedding
        """
        B = face_feats.size(0)
        device = face_feats.device
        d = self.config.d_model

        # Convert to float
        face_feats = face_feats.float()
        edge_feats = edge_feats.float()
        face_mask = face_mask.float()
        edge_mask = edge_mask.float()

        # Optional: Sort faces by centroid XYZ order
        if self.config.use_centroid_sorting and face_centroids is not None:
            face_feats, face_mask, _ = self._sort_faces_by_centroid(
                face_feats, face_mask, face_centroids
            )

        # Project to model dimension
        F_proj = self.face_proj(face_feats)  # (B, N_f, d)
        E_proj = self.edge_proj(edge_feats)  # (B, N_e, d)

        # Add type embeddings
        face_type = torch.zeros(B, F_proj.size(1), dtype=torch.long, device=device)
        edge_type = torch.ones(B, E_proj.size(1), dtype=torch.long, device=device)
        F_proj = F_proj + self.type_embedding(face_type)
        E_proj = E_proj + self.type_embedding(edge_type)

        # Concatenate faces and edges: [face_1, ..., face_N, edge_1, ..., edge_M]
        X = torch.cat([F_proj, E_proj], dim=1)  # (B, N_f + N_e, d)
        seq_mask = torch.cat([face_mask, edge_mask], dim=1)  # (B, N_f + N_e)

        # Add positional encoding
        X = self.pos_enc(X)

        # Apply transformer layers
        for layer in self.transformer_layers:
            X = layer(X, mask=seq_mask.bool())

        # Final norm
        X = self.final_norm(X)

        # Masked mean pooling
        mask_expanded = seq_mask.unsqueeze(-1)  # (B, T, 1)
        z_raw = (X * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        # Project
        z_proj = self.proj_head(z_raw)

        return z_raw, z_proj

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# === Simple contrastive model for testing ===

class SimpleCLIP4CAD_AutoBrep(nn.Module):
    """
    Simple CLIP4CAD model using AutoBrep-style BRep encoder.

    For testing/debugging purposes. Just BRep-PC alignment, no text.
    """

    def __init__(self, config: Optional[AutoBrepEncoderConfig] = None):
        super().__init__()
        if config is None:
            config = AutoBrepEncoderConfig()
        self.config = config

        # BRep encoder (AutoBrep-style)
        self.brep_encoder = AutoBrepEncoder(config)

        # PC encoder (simple projection)
        self.pc_encoder = nn.Sequential(
            nn.Linear(1024, config.d_model),  # ShapeLLM features are 1024d
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )

        # PC projection head
        self.pc_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_proj),
        )

        # Learnable temperature
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))

    @property
    def tau(self):
        return self.log_tau.exp().clamp(0.01, 1.0)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for BRep-PC alignment."""
        device = next(self.parameters()).device

        # Support both key naming conventions
        face_feats = batch.get('face_features', batch.get('brep_face_features'))
        edge_feats = batch.get('edge_features', batch.get('brep_edge_features'))
        face_mask = batch.get('face_mask', batch.get('brep_face_mask'))
        edge_mask = batch.get('edge_mask', batch.get('brep_edge_mask'))
        face_centroids = batch.get('face_centroids')

        # Encode BRep
        z_brep_raw, z_brep = self.brep_encoder(
            face_feats=face_feats.to(device),
            edge_feats=edge_feats.to(device),
            face_mask=face_mask.to(device),
            edge_mask=edge_mask.to(device),
            face_centroids=face_centroids.to(device) if face_centroids is not None else torch.zeros_like(face_feats[..., :3]).to(device),
        )

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

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class SimpleContrastiveLoss(nn.Module):
    """Simple InfoNCE contrastive loss."""

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        stage: int = 0,  # Ignored - for compatibility with staged training
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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


# === Test code ===
if __name__ == "__main__":
    print("Testing AutoBrep-style BRep encoder...")

    config = AutoBrepEncoderConfig()
    encoder = AutoBrepEncoder(config)
    print(f"Encoder parameters: {encoder.count_parameters():,}")

    # Dummy batch
    B = 4
    batch = {
        'face_features': torch.randn(B, 192, 48),
        'edge_features': torch.randn(B, 512, 12),
        'face_mask': torch.ones(B, 192),
        'edge_mask': torch.ones(B, 512),
        'face_centroids': torch.randn(B, 192, 3),
    }

    # Set some padding
    batch['face_mask'][:, 50:] = 0
    batch['edge_mask'][:, 200:] = 0

    # Forward
    z_raw, z_proj = encoder(
        batch['face_features'],
        batch['edge_features'],
        batch['face_mask'],
        batch['edge_mask'],
        batch['face_centroids'],
    )
    print(f"z_raw: {z_raw.shape}")
    print(f"z_proj: {z_proj.shape}")

    # Test full model
    model = SimpleCLIP4CAD_AutoBrep(config)
    print(f"\nFull model parameters: {model.count_parameters():,}")

    batch['pc_local_features'] = torch.randn(B, 48, 1024)
    batch['pc_global_features'] = torch.randn(B, 1024)

    outputs = model(batch)
    print(f"z_brep: {outputs['z_brep'].shape}")
    print(f"z_pc: {outputs['z_pc'].shape}")

    # Test loss
    criterion = SimpleContrastiveLoss()
    loss, losses = criterion(outputs)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Gap: {losses['gap'].item():.4f}")
    print(f"Cosine: {losses['cosine'].item():.4f}")
    print(f"R@1: {losses['r_at_1'].item()*100:.1f}%")

    # Test backward
    loss.backward()
    print("\nBackward OK")

    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"Total gradient norm: {total_grad:.4f}")
