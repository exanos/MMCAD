"""
CLIP4CAD GFA v4.8.1 SIMPLE - Diagnostic Version

This is a SIMPLIFIED version to diagnose training issues.
Removes complexity to isolate where the problem is.

Key simplifications:
1. No topology message passing (just MLP projections)
2. No hierarchical codebook
3. Simple pooling
4. Near-identity initialization

If this trains, the issue is in the complex architecture.
If this doesn't train, the issue is in the data.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GFAv481SimpleConfig:
    """Simplified configuration."""
    d_face: int = 48
    d_edge: int = 12
    d_pc: int = 1024
    d_text: int = 3072
    d: int = 256
    d_proj: int = 128
    dropout: float = 0.1


class SimpleBRepEncoder(nn.Module):
    """
    Simplified BRep encoder - just MLPs, no topology.
    """
    def __init__(self, config: GFAv481SimpleConfig):
        super().__init__()
        d = config.d

        # Simple face projection
        self.face_proj = nn.Sequential(
            nn.Linear(config.d_face, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        # Simple edge projection
        self.edge_proj = nn.Sequential(
            nn.Linear(config.d_edge, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        # Initialize near-identity for first layer
        self._init_weights()

    def _init_weights(self):
        # Small random init - don't destroy input structure
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        face_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        **kwargs  # Ignore topology args
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple forward: project and pool.

        Returns:
            z_face: (B, d) - Pooled face features
            z_edge: (B, d) - Pooled edge features
        """
        # Project
        F = self.face_proj(face_feats.float())  # (B, N_f, d)
        E = self.edge_proj(edge_feats.float())  # (B, N_e, d)

        # Masked mean pooling
        face_mask_f = face_mask.float().unsqueeze(-1)
        z_face = (F * face_mask_f).sum(1) / face_mask_f.sum(1).clamp(min=1)

        edge_mask_f = edge_mask.float().unsqueeze(-1)
        z_edge = (E * edge_mask_f).sum(1) / edge_mask_f.sum(1).clamp(min=1)

        return z_face, z_edge


class SimplePCEncoder(nn.Module):
    """Simplified PC encoder."""
    def __init__(self, config: GFAv481SimpleConfig):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(config.d_pc, config.d),
            nn.LayerNorm(config.d),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pc_local: torch.Tensor, pc_global: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pc_local: (B, N, 1024)
            pc_global: (B, 1024)
        Returns:
            z_pc: (B, d)
        """
        # Combine local and global
        pc_all = torch.cat([pc_local, pc_global.unsqueeze(1)], dim=1)
        X = self.proj(pc_all.float())
        return X.mean(dim=1)


class SimpleTextEncoder(nn.Module):
    """Simplified text encoder."""
    def __init__(self, config: GFAv481SimpleConfig):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(config.d_text, config.d),
            nn.LayerNorm(config.d),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            X: (B, T, 3072)
            mask: (B, T)
        Returns:
            z_text: (B, d)
        """
        X = self.proj(X.float())

        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            return (X * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        return X.mean(dim=1)


class CLIP4CAD_GFA_v481_Simple(nn.Module):
    """
    SIMPLIFIED GFA v4.8.1 for diagnostics.

    If this trains but the full model doesn't, the issue is architecture complexity.
    If this doesn't train, the issue is in the data.
    """

    def __init__(self, config: GFAv481SimpleConfig):
        super().__init__()
        self.config = config
        d = config.d

        # Simple encoders
        self.brep_encoder = SimpleBRepEncoder(config)
        self.pc_encoder = SimplePCEncoder(config)
        self.text_encoder = SimpleTextEncoder(config)

        # Combination layer
        self.brep_combine = nn.Sequential(
            nn.Linear(d * 2, d),  # face + edge
            nn.LayerNorm(d),
            nn.GELU(),
        )

        # Output projection
        self.proj_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.d_proj)
        )

        # Temperature
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))

        self._init_weights()

    def _init_weights(self):
        for m in [self.brep_combine, self.proj_head]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    @property
    def tau(self):
        return self.log_tau.exp().clamp(0.01, 1.0)

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full forward pass."""
        device = next(self.parameters()).device

        # Encode BRep
        z_face, z_edge = self.brep_encoder(
            face_feats=batch['face_features'].to(device),
            edge_feats=batch['edge_features'].to(device),
            face_mask=batch['face_mask'].to(device),
            edge_mask=batch['edge_mask'].to(device),
        )
        z_brep_raw = self.brep_combine(torch.cat([z_face, z_edge], dim=-1))

        # Encode PC
        z_pc_raw = self.pc_encoder(
            batch['pc_local_features'].to(device),
            batch['pc_global_features'].to(device),
        )

        # Encode Text
        text_mask = batch.get('text_mask')
        if text_mask is not None:
            text_mask = text_mask.to(device)
        z_text_raw = self.text_encoder(
            batch['text_features'].to(device),
            text_mask
        )

        # Project to output
        z_brep = self.proj_head(z_brep_raw)
        z_pc = self.proj_head(z_pc_raw)
        z_text = self.proj_head(z_text_raw)

        return {
            'z_brep': z_brep,
            'z_pc': z_pc,
            'z_text': z_text,
            'z_brep_raw': z_brep_raw,
            'z_pc_raw': z_pc_raw,
            'z_text_raw': z_text_raw,
            'tau': self.tau,
        }

    def forward_brep_pc(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """BRep-PC only forward (like Stage 0)."""
        device = next(self.parameters()).device

        # Encode BRep
        z_face, z_edge = self.brep_encoder(
            face_feats=batch['face_features'].to(device),
            edge_feats=batch['edge_features'].to(device),
            face_mask=batch['face_mask'].to(device),
            edge_mask=batch['edge_mask'].to(device),
        )
        z_brep_raw = self.brep_combine(torch.cat([z_face, z_edge], dim=-1))

        # Encode PC
        z_pc_raw = self.pc_encoder(
            batch['pc_local_features'].to(device),
            batch['pc_global_features'].to(device),
        )

        # Project
        z_brep = self.proj_head(z_brep_raw)
        z_pc = self.proj_head(z_pc_raw)

        return {
            'z_brep': z_brep,
            'z_pc': z_pc,
            'z_brep_raw': z_brep_raw,
            'z_pc_raw': z_pc_raw,
            'tau': self.tau,
        }


class SimpleLoss(nn.Module):
    """Simple contrastive loss for diagnostics."""

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z_brep = outputs['z_brep']
        z_pc = outputs['z_pc']
        tau = outputs['tau']

        B = z_brep.shape[0]
        device = z_brep.device

        # Normalize
        z_brep = F.normalize(z_brep.float(), dim=-1)
        z_pc = F.normalize(z_pc.float(), dim=-1)

        # Compute similarity matrix
        sim = z_brep @ z_pc.T / tau
        labels = torch.arange(B, device=device)

        # InfoNCE in both directions
        loss_b2p = F.cross_entropy(sim, labels, label_smoothing=self.label_smoothing)
        loss_p2b = F.cross_entropy(sim.T, labels, label_smoothing=self.label_smoothing)
        loss = (loss_b2p + loss_p2b) / 2

        # Metrics
        with torch.no_grad():
            z_brep_raw = outputs['z_brep_raw']
            z_pc_raw = outputs['z_pc_raw']

            # Gap (L2 distance)
            gap = (z_brep_raw - z_pc_raw).pow(2).sum(-1).sqrt().mean()

            # True-pair cosine
            cos = F.cosine_similarity(z_brep_raw, z_pc_raw, dim=-1).mean()

            # Retrieval R@1
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
    print("Testing simplified model...")

    config = GFAv481SimpleConfig()
    model = CLIP4CAD_GFA_v481_Simple(config)
    print(f"Parameters: {model.count_parameters():,}")

    # Dummy batch
    B = 4
    batch = {
        'face_features': torch.randn(B, 192, 48),
        'edge_features': torch.randn(B, 512, 12),
        'face_mask': torch.ones(B, 192),
        'edge_mask': torch.ones(B, 512),
        'pc_local_features': torch.randn(B, 48, 1024),
        'pc_global_features': torch.randn(B, 1024),
        'text_features': torch.randn(B, 256, 3072),
        'text_mask': torch.ones(B, 256),
    }

    # Forward
    outputs = model(batch)
    print("Forward pass OK")

    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    # Loss
    criterion = SimpleLoss()
    loss, losses = criterion(outputs)
    print(f"\nLoss: {loss.item():.4f}")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")

    # Backward
    loss.backward()
    print("\nBackward pass OK")

    # Check gradients
    total_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.norm().item()
    print(f"Total gradient norm: {total_grad:.4f}")
