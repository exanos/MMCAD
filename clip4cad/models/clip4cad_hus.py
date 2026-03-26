"""
CLIP4CAD-HUS: Hierarchical Unified Space for CAD Multimodal Learning

This architecture implements a hierarchical approach with:
- 2 levels: Global (G=4 queries) + Detail (M=16 queries)
- Modality-aware queries with adapters
- Global-conditioned detail attention
- Gated fusion between levels

Key differences from GFA:
- Text is an EQUAL modality (not used to create grounding matrices)
- No self_ground_queries needed (queries work without text)
- Simpler loss (3 terms vs 8)
- Direct query attention instead of text-parsed grounding slots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from omegaconf import DictConfig


class ModalityAwareQueries(nn.Module):
    """
    Shared query structure with modality-specific adaptations.

    Base queries capture semantic structure (global=shape category, detail=features).
    Modality adapters provide slight modulation for each modality.
    """

    def __init__(self, d: int, num_global: int, num_detail: int):
        """
        Args:
            d: Query dimension
            num_global: Number of global queries (G)
            num_detail: Number of detail queries (M)
        """
        super().__init__()

        # Base queries (shared semantic structure)
        self.global_queries = nn.Parameter(torch.randn(1, num_global, d) * 0.02)
        self.detail_queries = nn.Parameter(torch.randn(1, num_detail, d) * 0.02)

        # Modality embeddings
        self.modality_embed = nn.Embedding(3, d)  # brep=0, pc=1, text=2

        # Modality-specific adapters (lightweight)
        # Tanh ensures adaptation is bounded (doesn't dominate base)
        self.adapter = nn.ModuleDict({
            'brep': nn.Sequential(nn.Linear(d, d), nn.Tanh()),
            'pc': nn.Sequential(nn.Linear(d, d), nn.Tanh()),
            'text': nn.Sequential(nn.Linear(d, d), nn.Tanh())
        })

        self.mod_idx = {'brep': 0, 'pc': 1, 'text': 2}

    def get_queries(self, modality: str, batch_size: int, device: torch.device):
        """
        Get modality-adapted queries.

        Args:
            modality: 'brep', 'pc', or 'text'
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            global_q: (B, G, d)
            detail_q: (B, M, d)
        """
        idx = torch.tensor([self.mod_idx[modality]], device=device)
        mod_embed = self.modality_embed(idx)  # (1, d)

        global_q = self.global_queries.expand(batch_size, -1, -1)
        detail_q = self.detail_queries.expand(batch_size, -1, -1)

        # Adapt: base + modality_specific_offset
        # Scale factor 0.1 ensures adaptation is subtle
        adapt = self.adapter[modality](mod_embed).unsqueeze(1)  # (1, 1, d)
        global_q = global_q + 0.1 * adapt
        detail_q = detail_q + 0.1 * adapt

        return global_q, detail_q


class GlobalConditionedDetailAttention(nn.Module):
    """
    Detail attention conditioned on global context.

    Key innovation: Global tells detail WHERE to look.
    "I know this is a gear (global) -> focus on teeth (detail)"
    """

    def __init__(self, d: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads

        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

        # Conditioning: global context modulates queries via additive bias
        self.condition_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            queries: Detail queries (B, M, d)
            keys, values: Input tokens (B, N, d)
            condition: Global tokens (B, G, d) - guides attention
            mask: Valid positions (B, N), True=valid, False=pad

        Returns:
            output: (B, M, d)
            attn_weights: (B, num_heads, M, N)
        """
        B, M, _ = queries.shape
        N = keys.shape[1]

        # Pool condition and compute modulation
        cond_pooled = condition.mean(dim=1)  # (B, d)
        modulation = self.condition_proj(cond_pooled)  # (B, d)

        # Apply modulation to queries (additive bias)
        q = self.W_q(queries) + modulation.unsqueeze(1)  # (B, M, d)
        k = self.W_k(keys)   # (B, N, d)
        v = self.W_v(values) # (B, N, d)

        # Reshape for multi-head attention
        q = q.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, M, d_h)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d_h)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d_h)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, M, N)

        # Apply mask if provided
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle all-masked rows
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, M, d_h)
        out = out.transpose(1, 2).contiguous().view(B, M, self.d)  # (B, M, d)
        out = self.out_proj(out)

        return out, attn


class CLIP4CAD_HUS_v2(nn.Module):
    """
    Hierarchical Unified Space v2 for CAD Multimodal Learning.

    Architecture:
    1. Input projections (modality-specific)
    2. Global queries attend to tokens -> z_global
    3. Detail queries attend to tokens (conditioned on z_global) -> z_detail
    4. Gated fusion -> z_unified

    All modalities follow the SAME pipeline, making text an equal modality.
    No grounding matrices, no self_ground_queries needed.
    """

    def __init__(self, config: DictConfig):
        """
        Args:
            config: OmegaConf configuration with model hyperparameters
        """
        super().__init__()

        # Extract dimensions from config
        d = config.get('d_unified', 256)
        G = config.get('num_global', 4)
        M = config.get('num_detail', 16)
        d_proj = config.get('d_proj', 128)
        num_heads = config.get('num_heads', 8)
        dropout = config.get('dropout', 0.1)

        # Input dimensions
        d_face = config.get('encoders', {}).get('brep', {}).get('face_dim', 48)
        d_edge = config.get('encoders', {}).get('brep', {}).get('edge_dim', 12)
        d_pc = config.get('encoders', {}).get('pointcloud', {}).get('output_dim', 1024)
        d_text = config.get('encoders', {}).get('text', {}).get('hidden_dim', 3072)

        self.d = d
        self.d_proj = d_proj
        self.num_global = G
        self.num_detail = M

        # =====================================================================
        # INPUT PROJECTIONS (Modality-Specific)
        # =====================================================================

        # B-Rep: faces and edges
        self.proj_face = nn.Sequential(nn.Linear(d_face, d), nn.LayerNorm(d))
        self.proj_edge = nn.Sequential(nn.Linear(d_edge, d), nn.LayerNorm(d))
        self.face_type_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # Point Cloud: local and global tokens
        self.proj_pc_local = nn.Sequential(nn.Linear(d_pc, d), nn.LayerNorm(d))
        self.proj_pc_global = nn.Sequential(nn.Linear(d_pc, d), nn.LayerNorm(d))

        # Text: description tokens
        self.proj_text = nn.Sequential(nn.Linear(d_text, d), nn.LayerNorm(d))

        # =====================================================================
        # MODALITY-AWARE QUERIES
        # =====================================================================

        self.query_bank = ModalityAwareQueries(d, G, M)

        # =====================================================================
        # LEVEL 1: GLOBAL EXTRACTION
        # =====================================================================

        self.global_attn = nn.MultiheadAttention(
            d, num_heads, batch_first=True, dropout=dropout
        )
        self.global_ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )
        self.global_norm1 = nn.LayerNorm(d)
        self.global_norm2 = nn.LayerNorm(d)

        # =====================================================================
        # LEVEL 2: DETAIL EXTRACTION (Conditioned on Global)
        # =====================================================================

        self.detail_extractor = GlobalConditionedDetailAttention(d, num_heads, dropout)
        self.detail_ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )
        self.detail_norm = nn.LayerNorm(d)

        # =====================================================================
        # FUSION (Gated combination of global and detail)
        # =====================================================================

        # Learn to weight global vs detail importance
        self.level_gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, 2),
            nn.Softmax(dim=-1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.LayerNorm(d)
        )

        # =====================================================================
        # PROJECTION HEADS
        # =====================================================================

        self.proj_global = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_proj)
        )
        self.proj_detail = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_proj)
        )
        self.proj_unified = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_proj)
        )

        # =====================================================================
        # LEARNABLE TEMPERATURES
        # =====================================================================

        self.log_tau_global = nn.Parameter(torch.log(torch.tensor(0.07)))
        self.log_tau_detail = nn.Parameter(torch.log(torch.tensor(0.05)))
        self.log_tau_unified = nn.Parameter(torch.log(torch.tensor(0.07)))

    def encode(
        self,
        X_tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        modality: str,
        global_hint: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified hierarchical encoding for any modality.

        Args:
            X_tokens: Input tokens (B, N, d)
            mask: Valid token mask (B, N), True=valid
            modality: 'brep', 'pc', or 'text'
            global_hint: Optional initialization hint (B, 1, d) or (B, G, d)

        Returns:
            Dict with z_global, z_detail, z_unified, attention maps, gate values
        """
        B = X_tokens.shape[0]
        device = X_tokens.device

        # Get modality-adapted queries
        global_q, detail_q = self.query_bank.get_queries(modality, B, device)

        # Add global hint if provided (e.g., from PC global token)
        if global_hint is not None:
            if global_hint.shape[1] != global_q.shape[1]:
                # Pool if dimensions don't match
                global_hint = global_hint.mean(dim=1, keepdim=True)
            global_q = global_q + 0.3 * global_hint

        # =====================================================================
        # LEVEL 1: Global Extraction
        # =====================================================================

        # Prepare mask for MultiheadAttention (expects True=pad, False=valid)
        attn_mask = ~mask if mask is not None else None

        z_global, attn_global = self.global_attn(
            global_q, X_tokens, X_tokens,
            key_padding_mask=attn_mask
        )
        z_global = self.global_norm1(z_global + global_q)  # Residual
        z_global = self.global_norm2(z_global + self.global_ffn(z_global))
        # z_global: (B, G, d)

        # =====================================================================
        # LEVEL 2: Detail Extraction (Conditioned on Global)
        # =====================================================================

        z_detail, attn_detail = self.detail_extractor(
            queries=detail_q,
            keys=X_tokens,
            values=X_tokens,
            condition=z_global,  # Global context guides detail attention
            mask=mask  # True=valid for our custom attention
        )
        z_detail = self.detail_norm(z_detail + self.detail_ffn(z_detail))
        # z_detail: (B, M, d)

        # =====================================================================
        # FUSION
        # =====================================================================

        z_global_pooled = z_global.mean(dim=1)   # (B, d)
        z_detail_pooled = z_detail.mean(dim=1)   # (B, d)

        # Learn importance weighting
        gate_input = torch.cat([z_global_pooled, z_detail_pooled], dim=-1)
        gate = self.level_gate(gate_input)  # (B, 2)

        # Weighted combination + fusion
        z_weighted = gate[:, 0:1] * z_global_pooled + gate[:, 1:2] * z_detail_pooled
        z_unified = self.fusion(gate_input) + z_weighted  # Residual

        return {
            'z_global': z_global_pooled,
            'z_detail': z_detail_pooled,
            'z_unified': z_unified,
            'z_global_tokens': z_global,
            'z_detail_tokens': z_detail,
            'attn_global': attn_global,
            'attn_detail': attn_detail,
            'gate': gate,
        }

    def encode_brep(
        self,
        face_features: torch.Tensor,
        edge_features: torch.Tensor,
        face_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode B-Rep features hierarchically.

        Args:
            face_features: (B, F, 48)
            edge_features: (B, E, 12)
            face_mask: (B, F), True=valid
            edge_mask: (B, E), True=valid
        """
        # Project and add type embeddings
        X_face = self.proj_face(face_features) + self.face_type_embed
        X_edge = self.proj_edge(edge_features) + self.edge_type_embed

        # Concatenate into single token sequence
        X_tokens = torch.cat([X_face, X_edge], dim=1)  # (B, F+E, d)

        # Combine masks
        if face_mask is not None and edge_mask is not None:
            # Convert to boolean if float
            face_mask_bool = face_mask.bool() if face_mask.dtype != torch.bool else face_mask
            edge_mask_bool = edge_mask.bool() if edge_mask.dtype != torch.bool else edge_mask
            mask = torch.cat([face_mask_bool, edge_mask_bool], dim=1)
        else:
            mask = None

        return self.encode(X_tokens, mask, 'brep')

    def encode_pc(
        self,
        local_features: torch.Tensor,
        global_token: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode point cloud features hierarchically.

        Args:
            local_features: (B, L, 1024) - local patch tokens
            global_token: (B, G_pc, 1024) - global tokens from ShapeLLM
        """
        # Project
        X_local = self.proj_pc_local(local_features)    # (B, L, d)
        X_global = self.proj_pc_global(global_token)    # (B, G_pc, d)

        # Concatenate all tokens
        X_tokens = torch.cat([X_local, X_global], dim=1)  # (B, L+G_pc, d)

        # Use global as hint for query initialization
        global_hint = X_global.mean(dim=1, keepdim=True)  # (B, 1, d)

        # No mask needed - all tokens valid
        return self.encode(X_tokens, mask=None, modality='pc', global_hint=global_hint)

    def encode_text(
        self,
        desc_embeddings: torch.Tensor,
        desc_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text description hierarchically.

        Args:
            desc_embeddings: (B, T, 3072) - LLM hidden states
            desc_mask: (B, T), True=valid token
        """
        # Project
        X_text = self.proj_text(desc_embeddings)  # (B, T, d)

        # Convert mask to boolean if needed
        if desc_mask is not None:
            mask = desc_mask.bool() if desc_mask.dtype != torch.bool else desc_mask
        else:
            mask = None

        return self.encode(X_text, mask, 'text')

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for all modalities.

        Args:
            batch: Dict with keys from GFAMappedDataset:
                - brep_face_features: (B, F, 48)
                - brep_edge_features: (B, E, 12)
                - brep_face_mask: (B, F)
                - brep_edge_mask: (B, E)
                - pc_features: (B, L+G, 1024) - concatenated local+global
                - desc_embedding: (B, T, 3072)
                - desc_mask: (B, T)

        Returns:
            Dict with all embeddings at all levels, temperatures, gates, attention maps
        """
        device = next(self.parameters()).device

        outputs = {}

        # =====================================================================
        # ENCODE B-REP
        # =====================================================================

        if 'brep_face_features' in batch:
            face_features = batch['brep_face_features'].to(device)
            edge_features = batch['brep_edge_features'].to(device)
            face_mask = batch.get('brep_face_mask')
            edge_mask = batch.get('brep_edge_mask')

            if face_mask is not None:
                face_mask = face_mask.to(device)
            if edge_mask is not None:
                edge_mask = edge_mask.to(device)

            brep_out = self.encode_brep(face_features, edge_features, face_mask, edge_mask)

            outputs['z_brep_global'] = self.proj_global(brep_out['z_global'])
            outputs['z_brep_detail'] = self.proj_detail(brep_out['z_detail'])
            outputs['z_brep'] = self.proj_unified(brep_out['z_unified'])
            outputs['gate_brep'] = brep_out['gate']
            outputs['attn_brep_global'] = brep_out['attn_global']
            outputs['attn_brep_detail'] = brep_out['attn_detail']

        # =====================================================================
        # ENCODE POINT CLOUD
        # =====================================================================

        if 'pc_features' in batch:
            pc_features = batch['pc_features'].to(device)  # (B, L+G, 1024)

            # Split into local and global (32 local + 16 global from ShapeLLM)
            local_features = pc_features[:, :32, :]   # (B, 32, 1024)
            global_token = pc_features[:, 32:, :]     # (B, 16, 1024)

            pc_out = self.encode_pc(local_features, global_token)

            outputs['z_pc_global'] = self.proj_global(pc_out['z_global'])
            outputs['z_pc_detail'] = self.proj_detail(pc_out['z_detail'])
            outputs['z_pc'] = self.proj_unified(pc_out['z_unified'])
            outputs['gate_pc'] = pc_out['gate']
            outputs['attn_pc_global'] = pc_out['attn_global']
            outputs['attn_pc_detail'] = pc_out['attn_detail']

        # =====================================================================
        # ENCODE TEXT
        # =====================================================================

        if 'desc_embedding' in batch:
            desc_embeddings = batch['desc_embedding'].to(device)
            desc_mask = batch.get('desc_mask')

            if desc_mask is not None:
                desc_mask = desc_mask.to(device)

            text_out = self.encode_text(desc_embeddings, desc_mask)

            outputs['z_text_global'] = self.proj_global(text_out['z_global'])
            outputs['z_text_detail'] = self.proj_detail(text_out['z_detail'])
            outputs['z_text'] = self.proj_unified(text_out['z_unified'])
            outputs['gate_text'] = text_out['gate']
            outputs['attn_text_global'] = text_out['attn_global']
            outputs['attn_text_detail'] = text_out['attn_detail']

        # =====================================================================
        # TEMPERATURES
        # =====================================================================

        # Clamp temperatures to prevent numerical instability
        # Min 0.02 (not 0.01) to prevent logit explosion with FP16
        outputs['tau_global'] = self.log_tau_global.exp().clamp(0.02, 1.0)
        outputs['tau_detail'] = self.log_tau_detail.exp().clamp(0.02, 1.0)
        outputs['tau_unified'] = self.log_tau_unified.exp().clamp(0.02, 1.0)

        return outputs

    def encode_geometry_inference(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode only geometry modalities (no text needed).

        This is the key advantage of HUS over GFA:
        - GFA needs self_ground_queries for text-free inference
        - HUS uses the same queries regardless of text presence

        Args:
            batch: Dict with brep and/or pc features

        Returns:
            Dict with z_brep and z_pc unified embeddings
        """
        outputs = {}
        device = next(self.parameters()).device

        if 'brep_face_features' in batch:
            face_features = batch['brep_face_features'].to(device)
            edge_features = batch['brep_edge_features'].to(device)
            face_mask = batch.get('brep_face_mask')
            edge_mask = batch.get('brep_edge_mask')

            if face_mask is not None:
                face_mask = face_mask.to(device)
            if edge_mask is not None:
                edge_mask = edge_mask.to(device)

            brep_out = self.encode_brep(face_features, edge_features, face_mask, edge_mask)
            outputs['z_brep'] = self.proj_unified(brep_out['z_unified'])

        if 'pc_features' in batch:
            pc_features = batch['pc_features'].to(device)
            local_features = pc_features[:, :32, :]
            global_token = pc_features[:, 32:, :]

            pc_out = self.encode_pc(local_features, global_token)
            outputs['z_pc'] = self.proj_unified(pc_out['z_unified'])

        return outputs

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {
            'projections': sum(
                p.numel() for name, p in self.named_parameters()
                if 'proj_' in name and 'query' not in name
            ),
            'query_bank': sum(
                p.numel() for name, p in self.named_parameters()
                if 'query_bank' in name
            ),
            'global_attn': sum(
                p.numel() for name, p in self.named_parameters()
                if 'global_attn' in name or 'global_ffn' in name or 'global_norm' in name
            ),
            'detail_attn': sum(
                p.numel() for name, p in self.named_parameters()
                if 'detail_' in name
            ),
            'fusion': sum(
                p.numel() for name, p in self.named_parameters()
                if 'fusion' in name or 'level_gate' in name
            ),
            'projection_heads': sum(
                p.numel() for name, p in self.named_parameters()
                if name.startswith('proj_global') or name.startswith('proj_detail')
                or name.startswith('proj_unified')
            ),
        }
        counts['total'] = sum(p.numel() for p in self.parameters())
        counts['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts
