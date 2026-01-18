"""
Hierarchical Compression Module

Key innovation: Two-level compression for multimodal alignment
- Global Structure Compression (GSC): Captures coarse structure (align with titles)
- Adaptive Detail Mining (ADM): Captures fine features (align with descriptions)

Based on HCC-CAD architecture concepts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer with pre-norm and FFN.
    Queries attend to key-value tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [B, Q, D] query tokens
            key_value: [B, K, D] key-value tokens
            key_padding_mask: [B, K] True = IGNORE this position
            return_attention: Whether to return attention weights

        Returns:
            output: [B, Q, D]
            attn_weights: [B, Q, K] if return_attention else None
        """
        # Pre-norm
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)

        # Attention
        attn_out, attn_weights = self.attn(
            query=q,
            key=kv,
            value=key_value,  # Use unnormalized values
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=True,  # Average over heads
        )

        # Residual
        query = query + attn_out

        # FFN with residual
        query = query + self.ffn(self.norm_ffn(query))

        return query, attn_weights


class GlobalStructureCompression(nn.Module):
    """
    Global Structure Compression (GSC)

    Uses learnable global queries to capture coarse structural information
    from the input token sequence via cross-attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_queries: int = 8,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_queries = n_queries

        # Learnable global queries
        self.global_queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.global_pos = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Self-attention among global queries
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compress tokens into global features.

        Args:
            tokens: [B, M, D] input token sequence
            mask: [B, M] validity mask (1=valid, 0=padding)

        Returns:
            global_features: [B, n_queries, D]
            attention_weights: List of [B, n_queries, M] attention weights per layer
        """
        B = tokens.shape[0]

        # Create key_padding_mask (True = IGNORE)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()

        # Initialize queries
        queries = self.global_queries.unsqueeze(0).expand(B, -1, -1) + self.global_pos

        # Collect attention weights for coverage computation
        all_attn_weights = []

        # Cross-attention to input tokens
        for layer in self.cross_attn_layers:
            queries, attn_w = layer(
                query=queries,
                key_value=tokens,
                key_padding_mask=key_padding_mask,
                return_attention=True,
            )
            if attn_w is not None:
                all_attn_weights.append(attn_w)

        # Self-attention among queries
        queries = self.self_attn(queries)

        # Normalize
        global_features = self.output_norm(queries)

        return global_features, all_attn_weights


class AdaptiveDetailMining(nn.Module):
    """
    Adaptive Detail Mining (ADM)

    Identifies important but under-attended regions through coverage analysis,
    then compresses selected tokens using detail queries.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_queries: int = 8,
        k_select: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_queries = n_queries
        self.k_select = k_select

        # Importance predictor
        self.importance_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Detail queries
        self.detail_queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.detail_pos = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

        # Cross-attention layers for detail compression
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

    def compute_coverage(
        self,
        attention_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute how well each input token is covered by global queries.

        Uses max attention across all layers and queries.

        Args:
            attention_weights: List of [B, n_global, M] attention weights

        Returns:
            coverage: [B, M] coverage scores in [0, 1]
        """
        # Stack: [n_layers, B, n_global, M]
        attn_stack = torch.stack(attention_weights, dim=0)

        # Max over queries: [n_layers, B, M]
        max_over_queries = attn_stack.max(dim=2)[0]

        # Max over layers: [B, M]
        coverage = max_over_queries.max(dim=0)[0]

        return coverage

    def forward(
        self,
        tokens: torch.Tensor,
        attention_weights: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine important details that were under-attended by global queries.

        Args:
            tokens: [B, M, D] input token sequence
            attention_weights: Attention weights from GSC
            mask: [B, M] validity mask (1=valid, 0=padding)

        Returns:
            detail_features: [B, n_queries, D]
            importance: [B, M] importance scores
            selected_indices: [B, K] indices of selected tokens
        """
        B, M, D = tokens.shape
        device = tokens.device

        # Compute coverage
        coverage = self.compute_coverage(attention_weights)  # [B, M]

        # Predict importance
        importance = self.importance_mlp(tokens).squeeze(-1)  # [B, M]

        # Complementary score: important but not well-covered
        complementary = importance * (1.0 - coverage)

        # Mask out padding tokens
        if mask is not None:
            complementary = complementary.masked_fill(~mask.bool(), float("-inf"))

        # Select top-K tokens
        K = min(self.k_select, M)
        _, selected_indices = complementary.topk(K, dim=1)  # [B, K]

        # Gather selected tokens
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
        selected_tokens = tokens[batch_indices, selected_indices]  # [B, K, D]

        # Initialize detail queries
        queries = self.detail_queries.unsqueeze(0).expand(B, -1, -1) + self.detail_pos

        # Cross-attention to selected tokens (no mask needed - all selected are valid)
        for layer in self.cross_attn_layers:
            queries, _ = layer(
                query=queries,
                key_value=selected_tokens,
                key_padding_mask=None,
                return_attention=False,
            )

        # Normalize
        detail_features = self.output_norm(queries)

        return detail_features, importance, selected_indices


class HierarchicalCompressionModule(nn.Module):
    """
    Hierarchical Compression Module

    Combines Global Structure Compression (GSC) and Adaptive Detail Mining (ADM)
    to produce multi-level features for hierarchical alignment.

    Output:
    - Global features: [B, n_global, D] - for title alignment
    - Detail features: [B, n_detail, D] - for description alignment
    - Unified representation: [B, n_global + n_detail, D]
    """

    def __init__(
        self,
        d_model: int = 256,
        n_global_queries: int = 8,
        n_detail_queries: int = 8,
        k_detail_select: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            n_global_queries: Number of global queries
            n_detail_queries: Number of detail queries
            k_detail_select: Number of tokens to select for detail mining
            n_heads: Number of attention heads
            n_layers: Number of cross-attention layers per module
            dropout: Dropout rate
        """
        super().__init__()

        self.n_global_queries = n_global_queries
        self.n_detail_queries = n_detail_queries

        # Global Structure Compression
        self.gsc = GlobalStructureCompression(
            d_model=d_model,
            n_queries=n_global_queries,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Adaptive Detail Mining
        self.adm = AdaptiveDetailMining(
            d_model=d_model,
            n_queries=n_detail_queries,
            k_select=k_detail_select,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compress input tokens hierarchically.

        Args:
            tokens: [B, M, D] input token sequence
            mask: [B, M] validity mask (1=valid, 0=padding)

        Returns:
            Dictionary containing:
                - global_features: [B, n_global, D]
                - detail_features: [B, n_detail, D]
                - unified: [B, n_global + n_detail, D]
                - z_global: [B, D] mean-pooled global embedding
                - coverage: [B, M] attention coverage
                - importance: [B, M] importance scores
                - detail_indices: [B, K] selected token indices
        """
        # Global Structure Compression
        global_features, attn_weights = self.gsc(tokens, mask)

        # Adaptive Detail Mining
        detail_features, importance, detail_indices = self.adm(
            tokens, attn_weights, mask
        )

        # Compute coverage for analysis
        coverage = self.adm.compute_coverage(attn_weights)

        # Unified representation
        unified = torch.cat([global_features, detail_features], dim=1)

        # Global embedding (mean pool)
        z_global = global_features.mean(dim=1)

        return {
            "global_features": global_features,
            "detail_features": detail_features,
            "unified": unified,
            "z_global": z_global,
            "coverage": coverage,
            "importance": importance,
            "detail_indices": detail_indices,
        }

    def get_global_embedding(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get only the global embedding (for inference)."""
        global_features, _ = self.gsc(tokens, mask)
        return global_features.mean(dim=1)
