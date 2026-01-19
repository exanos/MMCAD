"""
CLIP4CAD-GFA: Cross-Modal Representation Learning for CAD Models
via Grounded Feature Alignment

This architecture introduces:
1. Adaptive text feature parsing with confidence-weighted feature slots
2. Bi-directional grounding module with explicit grounding matrices
3. Cross-modal alignment layers for grounding consistency
4. Semantic importance weighting based on text grounding

Key differences from CLIP4CAD-H:
- Grounding matrices provide interpretable text-geometry correspondence
- Cross-modal alignment handles encoder representation differences
- Adaptive confidence weighting handles variable description complexity
- No hierarchical (title/description) split - unified text processing
"""

import math
from typing import Dict, Optional, Any, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer with pre-norm architecture.

    Follows the pre-norm transformer design (Xiong et al., 2020) where
    layer normalization is applied before attention and FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_expansion: int = 4
    ):
        super().__init__()

        # Pre-norm layers
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, Q, d)
            key_value: (B, K, d)
            key_padding_mask: (B, K), True = valid position

        Returns:
            output: (B, Q, d)
            attn_weights: (B, Q, K)
        """
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)

        # Convert mask: our convention is True=valid, PyTorch MHA expects True=ignore
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask

        attn_out, attn_weights = self.attn(
            q_norm, kv_norm, kv_norm,
            key_padding_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )

        x = query + attn_out
        x = x + self.ffn(self.norm_ffn(x))

        return x, attn_weights


class CrossAttentionBlock(nn.Module):
    """Stack of cross-attention layers for text feature parsing."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = query
        all_attn_weights = []

        for layer in self.layers:
            x, attn_weights = layer(x, key_value, key_padding_mask)
            all_attn_weights.append(attn_weights)

        return x, all_attn_weights


class AlignmentNetwork(nn.Module):
    """
    Modality-specific alignment network for cross-modal consistency.

    Different encoders (AutoBrep for B-Rep, Point-BERT for point cloud) produce
    representations with different structural biases. This network learns to
    project modality-specific features into a shared comparison space.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnifiedProjectionGFA(nn.Module):
    """
    Projects tokens from different encoders to unified dimension.
    Adapted for GFA architecture with separate face/edge handling.
    """

    def __init__(
        self,
        d_unified: int = 256,
        d_brep_face: int = 48,
        d_brep_edge: int = 12,
        d_pointbert: int = 768,
        d_text: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_unified = d_unified

        # B-Rep projections
        self.face_proj = nn.Sequential(
            nn.Linear(d_brep_face, d_unified),
            nn.LayerNorm(d_unified),
            nn.Dropout(dropout)
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(d_brep_edge, d_unified),
            nn.LayerNorm(d_unified),
            nn.Dropout(dropout)
        )

        # Point cloud projection
        self.pc_proj = nn.Sequential(
            nn.Linear(d_pointbert, d_unified),
            nn.LayerNorm(d_unified),
            nn.Dropout(dropout)
        )

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_unified),
            nn.LayerNorm(d_unified),
            nn.Dropout(dropout)
        )

        # Learnable type embeddings for B-Rep
        self.face_type_embed = nn.Parameter(torch.zeros(1, 1, d_unified))
        self.edge_type_embed = nn.Parameter(torch.zeros(1, 1, d_unified))

        # Initialize
        nn.init.trunc_normal_(self.face_type_embed, std=0.02)
        nn.init.trunc_normal_(self.edge_type_embed, std=0.02)

    def project_brep(
        self,
        face_features: torch.Tensor,
        edge_features: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project B-Rep features to unified space with type embeddings.

        Args:
            face_features: [B, F, face_dim]
            edge_features: [B, E, edge_dim]
            face_mask: [B, F]
            edge_mask: [B, E]

        Returns:
            tokens: [B, F+E, d_unified]
            mask: [B, F+E]
        """
        face_proj = self.face_proj(face_features) + self.face_type_embed
        edge_proj = self.edge_proj(edge_features) + self.edge_type_embed

        tokens = torch.cat([face_proj, edge_proj], dim=1)
        mask = torch.cat([face_mask, edge_mask], dim=1)

        # Zero out padding
        tokens = tokens * mask.unsqueeze(-1)

        return tokens, mask

    def project_pointcloud(self, pc_features: torch.Tensor) -> torch.Tensor:
        """Project point cloud features to unified space."""
        return self.pc_proj(pc_features)

    def project_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """Project text features to unified space."""
        return self.text_proj(text_features)


class CLIP4CAD_GFA(nn.Module):
    """
    CLIP4CAD with Grounded Feature Alignment.

    This model learns a unified embedding space for CAD models by:
    1. Parsing text descriptions into distinct feature mention embeddings
    2. Learning explicit grounding matrices mapping text to geometry
    3. Enforcing grounding consistency across B-Rep and point cloud modalities
    4. Using confidence-weighted aggregation for adaptive feature utilization
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # Core dimensions
        d_unified = config.d_unified
        d_proj = config.d_proj
        d_ground = config.get("d_ground", 128)
        d_align = config.get("d_align", 128)

        K = config.get("num_feature_slots", 12)
        self.num_feature_slots = K

        # Encoder dimensions (from pre-computed features)
        d_brep_face = config.encoders.brep.face_dim
        d_brep_edge = config.encoders.brep.edge_dim
        d_pc = config.encoders.pointcloud.output_dim
        d_text = config.encoders.text.hidden_dim

        self.d_unified = d_unified
        self.d_proj = d_proj
        self.d_ground = d_ground
        self.d_align = d_align

        # Max sequence lengths
        self.max_brep_tokens = config.get("max_brep_tokens", 192)  # 64 faces + 128 edges
        self.max_pc_tokens = config.encoders.pointcloud.get("num_tokens", 513)
        self.max_text_tokens = config.get("max_text_tokens", 512)

        # Confidence threshold for active slots
        self.confidence_threshold = config.get("confidence_threshold", 0.3)

        # =====================================================================
        # Unified Projection Layer
        # =====================================================================

        self.projection = UnifiedProjectionGFA(
            d_unified=d_unified,
            d_brep_face=d_brep_face,
            d_brep_edge=d_brep_edge,
            d_pointbert=d_pc,
            d_text=d_text,
            dropout=config.get("dropout", 0.1),
        )

        # =====================================================================
        # Adaptive Text Feature Parsing Module
        # =====================================================================

        # Feature slot queries and positional encodings
        self.feature_queries = nn.Parameter(torch.randn(K, d_unified) * 0.02)
        self.feature_pos_enc = nn.Parameter(torch.randn(K, d_unified) * 0.02)

        # Cross-attention block for text parsing
        self.text_parser = CrossAttentionBlock(
            d_model=d_unified,
            num_heads=config.get("num_attention_heads", 10),
            num_layers=config.get("num_parser_layers", 2),
            dropout=config.get("dropout", 0.1),
        )

        # Confidence predictor for each feature slot
        self.confidence_predictor = nn.Sequential(
            nn.Linear(d_unified, d_unified // 4),
            nn.GELU(),
            nn.Linear(d_unified // 4, 1),
            nn.Sigmoid()
        )

        # =====================================================================
        # Bi-directional Grounding Module
        # =====================================================================

        # Grounding space projections
        self.ground_text = nn.Linear(d_unified, d_ground)
        self.ground_geo = nn.Linear(d_unified, d_ground)  # Shared for both modalities

        # Learnable grounding temperature
        self.log_tau_ground = nn.Parameter(
            torch.log(torch.tensor(config.get("tau_ground_init", 0.1)))
        )

        # =====================================================================
        # Cross-Modal Alignment Module
        # =====================================================================

        self.align_brep = AlignmentNetwork(
            d_in=d_unified,
            d_hidden=d_unified,
            d_out=d_align,
            dropout=config.get("dropout", 0.1),
        )

        self.align_pc = AlignmentNetwork(
            d_in=d_unified,
            d_hidden=d_unified,
            d_out=d_align,
            dropout=config.get("dropout", 0.1),
        )

        # =====================================================================
        # Fusion Layer
        # =====================================================================

        self.fusion_norm = nn.LayerNorm(d_unified)

        # =====================================================================
        # Global Projection Heads
        # =====================================================================

        self.global_proj_head = nn.Sequential(
            nn.Linear(d_unified, d_unified),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(d_unified, d_proj)
        )

        self.local_proj_head = nn.Sequential(
            nn.Linear(d_unified, d_unified),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(d_unified, d_proj)
        )

        # =====================================================================
        # Self-Grounding Queries (for inference without text)
        # =====================================================================

        self.self_ground_queries = nn.Parameter(torch.randn(K, d_unified) * 0.02)

        # =====================================================================
        # Learnable Contrastive Temperature
        # =====================================================================

        self.log_tau_contrastive = nn.Parameter(
            torch.log(torch.tensor(config.get("tau_contrastive_init", 0.07)))
        )

    # =========================================================================
    # Property Accessors for Temperatures
    # =========================================================================

    @property
    def tau_ground(self) -> torch.Tensor:
        """Grounding temperature, clamped to valid range."""
        return self.log_tau_ground.exp().clamp(0.01, 1.0)

    @property
    def tau_contrastive(self) -> torch.Tensor:
        """Contrastive temperature, clamped to valid range."""
        return self.log_tau_contrastive.exp().clamp(0.01, 1.0)

    # =========================================================================
    # Core Forward Methods
    # =========================================================================

    def parse_text_features(
        self,
        X_text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Parse text into K feature mention embeddings with confidence scores.

        Args:
            X_text: Projected text tokens, (B, L, d)
            text_mask: Boolean mask, (B, L), True = valid token

        Returns:
            T_feat: Feature mention embeddings, (B, K, d)
            confidence: Confidence scores, (B, K)
            attn_weights: Attention weights from parser layers
        """
        B = X_text.shape[0]
        K = self.num_feature_slots

        # Expand queries for batch processing
        queries = self.feature_queries + self.feature_pos_enc  # (K, d)
        queries = queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)

        # Cross-attend to text tokens
        T_feat, attn_weights = self.text_parser(
            queries, X_text, key_padding_mask=text_mask
        )

        # Predict confidence for each slot
        confidence = self.confidence_predictor(T_feat).squeeze(-1)  # (B, K)

        return T_feat, confidence, attn_weights

    def compute_grounding_matrix(
        self,
        T_feat: torch.Tensor,
        X_geo: torch.Tensor,
        geo_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute grounding matrix between text features and geometric tokens.

        Args:
            T_feat: Feature mention embeddings, (B, K, d)
            X_geo: Geometric tokens, (B, N, d)
            geo_mask: Boolean mask, (B, N), True = valid token

        Returns:
            G: Grounding matrix, (B, K, N)
        """
        # Project to grounding space
        T_g = self.ground_text(T_feat)  # (B, K, d_ground)
        X_g = self.ground_geo(X_geo)  # (B, N, d_ground)

        # Compute scaled dot-product attention scores
        d_g = self.d_ground
        scores = torch.bmm(T_g, X_g.transpose(-2, -1))  # (B, K, N)
        scores = scores / math.sqrt(d_g)
        scores = scores / self.tau_ground

        # Mask invalid geometric tokens before softmax
        if geo_mask is not None:
            mask_expanded = geo_mask.unsqueeze(1)  # (B, 1, N)
            scores = scores.masked_fill(~mask_expanded.bool(), float('-inf'))

        # Softmax over geometric tokens
        G = F.softmax(scores, dim=-1)
        G = torch.nan_to_num(G, nan=0.0)

        return G

    def extract_grounded_features(
        self,
        G: torch.Tensor,
        X_geo: torch.Tensor,
        T_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract grounded features using the grounding matrix.

        Args:
            G: Grounding matrix, (B, K, N)
            X_geo: Geometric tokens, (B, N, d)
            T_feat: Feature mention embeddings, (B, K, d)

        Returns:
            F_geo: Geometry-only grounded features, (B, K, d)
            F_fused: Fused multimodal features, (B, K, d)
        """
        # Weighted aggregation of geometric tokens
        F_geo = torch.bmm(G, X_geo)  # (B, K, d)

        # Fuse with text features
        F_fused = self.fusion_norm(F_geo + T_feat)

        return F_geo, F_fused

    def compute_aligned_features(
        self,
        F_brep_geo: torch.Tensor,
        F_pc_geo: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project grounded features through modality-specific alignment networks.

        Args:
            F_brep_geo: B-Rep grounded features, (B, K, d)
            F_pc_geo: Point cloud grounded features, (B, K, d)

        Returns:
            F_brep_aligned: Aligned B-Rep features, (B, K, d_align)
            F_pc_aligned: Aligned point cloud features, (B, K, d_align)
        """
        B, K, d = F_brep_geo.shape

        F_brep_flat = F_brep_geo.view(B * K, d)
        F_pc_flat = F_pc_geo.view(B * K, d)

        F_brep_aligned = self.align_brep(F_brep_flat)
        F_pc_aligned = self.align_pc(F_pc_flat)

        F_brep_aligned = F_brep_aligned.view(B, K, self.d_align)
        F_pc_aligned = F_pc_aligned.view(B, K, self.d_align)

        return F_brep_aligned, F_pc_aligned

    def compute_semantic_importance(
        self,
        G: torch.Tensor,
        confidence: torch.Tensor,
        geo_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute semantic importance of geometric tokens for global aggregation.

        Unlike HCC-CAD's attention-coverage criterion, we use semantic grounding
        weighted by slot confidence.

        Args:
            G: Grounding matrix, (B, K, N)
            confidence: Slot confidence scores, (B, K)
            geo_mask: Boolean mask, (B, N)

        Returns:
            importance: Normalized importance scores, (B, N)
        """
        # Weight grounding by confidence
        weighted_G = G * confidence.unsqueeze(-1)  # (B, K, N)

        # Sum over slots to get per-token importance
        importance = weighted_G.sum(dim=1)  # (B, N)

        # Apply geometric mask
        if geo_mask is not None:
            importance = importance * geo_mask.float()

        # Normalize to distribution
        importance_sum = importance.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        importance = importance / importance_sum

        return importance

    def compute_global_embeddings(
        self,
        X_brep: torch.Tensor,
        X_pc: torch.Tensor,
        T_feat: torch.Tensor,
        imp_brep: torch.Tensor,
        imp_pc: torch.Tensor,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute global embeddings for retrieval.

        Args:
            X_brep, X_pc: Projected geometric tokens
            T_feat: Feature mention embeddings
            imp_brep, imp_pc: Semantic importance scores
            confidence: Slot confidence scores

        Returns:
            z_brep, z_pc, z_text: Global embeddings, each (B, d)
        """
        # Geometry: importance-weighted aggregation
        z_brep_geo = torch.sum(imp_brep.unsqueeze(-1) * X_brep, dim=1)
        z_pc_geo = torch.sum(imp_pc.unsqueeze(-1) * X_pc, dim=1)

        # Text: confidence-weighted aggregation of feature slots
        conf_sum = confidence.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        conf_norm = confidence / conf_sum
        z_text = torch.sum(conf_norm.unsqueeze(-1) * T_feat, dim=1)

        return z_brep_geo, z_pc_geo, z_text

    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Full forward pass computing all representations for training losses.

        Supports both cached features and raw inputs.

        Args:
            batch: Dictionary containing:
                - brep_face_features or brep_faces
                - brep_edge_features or brep_edges
                - brep_face_mask, brep_edge_mask
                - pc_features or points
                - text embeddings or tokens

        Returns:
            Dictionary containing all intermediate representations.
        """
        device = next(self.parameters()).device
        outputs = {}

        # Determine if using cached features
        use_cached_brep = batch.get("use_cached_brep_features", False)
        use_cached_pc = batch.get("use_cached_pc_features", False)
        use_cached_text = batch.get("use_cached_embeddings", False)

        has_brep = batch.get("has_brep", torch.tensor([True])).any()
        has_pc = batch.get("has_pointcloud", torch.tensor([True])).any()

        # =================================================================
        # Step 1: Project to unified dimension
        # =================================================================

        X_brep = None
        brep_mask = None
        if has_brep:
            if use_cached_brep:
                face_features = batch["brep_face_features"].to(device)
                edge_features = batch["brep_edge_features"].to(device)
            else:
                # Would need to run through encoder first
                raise NotImplementedError("Raw B-Rep encoding not implemented for GFA")

            face_mask = batch["brep_face_mask"].to(device)
            edge_mask = batch["brep_edge_mask"].to(device)

            X_brep, brep_mask = self.projection.project_brep(
                face_features, edge_features, face_mask, edge_mask
            )

        X_pc = None
        if has_pc:
            if use_cached_pc:
                pc_features = batch["pc_features"].to(device)
            else:
                raise NotImplementedError("Raw point cloud encoding not implemented for GFA")

            X_pc = self.projection.project_pointcloud(pc_features)

        # Text projection
        if use_cached_text:
            # Use full description embeddings for GFA
            text_features = batch["desc_embedding"].to(device)
            text_mask = batch["desc_mask"].to(device)
        else:
            raise NotImplementedError("Live text encoding not implemented for GFA")

        X_text = self.projection.project_text(text_features)

        # =================================================================
        # Step 2: Parse text into feature mentions with confidence
        # =================================================================

        T_feat, confidence, text_attn = self.parse_text_features(X_text, text_mask)
        outputs["confidence"] = confidence
        outputs["text_attn"] = text_attn

        # =================================================================
        # Step 3: Compute grounding matrices
        # =================================================================

        G_brep = None
        G_pc = None

        if has_brep and X_brep is not None:
            G_brep = self.compute_grounding_matrix(T_feat, X_brep, brep_mask)
            outputs["G_brep"] = G_brep

        if has_pc and X_pc is not None:
            G_pc = self.compute_grounding_matrix(T_feat, X_pc, None)
            outputs["G_pc"] = G_pc

        # =================================================================
        # Step 4: Extract grounded features
        # =================================================================

        F_brep_geo = None
        F_brep_fused = None
        F_pc_geo = None
        F_pc_fused = None

        if G_brep is not None:
            F_brep_geo, F_brep_fused = self.extract_grounded_features(
                G_brep, X_brep, T_feat
            )

        if G_pc is not None:
            F_pc_geo, F_pc_fused = self.extract_grounded_features(
                G_pc, X_pc, T_feat
            )

        # =================================================================
        # Step 5: Compute aligned features for consistency loss
        # =================================================================

        if F_brep_geo is not None and F_pc_geo is not None:
            F_brep_aligned, F_pc_aligned = self.compute_aligned_features(
                F_brep_geo, F_pc_geo
            )
            outputs["F_brep_aligned"] = F_brep_aligned
            outputs["F_pc_aligned"] = F_pc_aligned

        # =================================================================
        # Step 6: Compute semantic importance
        # =================================================================

        imp_brep = None
        imp_pc = None

        if G_brep is not None:
            imp_brep = self.compute_semantic_importance(G_brep, confidence, brep_mask)
            outputs["imp_brep"] = imp_brep

        if G_pc is not None:
            imp_pc = self.compute_semantic_importance(G_pc, confidence, None)
            outputs["imp_pc"] = imp_pc

        # =================================================================
        # Step 7: Compute global embeddings
        # =================================================================

        z_brep_geo = None
        z_pc_geo = None

        if has_brep and imp_brep is not None:
            z_brep_geo = torch.sum(imp_brep.unsqueeze(-1) * X_brep, dim=1)

        if has_pc and imp_pc is not None:
            z_pc_geo = torch.sum(imp_pc.unsqueeze(-1) * X_pc, dim=1)

        # Text global embedding
        conf_sum = confidence.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        conf_norm = confidence / conf_sum
        z_text = torch.sum(conf_norm.unsqueeze(-1) * T_feat, dim=1)

        # =================================================================
        # Step 8: Project for contrastive losses
        # =================================================================

        if z_brep_geo is not None:
            z_brep_proj = self.global_proj_head(z_brep_geo)
            outputs["z_brep"] = z_brep_proj

        if z_pc_geo is not None:
            z_pc_proj = self.global_proj_head(z_pc_geo)
            outputs["z_pc"] = z_pc_proj

        z_text_proj = self.global_proj_head(z_text)
        outputs["z_text"] = z_text_proj

        # Local projections (per feature slot)
        if F_brep_fused is not None:
            B, K, d = F_brep_fused.shape
            F_brep_local = self.local_proj_head(
                F_brep_fused.view(B * K, d)
            ).view(B, K, -1)
            outputs["F_brep_local"] = F_brep_local

        if F_pc_fused is not None:
            B, K, d = F_pc_fused.shape
            F_pc_local = self.local_proj_head(
                F_pc_fused.view(B * K, d)
            ).view(B, K, -1)
            outputs["F_pc_local"] = F_pc_local

        # Temperature
        outputs["temperature"] = self.tau_contrastive

        # Masks
        outputs["has_brep"] = batch.get("has_brep", torch.tensor([True])).to(device)
        outputs["has_pointcloud"] = batch.get("has_pointcloud", torch.tensor([True])).to(device)

        return outputs

    # =========================================================================
    # Inference Methods
    # =========================================================================

    @torch.no_grad()
    def encode_text(
        self,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text for retrieval (inference mode).

        Args:
            text_features: Pre-computed text features, (B, L, d_text)
            text_mask: Boolean mask, (B, L)

        Returns:
            z_text: Projected text embedding, (B, d_proj)
        """
        self.eval()
        X_text = self.projection.project_text(text_features)
        T_feat, confidence, _ = self.parse_text_features(X_text, text_mask)

        # Confidence-weighted aggregation
        conf_sum = confidence.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        conf_norm = confidence / conf_sum
        z_text = torch.sum(conf_norm.unsqueeze(-1) * T_feat, dim=1)

        z_text_proj = self.global_proj_head(z_text)
        return F.normalize(z_text_proj, dim=-1)

    @torch.no_grad()
    def encode_geometry(
        self,
        brep_face_features: Optional[torch.Tensor] = None,
        brep_edge_features: Optional[torch.Tensor] = None,
        brep_face_mask: Optional[torch.Tensor] = None,
        brep_edge_mask: Optional[torch.Tensor] = None,
        pc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode geometry for retrieval (inference mode, no text available).

        Uses self-grounding queries to extract summary features.

        Returns:
            z_geo: Projected geometric embedding, (B, d_proj)
        """
        self.eval()
        embeddings = []

        if brep_face_features is not None and brep_edge_features is not None:
            B = brep_face_features.shape[0]
            X_brep, brep_mask = self.projection.project_brep(
                brep_face_features, brep_edge_features,
                brep_face_mask, brep_edge_mask
            )

            # Self-grounding
            queries = self.self_ground_queries.unsqueeze(0).expand(B, -1, -1)
            G_self = self.compute_grounding_matrix(queries, X_brep, brep_mask)

            # Importance-weighted aggregation
            importance = G_self.sum(dim=1)  # (B, N)
            if brep_mask is not None:
                importance = importance * brep_mask.float()
            importance = importance / importance.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            z_brep = torch.sum(importance.unsqueeze(-1) * X_brep, dim=1)
            embeddings.append(z_brep)

        if pc_features is not None:
            B = pc_features.shape[0]
            X_pc = self.projection.project_pointcloud(pc_features)

            # Self-grounding
            queries = self.self_ground_queries.unsqueeze(0).expand(B, -1, -1)
            G_self = self.compute_grounding_matrix(queries, X_pc, None)

            importance = G_self.sum(dim=1)
            importance = importance / importance.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            z_pc = torch.sum(importance.unsqueeze(-1) * X_pc, dim=1)
            embeddings.append(z_pc)

        # Average embeddings if both modalities available
        z_geo = torch.stack(embeddings, dim=0).mean(dim=0)
        z_geo_proj = self.global_proj_head(z_geo)

        return F.normalize(z_geo_proj, dim=-1)

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config_path: str) -> "CLIP4CAD_GFA":
        """Create model from config file."""
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        return cls(config)
