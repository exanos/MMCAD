"""
CLIP4CAD-H: Main Model

Integrates all components:
- Pretrained encoders (B-Rep, Point-BERT, LLM)
- Unified projection
- Hierarchical compression (GSC + ADM)
- Hierarchical text encoder
- Reconstruction decoder (auxiliary)
- Projection heads for contrastive learning
"""

import math
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .encoders.brep_encoder import BRepEncoder
from .encoders.pointbert_encoder import PointBertEncoder
from .encoders.unified_projection import UnifiedInputProjection
from .hierarchical_compression import HierarchicalCompressionModule
from .text_encoder import HierarchicalTextEncoder
from .reconstruction_decoder import ReconstructionDecoder
from .projection_heads import ProjectionHead, LearnableTemperature


class CLIP4CAD_H(nn.Module):
    """
    CLIP4CAD-H: Hierarchical Multimodal Encoder for CAD Models

    Aligns B-Rep, point cloud, and text representations in a unified
    latent space with hierarchical structure (global + local).
    """

    def __init__(self, config: DictConfig):
        """
        Args:
            config: Model configuration (from configs/model/clip4cad_h.yaml)
        """
        super().__init__()
        self.config = config

        d_unified = config.d_unified
        d_proj = config.d_proj

        # ============================================================
        # Pretrained Encoders
        # ============================================================

        # B-Rep encoder (AutoBrep-style FSQ VAE)
        brep_cfg = config.encoders.brep
        self.brep_encoder = BRepEncoder(
            face_dim=brep_cfg.face_dim,
            edge_dim=brep_cfg.edge_dim,
            face_grid_size=brep_cfg.get("face_grid_size", 32),
            edge_curve_size=brep_cfg.get("edge_curve_size", 32),
            face_base_channels=brep_cfg.get("face_base_channels", 64),
            face_channel_mult=tuple(brep_cfg.get("face_channel_mult", [1, 2, 4, 8])),
            face_latent_channels=brep_cfg.get("face_latent_channels", 16),
            edge_base_channels=brep_cfg.get("edge_base_channels", 64),
            edge_channel_mult=tuple(brep_cfg.get("edge_channel_mult", [1, 2, 4])),
            edge_latent_channels=brep_cfg.get("edge_latent_channels", 4),
            fsq_levels=tuple(brep_cfg.get("fsq_levels", [8, 5, 5, 5])),
            use_fsq=brep_cfg.get("use_fsq", False),
            surface_checkpoint=brep_cfg.get("surface_checkpoint"),
            edge_checkpoint=brep_cfg.get("edge_checkpoint"),
            freeze=brep_cfg.get("freeze", False),
        )

        # Point cloud encoder (ULIP-2 Point-BERT)
        pc_cfg = config.encoders.pointcloud
        self.pc_encoder = PointBertEncoder(
            num_points=pc_cfg.get("num_points", 10000),
            in_channels=pc_cfg.get("in_channels", 6),
            embed_dim=pc_cfg.output_dim,
            depth=pc_cfg.get("num_layers", 12),
            num_heads=pc_cfg.get("num_heads", 12),
            num_groups=pc_cfg.get("num_groups", 512),
            group_size=pc_cfg.get("group_size", 32),
            mlp_ratio=pc_cfg.get("mlp_ratio", 4.0),
            drop_rate=pc_cfg.get("drop_rate", 0.0),
            drop_path_rate=pc_cfg.get("drop_path_rate", 0.1),
            checkpoint_path=pc_cfg.get("checkpoint"),
            freeze=pc_cfg.freeze,
        )

        # Text encoder (LLM) - initialized lazily
        self.text_encoder = HierarchicalTextEncoder(
            llm_name=config.encoders.text.model_name,
            d_unified=d_unified,
            n_local_features=config.compression.n_detail_queries,
            freeze_llm=config.encoders.text.freeze,
            use_fp16=config.encoders.text.get("use_fp16", True),
        )

        # ============================================================
        # Unified Projection
        # ============================================================

        self.projection = UnifiedInputProjection(
            d_unified=d_unified,
            d_brep_face=self.brep_encoder.face_dim,
            d_brep_edge=self.brep_encoder.edge_dim,
            d_pointbert=self.pc_encoder.output_dim,
        )

        # ============================================================
        # Hierarchical Compression
        # ============================================================

        self.compression = HierarchicalCompressionModule(
            d_model=d_unified,
            n_global_queries=config.compression.n_global_queries,
            n_detail_queries=config.compression.n_detail_queries,
            k_detail_select=config.compression.k_detail_select,
            n_heads=config.compression.n_heads,
            n_layers=config.compression.n_layers,
            dropout=config.compression.get("dropout", 0.1),
        )

        # ============================================================
        # Reconstruction Decoder (Auxiliary)
        # ============================================================

        if config.reconstruction.enabled:
            n_tokens = config.compression.n_global_queries + config.compression.n_detail_queries
            self.recon_decoder = ReconstructionDecoder(
                d_unified=d_unified,
                n_unified_tokens=n_tokens,
                max_faces=config.reconstruction.max_faces,
                max_edges=config.reconstruction.max_edges,
                d_face=config.encoders.brep.face_dim,
                d_edge=config.encoders.brep.edge_dim,
            )
        else:
            self.recon_decoder = None

        # ============================================================
        # Projection Heads
        # ============================================================

        self.global_proj = ProjectionHead(d_unified, d_proj)
        self.local_proj = ProjectionHead(d_unified, d_proj)

        # ============================================================
        # Learnable Temperature
        # ============================================================

        self.temperature = LearnableTemperature(
            init_value=config.get("temperature_init", 0.07)
        )

        # Track if LLM is loaded
        self._llm_loaded = False

    def load_llm(self, device: Optional[torch.device] = None):
        """
        Explicitly load the LLM for text encoding.

        Call this before training to initialize the text encoder.
        """
        if not self._llm_loaded:
            self.text_encoder.load_llm(device)
            self._llm_loaded = True

    def get_tokenizer(self):
        """Get the text tokenizer."""
        return self.text_encoder.get_tokenizer()

    def encode_brep(
        self,
        face_grids: torch.Tensor,
        edge_curves: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode B-Rep through encoder + compression.

        Args:
            face_grids: [B, F, H, W, 3]
            edge_curves: [B, E, L, 3]
            face_mask: [B, F]
            edge_mask: [B, E]

        Returns:
            Dictionary with embeddings
        """
        # Encode with B-Rep encoder
        face_tokens, edge_tokens = self.brep_encoder(
            face_grids, edge_curves, face_mask, edge_mask
        )

        # Project to unified space
        tokens, mask = self.projection.project_brep(
            face_tokens, edge_tokens, face_mask, edge_mask
        )

        # Hierarchical compression
        compressed = self.compression(tokens, mask)

        # Project for contrastive
        z_global_proj = self.global_proj(compressed["z_global"])
        z_local_proj = self.local_proj(compressed["detail_features"])

        return {
            "z_global": compressed["z_global"],
            "z_global_proj": z_global_proj,
            "z_local": compressed["detail_features"],
            "z_local_proj": z_local_proj,
            "unified": compressed["unified"],
            "global_features": compressed["global_features"],
            "coverage": compressed["coverage"],
        }

    def encode_pointcloud(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode point cloud through encoder + compression.

        Args:
            points: [B, N, 3]

        Returns:
            Dictionary with embeddings
        """
        # Encode with Point-BERT
        if self.pc_encoder.training:
            pc_tokens = self.pc_encoder(points)
        else:
            with torch.no_grad():
                pc_tokens = self.pc_encoder(points)

        # Project to unified space
        tokens = self.projection.project_pointcloud(pc_tokens)

        # Hierarchical compression (no mask for point cloud)
        compressed = self.compression(tokens, mask=None)

        # Project for contrastive
        z_global_proj = self.global_proj(compressed["z_global"])
        z_local_proj = self.local_proj(compressed["detail_features"])

        return {
            "z_global": compressed["z_global"],
            "z_global_proj": z_global_proj,
            "z_local": compressed["detail_features"],
            "z_local_proj": z_local_proj,
            "unified": compressed["unified"],
            "global_features": compressed["global_features"],
        }

    def encode_text(
        self,
        title_input_ids: Optional[torch.Tensor] = None,
        title_attention_mask: Optional[torch.Tensor] = None,
        desc_input_ids: Optional[torch.Tensor] = None,
        desc_attention_mask: Optional[torch.Tensor] = None,
        # Cached embeddings (alternative to input_ids)
        title_embedding: Optional[torch.Tensor] = None,
        desc_embedding: Optional[torch.Tensor] = None,
        desc_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode hierarchical text.

        Supports both live inference and cached embeddings.

        Args:
            title_input_ids: [B, L_title] (for live inference)
            title_attention_mask: [B, L_title] (for live inference)
            desc_input_ids: [B, L_desc] (for live inference)
            desc_attention_mask: [B, L_desc] (for live inference)
            title_embedding: [B, d_llm] (for cached mode)
            desc_embedding: [B, L_desc, d_llm] (for cached mode)
            desc_mask: [B, L_desc] (for cached mode)

        Returns:
            Dictionary with embeddings
        """
        text_out = self.text_encoder(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask,
            desc_input_ids=desc_input_ids,
            desc_attention_mask=desc_attention_mask,
            title_embedding=title_embedding,
            desc_embedding=desc_embedding,
            desc_mask=desc_mask,
        )

        # Project for contrastive
        t_global_proj = self.global_proj(text_out["t_global"])
        t_local_proj = self.local_proj(text_out["t_local"])

        return {
            "t_global": text_out["t_global"],
            "t_global_proj": t_global_proj,
            "t_local": text_out["t_local"],
            "t_local_proj": t_local_proj,
            "t_local_confidence": text_out["t_local_confidence"],
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Full forward pass.

        Handles missing modalities gracefully.

        Args:
            batch: Batched input data

        Returns:
            Dictionary with all embeddings and outputs
        """
        outputs = {}
        device = next(self.parameters()).device

        # Move masks to device
        has_brep = batch.get("has_brep", torch.tensor([False]))
        has_pc = batch.get("has_pointcloud", torch.tensor([False]))

        if isinstance(has_brep, torch.Tensor):
            has_brep = has_brep.to(device)
        if isinstance(has_pc, torch.Tensor):
            has_pc = has_pc.to(device)

        # ============================================================
        # Encode B-Rep (if available)
        # ============================================================

        if has_brep.any():
            brep_out = self.encode_brep(
                batch["brep_faces"].to(device),
                batch["brep_edges"].to(device),
                batch["brep_face_mask"].to(device),
                batch["brep_edge_mask"].to(device),
            )
            outputs["brep"] = brep_out

        # ============================================================
        # Encode Point Cloud (if available)
        # ============================================================

        if has_pc.any():
            pc_out = self.encode_pointcloud(batch["points"].to(device))
            outputs["pointcloud"] = pc_out

        # ============================================================
        # Encode Text (supports both cached embeddings and live inference)
        # ============================================================

        use_cached = batch.get("use_cached_embeddings", False)

        if use_cached:
            # Use pre-computed embeddings
            text_out = self.encode_text(
                title_embedding=batch["title_embedding"].to(device),
                desc_embedding=batch["desc_embedding"].to(device),
                desc_mask=batch["desc_mask"].to(device),
            )
        else:
            # Use live LLM inference
            text_out = self.encode_text(
                title_input_ids=batch["title_input_ids"].to(device),
                title_attention_mask=batch["title_attention_mask"].to(device),
                desc_input_ids=batch["desc_input_ids"].to(device),
                desc_attention_mask=batch["desc_attention_mask"].to(device),
            )
        outputs["text"] = text_out

        # ============================================================
        # Reconstruction (from B-Rep if available, else point cloud)
        # ============================================================

        if self.recon_decoder is not None:
            if "brep" in outputs:
                recon_out = self.recon_decoder(outputs["brep"]["unified"])
                outputs["reconstruction"] = recon_out
            elif "pointcloud" in outputs:
                recon_out = self.recon_decoder(outputs["pointcloud"]["unified"])
                outputs["reconstruction"] = recon_out

        # Temperature
        outputs["temperature"] = self.temperature()

        # Masks for loss computation
        outputs["has_brep"] = has_brep
        outputs["has_pointcloud"] = has_pc

        return outputs

    def get_text_embedding(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get text embedding for inference (title only)."""
        t_global = self.text_encoder.encode_title_only(
            title_input_ids, title_attention_mask
        )
        return self.global_proj(t_global)

    def get_brep_embedding(
        self,
        face_grids: torch.Tensor,
        edge_curves: torch.Tensor,
        face_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get B-Rep embedding for inference."""
        brep_out = self.encode_brep(face_grids, edge_curves, face_mask, edge_mask)
        return brep_out["z_global_proj"]

    def get_pointcloud_embedding(self, points: torch.Tensor) -> torch.Tensor:
        """Get point cloud embedding for inference."""
        pc_out = self.encode_pointcloud(points)
        return pc_out["z_global_proj"]

    @classmethod
    def from_config(cls, config_path: str) -> "CLIP4CAD_H":
        """Create model from config file."""
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        return cls(config)

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
