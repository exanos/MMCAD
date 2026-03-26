"""
Ablation Model Variants

Provides CLIP4CAD_GFA_Ablation which extends the base model to support:
- Uniform confidence weighting (for no_confidence ablation)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from omegaconf import DictConfig

from ..models.clip4cad_gfa import CLIP4CAD_GFA


class CLIP4CAD_GFA_Ablation(CLIP4CAD_GFA):
    """
    CLIP4CAD-GFA model variant supporting ablation configurations.

    Extends the base model to support:
    - use_uniform_confidence: Replace learned confidence with fixed 1/K

    All other functionality is inherited from the base class.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize ablation model.

        Args:
            config: Model configuration (may include ablation-specific flags)
        """
        super().__init__(config)

        # Ablation flags
        self.use_uniform_confidence = config.get("use_uniform_confidence", False)
        self.ablation_type = config.get("ablation_type", "baseline")

        if self.use_uniform_confidence:
            print(f"  [Ablation] Using UNIFORM confidence (all slots = 1/K)")
        else:
            print(f"  [Ablation] Using LEARNED confidence")

    def parse_text_features(
        self,
        X_text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Parse text into K feature mention embeddings with confidence scores.

        For no_confidence ablation, returns uniform 1/K confidence instead
        of learned confidence.

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

        # Compute confidence based on ablation mode
        if self.use_uniform_confidence:
            # Fixed uniform confidence: all slots contribute equally
            confidence = torch.ones(B, K, device=X_text.device, dtype=X_text.dtype) / K
        else:
            # Learned confidence (original behavior)
            # Predict confidence for each slot with FP16-safe sigmoid
            logits = self.confidence_predictor(T_feat).squeeze(-1)  # (B, K)
            logits = logits.clamp(min=-5, max=5)  # FP16 safety
            confidence = torch.sigmoid(logits)

        return T_feat, confidence, attn_weights

    def get_ablation_info(self) -> Dict[str, Any]:
        """Return information about the ablation configuration."""
        return {
            "ablation_type": self.ablation_type,
            "use_uniform_confidence": self.use_uniform_confidence,
            "num_feature_slots": self.num_feature_slots,
        }


def create_ablation_model(config: DictConfig) -> CLIP4CAD_GFA_Ablation:
    """
    Factory function to create ablation model.

    Args:
        config: Model configuration with ablation flags

    Returns:
        CLIP4CAD_GFA_Ablation instance
    """
    model = CLIP4CAD_GFA_Ablation(config)

    # Print ablation info
    info = model.get_ablation_info()
    print(f"\nCreated ablation model:")
    print(f"  Type: {info['ablation_type']}")
    print(f"  Uniform confidence: {info['use_uniform_confidence']}")
    print(f"  Feature slots: {info['num_feature_slots']}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Trainable: {model.count_parameters(trainable_only=True):,}")

    return model
