"""
Loss Functions for CLIP4CAD Training

Three loss configurations available:
1. CLIP4CADLoss: For CLIP4CAD-H (hierarchical compression)
2. GFALoss: For CLIP4CAD-GFA (grounded feature alignment)
3. HUSLoss: For CLIP4CAD-HUS (hierarchical unified space)

Components:
- Global contrastive loss (InfoNCE across modalities)
- Local contrastive loss (Hungarian matching with confidence)
- Reconstruction loss (auxiliary regularization)
- Grounding consistency loss (for GFA)
- Grounding diversity loss (for GFA)
"""

from .combined import CLIP4CADLoss
from .infonce import InfoNCELoss
from .local_matching import LocalMatchingLoss
from .reconstruction import ReconstructionLoss
from .gfa_losses import (
    GFALoss,
    GroundingConsistencyLoss,
    GroundingDiversityLoss,
    SlotContrastiveLoss,
)
from .gfa_v2_losses import GFAv2Loss, compute_self_grounding_quality
from .gfa_v2_4_losses import GFAv2_4Loss, compute_self_grounding_quality as compute_self_grounding_quality_v2_4
from .gfa_v4_losses import GFAv4Loss, compute_self_grounding_quality as compute_self_grounding_quality_v4, compute_query_alignment
from .gfa_v4_2_losses import GFAv4_2Loss, compute_self_grounding_quality as compute_self_grounding_quality_v4_2, compute_query_alignment as compute_query_alignment_v4_2
from .hus_losses import HUSLoss, HUSLossConfig

__all__ = [
    # Main loss functions
    "CLIP4CADLoss",
    "GFALoss",
    "GFAv2Loss",
    "GFAv2_4Loss",
    "GFAv4Loss",
    "GFAv4_2Loss",
    "HUSLoss",
    "HUSLossConfig",
    # Component losses
    "InfoNCELoss",
    "LocalMatchingLoss",
    "ReconstructionLoss",
    "GroundingConsistencyLoss",
    "GroundingDiversityLoss",
    "SlotContrastiveLoss",
    # Utilities
    "compute_self_grounding_quality",
    "compute_query_alignment",
]
