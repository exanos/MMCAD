"""
Loss Functions for CLIP4CAD Training

Two loss configurations available:
1. CLIP4CADLoss: For CLIP4CAD-H (hierarchical compression)
2. GFALoss: For CLIP4CAD-GFA (grounded feature alignment)

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

__all__ = [
    # Main loss functions
    "CLIP4CADLoss",
    "GFALoss",
    # Component losses
    "InfoNCELoss",
    "LocalMatchingLoss",
    "ReconstructionLoss",
    "GroundingConsistencyLoss",
    "GroundingDiversityLoss",
    "SlotContrastiveLoss",
]
