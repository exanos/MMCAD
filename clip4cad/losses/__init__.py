"""
Loss Functions for CLIP4CAD Training

Components:
- Global contrastive loss (InfoNCE across modalities)
- Local contrastive loss (Hungarian matching with confidence)
- Reconstruction loss (auxiliary regularization)
"""

from .combined import CLIP4CADLoss
from .infonce import InfoNCELoss
from .local_matching import LocalMatchingLoss
from .reconstruction import ReconstructionLoss

__all__ = [
    "CLIP4CADLoss",
    "InfoNCELoss",
    "LocalMatchingLoss",
    "ReconstructionLoss",
]
