"""
Data Loading Pipeline for CLIP4CAD

Handles:
- Multimodal CAD data (B-Rep, point cloud, text)
- Variable numbers of faces/edges with padding
- Consistent augmentation across modalities
- Missing modality handling
"""

from .dataset import MMCADDataset, collate_fn
from .augmentation import apply_rotation_augmentation

__all__ = [
    "MMCADDataset",
    "collate_fn",
    "apply_rotation_augmentation",
]
