"""
Data Loading Pipeline for CLIP4CAD

Handles:
- Multimodal CAD data (B-Rep, point cloud, text)
- Variable numbers of faces/edges with padding
- Consistent augmentation across modalities
- Missing modality handling

Two dataset types:
1. MMCADDataset: Standard dataset for CLIP4CAD-H
2. GFADataset: Multi-rotation dataset for CLIP4CAD-GFA
"""

from .dataset import MMCADDataset, collate_fn
from .augmentation import apply_rotation_augmentation
from .gfa_dataset import (
    GFADataset,
    gfa_collate_fn,
    create_gfa_dataloader,
    HardNegativeSampler,
)

__all__ = [
    # Standard dataset
    "MMCADDataset",
    "collate_fn",
    "apply_rotation_augmentation",
    # GFA dataset
    "GFADataset",
    "gfa_collate_fn",
    "create_gfa_dataloader",
    "HardNegativeSampler",
]
