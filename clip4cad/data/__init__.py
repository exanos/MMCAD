"""
Data Loading Pipeline for CLIP4CAD

Handles:
- Multimodal CAD data (B-Rep, point cloud, text)
- Variable numbers of faces/edges with padding
- Consistent augmentation across modalities
- Missing modality handling

Dataset types:
1. MMCADDataset: Standard dataset for CLIP4CAD-H
2. GFADataset: Multi-rotation dataset for CLIP4CAD-GFA
3. AutoBrepDataset: BFS-ordered features with full topology for attention visualization
"""

from .dataset import MMCADDataset, collate_fn
from .augmentation import apply_rotation_augmentation
from .gfa_dataset import (
    GFADataset,
    gfa_collate_fn,
    create_gfa_dataloader,
    HardNegativeSampler,
)
from .shapellm_cache import ShapeLLMFeatureCache, verify_shapellm_cache
from .autobrep_dataset import (
    AutoBrepDataset,
    autobrep_collate_fn,
    create_autobrep_dataloader,
)
from .autobrep_utils import (
    bfs_order_faces_with_parents,
    bfs_order_edges,
    extract_spatial_properties,
    extract_geometry_types,
    build_face_adjacency_matrix,
    FACE_TYPE_NAMES,
    EDGE_TYPE_NAMES,
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
    # ShapeLLM features
    "ShapeLLMFeatureCache",
    "verify_shapellm_cache",
    # AutoBrep BFS-ordered dataset
    "AutoBrepDataset",
    "autobrep_collate_fn",
    "create_autobrep_dataloader",
    # AutoBrep utilities
    "bfs_order_faces_with_parents",
    "bfs_order_edges",
    "extract_spatial_properties",
    "extract_geometry_types",
    "build_face_adjacency_matrix",
    "FACE_TYPE_NAMES",
    "EDGE_TYPE_NAMES",
]
