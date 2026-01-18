"""
Pretrained Encoder Wrappers

Provides unified interface to:
- AutoBrep-style B-Rep encoder (face grids + edge curves)
- Point-BERT point cloud encoder with ULIP-2 weights
- Unified projection to common embedding dimension
"""

from .brep_encoder import BRepEncoder, FaceEncoder, EdgeEncoder
from .pointbert_encoder import PointBertEncoder
from .unified_projection import UnifiedInputProjection

__all__ = [
    "BRepEncoder",
    "FaceEncoder",
    "EdgeEncoder",
    "PointBertEncoder",
    "UnifiedInputProjection",
]
