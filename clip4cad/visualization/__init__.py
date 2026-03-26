"""
Visualization tools for CLIP4CAD.

Provides:
- AttentionVisualizer: Trace attention back to specific faces/edges
- 3D attention heatmaps
- Point cloud rendering of attended faces
"""

from .attention_viz import (
    AttentionVisualizer,
    FACE_TYPE_NAMES,
    EDGE_TYPE_NAMES,
)

__all__ = [
    "AttentionVisualizer",
    "FACE_TYPE_NAMES",
    "EDGE_TYPE_NAMES",
]
