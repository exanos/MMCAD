"""
CLIP4CAD Model Components

Two architectures available:
1. CLIP4CAD-H: Hierarchical compression (GSC + ADM)
2. CLIP4CAD-GFA: Grounded Feature Alignment

Components:
- Pretrained encoders (B-Rep, Point Cloud, LLM)
- Unified projection layer
- Hierarchical compression (GSC + ADM) for CLIP4CAD-H
- Grounding module for CLIP4CAD-GFA
- Contrastive projection heads
"""

from .clip4cad_h import CLIP4CAD_H
from .clip4cad_gfa import CLIP4CAD_GFA
from .hierarchical_compression import HierarchicalCompressionModule
from .text_encoder import HierarchicalTextEncoder
from .reconstruction_decoder import ReconstructionDecoder
from .fsq import FSQ, SurfaceFSQEncoder, EdgeFSQEncoder

__all__ = [
    # Main models
    "CLIP4CAD_H",
    "CLIP4CAD_GFA",
    # Components
    "HierarchicalCompressionModule",
    "HierarchicalTextEncoder",
    "ReconstructionDecoder",
    "FSQ",
    "SurfaceFSQEncoder",
    "EdgeFSQEncoder",
]
