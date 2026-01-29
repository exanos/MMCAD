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
from .clip4cad_gfa_v2 import CLIP4CAD_GFA_v2, GFAv2Config
from .clip4cad_gfa_v2_4 import CLIP4CAD_GFA_v2_4, GFAv2_4Config
from .clip4cad_gfa_v4 import CLIP4CAD_GFA_v4, GFAv4Config
from .clip4cad_gfa_v4_2 import CLIP4CAD_GFA_v4_2, GFAv4_2Config, get_cond_dropout
from .hierarchical_compression import HierarchicalCompressionModule
from .text_encoder import HierarchicalTextEncoder
from .reconstruction_decoder import ReconstructionDecoder
from .fsq import FSQ, SurfaceFSQEncoder, EdgeFSQEncoder

__all__ = [
    # Main models
    "CLIP4CAD_H",
    "CLIP4CAD_GFA",
    "CLIP4CAD_GFA_v2",
    "GFAv2Config",
    "CLIP4CAD_GFA_v2_4",
    "GFAv2_4Config",
    "CLIP4CAD_GFA_v4",
    "GFAv4Config",
    "CLIP4CAD_GFA_v4_2",
    "GFAv4_2Config",
    "get_cond_dropout",
    # Components
    "HierarchicalCompressionModule",
    "HierarchicalTextEncoder",
    "ReconstructionDecoder",
    "FSQ",
    "SurfaceFSQEncoder",
    "EdgeFSQEncoder",
]
