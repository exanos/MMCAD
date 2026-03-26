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
from .clip4cad_gfa_v4_4 import CLIP4CAD_GFA_v44, GFAv44Config
from .clip4cad_gfa_v4_8 import CLIP4CAD_GFA_v48, GFAv48Config
from .clip4cad_gfa_v4_8_1 import CLIP4CAD_GFA_v481, GFAv481Config
from .clip4cad_gfa_v4_9 import CLIP4CAD_GFA_v49, GFAv49Config
from .brep_encoder_autobrep import (
    AutoBrepEncoderConfig,
    AutoBrepEncoder,
    SimpleCLIP4CAD_AutoBrep,
    SimpleContrastiveLoss,
)
from .autobrep_encoder import (
    AutoBrepEncoderConfig as AutoBrepStyleEncoderConfig,
    AutoBrepStyleEncoder,
    CLIP4CAD_AutoBrep,
    ContrastiveLoss as AutoBrepContrastiveLoss,
)
from .topology_encoder import (
    TopologyAwareBRepEncoder,
    HierarchicalPatternAggregator,
    SemanticQueryGenerator,
    TopologyMessageLayer,
)
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
    "CLIP4CAD_GFA_v44",
    "GFAv44Config",
    "CLIP4CAD_GFA_v48",
    "GFAv48Config",
    "CLIP4CAD_GFA_v481",
    "GFAv481Config",
    # v4.9: Direct contrastive (no codebook)
    "CLIP4CAD_GFA_v49",
    "GFAv49Config",
    # AutoBrep-style encoder (original simple version)
    "AutoBrepEncoderConfig",
    "AutoBrepEncoder",
    "SimpleCLIP4CAD_AutoBrep",
    "SimpleContrastiveLoss",
    # AutoBrep-style encoder (full version following AutoBrep paper)
    "AutoBrepStyleEncoderConfig",
    "AutoBrepStyleEncoder",
    "CLIP4CAD_AutoBrep",
    "AutoBrepContrastiveLoss",
    # Topology encoder components
    "TopologyAwareBRepEncoder",
    "HierarchicalPatternAggregator",
    "SemanticQueryGenerator",
    "TopologyMessageLayer",
    # Components
    "HierarchicalCompressionModule",
    "HierarchicalTextEncoder",
    "ReconstructionDecoder",
    "FSQ",
    "SurfaceFSQEncoder",
    "EdgeFSQEncoder",
]
