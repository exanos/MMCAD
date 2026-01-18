"""
CLIP4CAD Model Components

Hierarchical multimodal encoder architecture:
- Pretrained encoders (B-Rep, Point Cloud, LLM)
- Unified projection layer
- Hierarchical compression (GSC + ADM)
- Contrastive projection heads
"""

from .clip4cad_h import CLIP4CAD_H
from .hierarchical_compression import HierarchicalCompressionModule
from .text_encoder import HierarchicalTextEncoder
from .reconstruction_decoder import ReconstructionDecoder

__all__ = [
    "CLIP4CAD_H",
    "HierarchicalCompressionModule",
    "HierarchicalTextEncoder",
    "ReconstructionDecoder",
]
