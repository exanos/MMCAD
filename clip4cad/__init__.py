"""
CLIP4CAD: Unified Multimodal Encoder for CAD Models

A CLIP-style model that jointly represents multiple CAD modalities
(B-Rep, point cloud, text) in a unified latent space.
"""

__version__ = "0.1.0"
__author__ = "MMCAD Team"

from .models.clip4cad_h import CLIP4CAD_H

__all__ = ["CLIP4CAD_H"]
