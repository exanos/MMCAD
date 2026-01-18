"""
Training Pipeline for CLIP4CAD

Two-stage training:
- Stage 1: Global alignment + reconstruction
- Stage 2: Add local contrastive alignment
"""

from .trainer import CLIP4CADTrainer

__all__ = ["CLIP4CADTrainer"]
