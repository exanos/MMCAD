"""
Training Pipeline for CLIP4CAD

Two trainer options:
1. CLIP4CADTrainer: For CLIP4CAD-H (hierarchical compression)
2. GFATrainer: For CLIP4CAD-GFA (grounded feature alignment)

Two-stage training for GFA:
- Stage 1: Grounding establishment (reduced global loss)
- Stage 2: Global alignment with hard negative mining
"""

from .trainer import CLIP4CADTrainer
from .gfa_trainer import GFATrainer, train_gfa
from .hard_negative_mining import (
    HardNegativeMiner,
    mine_hard_negatives,
    extract_embeddings,
    save_hard_negatives,
    load_hard_negatives,
)

__all__ = [
    # Trainers
    "CLIP4CADTrainer",
    "GFATrainer",
    "train_gfa",
    # Hard negative mining
    "HardNegativeMiner",
    "mine_hard_negatives",
    "extract_embeddings",
    "save_hard_negatives",
    "load_hard_negatives",
]
