"""
Training Pipeline for CLIP4CAD

Three trainer options:
1. CLIP4CADTrainer: For CLIP4CAD-H (hierarchical compression)
2. GFATrainer: For CLIP4CAD-GFA (grounded feature alignment)
3. HUSTrainer: For CLIP4CAD-HUS (hierarchical unified space)

Two-stage training for GFA and HUS:
- Stage 1: Grounding/hierarchy establishment
- Stage 2: Hard negative mining for fine-grained discrimination
"""

from .trainer import CLIP4CADTrainer
from .gfa_trainer import GFATrainer, train_gfa
from .gfa_v2_trainer import GFAv2Trainer, train_step
from .hus_trainer import HUSTrainer, train_hus
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
    "GFAv2Trainer",
    "train_step",
    "HUSTrainer",
    "train_hus",
    # Hard negative mining
    "HardNegativeMiner",
    "mine_hard_negatives",
    "extract_embeddings",
    "save_hard_negatives",
    "load_hard_negatives",
]
