"""
CLIP4CAD-GFA Ablation Study Module

Provides tools for running architecture ablation experiments:
- No Consistency (λ_c=0)
- Global-Only (no local/grounding losses)
- No Confidence (fixed uniform weighting)

Usage:
    from clip4cad.ablations import AblationTrainer, get_ablation_config

    trainer = AblationTrainer(
        ablation_type='no_consistency',
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config_path='configs/model/clip4cad_gfa.yaml',
        output_dir='outputs/ablation_no_consistency',
    )

    for epoch in range(35):
        metrics = trainer.train_epoch()
        print(f"Epoch {epoch}: {metrics}")
"""

from .configs import get_ablation_config, ABLATION_TYPES
from .models import CLIP4CAD_GFA_Ablation
from .trainer import AblationTrainer

__all__ = [
    "get_ablation_config",
    "ABLATION_TYPES",
    "CLIP4CAD_GFA_Ablation",
    "AblationTrainer",
]
