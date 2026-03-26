"""
Ablation Configuration Generators

Provides configuration modifications for each ablation variant:
- baseline: Full model (no changes)
- no_consistency: λ_c=0 (disable grounding consistency loss)
- global_only: No local/grounding losses (only global contrastive)
- no_confidence: Fixed uniform confidence weighting
- asymmetric_grounding: Different grounding weights per modality
"""

from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
import copy


# Available ablation types and their descriptions
ABLATION_TYPES = {
    "baseline": "Full model with all losses and learned confidence",
    "no_consistency": "Disable grounding consistency loss (λ_c_brep=0, λ_c_pc=0)",
    "global_only": "Only global contrastive loss (no local/grounding)",
    "no_confidence": "Fixed uniform confidence (no learned weighting)",
    "weak_grounding": "Reduced grounding losses (λ_l=0.1, λ_c_brep=0.08, λ_c_pc=0.02, λ_d=0.02)",
    "asymmetric_grounding": "Asymmetric consistency (λ_c_brep=0.08, λ_c_pc=0.02)",
}


def get_ablation_config(
    base_config_path: str,
    ablation_type: str,
    custom_overrides: Optional[Dict[str, Any]] = None,
) -> DictConfig:
    """
    Generate configuration for a specific ablation variant.

    Args:
        base_config_path: Path to base clip4cad_gfa.yaml config
        ablation_type: One of 'baseline', 'no_consistency', 'global_only', 'no_confidence'
        custom_overrides: Additional config overrides (optional)

    Returns:
        OmegaConf DictConfig with ablation-specific modifications
    """
    if ablation_type not in ABLATION_TYPES:
        raise ValueError(
            f"Unknown ablation type: {ablation_type}. "
            f"Valid types: {list(ABLATION_TYPES.keys())}"
        )

    # Load base config
    base_path = Path(base_config_path)
    if not base_path.exists():
        # Try relative to common locations
        alt_paths = [
            Path("configs/model/clip4cad_gfa.yaml"),
            Path("../configs/model/clip4cad_gfa.yaml"),
        ]
        for alt in alt_paths:
            if alt.exists():
                base_path = alt
                break

    if not base_path.exists():
        raise FileNotFoundError(f"Config not found: {base_config_path}")

    config = OmegaConf.load(base_path)

    # Apply ablation-specific modifications
    if ablation_type == "baseline":
        # No changes - full model
        pass

    elif ablation_type == "no_consistency":
        # Disable grounding consistency loss (both modalities)
        config.training.lambda_consist_brep = 0.0
        config.training.lambda_consist_pc = 0.0

    elif ablation_type == "global_only":
        # Disable all local/grounding losses - only global contrastive
        config.training.lambda_local = 0.0
        config.training.lambda_consist_brep = 0.0
        config.training.lambda_consist_pc = 0.0
        config.training.lambda_diverse = 0.0

    elif ablation_type == "no_confidence":
        # Use fixed uniform confidence instead of learned
        config.use_uniform_confidence = True
        # Disable confidence-related losses (meaningless with uniform conf)
        config.training.lambda_conf_reg = 0.0
        # Keep other losses as-is

    elif ablation_type == "weak_grounding":
        # Reduced grounding losses with asymmetric consistency
        config.training.lambda_local = 0.1          # was 0.5
        config.training.lambda_consist_brep = 0.08  # B-Rep needs more grounding
        config.training.lambda_consist_pc = 0.02    # PC already multimodal
        config.training.lambda_diverse = 0.02       # was 0.2

    elif ablation_type == "asymmetric_grounding":
        # Only asymmetric consistency - keep other losses at default
        config.training.lambda_consist_brep = 0.08  # B-Rep needs more grounding
        config.training.lambda_consist_pc = 0.02    # PC already multimodal

    # Apply ablation training schedule (35 epochs: 15 stage1 + 20 stage2)
    config.training.num_epochs_stage1 = 15
    config.training.num_epochs_stage2 = 20

    # Apply custom overrides if provided
    if custom_overrides:
        config = OmegaConf.merge(config, OmegaConf.create(custom_overrides))

    # Add ablation metadata
    config.ablation_type = ablation_type
    config.ablation_description = ABLATION_TYPES[ablation_type]

    return config


def print_ablation_diff(config: DictConfig, ablation_type: str):
    """Print the key differences from baseline for this ablation."""
    print(f"\n{'='*50}")
    print(f"Ablation: {ablation_type}")
    print(f"Description: {ABLATION_TYPES.get(ablation_type, 'Unknown')}")
    print(f"{'='*50}")

    if ablation_type == "baseline":
        print("  No modifications (full model)")

    elif ablation_type == "no_consistency":
        print(f"  lambda_consist_brep: 0.0 (was 0.08)")
        print(f"  lambda_consist_pc: 0.0 (was 0.02)")

    elif ablation_type == "global_only":
        print(f"  lambda_local: 0.0 (was 0.5)")
        print(f"  lambda_consist_brep: 0.0 (was 0.08)")
        print(f"  lambda_consist_pc: 0.0 (was 0.02)")
        print(f"  lambda_diverse: 0.0 (was 0.2)")

    elif ablation_type == "no_confidence":
        print(f"  use_uniform_confidence: True")
        print(f"  lambda_conf_reg: 0.0 (was 0.1)")

    elif ablation_type == "weak_grounding":
        print(f"  lambda_local: 0.1 (was 0.5)")
        print(f"  lambda_consist_brep: 0.08")
        print(f"  lambda_consist_pc: 0.02")
        print(f"  lambda_diverse: 0.02 (was 0.2)")

    elif ablation_type == "asymmetric_grounding":
        print(f"  lambda_consist_brep: 0.08 (B-Rep needs stronger grounding)")
        print(f"  lambda_consist_pc: 0.02 (PC has multimodal features)")

    print(f"\nTraining schedule:")
    print(f"  Stage 1: {config.training.num_epochs_stage1} epochs")
    print(f"  Stage 2: {config.training.num_epochs_stage2} epochs")
    print(f"  Total: {config.training.num_epochs_stage1 + config.training.num_epochs_stage2} epochs")
    print()


def get_loss_weights(config: DictConfig) -> Dict[str, float]:
    """Extract loss weights from config for display."""
    return {
        "lambda_global": config.training.lambda_global,
        "lambda_local": config.training.lambda_local,
        "lambda_consist_brep": config.training.lambda_consist_brep,
        "lambda_consist_pc": config.training.lambda_consist_pc,
        "lambda_diverse": config.training.lambda_diverse,
        "lambda_conf_reg": config.training.lambda_conf_reg,
        "lambda_global_stage1": config.training.lambda_global_stage1,
    }
