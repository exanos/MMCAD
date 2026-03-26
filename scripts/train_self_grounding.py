#!/usr/bin/env python3
"""
Phase 2 Self-Grounding Training Script

This script trains self-grounding queries on a frozen Phase 1 model.

Usage:
    python scripts/train_self_grounding.py \
        --phase1-checkpoint outputs/ablations/asymmetric_grounding/checkpoint_epoch35.pt \
        --output-dir outputs/ablations/asymmetric_grounding \
        --num-epochs 15

The script will:
1. Load the Phase 1 checkpoint
2. Freeze all parameters except self_ground_queries
3. Train self_ground_queries to match text-grounded embeddings
4. Save checkpoint_phase2_final.pt in the output directory
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from clip4cad.models.clip4cad_gfa import CLIP4CAD_GFA
from clip4cad.data.gfa_dataset import GFAMappedDataset, gfa_collate_fn
from clip4cad.training.self_grounding_trainer import SelfGroundingTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2 Self-Grounding Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        required=True,
        help="Path to Phase 1 checkpoint (e.g., checkpoint_epoch35.pt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for Phase 2 checkpoints"
    )

    # Data paths
    parser.add_argument(
        "--data-root",
        type=str,
        default="d:/Defect_Det/MMCAD/data",
        help="Root directory for data splits"
    )
    parser.add_argument(
        "--pc-file",
        type=str,
        default="c:/Users/User/Desktop/pc_embeddings_full.h5",
        help="Path to point cloud embeddings HDF5"
    )
    parser.add_argument(
        "--brep-file",
        type=str,
        default="c:/Users/User/Desktop/brep_features.h5",
        help="Path to B-Rep features HDF5"
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default="c:/Users/User/Desktop/text_embeddings.h5",
        help="Path to text embeddings HDF5"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/model/clip4cad_gfa.yaml",
        help="Path to model config YAML"
    )

    # Training hyperparameters
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=15,
        help="Number of Phase 2 epochs (default: 15)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3, higher than Phase 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size (default: 512)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0 for memory-mapped data)"
    )

    # Optional
    parser.add_argument(
        "--validate-every",
        type=int,
        default=5,
        help="Validate every N epochs (default: 5)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PHASE 2: SELF-GROUNDING TRAINING")
    print("=" * 70)
    print(f"Phase 1 checkpoint: {args.phase1_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    # Load Phase 1 checkpoint
    print("\nLoading Phase 1 checkpoint...")
    ckpt_path = Path(args.phase1_checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    phase1_epoch = ckpt.get("epoch", "?")
    phase1_stage = ckpt.get("stage", "?")
    print(f"  Loaded epoch {phase1_epoch}, stage {phase1_stage}")

    # Load config
    config_path = Path(args.config_path)
    if not config_path.exists():
        config_path = project_root / args.config_path
    config = OmegaConf.load(config_path)

    # Create model and load weights
    print("\nCreating model...")
    model = CLIP4CAD_GFA(config)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys (random init): {len(missing)}")
        for key in missing[:3]:
            print(f"    - {key}")
        if len(missing) > 3:
            print(f"    ... and {len(missing) - 3} more")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = GFAMappedDataset(
        data_root=args.data_root,
        split="train",
        pc_file=args.pc_file,
        text_file=args.text_file,
        brep_file=args.brep_file,
        num_rotations=1,
        load_to_memory=False,  # Keep on disk for Phase 2
    )
    print(f"  Train: {len(train_dataset):,} samples")

    val_dataset = GFAMappedDataset(
        data_root=args.data_root,
        split="val",
        pc_file=args.pc_file,
        text_file=args.text_file,
        brep_file=args.brep_file,
        num_rotations=1,
        load_to_memory=False,
    )
    print(f"  Val: {len(val_dataset):,} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=gfa_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=gfa_collate_fn,
    )

    # Create trainer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SelfGroundingTrainer(
        model=model,
        config=OmegaConf.to_container(config),
        device=device,
        output_dir=str(output_dir),
        use_amp=not args.no_amp,
    )

    # Train Phase 2
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        validate_every=args.validate_every,
        save_every=args.save_every,
    )

    print("\n" + "=" * 70)
    print("PHASE 2 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final checkpoint: {output_dir / 'checkpoint_phase2_final.pt'}")

    # Print summary
    if history["val_brep_mean"]:
        print(f"\nFinal B-Rep alignment: {history['val_brep_mean'][-1]:.4f}")
    if history["val_pc_mean"]:
        print(f"Final PC alignment: {history['val_pc_mean'][-1]:.4f}")


if __name__ == "__main__":
    main()
