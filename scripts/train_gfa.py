#!/usr/bin/env python
"""
Training script for CLIP4CAD-GFA

Usage:
    python scripts/train_gfa.py --data-root data/mmcad --output-dir outputs/gfa

    # Resume training
    python scripts/train_gfa.py --data-root data/mmcad --resume outputs/gfa/checkpoint_epoch30.pt

    # With custom config
    python scripts/train_gfa.py --config configs/model/clip4cad_gfa.yaml --data-root data/mmcad
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

from clip4cad.models.clip4cad_gfa import CLIP4CAD_GFA
from clip4cad.data.gfa_dataset import GFADataset
from clip4cad.training.gfa_trainer import GFATrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CLIP4CAD-GFA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/clip4cad_gfa.yaml",
        help="Path to model configuration YAML",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gfa",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--epochs-stage1",
        type=int,
        default=None,
        help="Override stage 1 epochs from config",
    )
    parser.add_argument(
        "--epochs-stage2",
        type=int,
        default=None,
        help="Override stage 2 epochs from config",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="clip4cad-gfa",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to project root
        config_path = project_root / args.config

    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = OmegaConf.load(config_path)

    # Override config with command line arguments
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.epochs_stage1 is not None:
        config.training.num_epochs_stage1 = args.epochs_stage1
    if args.epochs_stage2 is not None:
        config.training.num_epochs_stage2 = args.epochs_stage2
    if args.use_wandb:
        config.training.use_wandb = True

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(config, output_dir / "config.yaml")

    # Check data directory
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data directory not found: {data_root}")
        sys.exit(1)

    embeddings_dir = data_root / "embeddings"
    if not embeddings_dir.exists():
        print(f"Error: Embeddings directory not found: {embeddings_dir}")
        print("Run pre-computation scripts first:")
        print("  python scripts/precompute_brep_features.py --data-root", args.data_root)
        print("  python scripts/precompute_pointcloud_features.py --data-root", args.data_root)
        print("  python scripts/precompute_text_embeddings.py --data-root", args.data_root)
        sys.exit(1)

    print("=" * 60)
    print("CLIP4CAD-GFA Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Initialize wandb if requested
    if config.training.get("use_wandb", False):
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=OmegaConf.to_container(config),
                dir=str(output_dir),
            )
            print("Weights & Biases initialized")
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
            config.training.use_wandb = False

    # Create model
    print("\nInitializing model...")
    model = CLIP4CAD_GFA(config)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Trainable parameters: {model.count_parameters(trainable_only=True):,}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = GFADataset(
        data_root=str(data_root),
        split="train",
        num_rotations=config.get("num_rotations", 8),
        use_single_rotation_cache=True,
    )
    print(f"Training samples: {len(train_dataset)}")

    val_dataset = None
    val_split_file = data_root / "splits" / "val.txt"
    if val_split_file.exists():
        val_dataset = GFADataset(
            data_root=str(data_root),
            split="val",
            num_rotations=1,
            use_single_rotation_cache=True,
        )
        print(f"Validation samples: {len(val_dataset)}")

    # Create trainer
    trainer = GFATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=OmegaConf.to_container(config.training),
        output_dir=str(output_dir),
        device=args.device,
    )

    # Resume if specified
    if args.resume:
        if not Path(args.resume).exists():
            print(f"Error: Checkpoint not found: {args.resume}")
            sys.exit(1)
        trainer.resume(args.resume)

    # Train
    trainer.train()

    # Cleanup
    if config.training.get("use_wandb", False):
        try:
            import wandb
            wandb.finish()
        except:
            pass

    print("\nDone!")


if __name__ == "__main__":
    main()
