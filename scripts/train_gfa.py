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
from clip4cad.data.gfa_dataset import GFADataset, GFAMappedDataset
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
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Override checkpoint frequency (epochs)",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (0 = disabled)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage",
    )
    parser.add_argument(
        "--pc-file",
        type=str,
        default=None,
        help="Path to PC HDF5 file (for mapped dataset mode)",
    )
    parser.add_argument(
        "--brep-file",
        type=str,
        default=None,
        help="Path to B-Rep HDF5 file (default: data_root/embeddings/brep_features.h5)",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to text embeddings HDF5 file (default: data_root/embeddings/text_embeddings.h5)",
    )
    parser.add_argument(
        "--use-live-text",
        action="store_true",
        help="Encode text at train-time with frozen LLM instead of using pre-computed embeddings",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to CSV file with uid, title, description columns (required if --use-live-text)",
    )
    parser.add_argument(
        "--mapping-dir",
        type=str,
        default=None,
        help="Directory containing uid_mapping.json (default: data_root/aligned)",
    )
    parser.add_argument(
        "--load-to-memory",
        action="store_true",
        help="Load all data to RAM for fast training (recommended if you have enough RAM)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loading workers (default: 0 if load-to-memory, else 4)",
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
    if args.save_every is not None:
        config.training.save_every = args.save_every
    if args.save_every_steps is not None:
        config.training.save_every_steps = args.save_every_steps
    if args.gradient_checkpointing:
        config.training.gradient_checkpointing = True

    # Live text encoding settings
    if args.use_live_text:
        config.encoders.text.use_live_text = True
        config.encoders.text.use_cached_embeddings = False
        if args.csv_path:
            config.encoders.text.csv_path = args.csv_path

    # Set num_workers for parallel data loading (prefetching)
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    # Note: even with load_to_memory, text is still on disk, so workers help with prefetching

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

    # Check if we should use mapped dataset (uid_mapping.json exists)
    mapping_dir = Path(args.mapping_dir) if args.mapping_dir else data_root / "aligned"
    use_mapped = (mapping_dir / "uid_mapping.json").exists()

    if use_mapped:
        print("Using GFAMappedDataset (uid_mapping.json found)")

        # Live text encoding settings
        use_live_text = config.encoders.text.get("use_live_text", False)
        csv_path = args.csv_path or config.encoders.text.get("csv_path", None)

        if use_live_text:
            print(f"  Live text encoding enabled (CSV: {csv_path})")
        else:
            print(f"  Pre-computed text embeddings: {args.text_file or 'default'}")

        train_dataset = GFAMappedDataset(
            data_root=str(data_root),
            split="train",
            pc_file=args.pc_file,
            text_file=args.text_file,
            brep_file=args.brep_file,
            mapping_dir=str(mapping_dir),
            num_rotations=config.get("num_rotations", 1),
            load_to_memory=args.load_to_memory,
            use_live_text=use_live_text,
            csv_path=csv_path,
        )
        val_dataset = GFAMappedDataset(
            data_root=str(data_root),
            split="val",
            pc_file=args.pc_file,
            text_file=args.text_file,
            brep_file=args.brep_file,
            mapping_dir=str(mapping_dir),
            num_rotations=1,
            load_to_memory=False,  # Val stays on disk to save RAM (~36GB saved, validated once per epoch)
            use_live_text=use_live_text,
            csv_path=csv_path,
        )
    else:
        print("Using GFADataset (standard mode)")
        train_dataset = GFADataset(
            data_root=str(data_root),
            split="train",
            num_rotations=config.get("num_rotations", 8),
            use_single_rotation_cache=True,
        )
        val_dataset = None
        val_split_file = data_root / "splits" / "val.txt"
        if val_split_file.exists():
            val_dataset = GFADataset(
                data_root=str(data_root),
                split="val",
                num_rotations=1,
                use_single_rotation_cache=True,
            )

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
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
