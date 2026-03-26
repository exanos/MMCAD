#!/usr/bin/env python3
"""
Train CLIP4CAD-HUS: Hierarchical Unified Space

Two-stage training:
- Stage 1 (epochs 1-15): Hierarchy establishment
- Stage 2 (epochs 16-35): Hard negative mining at detail level

Usage:
    python scripts/train_hus.py --config configs/model/clip4cad_hus.yaml --output-dir outputs/hus
    python scripts/train_hus.py --config configs/model/clip4cad_hus.yaml --resume outputs/hus/checkpoint_latest.pt

Key differences from train_gfa.py:
- 3-term loss (unified, global, detail) vs 8-term
- Detail embeddings used for hard negative mining
- Gate value monitoring for interpretability
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

from clip4cad.models.clip4cad_hus import CLIP4CAD_HUS_v2
from clip4cad.data.gfa_dataset import GFAMappedDataset
from clip4cad.training.hus_trainer import HUSTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CLIP4CAD-HUS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/clip4cad_hus.yaml",
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/hus",
        help="Output directory for checkpoints"
    )

    # Data paths (override config)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for data splits (overrides config)"
    )
    parser.add_argument(
        "--pc-file",
        type=str,
        default=None,
        help="Path to point cloud embeddings HDF5 (overrides config)"
    )
    parser.add_argument(
        "--brep-file",
        type=str,
        default=None,
        help="Path to B-Rep features HDF5 (overrides config)"
    )
    parser.add_argument(
        "--text-dir",
        type=str,
        default=None,
        help="Path to text embeddings directory (overrides config)"
    )

    # Training options
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--load-to-memory",
        action="store_true",
        default=True,
        help="Load data to RAM for faster training (default: True)"
    )
    parser.add_argument(
        "--no-load-to-memory",
        action="store_true",
        help="Don't load data to RAM (use disk access)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = project_root / args.config
    config = OmegaConf.load(config_path)

    # Get data paths from config or args
    data_config = config.get('data', {})
    data_root = args.data_root or data_config.get('data_root', 'd:/Defect_Det/MMCAD/data')
    pc_file = args.pc_file or data_config.get('pc_file', 'c:/Users/User/Desktop/pc_embeddings_full.h5')
    brep_file = args.brep_file or data_config.get('brep_file', 'c:/Users/User/Desktop/brep_features.h5')
    text_dir = args.text_dir or data_config.get('text_files', 'c:/Users/User/Desktop/text_splits/')

    # Determine text file path (pre-split format)
    text_file = text_dir  # Will be resolved by dataset

    # Override config with command line args
    train_config = OmegaConf.to_container(config.get('training', {}))
    if args.batch_size is not None:
        train_config['batch_size'] = args.batch_size
    if args.lr is not None:
        train_config['learning_rate'] = args.lr

    # Determine load_to_memory
    load_to_memory = args.load_to_memory and not args.no_load_to_memory

    print("=" * 70)
    print("CLIP4CAD-HUS Training")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Load to memory: {load_to_memory}")
    print(f"Data root: {data_root}")
    print(f"PC file: {pc_file}")
    print(f"B-Rep file: {brep_file}")
    print(f"Text dir: {text_dir}")
    print("=" * 70)

    # Create model
    print("\nCreating model...")
    model = CLIP4CAD_HUS_v2(config)
    param_counts = model.count_parameters()
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Query bank: {param_counts.get('query_bank', 0):,}")
    print(f"  Global attention: {param_counts.get('global_attn', 0):,}")
    print(f"  Detail attention: {param_counts.get('detail_attn', 0):,}")
    print(f"  Fusion: {param_counts.get('fusion', 0):,}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = GFAMappedDataset(
        data_root=data_root,
        split="train",
        pc_file=pc_file,
        brep_file=brep_file,
        text_file=text_file,
        load_to_memory=load_to_memory,
    )
    print(f"  Train: {len(train_dataset):,} samples")

    val_dataset = GFAMappedDataset(
        data_root=data_root,
        split="val",
        pc_file=pc_file,
        brep_file=brep_file,
        text_file=text_file,
        load_to_memory=load_to_memory,
    )
    print(f"  Val: {len(val_dataset):,} samples")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = HUSTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config,
        output_dir=args.output_dir,
        device=str(device),
    )

    # Resume if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        trainer.resume(args.resume)

    # Train
    print("\nStarting training...")
    trainer.train()


if __name__ == "__main__":
    main()
