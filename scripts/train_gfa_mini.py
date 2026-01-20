#!/usr/bin/env python
"""
Mini training script for CLIP4CAD-GFA with ShapeLLM 10% subset.

This is a simplified training script for debugging and testing the
ShapeLLM integration before scaling to the full dataset.

Features:
- Small batch sizes (8-16)
- Few epochs (5-10)
- Verbose logging every batch
- Validation every epoch
- Gradient checking

Usage:
    python scripts/train_gfa_mini.py \
        --config configs/model/clip4cad_gfa.yaml \
        --data-dir ../data \
        --epochs 5 \
        --batch-size 8 \
        --output-dir outputs/mini_run
"""

import argparse
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from omegaconf import OmegaConf

from clip4cad.models.clip4cad_gfa import CLIP4CAD_GFA
from clip4cad.data.gfa_dataset import GFADataset, gfa_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mini training for CLIP4CAD-GFA with ShapeLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/clip4cad_gfa.yaml",
        help="Path to model configuration YAML",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mini_run",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to use (for quick debugging)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log every N batches",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="splits_10pct",
        help="Splits directory name (relative to data-dir)",
    )

    return parser.parse_args()


def simple_contrastive_loss(z1, z2, temperature=0.07):
    """Simple InfoNCE loss for debugging."""
    z1 = nn.functional.normalize(z1, dim=-1)
    z2 = nn.functional.normalize(z2, dim=-1)

    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)

    loss_12 = nn.functional.cross_entropy(logits, labels)
    loss_21 = nn.functional.cross_entropy(logits.t(), labels)

    return (loss_12 + loss_21) / 2


def train_one_epoch(model, dataloader, optimizer, device, epoch, log_every=1):
    """Train for one epoch with verbose logging."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Simple loss: just use global contrastive between modalities
        loss = 0
        loss_parts = []

        # Text-PC loss
        if "z_text" in outputs and "z_pc" in outputs:
            loss_text_pc = simple_contrastive_loss(
                outputs["z_text"], outputs["z_pc"],
                temperature=outputs["temperature"].item()
            )
            loss += loss_text_pc
            loss_parts.append(f"t-pc:{loss_text_pc.item():.4f}")

        # Text-BRep loss
        if "z_text" in outputs and "z_brep" in outputs:
            loss_text_brep = simple_contrastive_loss(
                outputs["z_text"], outputs["z_brep"],
                temperature=outputs["temperature"].item()
            )
            loss += loss_text_brep
            loss_parts.append(f"t-brep:{loss_text_brep.item():.4f}")

        # PC-BRep loss
        if "z_pc" in outputs and "z_brep" in outputs:
            loss_pc_brep = simple_contrastive_loss(
                outputs["z_pc"], outputs["z_brep"],
                temperature=outputs["temperature"].item()
            )
            loss += loss_pc_brep
            loss_parts.append(f"pc-brep:{loss_pc_brep.item():.4f}")

        if loss == 0:
            print(f"  Batch {batch_idx}: No valid modalities found!")
            continue

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Logging
        if batch_idx % log_every == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"loss={loss.item():.4f} [{', '.join(loss_parts)}]")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device):
    """Simple validation."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        outputs = model(batch)

        loss = 0
        if "z_text" in outputs and "z_pc" in outputs:
            loss += simple_contrastive_loss(
                outputs["z_text"], outputs["z_pc"],
                temperature=outputs["temperature"].item()
            )
        if "z_text" in outputs and "z_brep" in outputs:
            loss += simple_contrastive_loss(
                outputs["z_text"], outputs["z_brep"],
                temperature=outputs["temperature"].item()
            )

        if loss > 0:
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def verify_gradients(model, batch):
    """Verify gradients flow correctly."""
    print("\nVerifying gradient flow...")

    model.train()
    model.zero_grad()

    outputs = model(batch)

    loss = 0
    if "z_text" in outputs and "z_pc" in outputs:
        loss += simple_contrastive_loss(outputs["z_text"], outputs["z_pc"])

    if loss > 0:
        loss.backward()

        # Check gradients
        params_with_grad = 0
        params_without_grad = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    params_with_grad += 1
                else:
                    params_without_grad += 1
                    print(f"  WARNING: No gradient for {name}")

        print(f"  Parameters with gradients: {params_with_grad}")
        print(f"  Parameters without gradients: {params_without_grad}")

        return params_without_grad == 0

    print("  No loss computed for gradient check")
    return False


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = project_root / args.config

    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = OmegaConf.load(config_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check data directory
    data_dir = Path(args.data_dir)
    splits_dir = data_dir / "splits_10pct"
    embeddings_dir = data_dir / "embeddings"

    print("=" * 60)
    print("CLIP4CAD-GFA Mini Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Data dir: {data_dir}")
    print(f"Splits dir: {splits_dir}")
    print(f"Embeddings dir: {embeddings_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Create model
    print("\nInitializing model...")
    model = CLIP4CAD_GFA(config)
    model = model.to(args.device)
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Trainable parameters: {model.count_parameters(trainable_only=True):,}")

    # Create dataset
    splits_dir = data_dir / args.splits_dir
    print(f"Splits dir: {splits_dir}")

    print("\nLoading dataset...")
    try:
        train_dataset = GFADataset(
            data_root=str(data_dir),
            split="train",
            num_rotations=1,  # No rotation augmentation
            splits_dir=str(splits_dir),
            use_single_rotation_cache=True,
        )
        print(f"Training samples: {len(train_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nMake sure you have run the data preparation scripts:")
        print("  1. python scripts/build_shapellm_uid_mapping.py ...")
        print("  2. python scripts/convert_shapellm_to_standard.py ...")
        print("  3. python scripts/generate_subset_splits.py ...")
        sys.exit(1)

    # Optionally limit samples
    if args.max_samples and args.max_samples < len(train_dataset):
        indices = list(range(args.max_samples))
        train_dataset = Subset(train_dataset, indices)
        print(f"Limited to {args.max_samples} samples")

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Simple for debugging
        collate_fn=gfa_collate_fn,
    )
    print(f"Batches per epoch: {len(train_loader)}")

    # Validation dataset
    val_loader = None
    val_split = splits_dir / "val.txt"
    if val_split.exists():
        try:
            val_dataset = GFADataset(
                data_root=str(data_dir),
                split="val",
                num_rotations=1,
                splits_dir=str(splits_dir),
                use_single_rotation_cache=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=gfa_collate_fn,
            )
            print(f"Validation samples: {len(val_dataset)}")
        except Exception as e:
            print(f"Warning: Could not load validation set: {e}")

    # Test data loading
    print("\nTesting data loading...")
    test_batch = next(iter(train_loader))
    print(f"Batch keys: {list(test_batch.keys())}")

    for key in test_batch:
        if isinstance(test_batch[key], torch.Tensor):
            print(f"  {key}: {test_batch[key].shape} ({test_batch[key].dtype})")
        else:
            print(f"  {key}: {type(test_batch[key])}")

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        try:
            outputs = model(test_batch)
            print(f"Output keys: {list(outputs.keys())}")
            for key in ["z_text", "z_pc", "z_brep"]:
                if key in outputs:
                    print(f"  {key}: {outputs[key].shape}")
        except Exception as e:
            print(f"Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Verify gradients
    if not verify_gradients(model, test_batch):
        print("\nWARNING: Some parameters have no gradients!")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, args.device,
            epoch, log_every=args.log_every
        )

        elapsed = time.time() - start_time

        print(f"Epoch {epoch + 1} complete in {elapsed:.1f}s")
        print(f"  Train loss: {train_loss:.4f}")

        # Validate
        if val_loader is not None:
            val_loss = validate(model, val_loader, args.device)
            print(f"  Val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, checkpoint_path)
                print(f"  Saved best model to {checkpoint_path}")

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
    }, final_path)
    print(f"\nSaved final model to {final_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
