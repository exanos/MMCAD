#!/usr/bin/env python3
"""
Training script for CLIP4CAD-H.

Usage:
    python scripts/train.py
    python scripts/train.py training.epochs=50 data.batch_size=16
    python scripts/train.py --config-name=custom_config
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from clip4cad.models.clip4cad_h import CLIP4CAD_H
from clip4cad.data.dataset import MMCADDataset, create_dataloader
from clip4cad.training.trainer import CLIP4CADTrainer
from clip4cad.utils.misc import set_seed, count_parameters, format_number, get_device
from clip4cad.utils.config import save_config


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(config: DictConfig):
    """Main training entry point."""
    print("=" * 60)
    print("CLIP4CAD-H Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(config))
    print("=" * 60)

    # Setup
    set_seed(config.experiment.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Output directory
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config
    save_config(config, output_dir / "config.yaml")

    # ============================================================
    # Model
    # ============================================================

    print("\nBuilding model...")
    model = CLIP4CAD_H(config.model)

    # Load LLM for text encoding
    print("Loading LLM for text encoding...")
    model.load_llm(device)

    # Move to device
    model = model.to(device)

    # Count parameters
    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)
    print(f"Total parameters: {format_number(total_params)}")
    print(f"Trainable parameters: {format_number(trainable_params)}")

    # ============================================================
    # Data
    # ============================================================

    print("\nLoading data...")
    tokenizer = model.get_tokenizer()

    train_dataset = MMCADDataset(
        data_root=config.data.data_root,
        split="train",
        tokenizer=tokenizer,
        max_faces=config.data.max_faces,
        max_edges=config.data.max_edges,
        num_points=config.data.num_points,
        max_title_len=config.data.max_title_len,
        max_desc_len=config.data.max_desc_len,
        rotation_augment=config.data.rotation_augment,
        point_jitter=config.data.get("point_jitter", 0.0),
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last,
    )

    # Validation data (optional)
    val_loader = None
    val_split_file = Path(config.data.data_root) / "splits" / "val.txt"
    if val_split_file.exists():
        val_dataset = MMCADDataset(
            data_root=config.data.data_root,
            split="val",
            tokenizer=tokenizer,
            max_faces=config.data.max_faces,
            max_edges=config.data.max_edges,
            num_points=config.data.num_points,
            max_title_len=config.data.max_title_len,
            max_desc_len=config.data.max_desc_len,
            rotation_augment=False,
            point_jitter=0.0,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        print(f"Validation samples: {len(val_dataset)}")

    print(f"Training samples: {len(train_dataset)}")

    # ============================================================
    # Logger
    # ============================================================

    logger = None
    if config.logging.use_wandb:
        try:
            import wandb

            logger = wandb.init(
                project=config.logging.wandb_project,
                name=config.experiment.name,
                config=OmegaConf.to_container(config),
            )
            print("WandB logging enabled")
        except Exception as e:
            print(f"WandB initialization failed: {e}")
            logger = None

    # ============================================================
    # Training
    # ============================================================

    # Merge training config
    train_config = OmegaConf.create({
        **OmegaConf.to_container(config.training),
        "output_dir": str(output_dir),
    })

    trainer = CLIP4CADTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        logger=logger,
    )

    # Resume if checkpoint exists
    latest_ckpt = output_dir / "checkpoints" / "latest.pt"
    if latest_ckpt.exists():
        print(f"Resuming from {latest_ckpt}")
        trainer.load_checkpoint(str(latest_ckpt))

    # Train
    trainer.train()

    # Cleanup
    if logger is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    print("\nTraining complete!")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
