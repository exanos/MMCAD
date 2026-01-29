"""
GFA Trainer: Two-Stage Training Pipeline for CLIP4CAD-GFA

Stage 1 (Epochs 1-30): Grounding Establishment
- Focus on learning meaningful text-geometry correspondences
- Reduced global contrastive loss weight
- Random batch sampling

Stage 2 (Epochs 31-70): Global Alignment with Hard Negatives
- Full contrastive learning with all losses
- Hard negative mining at stage transition
- Hard negative batch construction

Features:
- Frequent checkpointing (configurable, default every 5 epochs)
- GPU memory management (cache clearing after each epoch)
- Gradient checkpointing support for reduced memory usage
- Mixed precision training with automatic scaling
"""

import gc
import math
import time
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json

from ..models.clip4cad_gfa import CLIP4CAD_GFA
from ..losses.gfa_losses import GFALoss
from ..data.gfa_dataset import GFADataset, create_gfa_dataloader
from .hard_negative_mining import HardNegativeMiner, construct_hard_negative_batch


class GFATrainer:
    """
    Two-stage trainer for CLIP4CAD-GFA.

    Implements the training strategy from the GFA architecture spec:
    - Stage 1: Grounding establishment (reduced global loss)
    - Stage 2: Full alignment with hard negatives
    """

    def __init__(
        self,
        model: CLIP4CAD_GFA,
        train_dataset: GFADataset,
        val_dataset: Optional[GFADataset] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs/gfa",
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default config
        self.config = {
            # Training stages
            "num_epochs_stage1": 30,
            "num_epochs_stage2": 40,

            # Optimization
            "batch_size": 64,
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            "warmup_epochs": 3,
            "min_lr": 1e-6,
            "max_grad_norm": 1.0,

            # Stage 2 LR reduction
            "stage2_lr_factor": 0.5,

            # Loss weights
            "lambda_global": 1.0,
            "lambda_local": 0.5,
            "lambda_consist": 0.5,
            "lambda_diverse": 0.2,
            "lambda_conf_reg": 0.1,
            "lambda_global_stage1": 0.2,

            # Hard negative mining
            "hard_neg_k": 20,
            "hard_neg_text_threshold": 0.8,
            "hard_neg_num_seeds": 16,
            "hard_neg_per_seed": 3,

            # Data loading
            "num_workers": 4,
            "pin_memory": True,

            # Logging and checkpointing
            "log_every": 100,
            "save_every": 5,  # Save checkpoint every N epochs
            "save_every_steps": 0,  # Save checkpoint every N steps (0 = disabled)
            "use_wandb": False,

            # Hardware optimization
            "gradient_checkpointing": False,  # Enable to reduce memory usage
            "empty_cache_every_epoch": True,  # Clear GPU cache after each epoch
        }
        if config:
            self.config.update(config)

        # Initialize components
        self._init_optimizer()
        self._init_loss()
        self._init_dataloader()
        self._init_scaler()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.current_stage = 1
        self.hard_neg_dict = None
        self.best_val_loss = float("inf")

        # Hard negative miner (initialized when entering stage 2)
        self.hard_neg_miner = None

    def _init_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            betas=(0.9, 0.999),
        )

    def _init_loss(self):
        """Initialize loss function."""
        self.criterion = GFALoss(
            lambda_global=self.config["lambda_global"],
            lambda_local=self.config["lambda_local"],
            lambda_consist=self.config["lambda_consist"],
            lambda_diverse=self.config["lambda_diverse"],
            lambda_conf_reg=self.config["lambda_conf_reg"],
            lambda_global_stage1=self.config["lambda_global_stage1"],
        )

    def _init_dataloader(self):
        """Initialize data loaders."""
        # Get tokenizer from model if using live text encoding
        tokenizer = getattr(self.model, 'tokenizer', None)

        self.train_loader = create_gfa_dataloader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            drop_last=True,
            tokenizer=tokenizer,
        )

        if self.val_dataset:
            self.val_loader = create_gfa_dataloader(
                self.val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.config["num_workers"],
                pin_memory=self.config["pin_memory"],
                tokenizer=tokenizer,
            )
        else:
            self.val_loader = None

    def _init_scaler(self):
        """Initialize gradient scaler for mixed precision."""
        self.scaler = GradScaler()

    def _get_lr_scheduler(self, total_epochs: int) -> LambdaLR:
        """Create learning rate scheduler with warmup and cosine decay."""
        warmup_steps = self.config["warmup_epochs"] * len(self.train_loader)
        total_steps = total_epochs * len(self.train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(
                self.config["min_lr"] / self.config["learning_rate"],
                0.5 * (1 + math.cos(math.pi * progress))
            )

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Run full training (both stages)."""
        total_epochs = self.config["num_epochs_stage1"] + self.config["num_epochs_stage2"]

        # Initialize scheduler
        self.scheduler = self._get_lr_scheduler(total_epochs)

        # Enable gradient checkpointing if requested (saves memory)
        if self.config.get("gradient_checkpointing", False):
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                print("Gradient checkpointing enabled")

        print("=" * 60)
        print("CLIP4CAD-GFA Training")
        print("=" * 60)
        print(f"Total epochs: {total_epochs}")
        print(f"Stage 1: {self.config['num_epochs_stage1']} epochs (grounding)")
        print(f"Stage 2: {self.config['num_epochs_stage2']} epochs (alignment)")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Checkpoint every: {self.config['save_every']} epochs")
        print(f"Trainable parameters: {self.model.count_parameters(trainable_only=True):,}")

        # Resume from current_epoch if set by resume()
        start_epoch = self.current_epoch
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch + 1}")
        print("=" * 60 + "\n")

        for epoch in range(start_epoch, total_epochs):
            self.current_epoch = epoch + 1

            # Check for stage transition
            if epoch == self.config["num_epochs_stage1"]:
                self._transition_to_stage2()

            # Train one epoch
            train_metrics = self._train_epoch()

            # Clear GPU cache after each epoch to prevent memory fragmentation
            if self.config.get("empty_cache_every_epoch", True) and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Validation
            val_metrics = None
            if self.val_loader and (epoch + 1) % 5 == 0:
                val_metrics = self._validate()

            # Save checkpoint
            if (epoch + 1) % self.config["save_every"] == 0:
                self._save_checkpoint(epoch + 1, val_metrics)

            # Log epoch summary
            self._log_epoch(epoch + 1, train_metrics, val_metrics)

        # Save final model
        self._save_final()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

    def _transition_to_stage2(self):
        """Handle transition from stage 1 to stage 2."""
        # Save Stage 1 final checkpoint (for ablation studies)
        stage1_path = self.output_dir / "checkpoint_stage1_final.pt"
        torch.save({
            "epoch": self.current_epoch,
            "stage": 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, stage1_path)
        print(f"\n✓ Saved Stage 1 final checkpoint: {stage1_path}")

        print("\n" + "=" * 60)
        print("Transitioning to Stage 2")
        print("=" * 60)

        self.current_stage = 2

        # Mine hard negatives
        self.hard_neg_miner = HardNegativeMiner(
            model=self.model,
            train_dataloader=self.train_loader,
            cache_dir=str(self.output_dir / "hard_negatives"),
            k=self.config["hard_neg_k"],
            text_sim_threshold=self.config["hard_neg_text_threshold"],
            device=self.device,
        )
        self.hard_neg_dict = self.hard_neg_miner.mine(epoch=self.config["num_epochs_stage1"])

        # Recreate dataloader with hard negative sampling
        tokenizer = getattr(self.model, 'tokenizer', None)
        self.train_loader = create_gfa_dataloader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            drop_last=True,
            hard_neg_dict=self.hard_neg_dict,
            tokenizer=tokenizer,
        )

        # Reduce learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.config["stage2_lr_factor"]

        print(f"Learning rate reduced to: {self.optimizer.param_groups[0]['lr']:.2e}")
        print("=" * 60 + "\n")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            "total": 0.0,
            "global": 0.0,
            "local": 0.0,
            "consistency": 0.0,
            "diversity": 0.0,
            "conf_reg": 0.0,
            "conf_floor": 0.0,
        }
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} (Stage {self.current_stage})"
        )

        for batch_idx, batch in enumerate(pbar):
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(batch)
                loss, loss_dict = self.criterion(outputs, stage=self.current_stage)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["max_grad_norm"]
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.global_step += 1

            # Accumulate losses
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            num_batches += 1

            # Update progress bar (shortened keys to fit)
            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.3f}",
                "G": f"{loss_dict['global']:.3f}",
                "L": f"{loss_dict['local']:.3f}",
                "C": f"{loss_dict['consistency']:.3f}",
                "D": f"{loss_dict['diversity']:.3f}",
                "cf": f"{loss_dict.get('conf_floor', 0):.3f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.1e}",
            })

            # Monitor confidence every 100 batches (detect collapse early)
            if batch_idx % 100 == 0:
                conf = outputs["confidence"]
                num_active = outputs.get("num_active_slots", 0)
                print(f"\n  [Conf] min={conf.min():.3f} max={conf.max():.3f} mean={conf.mean():.3f} active={num_active:.1f}/12")

            # Periodic logging
            if self.global_step % self.config["log_every"] == 0:
                self._log_step(loss_dict)

            # Step-level checkpointing (if enabled)
            save_every_steps = self.config.get("save_every_steps", 0)
            if save_every_steps > 0 and self.global_step % save_every_steps == 0:
                pbar.set_postfix({"status": "saving checkpoint..."})
                self._save_checkpoint(self.current_epoch, is_step_checkpoint=True)
                pbar.set_postfix({
                    "loss": f"{loss_dict['total']:.4f}",
                    "saved": f"step {self.global_step}",
                })

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        val_losses = {
            "total": 0.0,
            "global": 0.0,
            "local": 0.0,
            "consistency": 0.0,
            "diversity": 0.0,
            "conf_floor": 0.0,
        }
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            outputs = self.model(batch)
            loss, loss_dict = self.criterion(outputs, stage=self.current_stage)

            for key in val_losses:
                if key in loss_dict:
                    val_losses[key] += loss_dict[key]
            num_batches += 1

        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        # Track best
        if val_losses["total"] < self.best_val_loss:
            self.best_val_loss = val_losses["total"]
            self._save_checkpoint(self.current_epoch, val_losses, is_best=True)

        return val_losses

    def _log_step(self, loss_dict: Dict[str, float]):
        """Log step metrics (for wandb or console)."""
        if self.config["use_wandb"]:
            try:
                import wandb
                wandb.log({
                    "train/loss": loss_dict["total"],
                    "train/global_loss": loss_dict["global"],
                    "train/local_loss": loss_dict["local"],
                    "train/consistency_loss": loss_dict["consistency"],
                    "train/diversity_loss": loss_dict["diversity"],
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "step": self.global_step,
                })
            except ImportError:
                pass

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log epoch summary."""
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['total']:.4f}")
        print(f"    Global: {train_metrics['global']:.4f}")
        print(f"    Local: {train_metrics['local']:.4f}")
        print(f"    Consistency: {train_metrics['consistency']:.4f}")
        print(f"    Diversity: {train_metrics['diversity']:.4f}")
        print(f"    Conf Floor: {train_metrics.get('conf_floor', 0):.4f}")

        if val_metrics:
            print(f"  Val Loss: {val_metrics['total']:.4f}")

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        is_step_checkpoint: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "stage": self.current_stage,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "val_metrics": val_metrics,
            "best_val_loss": self.best_val_loss,
        }

        # Save checkpoint (step-level or epoch-level)
        if is_step_checkpoint:
            ckpt_path = self.output_dir / f"checkpoint_step{self.global_step}.pt"
        else:
            ckpt_path = self.output_dir / f"checkpoint_epoch{epoch}.pt"

        torch.save(checkpoint, ckpt_path)

        # Always save a "latest" checkpoint for easy resumption
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best checkpoint (val_loss={val_metrics['total']:.4f})")

    def _save_final(self):
        """Save final model."""
        final_path = self.output_dir / "clip4cad_gfa_final.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }, final_path)
        print(f"Final model saved: {final_path}")

    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        print(f"Resuming from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.current_stage = checkpoint["stage"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        # Reinitialize scheduler
        total_epochs = self.config["num_epochs_stage1"] + self.config["num_epochs_stage2"]
        self.scheduler = self._get_lr_scheduler(total_epochs)
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load hard negatives if in stage 2
        if self.current_stage == 2:
            hard_neg_path = self.output_dir / "hard_negatives" / "hard_negatives.json"
            if hard_neg_path.exists():
                from .hard_negative_mining import load_hard_negatives
                self.hard_neg_dict = load_hard_negatives(str(hard_neg_path))

        print(f"Resumed at epoch {self.current_epoch}, stage {self.current_stage}")


def train_gfa(
    config_path: str,
    data_root: str,
    output_dir: str,
    device: str = "cuda",
    resume_from: Optional[str] = None,
):
    """
    Main training entry point.

    Args:
        config_path: Path to model configuration YAML
        data_root: Root directory of dataset
        output_dir: Output directory for checkpoints
        device: Device to train on
        resume_from: Optional checkpoint path to resume from
    """
    from omegaconf import OmegaConf

    # Load config
    config = OmegaConf.load(config_path)

    # Create model
    model = CLIP4CAD_GFA(config)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Trainable parameters: {model.count_parameters(trainable_only=True):,}")

    # Create datasets
    train_dataset = GFADataset(
        data_root=data_root,
        split="train",
        num_rotations=config.get("num_rotations", 8),
        use_single_rotation_cache=True,
    )

    val_dataset = GFADataset(
        data_root=data_root,
        split="val",
        num_rotations=1,
        use_single_rotation_cache=True,
    )

    # Create trainer
    trainer = GFATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=OmegaConf.to_container(config.get("training", {})),
        output_dir=output_dir,
        device=device,
    )

    # Resume if specified
    if resume_from:
        trainer.resume(resume_from)

    # Train
    trainer.train()
