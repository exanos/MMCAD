"""
Training Pipeline for CLIP4CAD

Two-stage training:
- Stage 1: Global alignment + reconstruction
- Stage 2: Add local contrastive alignment

Features:
- Frequent checkpointing (configurable)
- GPU memory management (cache clearing after each epoch)
- Gradient checkpointing support for reduced memory usage
- Mixed precision training with automatic scaling
"""

import gc
import time
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig

from ..models.clip4cad_h import CLIP4CAD_H
from ..losses.combined import CLIP4CADLoss
from ..utils.logging_utils import AverageMeter, MetricTracker
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.misc import move_to_device


class CLIP4CADTrainer:
    """
    Trainer for CLIP4CAD-H model.

    Handles:
    - Two-stage training
    - Mixed precision
    - Gradient accumulation
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: CLIP4CAD_H,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: DictConfig,
        device: torch.device,
        logger: Any = None,
    ):
        """
        Args:
            model: CLIP4CAD-H model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Training device
            logger: Logger instance (WandB, TensorBoard, etc.)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger

        # Training settings
        self.epochs = config.epochs
        self.stage1_epochs = config.stage1_epochs
        self.gradient_accumulation = config.gradient_accumulation

        # Loss function
        self.loss_fn = CLIP4CADLoss(
            lambda_global=config.loss.lambda_global,
            lambda_local=0.0,  # Start with 0, enable in stage 2
            lambda_recon=config.loss.lambda_recon,
            confidence_threshold=config.loss.confidence_threshold,
        )

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=tuple(config.optimizer.betas),
            eps=config.optimizer.get("eps", 1e-8),
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=config.scheduler.min_lr,
        )

        # Mixed precision
        self.use_amp = config.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.max_grad_norm = config.max_grad_norm

        # Checkpointing
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config.save_every
        self.eval_every = config.eval_every

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train(self):
        """Run full training loop."""
        print(f"Starting training for {self.epochs} epochs")
        print(f"Stage 1 (global + recon): epochs 1-{self.stage1_epochs}")
        print(f"Stage 2 (+ local): epochs {self.stage1_epochs + 1}-{self.epochs}")
        print(f"Checkpoint every: {self.save_every} epochs")

        # Enable gradient checkpointing if requested
        if self.config.get("gradient_checkpointing", False):
            if hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                print("Gradient checkpointing enabled")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Update loss weights for stage 2
            if epoch == self.stage1_epochs:
                print("\n=== Entering Stage 2: Enabling local contrastive loss ===\n")
                self.loss_fn.set_local_weight(self.config.loss.lambda_local)
                # Optionally reduce reconstruction weight
                self.loss_fn.set_recon_weight(self.config.loss.lambda_recon * 0.5)

            # Train epoch
            train_metrics = self.train_epoch()

            # Update scheduler
            self.scheduler.step()

            # Clear GPU cache after each epoch to prevent memory fragmentation
            if self.config.get("empty_cache_every_epoch", True) and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Validation
            val_metrics = None
            if self.val_loader is not None and (epoch + 1) % self.eval_every == 0:
                val_metrics = self.validate()

                # Check for best model
                if val_metrics["total"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total"]
                    self.save_checkpoint(is_best=True)

            # Logging
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint()
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Metrics
        loss_meter = AverageMeter("loss")
        metric_names = ["global_contrastive", "local_contrastive", "reconstruction"]
        metrics = MetricTracker(metric_names)

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
            leave=True,
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # Forward
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss, loss_dict = self.loss_fn(outputs, batch)
                    loss = loss / self.gradient_accumulation
            else:
                outputs = self.model(batch)
                loss, loss_dict = self.loss_fn(outputs, batch)
                loss = loss / self.gradient_accumulation

            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Update metrics
            loss_meter.update(loss_dict["total"])
            metrics.update(loss_dict)

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

            # Periodic logging
            if self.logger is not None and self.global_step % 100 == 0:
                self._log_step(loss_dict)

        return metrics.get_averages()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        loss_meter = AverageMeter("val_loss")
        metric_names = ["global_contrastive", "local_contrastive", "reconstruction"]
        metrics = MetricTracker(metric_names)

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            outputs = self.model(batch)
            loss, loss_dict = self.loss_fn(outputs, batch)

            loss_meter.update(loss_dict["total"])
            metrics.update(loss_dict)

        avg_metrics = metrics.get_averages()
        avg_metrics["total"] = loss_meter.avg

        print(f"Validation - Loss: {loss_meter.avg:.4f}")

        return avg_metrics

    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        save_path = self.output_dir / "checkpoints" / f"epoch_{self.current_epoch + 1}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            metrics={"best_val_loss": self.best_val_loss},
            config=self.config,
            save_path=save_path,
            is_best=is_best,
        )

        # Always save a "latest" checkpoint for easy resumption
        latest_path = self.output_dir / "checkpoints" / "latest.pt"
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            metrics={"best_val_loss": self.best_val_loss},
            config=self.config,
            save_path=latest_path,
            is_best=False,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )

        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("step", 0)
        self.best_val_loss = checkpoint.get("metrics", {}).get(
            "best_val_loss", float("inf")
        )

        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")

    def _log_step(self, loss_dict: Dict[str, float]):
        """Log step metrics."""
        if self.logger is None:
            return

        try:
            import wandb

            if isinstance(self.logger, wandb.wandb_run.Run):
                self.logger.log(
                    {f"train/{k}": v for k, v in loss_dict.items()},
                    step=self.global_step,
                )
        except Exception:
            pass

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ):
        """Log epoch metrics."""
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss = {train_metrics.get('total', 0):.4f}, "
            f"LR = {self.scheduler.get_last_lr()[0]:.2e}"
        )

        if val_metrics is not None:
            print(f"  Val Loss = {val_metrics['total']:.4f}")

        if self.logger is None:
            return

        try:
            import wandb

            if isinstance(self.logger, wandb.wandb_run.Run):
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics.get("total", 0),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if val_metrics is not None:
                    log_dict["val/loss"] = val_metrics["total"]
                self.logger.log(log_dict)
        except Exception:
            pass
