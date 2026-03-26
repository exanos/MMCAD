"""
GFA v2 Trainer: Two-Stage Training Pipeline for CLIP4CAD-GFA v2

Stage 1 (Epochs 1-15): Establish Alignment
- Focus on learning basic text-geometry alignment
- Mild self-grounding weight (lambda_self=0.3)
- No hard negatives, no detail loss
- Target: Text→BRep R@1 ≥ 50%, Self cosine ≥ 0.8

Stage 2 (Epochs 16-35): Hard Negatives + Fine-Grained
- Full self-grounding weight (lambda_self=0.5)
- Hard negative mining enabled
- Detail loss enabled (lambda_detail=0.3)
- Target: Text→BRep R@1 ≥ 55%, Self cosine ≥ 0.9

Key innovation: Joint self-grounding training
- Self-path learns via DIRECT contrastive loss (not just distillation)
- Monitor self-grounding quality (cosine similarity)

Based on CLIP4CAD_GFA_v2_Architecture.md
"""

import gc
import math
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ..models.clip4cad_gfa_v2 import CLIP4CAD_GFA_v2, GFAv2Config
from ..losses.gfa_v2_losses import GFAv2Loss, compute_self_grounding_quality


class GFAv2Trainer:
    """
    Two-stage trainer for CLIP4CAD-GFA v2.

    Implements joint self-grounding training with:
    - Stage 1: Establish alignment (mild self-grounding)
    - Stage 2: Fine-grained with hard negatives
    """

    def __init__(
        self,
        model: CLIP4CAD_GFA_v2,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs/gfa_v2",
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default config
        self.config = {
            # Training stages
            "stage1_epochs": 15,
            "stage2_epochs": 20,

            # Optimization
            "batch_size": 512,
            "stage1_lr": 3e-5,
            "stage2_lr": 1e-5,
            "weight_decay": 0.01,
            "warmup_epochs": 2,
            "min_lr": 1e-6,
            "max_grad_norm": 1.0,

            # Stage 1 loss weights
            "stage1_lambda_self": 0.3,
            "stage1_lambda_distill": 0.1,
            "stage1_lambda_detail": 0.0,

            # Stage 2 loss weights
            "stage2_lambda_self": 0.5,
            "stage2_lambda_distill": 0.2,
            "stage2_lambda_detail": 0.3,

            # Hard negative mining
            "hard_negative_k": 10,
            "mine_every_n_epochs": 5,

            # Logging and checkpointing
            "log_every": 100,
            "save_every": 5,
            "eval_every": 5,

            # Hardware optimization
            "empty_cache_every_epoch": True,
        }
        if config:
            self.config.update(config)

        # Initialize components
        self._init_optimizer()
        self._init_loss()
        self._init_scaler()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.current_stage = 1
        self.best_val_loss = float("inf")
        self.best_self_cosine = 0.0
        self.hard_neg_dict = None

    def _init_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["stage1_lr"],
            weight_decay=self.config["weight_decay"],
            betas=(0.9, 0.999),
        )

    def _init_loss(self):
        """Initialize loss function."""
        self.criterion = GFAv2Loss(
            lambda_self=self.config["stage1_lambda_self"],
            lambda_distill=self.config["stage1_lambda_distill"],
            lambda_detail=self.config["stage1_lambda_detail"],
        )

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
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(
                self.config["min_lr"] / self.config["stage1_lr"],
                0.5 * (1 + math.cos(math.pi * progress))
            )

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Run full training (both stages)."""
        total_epochs = self.config["stage1_epochs"] + self.config["stage2_epochs"]

        # Initialize scheduler
        self.scheduler = self._get_lr_scheduler(total_epochs)

        print("=" * 70)
        print("CLIP4CAD-GFA v2 Training")
        print("=" * 70)
        print(f"Total epochs: {total_epochs}")
        print(f"Stage 1: {self.config['stage1_epochs']} epochs (establish alignment)")
        print(f"Stage 2: {self.config['stage2_epochs']} epochs (hard negatives)")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Stage 1 LR: {self.config['stage1_lr']}")
        print(f"Stage 2 LR: {self.config['stage2_lr']}")
        print(f"Trainable parameters: {self.model.count_parameters(trainable_only=True):,}")
        print("=" * 70 + "\n")

        # Resume from current_epoch if set
        start_epoch = self.current_epoch

        for epoch in range(start_epoch, total_epochs):
            self.current_epoch = epoch + 1

            # Check for stage transition
            if epoch == self.config["stage1_epochs"]:
                self._transition_to_stage2()

            # Train one epoch
            train_metrics = self._train_epoch()

            # Clear GPU cache
            if self.config.get("empty_cache_every_epoch", True) and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Validation
            val_metrics = None
            if self.val_loader and (epoch + 1) % self.config["eval_every"] == 0:
                val_metrics = self._validate()

            # Save checkpoint
            if (epoch + 1) % self.config["save_every"] == 0:
                self._save_checkpoint(epoch + 1, val_metrics)

            # Log epoch summary
            self._log_epoch(epoch + 1, train_metrics, val_metrics)

        # Save final model
        self._save_final()

        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best self-grounding cosine: {self.best_self_cosine:.4f}")
        print("=" * 70)

    def _transition_to_stage2(self):
        """Handle transition from stage 1 to stage 2."""
        # Save Stage 1 final checkpoint
        stage1_path = self.output_dir / "checkpoint_stage1_final.pt"
        torch.save({
            "epoch": self.current_epoch,
            "stage": 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "best_self_cosine": self.best_self_cosine,
        }, stage1_path)
        print(f"\n✓ Saved Stage 1 final checkpoint: {stage1_path}")

        print("\n" + "=" * 70)
        print("Transitioning to Stage 2: Hard Negatives + Fine-Grained")
        print("=" * 70)

        self.current_stage = 2

        # Update loss weights for Stage 2
        self.criterion.update_weights(
            lambda_self=self.config["stage2_lambda_self"],
            lambda_distill=self.config["stage2_lambda_distill"],
            lambda_detail=self.config["stage2_lambda_detail"],
        )
        print(f"Loss weights updated:")
        print(f"  lambda_self: {self.config['stage1_lambda_self']} -> {self.config['stage2_lambda_self']}")
        print(f"  lambda_distill: {self.config['stage1_lambda_distill']} -> {self.config['stage2_lambda_distill']}")
        print(f"  lambda_detail: {self.config['stage1_lambda_detail']} -> {self.config['stage2_lambda_detail']}")

        # Reduce learning rate
        new_lr = self.config["stage2_lr"]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        print(f"Learning rate: {self.config['stage1_lr']} -> {new_lr}")

        # Mine hard negatives (if enabled)
        if self.config.get("hard_negative_k", 0) > 0:
            self._mine_hard_negatives()

        print("=" * 70 + "\n")

    def _mine_hard_negatives(self):
        """Mine hard negatives for Stage 2."""
        print("\nMining hard negatives...")

        try:
            from .hard_negative_mining import HardNegativeMiner
            self.hard_neg_miner = HardNegativeMiner(
                model=self.model,
                train_dataloader=self.train_loader,
                cache_dir=str(self.output_dir / "hard_negatives"),
                k=self.config["hard_negative_k"],
                device=self.device,
            )
            self.hard_neg_dict = self.hard_neg_miner.mine(epoch=self.config["stage1_epochs"])
            print(f"Mined hard negatives for {len(self.hard_neg_dict)} samples")
        except Exception as e:
            print(f"Hard negative mining failed: {e}")
            print("Continuing without hard negatives")
            self.hard_neg_dict = None

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            "total": 0.0,
            "guided": 0.0,
            "self": 0.0,
            "distill": 0.0,
            "detail": 0.0,
        }
        epoch_self_cosine_brep = []
        epoch_self_cosine_pc = []
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} (Stage {self.current_stage})"
        )

        for batch_idx, batch in enumerate(pbar):
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(batch)

                # Get hard negatives for this batch (if available)
                hard_negs = None
                if self.hard_neg_dict is not None and self.current_stage == 2:
                    # Construct hard negatives based on batch indices
                    batch_indices = batch.get("idx")
                    if batch_indices is not None:
                        hard_negs = [
                            self.hard_neg_dict.get(str(idx.item()), None)
                            for idx in batch_indices
                        ]

                loss, loss_dict = self.criterion(outputs, hard_negs, stage=self.current_stage)

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

            # Compute self-grounding quality
            if outputs.get('z_brep') is not None and outputs.get('z_brep_self') is not None:
                cos_brep = compute_self_grounding_quality(
                    outputs['z_brep'].detach(),
                    outputs['z_brep_self'].detach()
                )
                epoch_self_cosine_brep.append(cos_brep)

            if outputs.get('z_pc') is not None and outputs.get('z_pc_self') is not None:
                cos_pc = compute_self_grounding_quality(
                    outputs['z_pc'].detach(),
                    outputs['z_pc_self'].detach()
                )
                epoch_self_cosine_pc.append(cos_pc)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.3f}",
                "guided": f"{loss_dict['guided']:.3f}",
                "self": f"{loss_dict['self']:.3f}",
                "cos_b": f"{epoch_self_cosine_brep[-1]:.3f}" if epoch_self_cosine_brep else "N/A",
                "lr": f"{self.scheduler.get_last_lr()[0]:.1e}",
            })

            # Monitor confidence every 200 batches
            if batch_idx % 200 == 0 and batch_idx > 0:
                conf = outputs.get("confidence")
                if conf is not None:
                    print(f"\n  [Conf] mean={conf.mean():.3f} std={conf.std():.3f}")

                avg_cos_brep = sum(epoch_self_cosine_brep[-100:]) / min(len(epoch_self_cosine_brep), 100)
                print(f"  [Self-grounding] BRep cosine (last 100): {avg_cos_brep:.4f}")

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        # Average self-grounding quality
        if epoch_self_cosine_brep:
            epoch_losses["self_cosine_brep"] = sum(epoch_self_cosine_brep) / len(epoch_self_cosine_brep)
        if epoch_self_cosine_pc:
            epoch_losses["self_cosine_pc"] = sum(epoch_self_cosine_pc) / len(epoch_self_cosine_pc)

        # Track best self-grounding quality
        avg_self_cosine = epoch_losses.get("self_cosine_brep", 0)
        if avg_self_cosine > self.best_self_cosine:
            self.best_self_cosine = avg_self_cosine

        return epoch_losses

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        val_losses = {
            "total": 0.0,
            "guided": 0.0,
            "self": 0.0,
            "distill": 0.0,
        }
        val_self_cosine_brep = []
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            outputs = self.model(batch)
            loss, loss_dict = self.criterion(outputs, None, stage=self.current_stage)

            for key in val_losses:
                if key in loss_dict:
                    val_losses[key] += loss_dict[key]
            num_batches += 1

            # Self-grounding quality
            if outputs.get('z_brep') is not None and outputs.get('z_brep_self') is not None:
                cos_brep = compute_self_grounding_quality(
                    outputs['z_brep'],
                    outputs['z_brep_self']
                )
                val_self_cosine_brep.append(cos_brep)

        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        if val_self_cosine_brep:
            val_losses["self_cosine_brep"] = sum(val_self_cosine_brep) / len(val_self_cosine_brep)

        # Track best
        if val_losses["total"] < self.best_val_loss:
            self.best_val_loss = val_losses["total"]
            self._save_checkpoint(self.current_epoch, val_losses, is_best=True)

        return val_losses

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log epoch summary."""
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Summary (Stage {self.current_stage})")
        print(f"{'='*70}")
        print(f"  Train Loss: {train_metrics['total']:.4f}")
        print(f"    Guided: {train_metrics['guided']:.4f}")
        print(f"    Self: {train_metrics['self']:.4f}")
        print(f"    Distill: {train_metrics['distill']:.4f}")
        print(f"    Detail: {train_metrics.get('detail', 0):.4f}")

        if "self_cosine_brep" in train_metrics:
            print(f"  Self-grounding cosine (BRep): {train_metrics['self_cosine_brep']:.4f}")
        if "self_cosine_pc" in train_metrics:
            print(f"  Self-grounding cosine (PC): {train_metrics['self_cosine_pc']:.4f}")

        if val_metrics:
            print(f"  Val Loss: {val_metrics['total']:.4f}")
            if "self_cosine_brep" in val_metrics:
                print(f"  Val Self-grounding cosine: {val_metrics['self_cosine_brep']:.4f}")

        print(f"  Best self-grounding cosine: {self.best_self_cosine:.4f}")

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
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
            "best_self_cosine": self.best_self_cosine,
        }

        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, ckpt_path)

        # Always save latest
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best checkpoint (val_loss={val_metrics['total']:.4f})")

    def _save_final(self):
        """Save final model."""
        final_path = self.output_dir / "clip4cad_gfa_v2_final.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_self_cosine": self.best_self_cosine,
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
        self.best_self_cosine = checkpoint.get("best_self_cosine", 0.0)

        # Update loss weights if in Stage 2
        if self.current_stage == 2:
            self.criterion.update_weights(
                lambda_self=self.config["stage2_lambda_self"],
                lambda_distill=self.config["stage2_lambda_distill"],
                lambda_detail=self.config["stage2_lambda_detail"],
            )

        # Reinitialize scheduler
        total_epochs = self.config["stage1_epochs"] + self.config["stage2_epochs"]
        self.scheduler = self._get_lr_scheduler(total_epochs)
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Resumed at epoch {self.current_epoch}, stage {self.current_stage}")
        print(f"Best self-grounding cosine so far: {self.best_self_cosine:.4f}")


def train_step(
    model: CLIP4CAD_GFA_v2,
    batch: Dict[str, torch.Tensor],
    criterion: GFAv2Loss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: LambdaLR,
    max_grad_norm: float = 1.0,
    stage: int = 1,
    hard_negatives: Optional[List] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Single training step (for notebook usage).

    Returns:
        loss_dict: Loss values
        metrics: Self-grounding quality metrics
    """
    model.train()

    with autocast():
        outputs = model(batch)
        loss, loss_dict = criterion(outputs, hard_negatives, stage=stage)

    optimizer.zero_grad()
    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # Compute self-grounding quality
    metrics = {}
    if outputs.get('z_brep') is not None and outputs.get('z_brep_self') is not None:
        metrics['self_cosine_brep'] = compute_self_grounding_quality(
            outputs['z_brep'].detach(),
            outputs['z_brep_self'].detach()
        )
    if outputs.get('z_pc') is not None and outputs.get('z_pc_self') is not None:
        metrics['self_cosine_pc'] = compute_self_grounding_quality(
            outputs['z_pc'].detach(),
            outputs['z_pc_self'].detach()
        )

    return loss_dict, metrics
