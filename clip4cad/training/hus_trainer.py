"""
HUS Trainer: Two-Stage Training Pipeline for CLIP4CAD-HUS

Stage 1 (Epochs 1-15): Hierarchy Establishment
- Learn global and detail queries
- Standard InfoNCE at all levels
- Mild detail weight (0.2)

Stage 2 (Epochs 16-35): Fine-Grained Discrimination
- Hard negative mining at detail level
- Increased detail weight (1.0)
- Focus on fine-grained differences (32 vs 64 teeth)

Features:
- Simpler loss (3 terms vs GFA's 8)
- Gate value monitoring (interpretability)
- Same memory management as GFA trainer
- Mixed precision training with automatic scaling
"""

import gc
import math
import time
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json

from ..models.clip4cad_hus import CLIP4CAD_HUS_v2
from ..losses.hus_losses import HUSLoss, HUSLossConfig
from ..data.gfa_dataset import GFAMappedDataset, gfa_collate_fn, create_gfa_dataloader
from .hard_negative_mining import HardNegativeMiner, load_hard_negatives


class HUSTrainer:
    """
    Two-stage trainer for CLIP4CAD-HUS.

    Implements simplified training compared to GFA:
    - Stage 1: Hierarchy establishment (15 epochs)
    - Stage 2: Hard negatives + increased detail weight (20 epochs)
    """

    def __init__(
        self,
        model: CLIP4CAD_HUS_v2,
        train_dataset: GFAMappedDataset,
        val_dataset: Optional[GFAMappedDataset] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs/hus",
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
            "num_epochs_stage1": 15,
            "num_epochs_stage2": 20,

            # Optimization - AGGRESSIVE LR for faster convergence
            "batch_size": 64,
            "learning_rate": 3e-4,    # High LR for fast convergence
            "weight_decay": 0.01,     # Low to allow fast learning
            "warmup_epochs": 2,       # Slightly longer warmup for stability
            "min_lr": 1e-6,
            "max_grad_norm": 1.0,

            # Stage 2 LR reduction
            "stage2_lr_factor": 0.3,

            # Loss weights - UNIFIED-DOMINANT (what retrieval uses!)
            "lambda_unified": 1.0,         # PRIMARY
            "lambda_global": 0.2,          # Mild regularizer
            "lambda_detail": 0.2,          # Mild regularizer
            "label_smoothing": 0.05,       # Light smoothing
            # Stage 2 weights
            "lambda_unified_stage2": 1.0,  # Still primary
            "lambda_global_stage2": 0.1,   # Reduce
            "lambda_detail_stage2": 0.5,   # Increase for hard negatives

            # Hard negative mining
            "hard_neg_k": 20,
            "hard_neg_text_threshold": 0.8,
            "hard_neg_embedding_key": "z_brep_detail",  # Use detail embeddings!

            # Data loading
            "num_workers": 0,  # Memory-mapped dataset works best with 0
            "pin_memory": True,

            # Logging and checkpointing
            "log_every": 100,
            "save_every": 5,
            "validate_every": 5,
            "use_wandb": False,

            # Hardware optimization
            "empty_cache_every_epoch": True,
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
        # Use weights from config (unified-dominant)
        self.criterion = HUSLoss(
            lambda_unified=self.config.get("lambda_unified", 1.0),
            lambda_global=self.config.get("lambda_global", 0.2),
            lambda_detail=self.config.get("lambda_detail", 0.2),
            label_smoothing=self.config.get("label_smoothing", 0.05),
        )

    def _init_dataloader(self):
        """Initialize data loaders."""
        from functools import partial

        collate_fn = partial(gfa_collate_fn, tokenizer=None)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
            drop_last=True,
            collate_fn=collate_fn,
        )

        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.config["num_workers"],
                pin_memory=self.config["pin_memory"],
                collate_fn=collate_fn,
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

        # Count parameters
        param_counts = self.model.count_parameters()

        print("=" * 60)
        print("CLIP4CAD-HUS Training")
        print("=" * 60)
        print(f"Total epochs: {total_epochs}")
        print(f"Stage 1: {self.config['num_epochs_stage1']} epochs (hierarchy)")
        print(f"Stage 2: {self.config['num_epochs_stage2']} epochs (fine-grained)")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")
        print(f"Loss terms: 3 (unified, global, detail)")
        print(f"Output directory: {self.output_dir}")

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

            # Clear GPU cache after each epoch
            if self.config.get("empty_cache_every_epoch", True) and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Validation
            val_metrics = None
            if self.val_loader and (epoch + 1) % self.config["validate_every"] == 0:
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
        }, stage1_path)
        print(f"\n✓ Saved Stage 1 final checkpoint: {stage1_path}")

        print("\n" + "=" * 60)
        print("Transitioning to Stage 2")
        print("=" * 60)

        self.current_stage = 2

        # Update loss weights for Stage 2
        # Stage 2: Unified becomes primary, detail increased
        self.criterion.set_stage2_weights(
            lambda_unified=self.config.get("lambda_unified_stage2", 1.0),
            lambda_global=self.config.get("lambda_global_stage2", 0.5),
            lambda_detail=self.config.get("lambda_detail_stage2", 1.0),
        )
        print(f"  Loss weights updated for Stage 2:")
        print(f"    unified: {self.config.get('lambda_unified', 0.5)} -> {self.config.get('lambda_unified_stage2', 1.0)}")
        print(f"    global:  {self.config.get('lambda_global', 1.0)} -> {self.config.get('lambda_global_stage2', 0.5)}")
        print(f"    detail:  {self.config.get('lambda_detail', 0.1)} -> {self.config.get('lambda_detail_stage2', 1.0)}")

        # Mine hard negatives using unified embeddings
        print("  Mining hard negatives...")
        self.hard_neg_miner = HardNegativeMiner(
            model=self.model,
            train_dataloader=self.train_loader,
            cache_dir=str(self.output_dir / "hard_negatives"),
            k=self.config["hard_neg_k"],
            text_sim_threshold=self.config["hard_neg_text_threshold"],
            device=self.device,
        )
        self.hard_neg_dict = self.hard_neg_miner.mine(epoch=self.config["num_epochs_stage1"])

        # Reduce learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.config["stage2_lr_factor"]

        print(f"  Learning rate reduced to: {self.optimizer.param_groups[0]['lr']:.2e}")
        print("=" * 60 + "\n")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            "total": 0.0,
            "unified": 0.0,
            "global": 0.0,
            "detail": 0.0,
        }
        epoch_gates = {
            "brep_global": 0.0,
            "brep_detail": 0.0,
            "pc_global": 0.0,
            "pc_detail": 0.0,
            "text_global": 0.0,
            "text_detail": 0.0,
        }
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} (Stage {self.current_stage})"
        )

        nan_batches = 0
        max_nan_batches = 50  # Warn if too many NaN batches

        for batch_idx, batch in enumerate(pbar):
            # Prepare hard negatives for this batch (Stage 2 only)
            hard_negatives = None
            if self.current_stage == 2 and self.hard_neg_dict is not None:
                hard_negatives = self._get_batch_hard_negatives(batch)

            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(batch)
                losses = self.criterion(outputs, hard_negatives=hard_negatives)
                loss = losses['total']

            # Check for NaN loss BEFORE backward - skip entire update if NaN
            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                nan_batches += 1
                if nan_batches <= 5 or nan_batches % 100 == 0:
                    print(f"\n  [WARNING] NaN/Inf loss at batch {batch_idx}, skipping update (total: {nan_batches})")
                self.optimizer.zero_grad()  # Clear any accumulated gradients
                continue  # Skip this batch entirely

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Check for NaN gradients
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["max_grad_norm"]
            )

            # Skip optimizer step if gradients are NaN
            if math.isnan(grad_norm.item()) or math.isinf(grad_norm.item()):
                nan_batches += 1
                if nan_batches <= 5 or nan_batches % 100 == 0:
                    print(f"\n  [WARNING] NaN/Inf gradients at batch {batch_idx}, skipping update")
                self.scaler.update()  # Still need to update scaler
                continue

            # Optimizer step (only if loss and gradients are valid)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.global_step += 1

            # Accumulate losses (NaN-safe)
            batch_valid = True
            for key in epoch_losses:
                if key in losses:
                    val = losses[key]
                    val_float = val.item() if torch.is_tensor(val) else val
                    if not math.isnan(val_float):
                        epoch_losses[key] += val_float
                    else:
                        batch_valid = False
            if batch_valid:
                num_batches += 1

            # Accumulate gate values (for monitoring, NaN-safe)
            if 'gate_brep' in outputs and outputs['gate_brep'] is not None:
                g_brep = outputs['gate_brep']
                if g_brep.dim() == 2 and g_brep.shape[1] >= 2:
                    val0 = g_brep[:, 0].mean().item()
                    val1 = g_brep[:, 1].mean().item()
                    if not (math.isnan(val0) or math.isnan(val1)):
                        epoch_gates['brep_global'] += val0
                        epoch_gates['brep_detail'] += val1
            if 'gate_pc' in outputs and outputs['gate_pc'] is not None:
                g_pc = outputs['gate_pc']
                if g_pc.dim() == 2 and g_pc.shape[1] >= 2:
                    val0 = g_pc[:, 0].mean().item()
                    val1 = g_pc[:, 1].mean().item()
                    if not (math.isnan(val0) or math.isnan(val1)):
                        epoch_gates['pc_global'] += val0
                        epoch_gates['pc_detail'] += val1
            if 'gate_text' in outputs and outputs['gate_text'] is not None:
                g_text = outputs['gate_text']
                if g_text.dim() == 2 and g_text.shape[1] >= 2:
                    val0 = g_text[:, 0].mean().item()
                    val1 = g_text[:, 1].mean().item()
                    if not (math.isnan(val0) or math.isnan(val1)):
                        epoch_gates['text_global'] += val0
                        epoch_gates['text_detail'] += val1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total']:.3f}",
                "U": f"{losses['unified']:.3f}",
                "G": f"{losses['global']:.3f}",
                "D": f"{losses['detail']:.3f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.1e}",
            })

            # Monitor gate values periodically
            if batch_idx % 200 == 0 and 'gate_brep' in outputs:
                gate_b = outputs['gate_brep'].mean(dim=0)
                gate_p = outputs['gate_pc'].mean(dim=0)
                gate_t = outputs['gate_text'].mean(dim=0)
                print(f"\n  [Gates] brep={gate_b[0]:.2f}/{gate_b[1]:.2f} "
                      f"pc={gate_p[0]:.2f}/{gate_p[1]:.2f} "
                      f"text={gate_t[0]:.2f}/{gate_t[1]:.2f}")

            # Periodic logging
            if self.global_step % self.config["log_every"] == 0:
                self._log_step(losses)

        # Average losses and gates
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        for key in epoch_gates:
            epoch_gates[key] /= max(num_batches, 1)

        epoch_losses['gates'] = epoch_gates
        epoch_losses['nan_batches'] = nan_batches
        epoch_losses['valid_batches'] = num_batches

        # Warn if too many NaN batches
        if nan_batches > 0:
            total_batches = num_batches + nan_batches
            nan_pct = 100 * nan_batches / total_batches
            print(f"\n  [NaN Summary] {nan_batches}/{total_batches} batches had NaN ({nan_pct:.1f}%)")
            if nan_pct > 10:
                print("  [WARNING] High NaN rate! Consider: lower LR, check data, or increase temp clamp")

        return epoch_losses

    def _get_batch_hard_negatives(self, batch: Dict) -> Optional[Dict]:
        """Get hard negative indices for current batch."""
        if self.hard_neg_dict is None:
            return None

        batch_indices = batch.get('idx', None)
        if batch_indices is None:
            return None

        # Convert to list if tensor
        if torch.is_tensor(batch_indices):
            batch_indices = batch_indices.tolist()

        # Get hard negatives for each sample
        hard_neg_indices = []
        for idx in batch_indices:
            if idx in self.hard_neg_dict:
                hard_neg_indices.append(self.hard_neg_dict[idx])
            else:
                hard_neg_indices.append(None)

        return {'indices': hard_neg_indices, 'weight': 2.0}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        val_losses = {
            "total": 0.0,
            "unified": 0.0,
            "global": 0.0,
            "detail": 0.0,
        }
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            outputs = self.model(batch)
            losses = self.criterion(outputs)

            for key in val_losses:
                if key in losses:
                    val = losses[key]
                    val_losses[key] += val.item() if torch.is_tensor(val) else val
            num_batches += 1

        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        # Track best
        if val_losses["total"] < self.best_val_loss:
            self.best_val_loss = val_losses["total"]
            self._save_checkpoint(self.current_epoch, val_losses, is_best=True)

        return val_losses

    def _log_step(self, losses: Dict[str, Any]):
        """Log step metrics."""
        if self.config["use_wandb"]:
            try:
                import wandb
                wandb.log({
                    "train/loss": losses["total"],
                    "train/unified_loss": losses["unified"],
                    "train/global_loss": losses["global"],
                    "train/detail_loss": losses["detail"],
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
        def fmt(val, precision=4):
            """Format value, handling nan gracefully."""
            if math.isnan(val) or math.isinf(val):
                return "nan" if math.isnan(val) else "inf"
            return f"{val:.{precision}f}"

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {fmt(train_metrics['total'])}")
        print(f"    Unified: {fmt(train_metrics['unified'])}")
        print(f"    Global:  {fmt(train_metrics['global'])}")
        print(f"    Detail:  {fmt(train_metrics['detail'])}")

        if 'gates' in train_metrics:
            gates = train_metrics['gates']
            print(f"  Gate Values (global/detail):")
            print(f"    B-Rep: {fmt(gates['brep_global'], 3)}/{fmt(gates['brep_detail'], 3)}")
            print(f"    PC:    {fmt(gates['pc_global'], 3)}/{fmt(gates['pc_detail'], 3)}")
            print(f"    Text:  {fmt(gates['text_global'], 3)}/{fmt(gates['text_detail'], 3)}")

        if val_metrics:
            print(f"  Val Loss: {fmt(val_metrics['total'])}")

    def _save_checkpoint(
        self,
        epoch: int,
        val_metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
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

        # Save epoch checkpoint
        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, ckpt_path)

        # Always save latest
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best checkpoint (val_loss={val_metrics['total']:.4f})")

    def _save_final(self):
        """Save final model."""
        final_path = self.output_dir / "clip4cad_hus_final.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }, final_path)
        print(f"Final model saved: {final_path}")

    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        print(f"Resuming from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

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

        # Update loss weights if in Stage 2
        if self.current_stage == 2:
            self.criterion.set_stage2_weights(
                lambda_detail=self.config["lambda_detail_stage2"]
            )

        # Load hard negatives if in stage 2
        if self.current_stage == 2:
            hard_neg_path = self.output_dir / "hard_negatives" / "hard_negatives.json"
            if hard_neg_path.exists():
                self.hard_neg_dict = load_hard_negatives(str(hard_neg_path))
                print(f"  Loaded hard negatives: {len(self.hard_neg_dict)} samples")

        print(f"Resumed at epoch {self.current_epoch}, stage {self.current_stage}")


def train_hus(
    config_path: str,
    data_root: str,
    pc_file: str,
    brep_file: str,
    text_file: str,
    output_dir: str,
    device: str = "cuda",
    resume_from: Optional[str] = None,
    load_to_memory: bool = True,
):
    """
    Main training entry point for HUS.

    Args:
        config_path: Path to model configuration YAML
        data_root: Root directory for data splits
        pc_file: Path to point cloud embeddings HDF5
        brep_file: Path to B-Rep features HDF5
        text_file: Path to text embeddings directory
        output_dir: Output directory for checkpoints
        device: Device to train on
        resume_from: Optional checkpoint path to resume from
        load_to_memory: Whether to load data to RAM for faster training
    """
    from omegaconf import OmegaConf

    # Load config
    config = OmegaConf.load(config_path)

    # Create model
    model = CLIP4CAD_HUS_v2(config)
    param_counts = model.count_parameters()
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")

    # Create datasets (reuse GFAMappedDataset)
    train_dataset = GFAMappedDataset(
        data_root=data_root,
        split="train",
        pc_file=pc_file,
        brep_file=brep_file,
        text_file=text_file,
        load_to_memory=load_to_memory,
    )

    val_dataset = GFAMappedDataset(
        data_root=data_root,
        split="val",
        pc_file=pc_file,
        brep_file=brep_file,
        text_file=text_file,
        load_to_memory=load_to_memory,
    )

    # Create trainer
    trainer = HUSTrainer(
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
