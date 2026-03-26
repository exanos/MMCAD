"""
Ablation Trainer

Notebook-friendly trainer for ablation experiments. Provides:
- Epoch-level control (call train_epoch() in a loop)
- Metrics history for plotting training curves
- Checkpoint saving/loading
- Evaluation utilities
"""

import gc
import math
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from omegaconf import DictConfig

from .configs import get_ablation_config, print_ablation_diff, get_loss_weights
from .models import CLIP4CAD_GFA_Ablation
from ..losses.gfa_losses import GFALoss
from ..data.gfa_dataset import GFAMappedDataset, gfa_collate_fn


class AblationTrainer:
    """
    Trainer for ablation experiments, designed for notebook use.

    Features:
    - Epoch-level control via train_epoch()
    - Automatic 2-stage curriculum (configurable)
    - Metrics history for plotting
    - Checkpoint management
    """

    def __init__(
        self,
        ablation_type: str,
        train_dataset: GFAMappedDataset,
        val_dataset: Optional[GFAMappedDataset],
        config_path: str,
        output_dir: str,
        device: str = "cuda",
        custom_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ablation trainer.

        Args:
            ablation_type: One of 'baseline', 'no_consistency', 'global_only', 'no_confidence'
            train_dataset: Training dataset (already loaded)
            val_dataset: Validation dataset (optional)
            config_path: Path to base config YAML
            output_dir: Directory for checkpoints and logs
            device: Device to train on
            custom_config: Additional config overrides
        """
        self.ablation_type = ablation_type
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load ablation config
        self.config = get_ablation_config(config_path, ablation_type, custom_config)
        print_ablation_diff(self.config, ablation_type)

        # Create model
        print("Creating model...")
        self.model = CLIP4CAD_GFA_Ablation(self.config).to(device)

        # Create loss function with ablation-specific weights
        self.criterion = GFALoss(
            lambda_global=self.config.training.lambda_global,
            lambda_local=self.config.training.lambda_local,
            lambda_consist=self.config.training.lambda_consist,
            lambda_diverse=self.config.training.lambda_diverse,
            lambda_conf_reg=self.config.training.lambda_conf_reg,
            lambda_global_stage1=self.config.training.lambda_global_stage1,
        )

        # Print loss weights
        weights = get_loss_weights(self.config)
        print(f"\nLoss weights:")
        for name, val in weights.items():
            print(f"  {name}: {val}")

        # Create dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=gfa_collate_fn,
            persistent_workers=True,
        )

        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=gfa_collate_fn,
                persistent_workers=True,
            )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        self._init_scheduler()

        # Mixed precision
        self.scaler = GradScaler()

        # Training state
        self.current_epoch = 0
        self.current_stage = 1
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Metrics history for plotting
        self.history = {
            "train_loss": [],
            "train_global": [],
            "train_local": [],
            "train_consist": [],
            "train_diverse": [],
            "train_conf_reg": [],
            "train_conf_floor": [],
            "val_loss": [],
            "learning_rate": [],
            "stage": [],
            "epoch": [],
        }

        # Total epochs
        self.num_epochs_stage1 = self.config.training.num_epochs_stage1
        self.num_epochs_stage2 = self.config.training.num_epochs_stage2
        self.total_epochs = self.num_epochs_stage1 + self.num_epochs_stage2

        print(f"\nTrainer initialized:")
        print(f"  Ablation: {ablation_type}")
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Val samples: {len(val_dataset) if val_dataset else 0:,}")
        print(f"  Total epochs: {self.total_epochs} (stage1: {self.num_epochs_stage1}, stage2: {self.num_epochs_stage2})")
        print(f"  Output: {self.output_dir}")

    def _init_scheduler(self):
        """Initialize learning rate scheduler with warmup and cosine decay."""
        warmup_steps = self.config.training.warmup_epochs * len(self.train_loader)
        total_steps = (
            self.config.training.num_epochs_stage1 +
            self.config.training.num_epochs_stage2
        ) * len(self.train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(
                self.config.training.min_lr / self.config.training.learning_rate,
                0.5 * (1 + math.cos(math.pi * progress))
            )

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with loss breakdown
        """
        # Check for stage transition
        if self.current_epoch == self.num_epochs_stage1 and self.current_stage == 1:
            self._transition_to_stage2()

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
            desc=f"[{self.ablation_type}] Epoch {self.current_epoch+1}/{self.total_epochs} (S{self.current_stage})",
            leave=False,
        )

        for batch in pbar:
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
                self.config.training.max_grad_norm
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

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.3f}",
                "G": f"{loss_dict['global']:.3f}",
                "L": f"{loss_dict['local']:.3f}",
                "C": f"{loss_dict['consistency']:.3f}",
            })

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        # Record history
        self.history["train_loss"].append(epoch_losses["total"])
        self.history["train_global"].append(epoch_losses["global"])
        self.history["train_local"].append(epoch_losses["local"])
        self.history["train_consist"].append(epoch_losses["consistency"])
        self.history["train_diverse"].append(epoch_losses["diversity"])
        self.history["train_conf_reg"].append(epoch_losses["conf_reg"])
        self.history["train_conf_floor"].append(epoch_losses["conf_floor"])
        self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])
        self.history["stage"].append(self.current_stage)
        self.history["epoch"].append(self.current_epoch)

        self.current_epoch += 1

        # Save checkpoint every 5 epochs
        if self.current_epoch % 5 == 0:
            self.save_checkpoint()

        return epoch_losses

    def _transition_to_stage2(self):
        """Handle transition from stage 1 to stage 2."""
        print(f"\n{'='*50}")
        print(f"[{self.ablation_type}] Transitioning to Stage 2")
        print(f"{'='*50}")

        self.current_stage = 2

        # Reduce learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.config.training.stage2_lr_factor

        print(f"Learning rate reduced to: {self.optimizer.param_groups[0]['lr']:.2e}")

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and return loss dict."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {
            "total": 0.0,
            "global": 0.0,
            "local": 0.0,
            "consistency": 0.0,
            "diversity": 0.0,
        }
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            with autocast():
                outputs = self.model(batch)
                loss, loss_dict = self.criterion(outputs, stage=self.current_stage)

            for key in val_losses:
                if key in loss_dict:
                    val_losses[key] += loss_dict[key]
            num_batches += 1

        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        self.history["val_loss"].append(val_losses["total"])

        # Track best
        if val_losses["total"] < self.best_val_loss:
            self.best_val_loss = val_losses["total"]
            self.save_checkpoint(is_best=True)

        return val_losses

    @torch.no_grad()
    def evaluate_retrieval(self, loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval metrics on a dataloader.

        Returns:
            Dictionary mapping task names to metric dicts
        """
        self.model.eval()

        # Generate embeddings
        text_emb, brep_emb, pc_emb, ids = [], [], [], []

        for batch in tqdm(loader, desc="Generating embeddings", leave=False):
            ids.extend(batch["sample_id"])

            with autocast():
                outputs = self.model(batch)

            text_emb.append(F.normalize(outputs["z_text"].float(), p=2, dim=-1).cpu())
            if "z_brep" in outputs:
                brep_emb.append(F.normalize(outputs["z_brep"].float(), p=2, dim=-1).cpu())
            if "z_pc" in outputs:
                pc_emb.append(F.normalize(outputs["z_pc"].float(), p=2, dim=-1).cpu())

        text_emb = torch.cat(text_emb, dim=0)
        brep_emb = torch.cat(brep_emb, dim=0) if brep_emb else None
        pc_emb = torch.cat(pc_emb, dim=0) if pc_emb else None

        # Compute retrieval metrics
        results = {}
        k_values = [1, 5, 10]

        tasks = [
            ("Text→BRep", text_emb, brep_emb),
            ("Text→PC", text_emb, pc_emb),
            ("PC→BRep", pc_emb, brep_emb),
            ("BRep→PC", brep_emb, pc_emb),
        ]

        for task_name, query_emb, gallery_emb in tasks:
            if query_emb is None or gallery_emb is None:
                continue

            # Compute similarities and rankings
            sim = torch.mm(query_emb, gallery_emb.T)
            _, rankings = torch.topk(sim, k=max(k_values), dim=1)

            # Compute metrics
            task_metrics = {}
            for k in k_values:
                hits = 0
                ap_sum = 0.0
                n = len(ids)

                for i in range(n):
                    qid = str(ids[i])
                    for rank, idx in enumerate(rankings[i, :k]):
                        if str(ids[idx]) == qid:
                            hits += 1
                            ap_sum += 1.0 / (rank + 1)
                            break

                task_metrics[f"R@{k}"] = hits / n
                task_metrics[f"mAP@{k}"] = ap_sum / n

            results[task_name] = task_metrics

        return results

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "stage": self.current_stage,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": dict(self.config),
            "ablation_type": self.ablation_type,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        # Regular checkpoint
        ckpt_path = self.output_dir / f"checkpoint_epoch{self.current_epoch}.pt"
        torch.save(checkpoint, ckpt_path)

        # Latest checkpoint (for easy resumption)
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Best checkpoint
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best checkpoint (val_loss={self.best_val_loss:.4f})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training state."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.current_stage = checkpoint["stage"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)

        print(f"  Resumed at epoch {self.current_epoch}, stage {self.current_stage}")

    def get_history(self) -> Dict[str, List[float]]:
        """Return training history for plotting."""
        return self.history

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to: {history_path}")

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{self.ablation_type}] Model cleaned up")
