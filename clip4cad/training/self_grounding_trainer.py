"""
Self-Grounding Trainer for Phase 2 Training

This trainer implements Phase 2 of the sequential training strategy:
- Phase 1: Text-grounding training (completed separately)
- Phase 2: Self-grounding training (this trainer)

Phase 2 freezes all parameters except self_ground_queries and trains them
to produce embeddings that match text-grounded embeddings via cosine alignment loss.

Benefits:
- Fast training (~30 min vs 4 hours for Phase 1)
- Only ~3K trainable parameters (12 queries × 256 dim)
- Clean separation from Phase 1
- Easy to iterate without retraining main model
"""

import gc
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm


class SelfGroundingTrainer:
    """
    Phase 2 Trainer: Train self-grounding queries on frozen text-grounded model.

    This trainer:
    1. Loads a Phase 1 checkpoint (text-grounded model)
    2. Freezes all parameters except self_ground_queries
    3. Trains self_ground_queries to produce embeddings matching text-grounded ones
    4. Saves Phase 2 checkpoint with trained self-grounding

    Usage:
        trainer = SelfGroundingTrainer(model, config, device)
        trainer.train(train_loader, val_loader, num_epochs=15)
        trainer.save_checkpoint("path/to/phase2_checkpoint.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: Optional[str] = None,
        use_amp: bool = True,
    ):
        """
        Initialize Phase 2 trainer.

        Args:
            model: CLIP4CAD_GFA model (loaded from Phase 1 checkpoint)
            config: Training configuration dict
            device: Device to train on
            output_dir: Optional output directory for checkpoints
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_amp = use_amp

        # Initialize optimizer and scaler (set during train())
        self.optimizer = None
        self.scaler = GradScaler() if use_amp else None

        # Freeze all except self_ground_queries
        self._freeze_except_self_grounding()

        # Create output directory
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _freeze_except_self_grounding(self):
        """Freeze all parameters except self_ground_queries."""
        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only self-grounding queries
        self.model.self_ground_queries.requires_grad = True

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        print(f"Phase 2 Self-Grounding Training:")
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Frozen parameters: {total - trainable:,}")
        print(f"  Self-ground queries shape: {self.model.self_ground_queries.shape}")

    def compute_alignment_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute cosine alignment loss between text-grounded and self-grounded embeddings.

        Args:
            batch: Batch dict from dataloader

        Returns:
            Dict with 'total', 'brep', 'pc' losses
        """
        losses = {}

        # Get text-grounded embeddings (frozen, detached - serves as target)
        with torch.no_grad():
            text_grounded = self.model.compute_text_grounded_embeddings(batch)

        # Get self-grounded embeddings (trainable - gradients flow through self_ground_queries)
        self_grounded = self.model.compute_self_grounded_embeddings(batch)

        total_loss = torch.tensor(0.0, device=self.device)
        num_modalities = 0

        # B-Rep alignment
        if "z_brep" in text_grounded and "z_brep" in self_grounded:
            loss_brep = 1 - F.cosine_similarity(
                text_grounded["z_brep"], self_grounded["z_brep"], dim=-1
            ).mean()
            losses["brep"] = loss_brep.item()
            total_loss = total_loss + loss_brep
            num_modalities += 1

        # PC alignment
        if "z_pc" in text_grounded and "z_pc" in self_grounded:
            loss_pc = 1 - F.cosine_similarity(
                text_grounded["z_pc"], self_grounded["z_pc"], dim=-1
            ).mean()
            losses["pc"] = loss_pc.item()
            total_loss = total_loss + loss_pc
            num_modalities += 1

        # Average across modalities
        if num_modalities > 0:
            total_loss = total_loss / num_modalities

        losses["total"] = total_loss

        return losses

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train one epoch.

        Args:
            dataloader: Training dataloader

        Returns:
            Dict with average losses for the epoch
        """
        self.model.train()
        epoch_losses = {"total": 0.0, "brep": 0.0, "pc": 0.0}
        num_batches = 0

        pbar = tqdm(dataloader, desc="  Training", leave=False)
        for batch in pbar:
            # Compute loss with optional AMP
            if self.use_amp:
                with autocast():
                    losses = self.compute_alignment_loss(batch)
                    loss = losses["total"]

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.compute_alignment_loss(batch)
                loss = losses["total"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Accumulate losses
            epoch_losses["total"] += losses["total"].item() if isinstance(losses["total"], torch.Tensor) else losses["total"]
            epoch_losses["brep"] += losses.get("brep", 0.0)
            epoch_losses["pc"] += losses.get("pc", 0.0)
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate self-grounding quality.

        Computes cosine similarity between text-grounded and self-grounded embeddings.
        Good alignment: mean > 0.95, std < 0.05

        Args:
            dataloader: Validation dataloader

        Returns:
            Dict with similarity statistics
        """
        self.model.eval()
        cosine_sims_brep = []
        cosine_sims_pc = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="  Validating", leave=False):
                text_grounded = self.model.compute_text_grounded_embeddings(batch)
                self_grounded = self.model.compute_self_grounded_embeddings(batch)

                # B-Rep similarity
                if "z_brep" in text_grounded and "z_brep" in self_grounded:
                    sim_brep = F.cosine_similarity(
                        text_grounded["z_brep"], self_grounded["z_brep"], dim=-1
                    )
                    cosine_sims_brep.extend(sim_brep.cpu().tolist())

                # PC similarity
                if "z_pc" in text_grounded and "z_pc" in self_grounded:
                    sim_pc = F.cosine_similarity(
                        text_grounded["z_pc"], self_grounded["z_pc"], dim=-1
                    )
                    cosine_sims_pc.extend(sim_pc.cpu().tolist())

        metrics = {}
        if cosine_sims_brep:
            metrics["brep_mean"] = np.mean(cosine_sims_brep)
            metrics["brep_std"] = np.std(cosine_sims_brep)
        if cosine_sims_pc:
            metrics["pc_mean"] = np.mean(cosine_sims_pc)
            metrics["pc_std"] = np.std(cosine_sims_pc)

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 15,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        validate_every: int = 5,
        save_every: int = 5,
    ) -> Dict[str, list]:
        """
        Full training loop for Phase 2.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs to train
            lr: Learning rate (higher than Phase 1 since only 3K params)
            weight_decay: Weight decay for AdamW
            validate_every: Validate every N epochs
            save_every: Save checkpoint every N epochs

        Returns:
            Dict with training history
        """
        print(f"\nStarting Phase 2 Self-Grounding Training")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print("=" * 60)

        # Initialize optimizer - only for self_ground_queries
        self.optimizer = torch.optim.AdamW(
            [self.model.self_ground_queries],
            lr=lr,
            weight_decay=weight_decay
        )

        # Training history
        history = {
            "train_loss": [],
            "train_loss_brep": [],
            "train_loss_pc": [],
            "val_brep_mean": [],
            "val_brep_std": [],
            "val_pc_mean": [],
            "val_pc_std": [],
        }

        # Initial validation
        print("\nInitial validation (before training):")
        val_metrics = self.validate(val_loader)
        self._print_val_metrics(val_metrics, prefix="  Before: ")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_losses = self.train_epoch(train_loader)
            history["train_loss"].append(train_losses["total"])
            history["train_loss_brep"].append(train_losses["brep"])
            history["train_loss_pc"].append(train_losses["pc"])

            print(f"  Loss: {train_losses['total']:.4f} "
                  f"(brep: {train_losses['brep']:.4f}, pc: {train_losses['pc']:.4f})")

            # Validate
            if (epoch + 1) % validate_every == 0:
                val_metrics = self.validate(val_loader)
                self._print_val_metrics(val_metrics)

                history["val_brep_mean"].append(val_metrics.get("brep_mean", 0))
                history["val_brep_std"].append(val_metrics.get("brep_std", 0))
                history["val_pc_mean"].append(val_metrics.get("pc_mean", 0))
                history["val_pc_std"].append(val_metrics.get("pc_std", 0))

            # Save checkpoint
            if self.output_dir and (epoch + 1) % save_every == 0:
                ckpt_path = self.output_dir / f"checkpoint_phase2_epoch{epoch + 1}.pt"
                self.save_checkpoint(ckpt_path, epoch + 1)

        # Final validation
        print("\n" + "=" * 60)
        print("Final validation:")
        val_metrics = self.validate(val_loader)
        self._print_val_metrics(val_metrics, prefix="  Final: ")

        # Save final checkpoint
        if self.output_dir:
            final_path = self.output_dir / "checkpoint_phase2_final.pt"
            self.save_checkpoint(final_path, num_epochs)
            print(f"\nFinal checkpoint saved to: {final_path}")

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        return history

    def _print_val_metrics(self, metrics: Dict[str, float], prefix: str = "  "):
        """Print validation metrics."""
        parts = []
        if "brep_mean" in metrics:
            parts.append(f"B-Rep: {metrics['brep_mean']:.4f}±{metrics['brep_std']:.4f}")
        if "pc_mean" in metrics:
            parts.append(f"PC: {metrics['pc_mean']:.4f}±{metrics['pc_std']:.4f}")

        if parts:
            print(f"{prefix}Cosine similarity: {', '.join(parts)}")

    def save_checkpoint(self, path: str, epoch: int):
        """Save checkpoint with Phase 2 metadata."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "phase": 2,
            "self_ground_queries": self.model.self_ground_queries.data.clone(),
            "config": self.config,
        }, path)


def train_self_grounding(
    phase1_checkpoint: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str,
    config: Dict[str, Any],
    num_epochs: int = 15,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> str:
    """
    Convenience function to run Phase 2 self-grounding training.

    Args:
        phase1_checkpoint: Path to Phase 1 checkpoint
        train_loader: Training dataloader
        val_loader: Validation dataloader
        output_dir: Output directory for checkpoints
        config: Model configuration
        num_epochs: Number of Phase 2 epochs
        lr: Learning rate
        device: Device to use

    Returns:
        Path to final Phase 2 checkpoint
    """
    from clip4cad.models.clip4cad_gfa import CLIP4CAD_GFA

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Phase 1 checkpoint: {phase1_checkpoint}")
    ckpt = torch.load(phase1_checkpoint, map_location=device, weights_only=False)

    # Create model and load weights
    model = CLIP4CAD_GFA(config)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys (random init): {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    # Create trainer and run
    trainer = SelfGroundingTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=lr,
    )

    return str(Path(output_dir) / "checkpoint_phase2_final.pt")
