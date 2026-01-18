"""
Checkpoint management utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from omegaconf import DictConfig, OmegaConf


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    config: DictConfig,
    save_path: Union[str, Path],
    is_best: bool = False
) -> None:
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        step: Current global step
        metrics: Current metrics
        config: Training configuration
        save_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": OmegaConf.to_container(config),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)

    # Also save as best if specified
    if is_best:
        best_path = save_path.parent / "best.pt"
        torch.save(checkpoint, best_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        strict: Whether to strictly enforce state dict matching
        map_location: Device mapping for loading

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.pt"))

    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return checkpoints[0]


def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 3,
    keep_best: bool = True
) -> None:
    """
    Remove old checkpoints, keeping only the most recent.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to always keep best.pt
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return

    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))

    if len(checkpoints) <= keep_last_n:
        return

    # Sort by modification time (oldest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime)

    # Remove oldest checkpoints
    for ckpt in checkpoints[:-keep_last_n]:
        ckpt.unlink()
