"""
Logging utilities for training.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricTracker:
    """Track multiple metrics during training."""

    def __init__(self, metric_names: list):
        self.meters = {name: AverageMeter(name) for name in metric_names}

    def update(self, metrics: Dict[str, float], n: int = 1):
        """Update all metrics."""
        for name, value in metrics.items():
            if name in self.meters:
                self.meters[name].update(value, n)

    def reset(self):
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()

    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        return {name: meter.avg for name, meter in self.meters.items()}

    def __repr__(self):
        return " | ".join(str(m) for m in self.meters.values())


def setup_logging(
    output_dir: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    log_file: str = "train.log"
) -> logging.Logger:
    """
    Set up logging to console and file.

    Args:
        output_dir: Directory for log file
        log_level: Logging level
        log_file: Log file name

    Returns:
        Configured logger
    """
    logger = logging.getLogger("clip4cad")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def log_config(logger: logging.Logger, config: Dict[str, Any], prefix: str = ""):
    """Log configuration parameters."""
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            log_config(logger, value, full_key)
        else:
            logger.info(f"  {full_key}: {value}")
