"""
Configuration management utilities.

Uses Hydra/OmegaConf for structured configuration.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file
        overrides: Dictionary of values to override

    Returns:
        Merged configuration as DictConfig
    """
    config = OmegaConf.load(config_path)

    if overrides:
        override_conf = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_conf)

    return config


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        save_path: Path to save to
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)


def flatten_config(config: DictConfig, sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested config to single-level dict for logging.

    Args:
        config: Nested configuration
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    def _flatten(d: DictConfig, parent_key: str = "") -> Dict[str, Any]:
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, DictConfig):
                items.update(_flatten(v, new_key))
            else:
                items[new_key] = v
        return items

    return _flatten(config)


def get_config_diff(config1: DictConfig, config2: DictConfig) -> Dict[str, Any]:
    """
    Get differences between two configurations.

    Args:
        config1: First config
        config2: Second config

    Returns:
        Dictionary of differing values
    """
    flat1 = flatten_config(config1)
    flat2 = flatten_config(config2)

    all_keys = set(flat1.keys()) | set(flat2.keys())
    diff = {}

    for key in all_keys:
        v1 = flat1.get(key)
        v2 = flat2.get(key)
        if v1 != v2:
            diff[key] = {"config1": v1, "config2": v2}

    return diff
