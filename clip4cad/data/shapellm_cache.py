"""
ShapeLLM Feature Cache for CLIP4CAD-GFA.

Handles loading pre-computed ShapeLLM/ReCon++ features from HDF5 files.
Supports both:
1. Standard format (output of convert_shapellm_to_standard.py)
2. Direct ShapeLLM format with on-the-fly concatenation
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import h5py
import numpy as np


class ShapeLLMFeatureCache:
    """
    Cache for ShapeLLM pre-computed point cloud features.

    Features are stored as:
    - Standard format: features [N, 48, 1024], sample_ids [N]
    - Direct format: local_features [N, 32, 1024], global_token [N, 16, 1024], filenames [N]

    Provides UID-based lookup for training.
    """

    def __init__(
        self,
        hdf5_path: Union[str, Path],
        mapping_json: Optional[Union[str, Path]] = None,
        load_to_memory: bool = False,
    ):
        """
        Initialize the cache.

        Args:
            hdf5_path: Path to HDF5 file (standard or direct format)
            mapping_json: Path to UID mapping JSON (required for direct format)
            load_to_memory: If True, load all features to memory (faster but uses more RAM)
        """
        self.hdf5_path = Path(hdf5_path)
        self.load_to_memory = load_to_memory

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Detect format and initialize
        self._detect_format()

        if self.format == "standard":
            self._init_standard()
        else:
            if mapping_json is None:
                raise ValueError("mapping_json required for direct ShapeLLM format")
            self._init_direct(mapping_json)

        # Optionally load to memory
        if load_to_memory:
            self._load_to_memory()

    def _detect_format(self) -> None:
        """Detect HDF5 format (standard vs direct)."""
        with h5py.File(self.hdf5_path, "r") as f:
            if "sample_ids" in f and "features" in f:
                self.format = "standard"
                self.num_tokens = f["features"].shape[1]
                self.embed_dim = f["features"].shape[2]
            elif "local_features" in f and "global_token" in f:
                self.format = "direct"
                self.num_tokens = f["local_features"].shape[1] + f["global_token"].shape[1]
                self.embed_dim = f["local_features"].shape[2]
            else:
                raise ValueError(f"Unrecognized HDF5 format in {self.hdf5_path}")

        print(f"ShapeLLMFeatureCache: format={self.format}, tokens={self.num_tokens}, dim={self.embed_dim}")

    def _init_standard(self) -> None:
        """Initialize from standard format."""
        with h5py.File(self.hdf5_path, "r") as f:
            # Build UID -> index mapping
            sample_ids = f["sample_ids"][:]
            self.uid_to_idx = {}
            for idx, uid in enumerate(sample_ids):
                if isinstance(uid, bytes):
                    uid = uid.decode("utf-8")
                self.uid_to_idx[str(uid)] = idx

            self.num_samples = len(self.uid_to_idx)

        print(f"  Loaded {self.num_samples} samples from standard format")

    def _init_direct(self, mapping_json: Union[str, Path]) -> None:
        """Initialize from direct ShapeLLM format with mapping."""
        # Load mapping
        with open(mapping_json) as f:
            mapping = json.load(f)

        self.uid_to_h5_idx = {}
        for uid, h5_idx in mapping["uid_to_h5_idx"].items():
            self.uid_to_h5_idx[str(uid)] = h5_idx

        self.uid_to_idx = self.uid_to_h5_idx
        self.num_samples = len(self.uid_to_idx)

        print(f"  Loaded mapping for {self.num_samples} samples (direct format)")

    def _load_to_memory(self) -> None:
        """Load all features to memory."""
        print("  Loading features to memory...")

        with h5py.File(self.hdf5_path, "r") as f:
            if self.format == "standard":
                self._features = f["features"][:]
            else:
                local = f["local_features"][:]
                global_ = f["global_token"][:]
                self._features = np.concatenate([local, global_], axis=1)

        print(f"  Loaded features shape: {self._features.shape}")

    def get(self, sample_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get features for a sample UID.

        Args:
            sample_id: UID string

        Returns:
            Dictionary with:
                - pc_features: np.ndarray (48, 1024)
            Or None if sample_id not found
        """
        sample_id = str(sample_id)

        if sample_id not in self.uid_to_idx:
            return None

        idx = self.uid_to_idx[sample_id]

        if self.load_to_memory:
            features = self._features[idx]
        else:
            features = self._get_features_from_disk(idx)

        return {
            "pc_features": features,
        }

    def _get_features_from_disk(self, idx: int) -> np.ndarray:
        """Load features for a single index from disk."""
        with h5py.File(self.hdf5_path, "r") as f:
            if self.format == "standard":
                return f["features"][idx]
            else:
                local = f["local_features"][idx]
                global_ = f["global_token"][idx]
                return np.concatenate([local, global_], axis=0)

    def __len__(self) -> int:
        return self.num_samples

    def __contains__(self, sample_id: str) -> bool:
        return str(sample_id) in self.uid_to_idx

    def get_all_uids(self) -> list:
        """Get list of all available UIDs."""
        return list(self.uid_to_idx.keys())


def verify_shapellm_cache(cache_path: str, test_uids: list = None, n_test: int = 5) -> bool:
    """
    Verify ShapeLLM cache is working correctly.

    Args:
        cache_path: Path to H5 file
        test_uids: Optional list of UIDs to test
        n_test: Number of samples to test

    Returns:
        True if verification passes
    """
    print(f"Verifying ShapeLLM cache: {cache_path}")

    try:
        cache = ShapeLLMFeatureCache(cache_path)
        print(f"  Loaded cache with {len(cache)} samples")

        if test_uids is None:
            test_uids = cache.get_all_uids()[:n_test]

        for uid in test_uids:
            result = cache.get(uid)
            if result is None:
                print(f"  FAIL: Could not get features for UID {uid}")
                return False

            features = result["pc_features"]
            if features.shape != (cache.num_tokens, cache.embed_dim):
                print(f"  FAIL: Wrong shape for UID {uid}: {features.shape}")
                return False

            print(f"  OK: UID {uid} -> shape {features.shape}, mean={features.mean():.4f}")

        print("  Verification PASSED")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        return False
