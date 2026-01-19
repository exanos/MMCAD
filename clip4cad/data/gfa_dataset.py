"""
GFA Dataset with Multi-Rotation Pre-computed Features

This dataset supports the CLIP4CAD-GFA training pipeline with:
1. Pre-computed features for multiple rotations
2. Consistent rotation sampling across modalities
3. Hard negative batch construction
4. Efficient HDF5 feature loading

The key difference from the standard dataset is that rotation augmentation
is applied during feature pre-computation rather than at runtime.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import json


class MultiRotationFeatureCache:
    """
    Manages pre-computed features with multiple rotations stored in HDF5.

    Expected HDF5 structure:
        features/           # (N, R, ...) - features for N samples, R rotations
        sample_ids/         # (N,) - sample identifiers
        rotation_matrices/  # (R, 3, 3) - rotation matrices used

    Attributes:
        num_rotations: Number of pre-computed rotations
    """

    def __init__(
        self,
        hdf5_path: str,
        modality: str,  # 'brep', 'pc', or 'text'
        load_to_memory: bool = False,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.modality = modality
        self.load_to_memory = load_to_memory

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self.file = h5py.File(hdf5_path, "r")

        # Build sample ID index
        sample_ids = self.file["sample_ids"][:]
        self.id_to_idx = {
            (sid.decode("utf-8") if isinstance(sid, bytes) else sid): i
            for i, sid in enumerate(sample_ids)
        }

        self.num_samples = len(self.id_to_idx)

        # Get rotation count
        if "rotation_matrices" in self.file:
            self.num_rotations = self.file["rotation_matrices"].shape[0]
        elif "features" in self.file and len(self.file["features"].shape) > 2:
            # Infer from feature shape (N, R, ...)
            self.num_rotations = self.file["features"].shape[1]
        else:
            self.num_rotations = 1

        # Load metadata
        self.metadata = dict(self.file.attrs)

        # Optionally load to memory
        self._cached_data = None
        if load_to_memory:
            self._load_to_memory()

        print(f"Loaded {modality} feature cache: {hdf5_path}")
        print(f"  - {self.num_samples} samples, {self.num_rotations} rotations")
        print(f"  - metadata: {self.metadata}")

    def _load_to_memory(self):
        """Load all data to memory for faster access."""
        self._cached_data = {}
        for key in self.file.keys():
            self._cached_data[key] = self.file[key][:]
        print(f"  - Loaded to memory")

    def get(
        self,
        sample_id: str,
        rotation_idx: int = 0
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Get features for a sample at a specific rotation.

        Args:
            sample_id: Sample identifier
            rotation_idx: Rotation index (0 to num_rotations-1)

        Returns:
            Dictionary with feature arrays
        """
        if sample_id not in self.id_to_idx:
            return None

        idx = self.id_to_idx[sample_id]

        data_source = self._cached_data if self._cached_data else self.file

        if self.modality == "brep":
            return {
                "face_features": data_source["face_features"][idx, rotation_idx],
                "edge_features": data_source["edge_features"][idx, rotation_idx],
                "face_mask": data_source["face_masks"][idx],
                "edge_mask": data_source["edge_masks"][idx],
                "adjacency": data_source["adjacency"][idx] if "adjacency" in data_source else None,
            }
        elif self.modality == "pc":
            return {
                "features": data_source["features"][idx, rotation_idx],
            }
        elif self.modality == "text":
            # Text is rotation-invariant
            return {
                "title_embedding": data_source["title_embeddings"][idx],
                "desc_embedding": data_source["desc_embeddings"][idx],
                "desc_mask": data_source["desc_masks"][idx],
            }

        return None

    def get_by_index(
        self,
        idx: int,
        rotation_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """Get features by numerical index."""
        data_source = self._cached_data if self._cached_data else self.file

        if self.modality == "brep":
            return {
                "face_features": data_source["face_features"][idx, rotation_idx],
                "edge_features": data_source["edge_features"][idx, rotation_idx],
                "face_mask": data_source["face_masks"][idx],
                "edge_mask": data_source["edge_masks"][idx],
            }
        elif self.modality == "pc":
            return {
                "features": data_source["features"][idx, rotation_idx],
            }
        elif self.modality == "text":
            return {
                "title_embedding": data_source["title_embeddings"][idx],
                "desc_embedding": data_source["desc_embeddings"][idx],
                "desc_mask": data_source["desc_masks"][idx],
            }

        return {}

    def __len__(self) -> int:
        return self.num_samples

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self.id_to_idx

    def close(self):
        if hasattr(self, "file") and self.file:
            self.file.close()

    def __del__(self):
        self.close()


class GFADataset(Dataset):
    """
    Dataset for CLIP4CAD-GFA with multi-rotation feature support.

    Loads pre-computed features for B-Rep, point cloud, and text,
    with support for multiple pre-computed rotations per sample.

    During training, one rotation is randomly selected per sample,
    applied consistently across B-Rep and point cloud modalities.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        num_rotations: int = 8,
        embeddings_dir: Optional[str] = None,
        use_single_rotation_cache: bool = True,
        load_to_memory: bool = False,
    ):
        """
        Args:
            data_root: Root directory of dataset
            split: 'train', 'val', or 'test'
            num_rotations: Number of rotations per sample
            embeddings_dir: Directory containing pre-computed features
            use_single_rotation_cache: If True, use single-rotation caches
                                       (standard cache format)
            load_to_memory: Load all features to memory
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_rotations = num_rotations
        self.use_single_rotation = use_single_rotation_cache

        # Embeddings directory
        if embeddings_dir is None:
            embeddings_dir = self.data_root / "embeddings"
        self.embeddings_dir = Path(embeddings_dir)

        # Load sample IDs
        split_file = self.data_root / "splits" / f"{split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                self.sample_ids = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.num_samples = len(self.sample_ids)

        # Initialize feature caches
        self._init_caches(load_to_memory)

        print(f"GFA Dataset: {split} split with {self.num_samples} samples")

    def _init_caches(self, load_to_memory: bool):
        """Initialize feature caches."""
        # For standard single-rotation caches, we just use the existing cache format
        # For multi-rotation, we'd use the MultiRotationFeatureCache

        if self.use_single_rotation:
            self._init_single_rotation_caches(load_to_memory)
        else:
            self._init_multi_rotation_caches(load_to_memory)

    def _init_single_rotation_caches(self, load_to_memory: bool):
        """Use existing single-rotation cache format."""
        import h5py

        # B-Rep cache
        brep_path = self.embeddings_dir / f"{self.split}_brep_features.h5"
        self.brep_cache = None
        if brep_path.exists():
            self.brep_cache = h5py.File(brep_path, "r")
            sample_ids = self.brep_cache["sample_ids"][:]
            self.brep_id_to_idx = {
                (sid.decode("utf-8") if isinstance(sid, bytes) else sid): i
                for i, sid in enumerate(sample_ids)
            }
            print(f"  - B-Rep cache: {len(self.brep_id_to_idx)} samples")

        # Point cloud cache
        pc_path = self.embeddings_dir / f"{self.split}_pointcloud_features.h5"
        self.pc_cache = None
        if pc_path.exists():
            self.pc_cache = h5py.File(pc_path, "r")
            sample_ids = self.pc_cache["sample_ids"][:]
            self.pc_id_to_idx = {
                (sid.decode("utf-8") if isinstance(sid, bytes) else sid): i
                for i, sid in enumerate(sample_ids)
            }
            print(f"  - PC cache: {len(self.pc_id_to_idx)} samples")

        # Text cache
        text_path = self.embeddings_dir / f"{self.split}_text_embeddings.h5"
        self.text_cache = None
        if text_path.exists():
            self.text_cache = h5py.File(text_path, "r")
            sample_ids = self.text_cache["sample_ids"][:]
            self.text_id_to_idx = {
                (sid.decode("utf-8") if isinstance(sid, bytes) else sid): i
                for i, sid in enumerate(sample_ids)
            }
            print(f"  - Text cache: {len(self.text_id_to_idx)} samples")

    def _init_multi_rotation_caches(self, load_to_memory: bool):
        """Initialize multi-rotation caches."""
        # Multi-rotation B-Rep
        brep_path = self.embeddings_dir / f"{self.split}_brep_multirot.h5"
        self.brep_cache = None
        if brep_path.exists():
            self.brep_cache = MultiRotationFeatureCache(
                str(brep_path), "brep", load_to_memory
            )

        # Multi-rotation PC
        pc_path = self.embeddings_dir / f"{self.split}_pc_multirot.h5"
        self.pc_cache = None
        if pc_path.exists():
            self.pc_cache = MultiRotationFeatureCache(
                str(pc_path), "pc", load_to_memory
            )

        # Text (rotation-invariant)
        text_path = self.embeddings_dir / f"{self.split}_text_embeddings.h5"
        self.text_cache = None
        if text_path.exists():
            self.text_cache = MultiRotationFeatureCache(
                str(text_path), "text", load_to_memory
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[idx]

        # Random rotation selection (consistent across modalities)
        if self.split == "train":
            rot_idx = np.random.randint(0, self.num_rotations)
        else:
            rot_idx = 0  # Use canonical orientation for eval

        output = {
            "sample_id": sample_id,
            "idx": idx,
            "rot_idx": rot_idx,
        }

        # Load B-Rep features
        if self.brep_cache is not None:
            if self.use_single_rotation:
                brep_data = self._load_brep_single(sample_id)
            else:
                brep_data = self.brep_cache.get(sample_id, rot_idx)

            if brep_data is not None:
                output["brep_face_features"] = torch.from_numpy(
                    brep_data["face_features"].astype(np.float32)
                )
                output["brep_edge_features"] = torch.from_numpy(
                    brep_data["edge_features"].astype(np.float32)
                )
                output["brep_face_mask"] = torch.from_numpy(
                    brep_data["face_mask"].astype(np.float32)
                )
                output["brep_edge_mask"] = torch.from_numpy(
                    brep_data["edge_mask"].astype(np.float32)
                )
                output["has_brep"] = True
                output["use_cached_brep_features"] = True
            else:
                output["has_brep"] = False
        else:
            output["has_brep"] = False

        # Load PC features
        if self.pc_cache is not None:
            if self.use_single_rotation:
                pc_data = self._load_pc_single(sample_id)
            else:
                pc_data = self.pc_cache.get(sample_id, rot_idx)

            if pc_data is not None:
                output["pc_features"] = torch.from_numpy(
                    pc_data["features"].astype(np.float32)
                )
                output["has_pointcloud"] = True
                output["use_cached_pc_features"] = True
            else:
                output["has_pointcloud"] = False
        else:
            output["has_pointcloud"] = False

        # Load text features
        if self.text_cache is not None:
            if self.use_single_rotation:
                text_data = self._load_text_single(sample_id)
            else:
                text_data = self.text_cache.get(sample_id, 0)

            if text_data is not None:
                output["title_embedding"] = torch.from_numpy(
                    text_data["title_embedding"].astype(np.float32)
                )
                output["desc_embedding"] = torch.from_numpy(
                    text_data["desc_embedding"].astype(np.float32)
                )
                output["desc_mask"] = torch.from_numpy(
                    text_data["desc_mask"].astype(np.float32)
                )
                output["has_text"] = True
                output["use_cached_embeddings"] = True
            else:
                output["has_text"] = False
        else:
            output["has_text"] = False

        return output

    def _load_brep_single(self, sample_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load B-Rep from single-rotation cache."""
        if sample_id not in self.brep_id_to_idx:
            return None

        idx = self.brep_id_to_idx[sample_id]
        return {
            "face_features": self.brep_cache["face_features"][idx],
            "edge_features": self.brep_cache["edge_features"][idx],
            "face_mask": self.brep_cache["face_masks"][idx],
            "edge_mask": self.brep_cache["edge_masks"][idx],
        }

    def _load_pc_single(self, sample_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load PC from single-rotation cache."""
        if sample_id not in self.pc_id_to_idx:
            return None

        idx = self.pc_id_to_idx[sample_id]
        return {
            "features": self.pc_cache["features"][idx],
        }

    def _load_text_single(self, sample_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load text from single-rotation cache."""
        if sample_id not in self.text_id_to_idx:
            return None

        idx = self.text_id_to_idx[sample_id]
        return {
            "title_embedding": self.text_cache["title_embeddings"][idx],
            "desc_embedding": self.text_cache["desc_embeddings"][idx],
            "desc_mask": self.text_cache["desc_masks"][idx],
        }

    def get_sample_by_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get sample by ID."""
        if sample_id not in self.sample_ids:
            return None
        idx = self.sample_ids.index(sample_id)
        return self.__getitem__(idx)

    def close(self):
        """Close HDF5 files."""
        if hasattr(self, "brep_cache") and self.brep_cache:
            if self.use_single_rotation:
                self.brep_cache.close()
            else:
                self.brep_cache.close()

        if hasattr(self, "pc_cache") and self.pc_cache:
            if self.use_single_rotation:
                self.pc_cache.close()
            else:
                self.pc_cache.close()

        if hasattr(self, "text_cache") and self.text_cache:
            if self.use_single_rotation:
                self.text_cache.close()
            else:
                self.text_cache.close()

    def __del__(self):
        self.close()


def gfa_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for GFA dataset.
    """
    collated = {}

    # Flags
    has_brep = [s.get("has_brep", False) for s in batch]
    has_pc = [s.get("has_pointcloud", False) for s in batch]
    has_text = [s.get("has_text", False) for s in batch]

    collated["has_brep"] = torch.tensor(has_brep)
    collated["has_pointcloud"] = torch.tensor(has_pc)
    collated["has_text"] = torch.tensor(has_text)
    collated["use_cached_embeddings"] = True
    collated["use_cached_pc_features"] = True
    collated["use_cached_brep_features"] = True

    # B-Rep features
    if any(has_brep):
        brep_keys = [
            "brep_face_features",
            "brep_edge_features",
            "brep_face_mask",
            "brep_edge_mask",
        ]
        for key in brep_keys:
            if key in batch[0]:
                collated[key] = torch.stack([s[key] for s in batch])

    # PC features
    if any(has_pc) and "pc_features" in batch[0]:
        collated["pc_features"] = torch.stack([s["pc_features"] for s in batch])

    # Text features
    if any(has_text):
        text_keys = ["title_embedding", "desc_embedding", "desc_mask"]
        for key in text_keys:
            if key in batch[0]:
                collated[key] = torch.stack([s[key] for s in batch])

    # Metadata
    collated["sample_id"] = [s["sample_id"] for s in batch]
    collated["idx"] = torch.tensor([s["idx"] for s in batch])
    collated["rot_idx"] = torch.tensor([s.get("rot_idx", 0) for s in batch])

    return collated


class HardNegativeSampler(Sampler):
    """
    Batch sampler that includes hard negatives.

    Given pre-computed hard negative indices for each sample,
    constructs batches that include geometrically similar but
    semantically different samples.
    """

    def __init__(
        self,
        dataset: GFADataset,
        hard_neg_dict: Dict[int, List[int]],
        batch_size: int = 64,
        num_seeds: int = 16,
        negs_per_seed: int = 3,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset: GFA dataset
            hard_neg_dict: Mapping from sample index to hard negative indices
            batch_size: Batch size
            num_seeds: Number of seed samples per batch
            negs_per_seed: Number of hard negatives per seed
            shuffle: Whether to shuffle seeds
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.hard_neg_dict = hard_neg_dict
        self.batch_size = batch_size
        self.num_seeds = num_seeds
        self.negs_per_seed = negs_per_seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.all_indices = np.arange(len(dataset))
        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.all_indices)

        for _ in range(self.num_batches):
            batch_indices = self._construct_batch()
            yield batch_indices

    def _construct_batch(self) -> List[int]:
        """Construct a single batch with hard negatives."""
        # Sample seeds
        seeds = np.random.choice(
            self.all_indices, self.num_seeds, replace=False
        ).tolist()
        batch_indices = seeds.copy()

        # Add hard negatives for each seed
        for seed in seeds:
            if seed in self.hard_neg_dict:
                available = [
                    n for n in self.hard_neg_dict[seed]
                    if n not in batch_indices
                ]
                if available:
                    num_add = min(self.negs_per_seed, len(available))
                    negs = np.random.choice(available, num_add, replace=False)
                    batch_indices.extend(negs.tolist())

        # Fill remaining with random samples
        remaining = self.batch_size - len(batch_indices)
        if remaining > 0:
            available = [i for i in self.all_indices if i not in batch_indices]
            if len(available) >= remaining:
                fill = np.random.choice(available, remaining, replace=False)
            else:
                fill = np.random.choice(self.all_indices, remaining, replace=True)
            batch_indices.extend(fill.tolist())

        return batch_indices[:self.batch_size]

    def __len__(self) -> int:
        return self.num_batches


def create_gfa_dataloader(
    dataset: GFADataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    hard_neg_dict: Optional[Dict[int, List[int]]] = None,
) -> DataLoader:
    """
    Create DataLoader for GFA dataset.

    Args:
        dataset: GFA dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        hard_neg_dict: Hard negative indices (for stage 2 training)

    Returns:
        DataLoader instance
    """
    if hard_neg_dict is not None:
        sampler = HardNegativeSampler(
            dataset,
            hard_neg_dict,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=gfa_collate_fn,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=gfa_collate_fn,
        )
