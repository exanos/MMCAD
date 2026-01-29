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
        splits_dir: Optional[str] = None,
        use_single_rotation_cache: bool = True,
        load_to_memory: bool = False,
    ):
        """
        Args:
            data_root: Root directory of dataset
            split: 'train', 'val', or 'test'
            num_rotations: Number of rotations per sample
            embeddings_dir: Directory containing pre-computed features
            splits_dir: Directory containing split files (default: data_root/splits)
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

        # Splits directory
        if splits_dir is None:
            splits_dir = self.data_root / "splits"
        else:
            splits_dir = Path(splits_dir)

        # Load sample IDs (try multiple naming conventions)
        split_file = splits_dir / f"{split}_uids.txt"
        if not split_file.exists():
            split_file = splits_dir / f"{split}.txt"

        if split_file.exists():
            with open(split_file, "r") as f:
                self.sample_ids = [line.strip() for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Split file not found in {splits_dir} (tried {split}_uids.txt and {split}.txt)")

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

        # Try split-prefixed files first, then fall back to single files
        # B-Rep cache
        brep_path = self.embeddings_dir / f"{self.split}_brep_features.h5"
        if not brep_path.exists():
            brep_path = self.embeddings_dir / "brep_features.h5"

        self.brep_cache = None
        if brep_path.exists():
            self.brep_cache = h5py.File(brep_path, "r")
            # Try different key names
            uid_key = "sample_ids" if "sample_ids" in self.brep_cache else "uids"
            sample_ids = self.brep_cache[uid_key][:]
            self.brep_id_to_idx = {
                (sid.decode("utf-8") if isinstance(sid, bytes) else str(sid)): i
                for i, sid in enumerate(sample_ids)
            }
            print(f"  - B-Rep cache: {len(self.brep_id_to_idx)} samples from {brep_path.name}")

        # Point cloud cache (try multiple file naming conventions)
        pc_path = self.embeddings_dir / f"{self.split}_pointcloud_features.h5"
        if not pc_path.exists():
            pc_path = self.embeddings_dir / "pc_features.h5"  # Aligned format
        if not pc_path.exists():
            pc_path = self.embeddings_dir / "shapellm_pc_features.h5"
        if not pc_path.exists():
            pc_path = self.embeddings_dir / "pointcloud_features.h5"

        self.pc_cache = None
        if pc_path.exists():
            self.pc_cache = h5py.File(pc_path, "r")
            uid_key = "sample_ids" if "sample_ids" in self.pc_cache else "uids"
            sample_ids = self.pc_cache[uid_key][:]
            self.pc_id_to_idx = {
                (sid.decode("utf-8") if isinstance(sid, bytes) else str(sid)): i
                for i, sid in enumerate(sample_ids)
            }
            print(f"  - PC cache: {len(self.pc_id_to_idx)} samples from {pc_path.name}")

        # Text cache
        text_path = self.embeddings_dir / f"{self.split}_text_embeddings.h5"
        if not text_path.exists():
            text_path = self.embeddings_dir / "text_embeddings.h5"

        self.text_cache = None
        if text_path.exists():
            self.text_cache = h5py.File(text_path, "r")
            uid_key = "sample_ids" if "sample_ids" in self.text_cache else "uids"
            sample_ids = self.text_cache[uid_key][:]
            self.text_id_to_idx = {
                (sid.decode("utf-8") if isinstance(sid, bytes) else str(sid)): i
                for i, sid in enumerate(sample_ids)
            }
            print(f"  - Text cache: {len(self.text_id_to_idx)} samples from {text_path.name}")

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
                # Title embedding is optional
                if "title_embedding" in text_data:
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
        result = {
            "desc_embedding": self.text_cache["desc_embeddings"][idx],
            "desc_mask": self.text_cache["desc_masks"][idx],
        }
        # Title embeddings are optional
        if "title_embeddings" in self.text_cache:
            result["title_embedding"] = self.text_cache["title_embeddings"][idx]
        return result

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


def gfa_collate_fn(batch: List[Dict], tokenizer=None, max_text_len: int = 512) -> Dict[str, Any]:
    """
    Collate function for GFA dataset.

    Args:
        batch: List of sample dicts
        tokenizer: HuggingFace tokenizer (required if use_cached_embeddings=False)
        max_text_len: Maximum text sequence length for tokenization
    """
    collated = {}

    # Flags
    has_brep = [s.get("has_brep", False) for s in batch]
    has_pc = [s.get("has_pointcloud", False) for s in batch]
    has_text = [s.get("has_text", False) for s in batch]
    use_cached = batch[0].get("use_cached_embeddings", True)

    collated["has_brep"] = torch.tensor(has_brep)
    collated["has_pointcloud"] = torch.tensor(has_pc)
    collated["has_text"] = torch.tensor(has_text)
    collated["use_cached_embeddings"] = use_cached
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
        if use_cached:
            # Pre-computed embeddings
            text_keys = ["title_embedding", "desc_embedding", "desc_mask"]
            for key in text_keys:
                if key in batch[0]:
                    collated[key] = torch.stack([s[key] for s in batch])
        else:
            # Live text: tokenize here
            if tokenizer is None:
                raise ValueError("Tokenizer required for live text encoding")

            # Combine title + description
            texts = []
            for s in batch:
                title = s.get("title", "")
                desc = s.get("description", "")
                if title:
                    text = f"{title}. {desc}" if desc else title
                else:
                    text = desc if desc else "A CAD model"
                texts.append(text)

            # Tokenize
            tokens = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_text_len,
                return_tensors="pt",
            )
            collated["text_input_ids"] = tokens["input_ids"]
            collated["text_attention_mask"] = tokens["attention_mask"]

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


class GFAMappedDataset(Dataset):
    """
    GFA Dataset using UID mapping to load from original HDF5 files.

    Uses uid_mapping.json to map canonical UIDs to indices in each modality's HDF5 file.
    This avoids duplicating data while enabling aligned training across modalities.

    Set load_to_memory=True for fast training (recommended if you have enough RAM).
    Set use_live_text=True for train-time text encoding (requires csv_path).
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        pc_file: Optional[str] = None,
        text_file: Optional[str] = None,
        brep_file: Optional[str] = None,
        mapping_dir: Optional[str] = None,
        num_rotations: int = 1,
        load_to_memory: bool = False,
        use_live_text: bool = False,
        csv_path: Optional[str] = None,
    ):
        """
        Args:
            data_root: Root directory containing embeddings/
            split: 'train' or 'val'
            pc_file: Path to PC HDF5 file (with local_features, global_token)
            text_file: Path to text embeddings HDF5 file (default: data_root/embeddings/text_embeddings.h5)
            brep_file: Path to B-Rep HDF5 file (default: data_root/embeddings/brep_features.h5)
            mapping_dir: Directory containing uid_mapping.json and splits/
            num_rotations: Number of rotations (1 = no rotation augmentation)
            load_to_memory: If True, load B-Rep and PC to RAM for fast access
            use_live_text: If True, load raw text from CSV for train-time encoding
            csv_path: Path to CSV file with uid, title, description columns
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_rotations = num_rotations
        self.load_to_memory = load_to_memory
        self.use_live_text = use_live_text

        # Initialize file handles early to avoid AttributeError in __del__
        self._data = None
        self._brep_file = None
        self._text_file = None
        self._pc_file = None
        self._using_presplit_text = False  # Track if using pre-split text file

        # Mapping directory (default: data_root/aligned)
        if mapping_dir is None:
            mapping_dir = self.data_root / "aligned"
        self.mapping_dir = Path(mapping_dir)

        # Load UID mapping
        mapping_file = self.mapping_dir / "uid_mapping.json"
        if not mapping_file.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

        with open(mapping_file, 'r') as f:
            self.uid_mapping = {int(k): v for k, v in json.load(f).items()}

        # Load split UIDs
        splits_dir = self.mapping_dir / "splits"
        split_file = splits_dir / f"{split}_uids.txt"
        if not split_file.exists():
            split_file = splits_dir / f"{split}.txt"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.sample_ids = np.loadtxt(split_file, dtype=int).tolist()
        self.num_samples = len(self.sample_ids)

        # Store file paths
        self.brep_path = brep_file if brep_file else str(self.data_root / "embeddings" / "brep_features.h5")
        self.text_path = text_file if text_file else str(self.data_root / "embeddings" / "text_embeddings.h5")

        # Validate text file exists when using pre-computed embeddings
        # But allow pre-split files to be used instead
        if not use_live_text and not Path(self.text_path).exists():
            # Check if pre-split files exist before raising error (prioritize SSD first!)
            text_path = Path(self.text_path)
            possible_presplit = [
                Path("c:/Users/User/Desktop/text_splits") / f"{split}_text_embeddings.h5",  # C: drive (SSD) - CHECK FIRST!
                text_path.parent / f"{split}_text_embeddings.h5",
                text_path.parent / "text_splits" / f"{split}_text_embeddings.h5",
                Path("d:/Defect_Det/MMCAD/data/aligned/text_splits") / f"{split}_text_embeddings.h5",  # D: drive (HDD)
            ]

            has_presplit = any(loc.exists() for loc in possible_presplit)

            if not has_presplit:
                raise FileNotFoundError(
                    f"Text embeddings file not found: {self.text_path}\n"
                    f"Also checked for pre-split files but none found.\n"
                    "Either:\n"
                    "  1. Provide a valid text_file path, OR\n"
                    "  2. Run preprocessing: python scripts/preprocess_text_splits.py, OR\n"
                    "  3. Use use_live_text=True with csv_path to encode text at train-time"
                )

        # PC file path
        if pc_file is None:
            info_file = splits_dir / "split_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                    pc_file = info.get('pc_file')

        if pc_file is None:
            raise ValueError("PC file path must be provided")

        self.pc_path = str(pc_file)

        # Live text encoding: load raw text from CSV
        self.uid_to_text = None
        if use_live_text:
            if csv_path is None:
                # Try default location
                csv_path = self.data_root / "169k.csv"
            self._load_text_csv(csv_path)

        if load_to_memory:
            self._load_to_memory()

        mode_str = " (in memory)" if load_to_memory else ""
        text_str = " (live text)" if use_live_text else ""
        print(f"GFAMappedDataset: {split} with {self.num_samples} samples{mode_str}{text_str}")

    def _load_to_memory(self):
        """Load B-Rep, PC, and Text to RAM (text uses chunked loading to reduce peak RAM)."""
        print(f"  Loading {self.split} data to memory (B-Rep + PC + Text)...")

        n = self.num_samples

        # Build index arrays
        brep_indices = np.array([self.uid_mapping[uid]['brep_idx'] for uid in self.sample_ids])
        pc_indices = np.array([self.uid_mapping[uid]['pc_idx'] for uid in self.sample_ids])

        # Load B-Rep (~3GB)
        print("    Loading B-Rep (3GB)...")
        with h5py.File(self.brep_path, 'r') as f:
            all_face = f['face_features'][:]
            all_edge = f['edge_features'][:]
            all_face_mask = f['face_masks'][:]
            all_edge_mask = f['edge_masks'][:]

        self._data = {
            'face_features': all_face[brep_indices].astype(np.float32),
            'edge_features': all_edge[brep_indices].astype(np.float32),
            'face_masks': all_face_mask[brep_indices].astype(np.float32),
            'edge_masks': all_edge_mask[brep_indices].astype(np.float32),
        }
        del all_face, all_edge, all_face_mask, all_edge_mask

        # Load PC (~50GB)
        print("    Loading PC (50GB)...")
        with h5py.File(self.pc_path, 'r') as f:
            all_local = f['local_features'][:]
            all_global = f['global_token'][:]

        self._data['pc_features'] = np.concatenate([
            all_local[pc_indices],
            all_global[pc_indices]
        ], axis=1).astype(np.float32)
        del all_local, all_global

        # Load text to RAM in FP16 (saves 50% RAM, autocast handles conversion)
        if not self.use_live_text:
            # Check if pre-split file exists (created by scripts/preprocess_text_splits.py)
            text_path = Path(self.text_path)

            # Check multiple possible locations for pre-split files (prioritize SSD first!)
            possible_locations = [
                Path("c:/Users/User/Desktop/text_splits") / f"{self.split}_text_embeddings.h5",  # C: drive (SSD) - CHECK FIRST!
                text_path.parent / f"{self.split}_text_embeddings.h5",  # Same dir as original
                text_path.parent / "text_splits" / f"{self.split}_text_embeddings.h5",  # text_splits subdir
                Path("d:/Defect_Det/MMCAD/data/aligned/text_splits") / f"{self.split}_text_embeddings.h5",  # D: drive (HDD)
            ]

            presplit_file = None
            for loc in possible_locations:
                if loc.exists():
                    presplit_file = loc
                    break

            if presplit_file:
                # Fast path: Load pre-extracted split directly
                print(f"    Loading Text from: {presplit_file}")
                import time
                start = time.time()
                with h5py.File(presplit_file, 'r') as f:
                    # Already in FP16, direct load
                    text_embs = f['desc_embeddings'][:]
                    text_masks = f['desc_masks'][:]
                elapsed = time.time() - start
                print(f"    ✓ Text loaded: {text_embs.nbytes/1e9:.1f}GB in {elapsed:.1f}s")

                # Handle reduced sample size (from --max-samples during preprocessing)
                if text_embs.shape[0] != n:
                    if text_embs.shape[0] < n:
                        print(f"    ⚠️  Pre-split has {text_embs.shape[0]} samples, dataset expected {n}")
                        print(f"    Using first {text_embs.shape[0]} samples to match pre-split file")
                        # Update dataset to match pre-split file size
                        self.sample_ids = self.sample_ids[:text_embs.shape[0]]
                        self.num_samples = text_embs.shape[0]
                        n = text_embs.shape[0]
                        # Also update BREP and PC to match
                        if hasattr(self, '_data') and self._data is not None:
                            for key in ['face_features', 'edge_features', 'face_masks', 'edge_masks', 'pc_features']:
                                if key in self._data:
                                    self._data[key] = self._data[key][:n]
                    else:
                        raise ValueError(f"Pre-split file has MORE samples ({text_embs.shape[0]}) than expected ({n})!")
            else:
                # Slow path: Extract from full dataset (happens once, then use preprocessing script)
                print(f"    Pre-split file not found: {presplit_file}")
                print(f"    Loading Text ({n} samples) via sequential scan...")
                print(f"    TIP: Run 'python scripts/preprocess_text_splits.py' to speed this up!")

                text_indices = np.array([self.uid_mapping[uid]['text_idx'] for uid in self.sample_ids])

                with h5py.File(self.text_path, 'r') as f:
                    seq_len = f['desc_embeddings'].shape[1]  # 256
                    emb_dim = f['desc_embeddings'].shape[2]  # 3072

                    # Pre-allocate output arrays
                    text_embs = np.empty((n, seq_len, emb_dim), dtype=np.float16)
                    text_masks = np.empty((n, seq_len), dtype=np.float16)

                    # Sequential scan (slow but works)
                    import time
                    start_time = time.time()

                    total_samples = f['desc_embeddings'].shape[0]  # 169k
                    chunk_size = 10000
                    loaded_count = 0

                    for chunk_start in range(0, total_samples, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, total_samples)

                        # Find which needed indices are in this chunk
                        mask = (text_indices >= chunk_start) & (text_indices < chunk_end)

                        if mask.any():
                            chunk_embs = f['desc_embeddings'][chunk_start:chunk_end]
                            chunk_masks = f['desc_masks'][chunk_start:chunk_end]

                            local_indices = text_indices[mask] - chunk_start
                            text_embs[mask] = chunk_embs[local_indices].astype(np.float16)
                            text_masks[mask] = chunk_masks[local_indices].astype(np.float16)

                            loaded_count += mask.sum()
                            del chunk_embs, chunk_masks

                        # Progress
                        if chunk_end % 50000 == 0 or chunk_end == total_samples:
                            elapsed = time.time() - start_time
                            pct = chunk_end / total_samples * 100
                            print(f"      {chunk_end}/{total_samples} ({pct:.1f}%) | {loaded_count}/{n} found | {elapsed:.1f}s")

                elapsed = time.time() - start_time
                print(f"    ✓ Text loaded: {text_embs.nbytes/1e9:.1f}GB in {elapsed:.1f}s")

            self._data['text_embeddings'] = text_embs
            self._data['text_masks'] = text_masks

        total_gb = sum(arr.nbytes for arr in self._data.values()) / 1e9
        print(f"  ✓ Loaded {n} samples: {total_gb:.1f}GB in RAM (B-Rep + PC + Text)")

    def _load_text_csv(self, csv_path):
        """Load raw text from CSV for train-time encoding."""
        import pandas as pd
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"  Loading text from CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        # Build UID to text mapping
        self.uid_to_text = {}
        for _, row in df.iterrows():
            uid = str(row.get('uid', ''))
            if uid:
                self.uid_to_text[uid] = {
                    'title': str(row.get('title', '') or ''),
                    'description': str(row.get('description', '') or ''),
                }

        print(f"    Loaded {len(self.uid_to_text)} text entries")

    def _open_files(self):
        """Open HDF5 files lazily (for non-memory mode)."""
        if self._brep_file is None:
            self._brep_file = h5py.File(self.brep_path, "r")
        if self._text_file is None and not self.use_live_text:
            # Check for pre-split file first (same logic as _load_to_memory)
            text_path = Path(self.text_path)
            possible_locations = [
                Path("c:/Users/User/Desktop/text_splits") / f"{self.split}_text_embeddings.h5",  # C: drive (SSD) - CHECK FIRST!
                text_path.parent / f"{self.split}_text_embeddings.h5",
                text_path.parent / "text_splits" / f"{self.split}_text_embeddings.h5",
                Path("d:/Defect_Det/MMCAD/data/aligned/text_splits") / f"{self.split}_text_embeddings.h5",  # D: drive (HDD)
            ]

            presplit_file = None
            for loc in possible_locations:
                if loc.exists():
                    presplit_file = loc
                    break

            # Use pre-split file if found, otherwise fall back to original path
            if presplit_file:
                text_file_path = str(presplit_file)
                self._using_presplit_text = True  # Track that we're using pre-split
            else:
                text_file_path = self.text_path
                self._using_presplit_text = False
            self._text_file = h5py.File(text_file_path, "r")
        if self._pc_file is None:
            self._pc_file = h5py.File(self.pc_path, "r")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        canonical_uid = self.sample_ids[idx]
        mapping = self.uid_mapping[canonical_uid]
        text_idx = mapping['text_idx']

        output = {
            "sample_id": str(canonical_uid),
            "idx": idx,
            "rot_idx": 0,
        }

        if self._data is not None:
            # B-Rep and PC from memory
            output["brep_face_features"] = torch.from_numpy(self._data['face_features'][idx])
            output["brep_edge_features"] = torch.from_numpy(self._data['edge_features'][idx])
            output["brep_face_mask"] = torch.from_numpy(self._data['face_masks'][idx])
            output["brep_edge_mask"] = torch.from_numpy(self._data['edge_masks'][idx])
            output["pc_features"] = torch.from_numpy(self._data['pc_features'][idx])

            # Text: read from RAM (FP16, autocast handles conversion in forward pass)
            if not self.use_live_text:
                output["desc_embedding"] = torch.from_numpy(self._data['text_embeddings'][idx])
                output["desc_mask"] = torch.from_numpy(self._data['text_masks'][idx])
        else:
            # All from disk
            self._open_files()

            brep_idx = mapping['brep_idx']
            pc_idx = mapping['pc_idx']

            output["brep_face_features"] = torch.from_numpy(
                self._brep_file["face_features"][brep_idx].astype(np.float32)
            )
            output["brep_edge_features"] = torch.from_numpy(
                self._brep_file["edge_features"][brep_idx].astype(np.float32)
            )
            output["brep_face_mask"] = torch.from_numpy(
                self._brep_file["face_masks"][brep_idx].astype(np.float32)
            )
            output["brep_edge_mask"] = torch.from_numpy(
                self._brep_file["edge_masks"][brep_idx].astype(np.float32)
            )

            local_feat = self._pc_file["local_features"][pc_idx]
            global_tok = self._pc_file["global_token"][pc_idx]
            pc_features = np.concatenate([local_feat, global_tok], axis=0)
            output["pc_features"] = torch.from_numpy(pc_features.astype(np.float32))

            # Text: skip reading from disk if using live text encoding
            if not self.use_live_text:
                # If using pre-split file, use dataset idx instead of original text_idx
                text_file_idx = idx if self._using_presplit_text else text_idx
                output["desc_embedding"] = torch.from_numpy(
                    self._text_file["desc_embeddings"][text_file_idx].astype(np.float32)
                )
                output["desc_mask"] = torch.from_numpy(
                    self._text_file["desc_masks"][text_file_idx].astype(np.float32)
                )
                if "title_embeddings" in self._text_file:
                    output["title_embedding"] = torch.from_numpy(
                        self._text_file["title_embeddings"][text_file_idx].astype(np.float32)
                    )

        output["has_brep"] = True
        output["use_cached_brep_features"] = True
        output["has_pointcloud"] = True
        output["use_cached_pc_features"] = True
        output["has_text"] = True

        # Live text: return raw strings instead of embeddings
        if self.use_live_text and self.uid_to_text is not None:
            text_data = self.uid_to_text.get(str(canonical_uid), {})
            output["title"] = text_data.get("title", "")
            output["description"] = text_data.get("description", "")
            output["use_cached_embeddings"] = False
            # Remove pre-computed embeddings if they were loaded
            output.pop("desc_embedding", None)
            output.pop("desc_mask", None)
            output.pop("title_embedding", None)
        else:
            output["use_cached_embeddings"] = True

        return output

    def close(self):
        """Close HDF5 files."""
        if self._brep_file is not None:
            self._brep_file.close()
            self._brep_file = None
        if self._text_file is not None:
            self._text_file.close()
            self._text_file = None
        if self._pc_file is not None:
            self._pc_file.close()
            self._pc_file = None

    def __del__(self):
        self.close()


def create_gfa_dataloader(
    dataset: GFADataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    hard_neg_dict: Optional[Dict[int, List[int]]] = None,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    tokenizer=None,
    max_text_len: int = 512,
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
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        tokenizer: Tokenizer for live text encoding (required if dataset.use_live_text=True)
        max_text_len: Max sequence length for tokenization

    Returns:
        DataLoader instance
    """
    from functools import partial

    # Create collate function with tokenizer if provided
    collate_fn = partial(gfa_collate_fn, tokenizer=tokenizer, max_text_len=max_text_len)
    # Extra args for multiprocessing
    mp_kwargs = {}
    if num_workers > 0:
        mp_kwargs['prefetch_factor'] = prefetch_factor
        mp_kwargs['persistent_workers'] = persistent_workers

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
            collate_fn=collate_fn,
            **mp_kwargs,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **mp_kwargs,
        )
