"""
MM-CAD Dataset for CLIP4CAD-H

Expected data organization:
    data_root/
    ├── brep/
    │   ├── {id}_faces.npy      # [F, 32, 32, 3] face point grids
    │   ├── {id}_edges.npy      # [E, 32, 3] edge curves
    │   └── {id}_adjacency.npy  # [F, E] adjacency matrix
    ├── pointcloud/
    │   └── {id}.npy            # [N, 3] or [N, 6] with normals
    ├── text/
    │   ├── titles.json         # {id: "title"}
    │   └── descriptions.json   # {id: "description"}
    └── splits/
        ├── train.txt
        ├── val.txt
        └── test.txt
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset

from .augmentation import (
    sample_discrete_rotation_matrix,
    rotate_face_grids,
    rotate_edge_curves,
    rotate_pointcloud,
    random_point_jitter,
    normalize_to_bbox,
    normalize_pointcloud,
)


class MMCADDataset(Dataset):
    """
    Multimodal CAD dataset with B-Rep, point cloud, and hierarchical text.

    Handles:
    - Variable numbers of faces/edges with padding and masking
    - Missing modalities (not all samples have all modalities)
    - Consistent rotation augmentation across modalities
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: Any,
        max_faces: int = 64,
        max_edges: int = 128,
        num_points: int = 2048,
        face_grid_size: int = 32,
        edge_curve_size: int = 32,
        max_title_len: int = 64,
        max_desc_len: int = 512,
        rotation_augment: bool = False,
        point_jitter: float = 0.0,
    ):
        """
        Args:
            data_root: Root directory of dataset
            split: 'train', 'val', or 'test'
            tokenizer: HuggingFace tokenizer for text
            max_faces: Maximum number of faces to load
            max_edges: Maximum number of edges to load
            num_points: Number of points in point cloud
            face_grid_size: Size of face UV grid (face_grid_size x face_grid_size)
            edge_curve_size: Number of points per edge curve
            max_title_len: Maximum title token length
            max_desc_len: Maximum description token length
            rotation_augment: Whether to apply rotation augmentation
            point_jitter: Standard deviation for point jitter
        """
        self.data_root = Path(data_root)
        self.split = split
        self.tokenizer = tokenizer
        self.max_faces = max_faces
        self.max_edges = max_edges
        self.num_points = num_points
        self.face_grid_size = face_grid_size
        self.edge_curve_size = edge_curve_size
        self.max_title_len = max_title_len
        self.max_desc_len = max_desc_len
        self.augment = rotation_augment and (split == "train")
        self.point_jitter = point_jitter if split == "train" else 0.0

        # Load split
        split_file = self.data_root / "splits" / f"{split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                self.sample_ids = [line.strip() for line in f if line.strip()]
        else:
            # If no split file, try to find all samples
            self.sample_ids = self._discover_samples()

        # Load text data
        titles_file = self.data_root / "text" / "titles.json"
        descriptions_file = self.data_root / "text" / "descriptions.json"

        self.titles = {}
        self.descriptions = {}

        if titles_file.exists():
            with open(titles_file, "r") as f:
                self.titles = json.load(f)

        if descriptions_file.exists():
            with open(descriptions_file, "r") as f:
                self.descriptions = json.load(f)

        print(f"Loaded {len(self.sample_ids)} samples for {split}")

    def _discover_samples(self) -> List[str]:
        """Discover sample IDs from available data."""
        sample_ids = set()

        # Check B-Rep directory
        brep_dir = self.data_root / "brep"
        if brep_dir.exists():
            for f in brep_dir.glob("*_faces.npy"):
                sample_id = f.stem.replace("_faces", "")
                sample_ids.add(sample_id)

        # Check point cloud directory
        pc_dir = self.data_root / "pointcloud"
        if pc_dir.exists():
            for f in pc_dir.glob("*.npy"):
                sample_ids.add(f.stem)

        return sorted(list(sample_ids))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[idx]
        output = {"sample_id": sample_id}

        # Sample rotation matrix for augmentation (shared across modalities)
        rotation_matrix = None
        if self.augment:
            rotation_matrix = sample_discrete_rotation_matrix()

        # Load B-Rep (if available)
        brep_data = self._load_brep(sample_id, rotation_matrix)
        if brep_data is not None:
            output.update(brep_data)
            output["has_brep"] = True
        else:
            output["has_brep"] = False
            output.update(self._get_empty_brep())

        # Load point cloud (if available)
        pc_data = self._load_pointcloud(sample_id, rotation_matrix)
        if pc_data is not None:
            output.update(pc_data)
            output["has_pointcloud"] = True
        else:
            output["has_pointcloud"] = False
            output.update(self._get_empty_pointcloud())

        # Load and tokenize text
        text_data = self._load_text(sample_id)
        output.update(text_data)
        output["has_text"] = True

        return output

    def _get_empty_brep(self) -> Dict[str, torch.Tensor]:
        """Return empty B-Rep tensors for missing data."""
        return {
            "brep_faces": torch.zeros(self.max_faces, self.face_grid_size, self.face_grid_size, 3),
            "brep_edges": torch.zeros(self.max_edges, self.edge_curve_size, 3),
            "brep_face_mask": torch.zeros(self.max_faces),
            "brep_edge_mask": torch.zeros(self.max_edges),
            "brep_adjacency": torch.zeros(self.max_faces, self.max_edges),
            "brep_num_faces": 0,
            "brep_num_edges": 0,
        }

    def _get_empty_pointcloud(self) -> Dict[str, torch.Tensor]:
        """Return empty point cloud tensor for missing data."""
        return {"points": torch.zeros(self.num_points, 3)}

    def _load_brep(
        self, sample_id: str, rotation_matrix: Optional[torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Load B-Rep face grids, edge curves, and adjacency."""
        face_path = self.data_root / "brep" / f"{sample_id}_faces.npy"
        edge_path = self.data_root / "brep" / f"{sample_id}_edges.npy"
        adj_path = self.data_root / "brep" / f"{sample_id}_adjacency.npy"

        if not face_path.exists():
            return None

        # Load raw data
        faces = np.load(face_path).astype(np.float32)  # [F, H, W, 3]

        # Handle different face grid sizes
        F = faces.shape[0]
        if faces.shape[1] != self.face_grid_size or faces.shape[2] != self.face_grid_size:
            # Resample if needed (simple approach - could use interpolation)
            pass

        edges = (
            np.load(edge_path).astype(np.float32)
            if edge_path.exists()
            else np.zeros((0, self.edge_curve_size, 3), dtype=np.float32)
        )
        E = edges.shape[0]

        adjacency = (
            np.load(adj_path).astype(np.float32)
            if adj_path.exists()
            else np.zeros((F, E), dtype=np.float32)
        )

        # Convert to tensors
        faces = torch.from_numpy(faces)
        edges = torch.from_numpy(edges)
        adjacency = torch.from_numpy(adjacency)

        # Apply rotation augmentation
        if rotation_matrix is not None:
            faces = rotate_face_grids(faces, rotation_matrix)
            if E > 0:
                edges = rotate_edge_curves(edges, rotation_matrix)

        # Normalize each face/edge to [-1, 1] bounding box
        for i in range(F):
            faces[i] = normalize_to_bbox(faces[i])
        for i in range(E):
            edges[i] = normalize_to_bbox(edges[i])

        # Create masks
        face_mask = torch.zeros(self.max_faces)
        face_mask[: min(F, self.max_faces)] = 1.0

        edge_mask = torch.zeros(self.max_edges)
        edge_mask[: min(E, self.max_edges)] = 1.0

        # Pad faces
        if F < self.max_faces:
            pad_faces = torch.zeros(
                self.max_faces - F, self.face_grid_size, self.face_grid_size, 3
            )
            faces = torch.cat([faces, pad_faces], dim=0)
        else:
            faces = faces[: self.max_faces]

        # Pad edges
        if E < self.max_edges:
            pad_edges = torch.zeros(self.max_edges - E, self.edge_curve_size, 3)
            edges = torch.cat([edges, pad_edges], dim=0)
        else:
            edges = edges[: self.max_edges]

        # Pad adjacency
        padded_adj = torch.zeros(self.max_faces, self.max_edges)
        f_end, e_end = min(F, self.max_faces), min(E, self.max_edges)
        padded_adj[:f_end, :e_end] = adjacency[:f_end, :e_end]

        return {
            "brep_faces": faces,
            "brep_edges": edges,
            "brep_face_mask": face_mask,
            "brep_edge_mask": edge_mask,
            "brep_adjacency": padded_adj,
            "brep_num_faces": min(F, self.max_faces),
            "brep_num_edges": min(E, self.max_edges),
        }

    def _load_pointcloud(
        self, sample_id: str, rotation_matrix: Optional[torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Load and process point cloud."""
        pc_path = self.data_root / "pointcloud" / f"{sample_id}.npy"

        if not pc_path.exists():
            return None

        points = np.load(pc_path).astype(np.float32)

        # Handle points with normals [N, 6] vs just positions [N, 3]
        if points.shape[1] >= 6:
            positions = points[:, :3]
        else:
            positions = points[:, :3] if points.shape[1] >= 3 else points

        N = positions.shape[0]

        # Subsample or pad to exact size
        if N > self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False)
            positions = positions[idx]
        elif N < self.num_points:
            pad_idx = np.random.choice(N, self.num_points - N, replace=True)
            positions = np.concatenate([positions, positions[pad_idx]], axis=0)

        # Convert to tensor
        positions = torch.from_numpy(positions)

        # Apply rotation
        if rotation_matrix is not None:
            positions = rotate_pointcloud(positions, rotation_matrix)

        # Normalize to unit sphere
        positions = normalize_pointcloud(positions)

        # Apply jitter
        if self.point_jitter > 0:
            positions = random_point_jitter(positions, std=self.point_jitter)

        return {"points": positions}

    def _load_text(self, sample_id: str) -> Dict[str, torch.Tensor]:
        """Load and tokenize title and description."""
        title = self.titles.get(sample_id, "CAD model")
        description = self.descriptions.get(sample_id, "A CAD model.")

        # Tokenize title
        title_enc = self.tokenizer(
            title,
            max_length=self.max_title_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize description
        desc_enc = self.tokenizer(
            description,
            max_length=self.max_desc_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "title_input_ids": title_enc["input_ids"].squeeze(0),
            "title_attention_mask": title_enc["attention_mask"].squeeze(0),
            "desc_input_ids": desc_enc["input_ids"].squeeze(0),
            "desc_attention_mask": desc_enc["attention_mask"].squeeze(0),
            "title_text": title,
            "desc_text": description,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function handling missing modalities.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with tensors stacked
    """
    collated = {}

    # Identify which samples have which modalities
    has_brep = [s["has_brep"] for s in batch]
    has_pc = [s["has_pointcloud"] for s in batch]
    has_text = [s["has_text"] for s in batch]

    collated["has_brep"] = torch.tensor(has_brep)
    collated["has_pointcloud"] = torch.tensor(has_pc)
    collated["has_text"] = torch.tensor(has_text)

    # Tensor keys to stack
    tensor_keys = [
        "brep_faces",
        "brep_edges",
        "brep_face_mask",
        "brep_edge_mask",
        "brep_adjacency",
        "points",
        "title_input_ids",
        "title_attention_mask",
        "desc_input_ids",
        "desc_attention_mask",
    ]

    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([s[key] for s in batch])

    # Integer keys
    int_keys = ["brep_num_faces", "brep_num_edges"]
    for key in int_keys:
        if key in batch[0]:
            collated[key] = torch.tensor([s[key] for s in batch])

    # Keep string lists
    collated["sample_id"] = [s["sample_id"] for s in batch]
    collated["title_text"] = [s["title_text"] for s in batch]
    collated["desc_text"] = [s["desc_text"] for s in batch]

    return collated


def create_dataloader(
    dataset: MMCADDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with the custom collate function.

    Args:
        dataset: MMCADDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
