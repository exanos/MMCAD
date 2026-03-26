"""
AutoBrep Dataset with BFS-Ordered Features

Two-file design optimized for fast training:
- Main file: Features loaded every batch (small, fast)
- Raw file: Geometry loaded only for visualization (large, lazy)

Usage:
    # Training (fast)
    ds = AutoBrepDataset("brep_autobrep.h5", load_main_to_memory=True)
    sample = ds[0]

    # With visualization support
    ds = AutoBrepDataset(
        "brep_autobrep.h5",
        raw_hdf5="brep_autobrep_raw.h5",
        load_main_to_memory=True
    )
    raw_geom = ds.get_raw_geometry(0)
"""

from typing import Dict, Optional, List, Union
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class AutoBrepDataset(Dataset):
    """
    Dataset with BFS-ordered AutoBrep features.

    Optimized for fast training:
    - Main file: Features loaded every batch (small, fast)
    - Raw file: Geometry loaded only for visualization (large, lazy)

    Args:
        main_hdf5: Path to main HDF5 file (brep_autobrep.h5)
        raw_hdf5: Optional path to raw HDF5 file (brep_autobrep_raw.h5)
        max_faces: Maximum faces per sample (for padding)
        max_edges: Maximum edges per sample (for padding)
        load_main_to_memory: Load main file to RAM for fast access
        return_bfs_info: Include BFS ordering info in samples
    """

    def __init__(
        self,
        main_hdf5: Union[str, Path],
        raw_hdf5: Optional[Union[str, Path]] = None,
        max_faces: int = 192,
        max_edges: int = 512,
        load_main_to_memory: bool = True,
        return_bfs_info: bool = True,
    ):
        self.main_hdf5_path = Path(main_hdf5)
        self.raw_hdf5_path = Path(raw_hdf5) if raw_hdf5 else None
        self.max_faces = max_faces
        self.max_edges = max_edges
        self.return_bfs_info = return_bfs_info

        # Open main file
        self.main_file = h5py.File(self.main_hdf5_path, 'r')
        self.n_samples = self.main_file['face_features'].shape[0]

        # Get dimensions from file
        self.face_dim = self.main_file.attrs.get('face_dim', 48)
        self.edge_dim = self.main_file.attrs.get('edge_dim', 12)

        # Optionally load to memory for fast access
        self.in_memory = load_main_to_memory
        if self.in_memory:
            print(f"Loading main features to RAM from {self.main_hdf5_path}...")
            self._load_to_memory()
            print(f"  Loaded {self.n_samples} samples to RAM")

        # Raw file: opened lazily for visualization
        self._raw_file = None

    def _load_to_memory(self):
        """Load main file datasets to memory."""
        self.face_features = self.main_file['face_features'][:]
        self.edge_features = self.main_file['edge_features'][:]
        self.face_masks = self.main_file['face_masks'][:]
        self.edge_masks = self.main_file['edge_masks'][:]
        self.num_faces_arr = self.main_file['num_faces'][:]
        self.num_edges_arr = self.main_file['num_edges'][:]

        # BFS info
        self.bfs_to_original_face = self.main_file['bfs_to_original_face'][:]
        self.bfs_to_original_edge = self.main_file['bfs_to_original_edge'][:]
        self.bfs_level = self.main_file['bfs_level'][:]
        self.bfs_parent_face = self.main_file['bfs_parent_face'][:]
        self.bfs_parent_edge = self.main_file['bfs_parent_edge'][:]
        self.edge_to_faces = self.main_file['edge_to_faces'][:]
        self.face_centroids = self.main_file['face_centroids'][:]

        # UIDs
        self.uids = [
            uid.decode() if isinstance(uid, bytes) else uid
            for uid in self.main_file['uids'][:]
        ]

    @property
    def raw_file(self):
        """Lazy load raw file only when needed."""
        if self._raw_file is None and self.raw_hdf5_path:
            if self.raw_hdf5_path.exists():
                self._raw_file = h5py.File(self.raw_hdf5_path, 'r')
            else:
                raise FileNotFoundError(f"Raw HDF5 file not found: {self.raw_hdf5_path}")
        return self._raw_file

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample (fast path).

        Returns dict with:
            brep_face_features: [max_faces, face_dim] FSQ latents
            brep_edge_features: [max_edges, edge_dim] FSQ latents
            brep_face_mask: [max_faces] boolean mask
            brep_edge_mask: [max_edges] boolean mask
            num_faces: int
            num_edges: int
            sample_id: str

            If return_bfs_info=True:
                face_bfs_level: [max_faces] BFS tree level
                bfs_to_original_face: [max_faces] index mapping
                bfs_to_original_edge: [max_edges] index mapping
                edge_to_faces: [max_edges, 2] topology
                face_centroids: [max_faces, 3] for quick viz
        """
        if self.in_memory:
            sample = {
                'brep_face_features': torch.from_numpy(self.face_features[idx].copy()),
                'brep_edge_features': torch.from_numpy(self.edge_features[idx].copy()),
                'brep_face_mask': torch.from_numpy(self.face_masks[idx].copy()).bool(),
                'brep_edge_mask': torch.from_numpy(self.edge_masks[idx].copy()).bool(),
                'num_faces': int(self.num_faces_arr[idx]),
                'num_edges': int(self.num_edges_arr[idx]),
                'sample_id': self.uids[idx],
            }

            if self.return_bfs_info:
                sample.update({
                    'face_bfs_level': torch.from_numpy(self.bfs_level[idx].copy()),
                    'bfs_to_original_face': torch.from_numpy(self.bfs_to_original_face[idx].copy()),
                    'bfs_to_original_edge': torch.from_numpy(self.bfs_to_original_edge[idx].copy()),
                    'edge_to_faces': torch.from_numpy(self.edge_to_faces[idx].copy()),
                    'face_centroids': torch.from_numpy(self.face_centroids[idx].copy()),
                    'bfs_parent_face': torch.from_numpy(self.bfs_parent_face[idx].copy()),
                    'bfs_parent_edge': torch.from_numpy(self.bfs_parent_edge[idx].copy()),
                })

            return sample
        else:
            # Read from disk (slower)
            sample = {
                'brep_face_features': torch.from_numpy(self.main_file['face_features'][idx]),
                'brep_edge_features': torch.from_numpy(self.main_file['edge_features'][idx]),
                'brep_face_mask': torch.from_numpy(self.main_file['face_masks'][idx]).bool(),
                'brep_edge_mask': torch.from_numpy(self.main_file['edge_masks'][idx]).bool(),
                'num_faces': int(self.main_file['num_faces'][idx]),
                'num_edges': int(self.main_file['num_edges'][idx]),
                'sample_id': self.main_file['uids'][idx].decode() if isinstance(
                    self.main_file['uids'][idx], bytes
                ) else self.main_file['uids'][idx],
            }

            if self.return_bfs_info:
                sample.update({
                    'face_bfs_level': torch.from_numpy(self.main_file['bfs_level'][idx]),
                    'bfs_to_original_face': torch.from_numpy(self.main_file['bfs_to_original_face'][idx]),
                    'bfs_to_original_edge': torch.from_numpy(self.main_file['bfs_to_original_edge'][idx]),
                    'edge_to_faces': torch.from_numpy(self.main_file['edge_to_faces'][idx]),
                    'face_centroids': torch.from_numpy(self.main_file['face_centroids'][idx]),
                    'bfs_parent_face': torch.from_numpy(self.main_file['bfs_parent_face'][idx]),
                    'bfs_parent_edge': torch.from_numpy(self.main_file['bfs_parent_edge'][idx]),
                })

            return sample

    def get_by_uid(self, uid: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get sample by UID."""
        if self.in_memory:
            try:
                idx = self.uids.index(uid)
                return self[idx]
            except ValueError:
                return None
        else:
            # Search through file
            for i in range(self.n_samples):
                sample_uid = self.main_file['uids'][i]
                if isinstance(sample_uid, bytes):
                    sample_uid = sample_uid.decode()
                if sample_uid == uid:
                    return self[i]
            return None

    def get_raw_geometry(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get raw geometry for visualization (slow path).

        Returns dict with:
            face_point_grids: [max_faces, 32, 32, 3] raw surface samples
            edge_point_grids: [max_edges, 32, 3] raw curve samples
            face_normals: [max_faces, 3]
            face_areas: [max_faces]
            face_bboxes: [max_faces, 6]
            face_types: [max_faces] (0=plane, 1=cylinder, etc.)
            edge_types: [max_edges] (0=line, 1=circle, etc.)
            edge_midpoints: [max_edges, 3]
            edge_directions: [max_edges, 3]
            edge_lengths: [max_edges]
            face_adjacency: [max_faces, max_faces]
            bfs_to_original_face: [max_faces]
            bfs_to_original_edge: [max_edges]
            num_faces: int
            num_edges: int
            sample_id: str
        """
        if self.raw_file is None:
            raise ValueError("Raw HDF5 file not provided")

        result = {
            'face_point_grids': self.raw_file['face_point_grids'][idx],
            'edge_point_grids': self.raw_file['edge_point_grids'][idx],
            'face_normals': self.raw_file['face_normals'][idx],
            'face_areas': self.raw_file['face_areas'][idx],
            'face_bboxes': self.raw_file['face_bboxes'][idx],
            'face_types': self.raw_file['face_types'][idx],
            'edge_types': self.raw_file['edge_types'][idx],
            'edge_midpoints': self.raw_file['edge_midpoints'][idx],
            'edge_directions': self.raw_file['edge_directions'][idx],
            'edge_lengths': self.raw_file['edge_lengths'][idx],
            'edge_bboxes': self.raw_file['edge_bboxes'][idx],
            'face_adjacency': self.raw_file['face_adjacency'][idx],
        }

        # Add BFS mapping from main file
        if self.in_memory:
            result['bfs_to_original_face'] = self.bfs_to_original_face[idx]
            result['bfs_to_original_edge'] = self.bfs_to_original_edge[idx]
            result['num_faces'] = int(self.num_faces_arr[idx])
            result['num_edges'] = int(self.num_edges_arr[idx])
            result['sample_id'] = self.uids[idx]
        else:
            result['bfs_to_original_face'] = self.main_file['bfs_to_original_face'][idx]
            result['bfs_to_original_edge'] = self.main_file['bfs_to_original_edge'][idx]
            result['num_faces'] = int(self.main_file['num_faces'][idx])
            result['num_edges'] = int(self.main_file['num_edges'][idx])
            uid = self.main_file['uids'][idx]
            result['sample_id'] = uid.decode() if isinstance(uid, bytes) else uid

        return result

    def get_face_point_cloud(self, idx: int, face_bfs_idx: int) -> np.ndarray:
        """
        Get point cloud for a specific face.

        Args:
            idx: Sample index
            face_bfs_idx: Face index in BFS order

        Returns:
            points: [32, 32, 3] point grid
        """
        if self.raw_file is None:
            raise ValueError("Raw HDF5 file not provided")

        return self.raw_file['face_point_grids'][idx, face_bfs_idx]

    def get_edge_curve(self, idx: int, edge_bfs_idx: int) -> np.ndarray:
        """
        Get curve points for a specific edge.

        Args:
            idx: Sample index
            edge_bfs_idx: Edge index in BFS order

        Returns:
            points: [32, 3] curve points
        """
        if self.raw_file is None:
            raise ValueError("Raw HDF5 file not provided")

        return self.raw_file['edge_point_grids'][idx, edge_bfs_idx]

    def close(self):
        """Clean up file handles."""
        self.main_file.close()
        if self._raw_file:
            self._raw_file.close()
            self._raw_file = None

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"AutoBrepDataset(\n"
            f"  main_hdf5={self.main_hdf5_path},\n"
            f"  raw_hdf5={self.raw_hdf5_path},\n"
            f"  n_samples={self.n_samples},\n"
            f"  in_memory={self.in_memory},\n"
            f"  face_dim={self.face_dim},\n"
            f"  edge_dim={self.edge_dim}\n"
            f")"
        )


def autobrep_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for AutoBrepDataset.

    Stacks tensors and handles sample_id separately.
    """
    result = {}

    # Get keys from first sample
    keys = batch[0].keys()

    for key in keys:
        if key == 'sample_id':
            result[key] = [b[key] for b in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch], dim=0)
        elif isinstance(batch[0][key], (int, float)):
            result[key] = torch.tensor([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]

    return result


def create_autobrep_dataloader(
    main_hdf5: Union[str, Path],
    raw_hdf5: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    load_main_to_memory: bool = True,
    return_bfs_info: bool = True,
    max_faces: int = 192,
    max_edges: int = 512,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for AutoBrepDataset.

    Args:
        main_hdf5: Path to main HDF5 file
        raw_hdf5: Optional path to raw HDF5 file
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        load_main_to_memory: Load main file to RAM
        return_bfs_info: Include BFS ordering info
        max_faces: Maximum faces per sample
        max_edges: Maximum edges per sample
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader instance
    """
    dataset = AutoBrepDataset(
        main_hdf5=main_hdf5,
        raw_hdf5=raw_hdf5,
        max_faces=max_faces,
        max_edges=max_edges,
        load_main_to_memory=load_main_to_memory,
        return_bfs_info=return_bfs_info,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=autobrep_collate_fn,
        **kwargs,
    )
