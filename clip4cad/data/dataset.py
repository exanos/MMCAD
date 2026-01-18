"""
MM-CAD Dataset for CLIP4CAD-H

Expected data organization:
    data_root/
    ├── brep/
    │   ├── {id}_faces.npy      # [F, 32, 32, 3] face point grids
    │   ├── {id}_edges.npy      # [E, 32, 3] edge curves
    │   └── {id}_adjacency.npy  # [F, E] adjacency matrix
    ├── pointcloud/
    │   └── {id}.ply            # PLY with 10K points (xyz + normals)
    │   or  {id}.npy            # [N, 3] or [N, 6] with normals (fallback)
    ├── text/
    │   ├── titles.json         # {id: "title"}
    │   └── descriptions.json   # {id: "description"}
    ├── embeddings/             # Pre-computed embeddings (optional)
    │   ├── train_text_embeddings.h5
    │   ├── val_text_embeddings.h5
    │   ├── test_text_embeddings.h5
    │   ├── train_pointcloud_features.h5  # Pre-computed PC features
    │   ├── val_pointcloud_features.h5
    │   ├── test_pointcloud_features.h5
    │   ├── train_brep_features.h5        # Pre-computed B-Rep features
    │   ├── val_brep_features.h5
    │   └── test_brep_features.h5
    └── splits/
        ├── train.txt
        ├── val.txt
        └── test.txt

PLY file format (binary_little_endian):
    ply
    format binary_little_endian 1.0
    element vertex 10000
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    end_header
    <binary data: 10000 vertices * 6 floats * 4 bytes = 240000 bytes>
"""

import json
import struct
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


def load_ply_file(filepath: str, num_points: int = 10000) -> np.ndarray:
    """
    Load PLY file with positions and normals.

    Supports binary_little_endian format with structure:
    - x, y, z (float)
    - nx, ny, nz (float)

    Args:
        filepath: Path to PLY file
        num_points: Number of points to return (subsample or pad)

    Returns:
        points: [num_points, 6] array (xyz + normals)
    """
    with open(filepath, "rb") as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode("utf-8").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header
        n_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                properties.append(line.split()[-1])

        # Determine format
        has_normals = "nx" in properties and "ny" in properties and "nz" in properties
        n_floats = len(properties)

        # Read binary data
        data = np.frombuffer(
            f.read(n_vertices * n_floats * 4),
            dtype=np.float32
        ).reshape(n_vertices, n_floats)

        # Extract xyz and normals
        xyz = data[:, :3]
        if has_normals:
            # Find indices for normals (in case there are other properties between)
            nx_idx = properties.index("nx")
            ny_idx = properties.index("ny")
            nz_idx = properties.index("nz")
            normals = data[:, [nx_idx, ny_idx, nz_idx]]
            points = np.concatenate([xyz, normals], axis=1)
        else:
            # Use zeros for normals
            normals = np.zeros_like(xyz)
            points = np.concatenate([xyz, normals], axis=1)

    # Handle point count
    n = points.shape[0]
    if n > num_points:
        # Random subsample
        idx = np.random.choice(n, num_points, replace=False)
        points = points[idx]
    elif n < num_points:
        # Pad by repeating
        pad_idx = np.random.choice(n, num_points - n, replace=True)
        points = np.concatenate([points, points[pad_idx]], axis=0)

    return points.astype(np.float32)


class TextEmbeddingCache:
    """
    Manages pre-computed text embeddings stored in HDF5 files.

    Provides efficient random access to embeddings without loading
    the entire file into memory.
    """

    def __init__(self, hdf5_path: str):
        import h5py

        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        # Open file and build index
        self.file = h5py.File(hdf5_path, "r")

        # Build sample_id -> index mapping
        sample_ids = self.file["sample_ids"][:]
        self.id_to_idx = {
            (sid.decode("utf-8") if isinstance(sid, bytes) else sid): i
            for i, sid in enumerate(sample_ids)
        }

        # Store metadata
        self.d_llm = self.file.attrs.get("d_llm", 3072)
        self.max_desc_len = self.file.attrs.get("max_desc_len", 256)
        self.model_name = self.file.attrs.get("model_name", "unknown")

        print(f"Loaded text embedding cache: {hdf5_path}")
        print(f"  - {len(self.id_to_idx)} samples")
        print(f"  - d_llm: {self.d_llm}")
        print(f"  - max_desc_len: {self.max_desc_len}")
        print(f"  - model: {self.model_name}")

    def __len__(self):
        return len(self.id_to_idx)

    def __contains__(self, sample_id: str):
        return sample_id in self.id_to_idx

    def get(self, sample_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get embeddings for a sample.

        Returns:
            Dictionary with:
                - title_embedding: [d_llm]
                - desc_embedding: [max_desc_len, d_llm]
                - desc_mask: [max_desc_len]
        """
        if sample_id not in self.id_to_idx:
            return None

        idx = self.id_to_idx[sample_id]

        return {
            "title_embedding": self.file["title_embeddings"][idx],
            "desc_embedding": self.file["desc_embeddings"][idx],
            "desc_mask": self.file["desc_masks"][idx],
        }

    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, "file") and self.file:
            self.file.close()

    def __del__(self):
        self.close()


class PointCloudFeatureCache:
    """
    Manages pre-computed point cloud features stored in HDF5 files.

    Stores Point-BERT encoder outputs (CLS token + group tokens) to
    avoid re-encoding during training when encoder is frozen.
    """

    def __init__(self, hdf5_path: str):
        import h5py

        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        # Open file and build index
        self.file = h5py.File(hdf5_path, "r")

        # Build sample_id -> index mapping
        sample_ids = self.file["sample_ids"][:]
        self.id_to_idx = {
            (sid.decode("utf-8") if isinstance(sid, bytes) else sid): i
            for i, sid in enumerate(sample_ids)
        }

        # Store metadata
        self.embed_dim = self.file.attrs.get("embed_dim", 768)
        self.num_tokens = self.file.attrs.get("num_tokens", 513)  # CLS + 512 groups
        self.model_config = self.file.attrs.get("model_config", "ulip2-pointbert")

        print(f"Loaded point cloud feature cache: {hdf5_path}")
        print(f"  - {len(self.id_to_idx)} samples")
        print(f"  - embed_dim: {self.embed_dim}")
        print(f"  - num_tokens: {self.num_tokens}")
        print(f"  - model: {self.model_config}")

    def __len__(self):
        return len(self.id_to_idx)

    def __contains__(self, sample_id: str):
        return sample_id in self.id_to_idx

    def get(self, sample_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get features for a sample.

        Returns:
            Dictionary with:
                - pc_features: [num_tokens, embed_dim] - all tokens (CLS first)
                - pc_cls: [embed_dim] - CLS token only (global feature)
        """
        if sample_id not in self.id_to_idx:
            return None

        idx = self.id_to_idx[sample_id]

        features = self.file["features"][idx]  # [num_tokens, embed_dim]

        return {
            "pc_features": features,
            "pc_cls": features[0],  # CLS token is first
        }

    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, "file") and self.file:
            self.file.close()

    def __del__(self):
        self.close()


class BRepFeatureCache:
    """
    Manages pre-computed B-Rep features stored in HDF5 files.

    Stores AutoBrep encoder outputs (face and edge latents) to
    avoid re-encoding during training when encoder is frozen.
    """

    def __init__(self, hdf5_path: str):
        import h5py

        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        # Open file and build index
        self.file = h5py.File(hdf5_path, "r")

        # Build sample_id -> index mapping
        sample_ids = self.file["sample_ids"][:]
        self.id_to_idx = {
            (sid.decode("utf-8") if isinstance(sid, bytes) else sid): i
            for i, sid in enumerate(sample_ids)
        }

        # Store metadata
        self.face_dim = self.file.attrs.get("face_dim", 48)
        self.edge_dim = self.file.attrs.get("edge_dim", 12)
        self.max_faces = self.file.attrs.get("max_faces", 64)
        self.max_edges = self.file.attrs.get("max_edges", 128)
        self.model_config = self.file.attrs.get("model_config", "autobrep")

        print(f"Loaded B-Rep feature cache: {hdf5_path}")
        print(f"  - {len(self.id_to_idx)} samples")
        print(f"  - face_dim: {self.face_dim}, edge_dim: {self.edge_dim}")
        print(f"  - max_faces: {self.max_faces}, max_edges: {self.max_edges}")
        print(f"  - model: {self.model_config}")

    def __len__(self):
        return len(self.id_to_idx)

    def __contains__(self, sample_id: str):
        return sample_id in self.id_to_idx

    def get(self, sample_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get features for a sample.

        Returns:
            Dictionary with:
                - face_features: [max_faces, face_dim] - face latents
                - edge_features: [max_edges, edge_dim] - edge latents
                - face_mask: [max_faces] - valid face mask
                - edge_mask: [max_edges] - valid edge mask
                - adjacency: [max_faces, max_edges] - face-edge adjacency
                - num_faces: int - actual number of faces
                - num_edges: int - actual number of edges
        """
        if sample_id not in self.id_to_idx:
            return None

        idx = self.id_to_idx[sample_id]

        return {
            "face_features": self.file["face_features"][idx],
            "edge_features": self.file["edge_features"][idx],
            "face_mask": self.file["face_masks"][idx],
            "edge_mask": self.file["edge_masks"][idx],
            "adjacency": self.file["adjacency"][idx],
            "num_faces": int(self.file["num_faces"][idx]),
            "num_edges": int(self.file["num_edges"][idx]),
        }

    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, "file") and self.file:
            self.file.close()

    def __del__(self):
        self.close()


class MMCADDataset(Dataset):
    """
    Multimodal CAD dataset with B-Rep, point cloud, and hierarchical text.

    Handles:
    - Variable numbers of faces/edges with padding and masking
    - Missing modalities (not all samples have all modalities)
    - Consistent rotation augmentation across modalities
    - Pre-computed embeddings/features (optional, for faster training)
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: Optional[Any] = None,
        max_faces: int = 64,
        max_edges: int = 128,
        num_points: int = 10000,  # 10K for ULIP-2 compatibility
        point_channels: int = 6,  # xyz + normals
        face_grid_size: int = 32,
        edge_curve_size: int = 32,
        max_title_len: int = 64,
        max_desc_len: int = 256,
        rotation_augment: bool = False,
        point_jitter: float = 0.0,
        # Pre-computed embeddings/features
        embeddings_dir: Optional[str] = None,
        use_cached_text_embeddings: bool = False,
        use_cached_pc_features: bool = False,
        use_cached_brep_features: bool = False,
    ):
        """
        Args:
            data_root: Root directory of dataset
            split: 'train', 'val', or 'test'
            tokenizer: HuggingFace tokenizer for text (not needed if using cached)
            max_faces: Maximum number of faces to load
            max_edges: Maximum number of edges to load
            num_points: Number of points in point cloud (default 10K for ULIP-2)
            point_channels: Number of point channels (3 for xyz, 6 for xyz+normals)
            face_grid_size: Size of face UV grid (face_grid_size x face_grid_size)
            edge_curve_size: Number of points per edge curve
            max_title_len: Maximum title token length
            max_desc_len: Maximum description token length
            rotation_augment: Whether to apply rotation augmentation
            point_jitter: Standard deviation for point jitter
            embeddings_dir: Directory containing pre-computed embeddings
            use_cached_text_embeddings: Whether to use pre-computed text embeddings
            use_cached_pc_features: Whether to use pre-computed point cloud features
            use_cached_brep_features: Whether to use pre-computed B-Rep features
        """
        self.data_root = Path(data_root)
        self.split = split
        self.tokenizer = tokenizer
        self.max_faces = max_faces
        self.max_edges = max_edges
        self.num_points = num_points
        self.point_channels = point_channels
        self.face_grid_size = face_grid_size
        self.edge_curve_size = edge_curve_size
        self.max_title_len = max_title_len
        self.max_desc_len = max_desc_len
        self.augment = rotation_augment and (split == "train")
        self.point_jitter = point_jitter if split == "train" else 0.0
        self.use_cached_text_embeddings = use_cached_text_embeddings
        self.use_cached_pc_features = use_cached_pc_features
        self.use_cached_brep_features = use_cached_brep_features

        # Load split
        split_file = self.data_root / "splits" / f"{split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                self.sample_ids = [line.strip() for line in f if line.strip()]
        else:
            # If no split file, try to find all samples
            self.sample_ids = self._discover_samples()

        # Embeddings directory
        if embeddings_dir is None:
            embeddings_dir = self.data_root / "embeddings"
        else:
            embeddings_dir = Path(embeddings_dir)

        # Load text embedding cache if requested
        self.text_cache = None
        if use_cached_text_embeddings:
            cache_file = embeddings_dir / f"{split}_text_embeddings.h5"
            if cache_file.exists():
                self.text_cache = TextEmbeddingCache(str(cache_file))
            else:
                print(f"Warning: Text embedding cache not found: {cache_file}")
                print("Falling back to tokenization (requires tokenizer)")
                self.use_cached_text_embeddings = False

        # Load point cloud feature cache if requested
        self.pc_cache = None
        if use_cached_pc_features:
            pc_cache_file = embeddings_dir / f"{split}_pointcloud_features.h5"
            if pc_cache_file.exists():
                self.pc_cache = PointCloudFeatureCache(str(pc_cache_file))
            else:
                print(f"Warning: Point cloud feature cache not found: {pc_cache_file}")
                print("Falling back to loading raw point clouds")
                self.use_cached_pc_features = False

        # Load B-Rep feature cache if requested
        self.brep_cache = None
        if use_cached_brep_features:
            brep_cache_file = embeddings_dir / f"{split}_brep_features.h5"
            if brep_cache_file.exists():
                self.brep_cache = BRepFeatureCache(str(brep_cache_file))
            else:
                print(f"Warning: B-Rep feature cache not found: {brep_cache_file}")
                print("Falling back to loading raw B-Rep data")
                self.use_cached_brep_features = False

        # Load text data (for fallback or if not using cache)
        if not self.use_cached_text_embeddings or self.text_cache is None:
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

            # Also try loading per-sample text files
            text_dir = self.data_root / "text"
            if text_dir.exists():
                for sample_id in self.sample_ids:
                    text_file = text_dir / f"{sample_id}.json"
                    if text_file.exists() and sample_id not in self.titles:
                        with open(text_file, "r") as f:
                            data = json.load(f)
                            self.titles[sample_id] = data.get("title", "")
                            self.descriptions[sample_id] = data.get("description", "")

        print(f"Loaded {len(self.sample_ids)} samples for {split}")
        if self.use_cached_text_embeddings and self.text_cache:
            print(f"Using pre-computed text embeddings")
        if self.use_cached_pc_features and self.pc_cache:
            print(f"Using pre-computed point cloud features")
        if self.use_cached_brep_features and self.brep_cache:
            print(f"Using pre-computed B-Rep features")

    def _discover_samples(self) -> List[str]:
        """Discover sample IDs from available data."""
        sample_ids = set()

        # Check B-Rep directory
        brep_dir = self.data_root / "brep"
        if brep_dir.exists():
            for f in brep_dir.glob("*_faces.npy"):
                sample_id = f.stem.replace("_faces", "")
                sample_ids.add(sample_id)

        # Check point cloud directory (PLY files preferred, NPY as fallback)
        pc_dir = self.data_root / "pointcloud"
        if pc_dir.exists():
            # PLY files (10K points with normals)
            for f in pc_dir.glob("*.ply"):
                sample_ids.add(f.stem)
            # NPY files (fallback)
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
        if self.use_cached_brep_features and self.brep_cache:
            brep_data = self._load_cached_brep_features(sample_id)
            if brep_data is not None:
                output.update(brep_data)
                output["has_brep"] = True
            else:
                output["has_brep"] = False
                output.update(self._get_empty_brep_features())
        else:
            brep_data = self._load_brep(sample_id, rotation_matrix)
            if brep_data is not None:
                output.update(brep_data)
                output["has_brep"] = True
            else:
                output["has_brep"] = False
                output.update(self._get_empty_brep())

        # Load point cloud (if available)
        if self.use_cached_pc_features and self.pc_cache:
            pc_data = self._load_cached_pc_features(sample_id)
            if pc_data is not None:
                output.update(pc_data)
                output["has_pointcloud"] = True
            else:
                output["has_pointcloud"] = False
                output.update(self._get_empty_pc_features())
        else:
            pc_data = self._load_pointcloud(sample_id, rotation_matrix)
            if pc_data is not None:
                output.update(pc_data)
                output["has_pointcloud"] = True
            else:
                output["has_pointcloud"] = False
                output.update(self._get_empty_pointcloud())

        # Load text (cached or tokenized)
        if self.use_cached_text_embeddings and self.text_cache:
            text_data = self._load_cached_text(sample_id)
        else:
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
        return {"points": torch.zeros(self.num_points, self.point_channels)}

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
        """Load and process point cloud from PLY or NPY file."""
        pc_dir = self.data_root / "pointcloud"

        # Try PLY first (preferred format for 10K points with normals)
        ply_path = pc_dir / f"{sample_id}.ply"
        npy_path = pc_dir / f"{sample_id}.npy"

        if ply_path.exists():
            # Load PLY file (returns [N, 6] with xyz + normals)
            points = load_ply_file(str(ply_path), num_points=self.num_points)
        elif npy_path.exists():
            points = np.load(npy_path).astype(np.float32)
        else:
            return None

        # Ensure we have the right shape
        N = points.shape[0]
        C = points.shape[1] if len(points.shape) > 1 else 1

        # Handle different input formats
        if C >= 6:
            # Already has normals: [N, 6+]
            positions = points[:, :3]
            normals = points[:, 3:6]
        elif C >= 3:
            # Just positions: [N, 3]
            positions = points[:, :3]
            # Zero normals (or could estimate them)
            normals = np.zeros_like(positions)
        else:
            return None

        # Subsample or pad to exact size
        if N > self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False)
            positions = positions[idx]
            normals = normals[idx]
        elif N < self.num_points:
            pad_idx = np.random.choice(N, self.num_points - N, replace=True)
            positions = np.concatenate([positions, positions[pad_idx]], axis=0)
            normals = np.concatenate([normals, normals[pad_idx]], axis=0)

        # Convert to tensors
        positions = torch.from_numpy(positions)
        normals = torch.from_numpy(normals)

        # Apply rotation to both positions and normals
        if rotation_matrix is not None:
            positions = rotate_pointcloud(positions, rotation_matrix)
            normals = rotate_pointcloud(normals, rotation_matrix)

        # Normalize positions to unit sphere
        positions = normalize_pointcloud(positions)

        # Re-normalize normals (they should stay unit vectors)
        normal_norms = torch.norm(normals, dim=-1, keepdim=True).clamp(min=1e-8)
        normals = normals / normal_norms

        # Apply jitter to positions only
        if self.point_jitter > 0:
            positions = random_point_jitter(positions, std=self.point_jitter)

        # Return based on configured channels
        if self.point_channels >= 6:
            points_out = torch.cat([positions, normals], dim=-1)
        else:
            points_out = positions

        return {"points": points_out}

    def _load_cached_brep_features(self, sample_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load pre-computed B-Rep features."""
        cached = self.brep_cache.get(sample_id)

        if cached is None:
            return None

        return {
            "brep_face_features": torch.from_numpy(cached["face_features"].astype(np.float32)),
            "brep_edge_features": torch.from_numpy(cached["edge_features"].astype(np.float32)),
            "brep_face_mask": torch.from_numpy(cached["face_mask"].astype(np.float32)),
            "brep_edge_mask": torch.from_numpy(cached["edge_mask"].astype(np.float32)),
            "brep_adjacency": torch.from_numpy(cached["adjacency"].astype(np.float32)),
            "brep_num_faces": cached["num_faces"],
            "brep_num_edges": cached["num_edges"],
            "use_cached_brep_features": True,
        }

    def _get_empty_brep_features(self) -> Dict[str, torch.Tensor]:
        """Return empty B-Rep features for missing data."""
        face_dim = self.brep_cache.face_dim if self.brep_cache else 48
        edge_dim = self.brep_cache.edge_dim if self.brep_cache else 12
        max_faces = self.brep_cache.max_faces if self.brep_cache else self.max_faces
        max_edges = self.brep_cache.max_edges if self.brep_cache else self.max_edges
        return {
            "brep_face_features": torch.zeros(max_faces, face_dim),
            "brep_edge_features": torch.zeros(max_edges, edge_dim),
            "brep_face_mask": torch.zeros(max_faces),
            "brep_edge_mask": torch.zeros(max_edges),
            "brep_adjacency": torch.zeros(max_faces, max_edges),
            "brep_num_faces": 0,
            "brep_num_edges": 0,
            "use_cached_brep_features": True,
        }

    def _load_cached_pc_features(self, sample_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load pre-computed point cloud features."""
        cached = self.pc_cache.get(sample_id)

        if cached is None:
            return None

        return {
            "pc_features": torch.from_numpy(cached["pc_features"].astype(np.float32)),
            "pc_cls": torch.from_numpy(cached["pc_cls"].astype(np.float32)),
            "use_cached_pc_features": True,
        }

    def _get_empty_pc_features(self) -> Dict[str, torch.Tensor]:
        """Return empty point cloud features for missing data."""
        embed_dim = self.pc_cache.embed_dim if self.pc_cache else 768
        num_tokens = self.pc_cache.num_tokens if self.pc_cache else 513
        return {
            "pc_features": torch.zeros(num_tokens, embed_dim),
            "pc_cls": torch.zeros(embed_dim),
            "use_cached_pc_features": True,
        }

    def _load_cached_text(self, sample_id: str) -> Dict[str, torch.Tensor]:
        """Load pre-computed text embeddings."""
        cached = self.text_cache.get(sample_id)

        if cached is None:
            # Fallback to empty embeddings
            d_llm = self.text_cache.d_llm if self.text_cache else 3072
            max_len = self.text_cache.max_desc_len if self.text_cache else self.max_desc_len
            return {
                "title_embedding": torch.zeros(d_llm),
                "desc_embedding": torch.zeros(max_len, d_llm),
                "desc_mask": torch.zeros(max_len),
                "use_cached_embeddings": True,
            }

        return {
            "title_embedding": torch.from_numpy(cached["title_embedding"].astype(np.float32)),
            "desc_embedding": torch.from_numpy(cached["desc_embedding"].astype(np.float32)),
            "desc_mask": torch.from_numpy(cached["desc_mask"].astype(np.float32)),
            "use_cached_embeddings": True,
        }

    def _load_text(self, sample_id: str) -> Dict[str, torch.Tensor]:
        """Load and tokenize title and description."""
        title = self.titles.get(sample_id, "CAD model")
        description = self.descriptions.get(sample_id, "A CAD model.")

        if self.tokenizer is None:
            raise ValueError("Tokenizer required when not using cached embeddings")

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
            "use_cached_embeddings": False,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function handling missing modalities and cached embeddings.

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
    use_cached_text = [s.get("use_cached_embeddings", False) for s in batch]
    use_cached_pc = [s.get("use_cached_pc_features", False) for s in batch]
    use_cached_brep = [s.get("use_cached_brep_features", False) for s in batch]

    collated["has_brep"] = torch.tensor(has_brep)
    collated["has_pointcloud"] = torch.tensor(has_pc)
    collated["has_text"] = torch.tensor(has_text)
    collated["use_cached_embeddings"] = all(use_cached_text)
    collated["use_cached_pc_features"] = all(use_cached_pc)
    collated["use_cached_brep_features"] = all(use_cached_brep)

    # B-Rep keys - either raw geometry or cached features
    if collated["use_cached_brep_features"]:
        # Stack cached features
        brep_feature_keys = [
            "brep_face_features",
            "brep_edge_features",
            "brep_face_mask",
            "brep_edge_mask",
            "brep_adjacency",
        ]
        for key in brep_feature_keys:
            if key in batch[0]:
                collated[key] = torch.stack([s[key] for s in batch])
    else:
        # Stack raw B-Rep geometry
        brep_keys = [
            "brep_faces",
            "brep_edges",
            "brep_face_mask",
            "brep_edge_mask",
            "brep_adjacency",
        ]
        for key in brep_keys:
            if key in batch[0]:
                collated[key] = torch.stack([s[key] for s in batch])

    # Point cloud keys - either raw points or cached features
    if collated["use_cached_pc_features"]:
        # Stack cached features
        collated["pc_features"] = torch.stack([s["pc_features"] for s in batch])
        collated["pc_cls"] = torch.stack([s["pc_cls"] for s in batch])
    else:
        # Stack raw point clouds
        if "points" in batch[0]:
            collated["points"] = torch.stack([s["points"] for s in batch])

    # Text keys - depends on whether using cached embeddings
    if collated["use_cached_embeddings"]:
        # Stack cached embeddings
        collated["title_embedding"] = torch.stack([s["title_embedding"] for s in batch])
        collated["desc_embedding"] = torch.stack([s["desc_embedding"] for s in batch])
        collated["desc_mask"] = torch.stack([s["desc_mask"] for s in batch])
    else:
        # Stack tokenized inputs
        text_keys = [
            "title_input_ids",
            "title_attention_mask",
            "desc_input_ids",
            "desc_attention_mask",
        ]
        for key in text_keys:
            if key in batch[0]:
                collated[key] = torch.stack([s[key] for s in batch])

        # Keep string lists
        if "title_text" in batch[0]:
            collated["title_text"] = [s["title_text"] for s in batch]
        if "desc_text" in batch[0]:
            collated["desc_text"] = [s["desc_text"] for s in batch]

    # Integer keys
    int_keys = ["brep_num_faces", "brep_num_edges"]
    for key in int_keys:
        if key in batch[0]:
            collated[key] = torch.tensor([s[key] for s in batch])

    # Keep sample IDs
    collated["sample_id"] = [s["sample_id"] for s in batch]

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
