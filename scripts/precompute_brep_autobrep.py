#!/usr/bin/env python3
"""
Pre-compute B-Rep features with AutoBrep BFS ordering.

This script reads STEP files and extracts B-Rep features with:
- BFS-ordered face/edge sequences (following AutoBrep paper)
- Full topology information (face adjacency, edge-to-face mappings)
- Spatial properties (centroids, normals, areas)
- Raw point grids for visualization

Features:
- Auto-download pretrained weights from HuggingFace (default behavior)
- Multiprocessing for parallel geometry extraction (8-16x speedup)
- Checkpointing every N batches (default 100) to allow resuming
- Resume from checkpoint with --resume flag
- Robust error handling for invalid geometry
- Two-file HDF5 output (main for training, raw for visualization)

Requirements:
    - pythonOCC-core (OpenCASCADE bindings)
    Install with: conda install -c conda-forge pythonocc-core

Usage:
    # Standard usage with BFS ordering
    python scripts/precompute_brep_autobrep.py \\
        --step-dir ../data/extracted_step_files \\
        --csv ../data/169k.csv \\
        --output-dir ../data/embeddings \\
        --num-workers 8

    # Resume from checkpoint
    python scripts/precompute_brep_autobrep.py \\
        --step-dir ../data/extracted_step_files \\
        --csv ../data/169k.csv \\
        --output-dir ../data/embeddings \\
        --resume

Output:
    Creates two HDF5 files:
    1. brep_autobrep.h5 (MAIN - for training, ~2GB)
        - face_features: [N, max_faces, 48] FSQ latents
        - edge_features: [N, max_edges, 12] FSQ latents
        - face_masks, edge_masks: [N, max_faces/edges]
        - bfs_to_original_face: [N, max_faces] index mapping
        - bfs_level: [N, max_faces] BFS tree level
        - edge_to_faces: [N, max_edges, 2] topology
        - face_centroids: [N, max_faces, 3] for quick viz
        - uids: [N] sample IDs

    2. brep_autobrep_raw.h5 (AUX - for visualization, ~50GB)
        - face_point_grids: [N, max_faces, 32, 32, 3] raw geometry
        - edge_point_grids: [N, max_edges, 32, 3] raw geometry
        - face_normals, face_areas, face_bboxes
        - edge_midpoints, edge_directions, edge_lengths
        - face_types, edge_types
        - face_adjacency: [N, max_faces, max_faces] sparse
"""

# CRITICAL: Fix OpenMP library conflict on Windows
# Must be set BEFORE importing numpy/torch/OCC
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import gc
import json
import sys
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def import_occ():
    """Import pythonOCC modules. Called in each worker process."""
    # Set environment variable in worker process too
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
        from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
        from OCC.Core.TopExp import topexp
        return {
            'STEPControl_Reader': STEPControl_Reader,
            'IFSelect_RetDone': IFSelect_RetDone,
            'TopExp_Explorer': TopExp_Explorer,
            'TopAbs_FACE': TopAbs_FACE,
            'TopAbs_EDGE': TopAbs_EDGE,
            'topods': topods,
            'BRepAdaptor_Surface': BRepAdaptor_Surface,
            'BRepAdaptor_Curve': BRepAdaptor_Curve,
            'TopTools_IndexedDataMapOfShapeListOfShape': TopTools_IndexedDataMapOfShapeListOfShape,
            'topexp': topexp,
        }
    except Exception as e:
        return None


def load_uid_list(csv_path: Path) -> list:
    """Load list of UIDs from CSV file."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["uid"])
    return [str(uid) for uid in df["uid"].tolist()]


def load_uid_mapping(mapping_csv: Path) -> dict:
    """Load UID mapping from combined_sorted_models.csv."""
    if not mapping_csv.exists():
        return {}

    df = pd.read_csv(mapping_csv, low_memory=False)
    df['modelname_clean'] = df['modelname'].str.replace('"', '')
    df['modelname_stripped'] = df['modelname_clean'].str.lstrip('0')

    return dict(zip(df['modelname_stripped'], df['uid']))


def scan_step_directory(step_dir: Path) -> Dict[str, Path]:
    """
    Scan STEP directory once and build a lookup dict.

    MUCH faster than checking file existence for each UID.

    Returns:
        dict mapping uid (without extension) -> full path
    """
    print(f"Scanning STEP directory: {step_dir}")
    step_files = {}

    # Scan for .step and .stp files
    for ext in ['*.step', '*.stp', '*.STEP', '*.STP']:
        for path in step_dir.glob(ext):
            uid = path.stem  # filename without extension
            step_files[uid] = path

    print(f"  Found {len(step_files)} STEP files")
    return step_files


def find_step_files(step_dir: Path, uids: list, uid_mapping: dict = None,
                    step_files_cache: Dict[str, Path] = None) -> list:
    """
    Find STEP files for given UIDs using combined lookup strategy.

    Uses pre-scanned directory for O(1) lookups instead of O(n) file checks.
    """
    # Scan directory if not provided
    if step_files_cache is None:
        step_files_cache = scan_step_directory(step_dir)

    samples = []
    via_mapping = 0
    via_direct = 0
    missing = 0

    for uid in tqdm(uids, desc="Matching UIDs to STEP files"):
        step_path = None
        step_uid = uid

        # Method 1: Try mapping
        if uid_mapping and uid in uid_mapping:
            mapped_uid = str(uid_mapping[uid])
            if mapped_uid in step_files_cache:
                step_path = step_files_cache[mapped_uid]
                step_uid = mapped_uid
                via_mapping += 1

        # Method 2: Try direct (use cache for O(1) lookup)
        if step_path is None:
            if uid in step_files_cache:
                step_path = step_files_cache[uid]
                step_uid = uid
                via_direct += 1

        if step_path:
            samples.append({
                "uid": uid,
                "step_uid": step_uid,
                "path": str(step_path),
            })
        else:
            missing += 1

    print(f"Found via mapping: {via_mapping}")
    print(f"Found via direct: {via_direct}")
    if missing > 0:
        print(f"Warning: {missing} STEP files not found")

    return samples


def sample_face_grid(face, grid_size, occ):
    """Sample a face as a UV grid of 3D points."""
    try:
        surface = occ['BRepAdaptor_Surface'](face)
        u_min = surface.FirstUParameter()
        u_max = surface.LastUParameter()
        v_min = surface.FirstVParameter()
        v_max = surface.LastVParameter()

        # Handle infinite parameters
        if abs(u_min) > 1e10: u_min = -100.0
        if abs(u_max) > 1e10: u_max = 100.0
        if abs(v_min) > 1e10: v_min = -100.0
        if abs(v_max) > 1e10: v_max = 100.0

        grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)

        for i in range(grid_size):
            for j in range(grid_size):
                u = u_min + (u_max - u_min) * i / max(grid_size - 1, 1)
                v = v_min + (v_max - v_min) * j / max(grid_size - 1, 1)

                try:
                    pnt = surface.Value(u, v)
                    grid[i, j] = [pnt.X(), pnt.Y(), pnt.Z()]
                except Exception:
                    grid[i, j] = [0.0, 0.0, 0.0]

        return grid

    except Exception:
        return None


def sample_edge_curve(edge, num_points, occ):
    """Sample an edge as a curve of 3D points."""
    try:
        curve = occ['BRepAdaptor_Curve'](edge)
        t_min = curve.FirstParameter()
        t_max = curve.LastParameter()

        # Handle infinite curves
        if abs(t_min) > 1e10: t_min = -100.0
        if abs(t_max) > 1e10: t_max = 100.0

        points = np.zeros((num_points, 3), dtype=np.float32)

        for i in range(num_points):
            t = t_min + (t_max - t_min) * i / max(num_points - 1, 1)

            try:
                pnt = curve.Value(t)
                points[i] = [pnt.X(), pnt.Y(), pnt.Z()]
            except Exception:
                points[i] = [0.0, 0.0, 0.0]

        return points

    except Exception:
        return None


def process_single_file(args):
    """
    Worker function: Extract geometry with BFS ordering from a single STEP file.

    This runs in a separate process with its own pythonOCC import.

    Returns dict with:
        - uid: sample identifier
        - faces: [F, 32, 32, 3] face point grids (original order)
        - edges: [E, 32, 3] edge point curves (original order)
        - face_edge_incidence: [F, E] binary incidence matrix
        - face_bboxes: [F, 6] bounding boxes
        - edge_bboxes: [E, 6] bounding boxes
        - num_faces, num_edges: actual counts
    """
    sample, max_faces, max_edges, face_grid_size, edge_curve_size = args

    # Import OCC in this worker process
    occ = import_occ()
    if occ is None:
        return None

    filepath = sample["path"]
    uid = sample["uid"]

    try:
        # Read STEP file
        reader = occ['STEPControl_Reader']()
        status = reader.ReadFile(filepath)

        if status != occ['IFSelect_RetDone']:
            return None

        reader.TransferRoots()
        shape = reader.OneShape()

        if shape.IsNull():
            return None

        # Extract faces with stable ordering
        faces = []
        face_shapes = []
        face_bboxes = []

        try:
            face_explorer = occ['TopExp_Explorer'](shape, occ['TopAbs_FACE'])
            while face_explorer.More():
                try:
                    face = occ['topods'].Face(face_explorer.Current())
                    grid = sample_face_grid(face, face_grid_size, occ)
                    if grid is not None:
                        # Compute bbox from grid
                        bbox_min = grid.min(axis=(0, 1))
                        bbox_max = grid.max(axis=(0, 1))
                        bbox = np.concatenate([bbox_min, bbox_max])

                        faces.append(grid)
                        face_shapes.append(face)
                        face_bboxes.append(bbox)
                except Exception:
                    pass
                face_explorer.Next()
        except Exception:
            pass

        # Extract edges with stable ordering
        edges = []
        edge_shapes = []
        edge_bboxes = []

        try:
            edge_explorer = occ['TopExp_Explorer'](shape, occ['TopAbs_EDGE'])
            while edge_explorer.More():
                try:
                    edge = occ['topods'].Edge(edge_explorer.Current())
                    curve = sample_edge_curve(edge, edge_curve_size, occ)
                    if curve is not None:
                        # Compute bbox from curve
                        bbox_min = curve.min(axis=0)
                        bbox_max = curve.max(axis=0)
                        bbox = np.concatenate([bbox_min, bbox_max])

                        edges.append(curve)
                        edge_shapes.append(edge)
                        edge_bboxes.append(bbox)
                except Exception:
                    pass
                edge_explorer.Next()
        except Exception:
            pass

        # Need at least some geometry
        if len(faces) == 0 and len(edges) == 0:
            return None

        num_faces = min(len(faces), max_faces)
        num_edges = min(len(edges), max_edges)

        # Build face-edge incidence matrix
        # For each edge, find which faces it belongs to
        face_edge_incidence = np.zeros((num_faces, num_edges), dtype=np.int8)

        try:
            # Build map of edge -> adjacent faces
            edge_face_map = occ['TopTools_IndexedDataMapOfShapeListOfShape']()
            occ['topexp'].MapShapesAndAncestors(
                shape, occ['TopAbs_EDGE'], occ['TopAbs_FACE'], edge_face_map
            )

            # For each edge, find its faces
            for e_idx, edge in enumerate(edge_shapes[:num_edges]):
                try:
                    # Find this edge in the map
                    edge_index = edge_face_map.FindIndex(edge)
                    if edge_index > 0:
                        face_list = edge_face_map.FindFromIndex(edge_index)
                        # Iterate through faces
                        it = face_list.begin()
                        while it != face_list.end():
                            adj_face = occ['topods'].Face(it.Value())
                            # Find face index
                            for f_idx, f in enumerate(face_shapes[:num_faces]):
                                if f.IsSame(adj_face):
                                    face_edge_incidence[f_idx, e_idx] = 1
                                    break
                            it.Next()
                except Exception:
                    pass
        except Exception:
            # If topology extraction fails, continue without incidence
            pass

        # Convert to padded arrays
        face_array = np.zeros((max_faces, face_grid_size, face_grid_size, 3), dtype=np.float32)
        edge_array = np.zeros((max_edges, edge_curve_size, 3), dtype=np.float32)
        face_bbox_array = np.zeros((max_faces, 6), dtype=np.float32)
        edge_bbox_array = np.zeros((max_edges, 6), dtype=np.float32)
        face_edge_inc_padded = np.zeros((max_faces, max_edges), dtype=np.int8)

        for i, face in enumerate(faces[:max_faces]):
            face_array[i] = face
            face_bbox_array[i] = face_bboxes[i]

        for i, edge in enumerate(edges[:max_edges]):
            edge_array[i] = edge
            edge_bbox_array[i] = edge_bboxes[i]

        face_edge_inc_padded[:num_faces, :num_edges] = face_edge_incidence

        return {
            "uid": uid,
            "faces": face_array,
            "edges": edge_array,
            "face_bboxes": face_bbox_array,
            "edge_bboxes": edge_bbox_array,
            "face_edge_incidence": face_edge_inc_padded,
            "num_faces": num_faces,
            "num_edges": num_edges,
        }

    except Exception as e:
        return None


def normalize_geometry(face_array, edge_array, face_mask, edge_mask):
    """Normalize each face/edge to [-1, 1] bounding box."""
    face_array = face_array.copy()
    edge_array = edge_array.copy()
    face_mask = face_mask.copy()
    edge_mask = edge_mask.copy()

    num_faces = int(face_mask.sum())
    num_edges = int(edge_mask.sum())

    for i in range(num_faces):
        face = face_array[i]
        face_min = face.min(axis=(0, 1), keepdims=True)
        face_max = face.max(axis=(0, 1), keepdims=True)
        face_range = face_max - face_min

        if face_range.max() < 1e-8:
            face_mask[i] = 0.0
            continue

        face_array[i] = 2.0 * (face - face_min) / np.maximum(face_range, 1e-8) - 1.0

    for i in range(num_edges):
        edge = edge_array[i]
        edge_min = edge.min(axis=0, keepdims=True)
        edge_max = edge.max(axis=0, keepdims=True)
        edge_range = edge_max - edge_min

        if edge_range.max() < 1e-8:
            edge_mask[i] = 0.0
            continue

        edge_array[i] = 2.0 * (edge - edge_min) / np.maximum(edge_range, 1e-8) - 1.0

    return face_array, edge_array, face_mask, edge_mask


def save_checkpoint(checkpoint_path: Path, processed_count: int, processed_uids: list):
    """Save checkpoint information."""
    checkpoint = {
        "processed_count": processed_count,
        "processed_uids": processed_uids[-10000:],
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint(checkpoint_path: Path):
    """Load checkpoint information."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None


def load_processed_uids_from_h5(h5_path: Path) -> set:
    """Load all processed UIDs from existing HDF5 file."""
    if not h5_path.exists():
        return set()
    try:
        with h5py.File(h5_path, "r") as f:
            if "uids" in f:
                uids = set(uid.decode() if isinstance(uid, bytes) else uid for uid in f["uids"][:])
                return uids
    except Exception as e:
        print(f"Warning: Could not read UIDs from HDF5: {e}")
    return set()


def create_main_hdf5(
    output_path: Path,
    max_faces: int,
    max_edges: int,
    face_dim: int,
    edge_dim: int,
    total_samples: int,
):
    """Create the main HDF5 file for training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Features (BFS ordered)
        f.create_dataset(
            "face_features",
            shape=(0, max_faces, face_dim),
            maxshape=(total_samples, max_faces, face_dim),
            dtype=np.float32,
            chunks=(10, max_faces, face_dim),
            compression="lzf",
        )
        f.create_dataset(
            "edge_features",
            shape=(0, max_edges, edge_dim),
            maxshape=(total_samples, max_edges, edge_dim),
            dtype=np.float32,
            chunks=(10, max_edges, edge_dim),
            compression="lzf",
        )

        # Masks
        f.create_dataset(
            "face_masks",
            shape=(0, max_faces),
            maxshape=(total_samples, max_faces),
            dtype=np.float32,
            chunks=(100, max_faces),
            compression="lzf",
        )
        f.create_dataset(
            "edge_masks",
            shape=(0, max_edges),
            maxshape=(total_samples, max_edges),
            dtype=np.float32,
            chunks=(100, max_edges),
            compression="lzf",
        )

        # BFS ordering info
        f.create_dataset(
            "bfs_to_original_face",
            shape=(0, max_faces),
            maxshape=(total_samples, max_faces),
            dtype=np.int32,
            chunks=(100, max_faces),
            compression="lzf",
        )
        f.create_dataset(
            "bfs_to_original_edge",
            shape=(0, max_edges),
            maxshape=(total_samples, max_edges),
            dtype=np.int32,
            chunks=(100, max_edges),
            compression="lzf",
        )
        f.create_dataset(
            "bfs_level",
            shape=(0, max_faces),
            maxshape=(total_samples, max_faces),
            dtype=np.int32,
            chunks=(100, max_faces),
            compression="lzf",
        )
        f.create_dataset(
            "bfs_parent_face",
            shape=(0, max_faces),
            maxshape=(total_samples, max_faces),
            dtype=np.int32,
            chunks=(100, max_faces),
            compression="lzf",
        )
        f.create_dataset(
            "bfs_parent_edge",
            shape=(0, max_faces),
            maxshape=(total_samples, max_faces),
            dtype=np.int32,
            chunks=(100, max_faces),
            compression="lzf",
        )

        # Topology
        f.create_dataset(
            "edge_to_faces",
            shape=(0, max_edges, 2),
            maxshape=(total_samples, max_edges, 2),
            dtype=np.int32,
            chunks=(100, max_edges, 2),
            compression="lzf",
        )

        # Spatial (for quick viz without raw file)
        f.create_dataset(
            "face_centroids",
            shape=(0, max_faces, 3),
            maxshape=(total_samples, max_faces, 3),
            dtype=np.float32,
            chunks=(100, max_faces, 3),
            compression="lzf",
        )

        # Counts
        f.create_dataset(
            "num_faces",
            shape=(0,),
            maxshape=(total_samples,),
            dtype=np.int32,
        )
        f.create_dataset(
            "num_edges",
            shape=(0,),
            maxshape=(total_samples,),
            dtype=np.int32,
        )

        # UIDs
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset(
            "uids",
            shape=(0,),
            maxshape=(total_samples,),
            dtype=dt,
        )

        # Metadata
        f.attrs["model_config"] = "autobrep_bfs"
        f.attrs["face_dim"] = face_dim
        f.attrs["edge_dim"] = edge_dim
        f.attrs["max_faces"] = max_faces
        f.attrs["max_edges"] = max_edges
        f.attrs["total_samples"] = total_samples
        f.attrs["n_samples"] = 0


def create_raw_hdf5(
    output_path: Path,
    max_faces: int,
    max_edges: int,
    face_grid_size: int,
    edge_curve_size: int,
    total_samples: int,
):
    """Create the raw HDF5 file for visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Raw point grids (BFS ordered)
        f.create_dataset(
            "face_point_grids",
            shape=(0, max_faces, face_grid_size, face_grid_size, 3),
            maxshape=(total_samples, max_faces, face_grid_size, face_grid_size, 3),
            dtype=np.float32,
            chunks=(1, max_faces, face_grid_size, face_grid_size, 3),
            compression="lzf",
        )
        f.create_dataset(
            "edge_point_grids",
            shape=(0, max_edges, edge_curve_size, 3),
            maxshape=(total_samples, max_edges, edge_curve_size, 3),
            dtype=np.float32,
            chunks=(10, max_edges, edge_curve_size, 3),
            compression="lzf",
        )

        # Spatial properties
        f.create_dataset(
            "face_normals",
            shape=(0, max_faces, 3),
            maxshape=(total_samples, max_faces, 3),
            dtype=np.float32,
            chunks=(100, max_faces, 3),
            compression="lzf",
        )
        f.create_dataset(
            "face_areas",
            shape=(0, max_faces),
            maxshape=(total_samples, max_faces),
            dtype=np.float32,
            chunks=(100, max_faces),
            compression="lzf",
        )
        f.create_dataset(
            "face_bboxes",
            shape=(0, max_faces, 6),
            maxshape=(total_samples, max_faces, 6),
            dtype=np.float32,
            chunks=(100, max_faces, 6),
            compression="lzf",
        )
        f.create_dataset(
            "edge_midpoints",
            shape=(0, max_edges, 3),
            maxshape=(total_samples, max_edges, 3),
            dtype=np.float32,
            chunks=(100, max_edges, 3),
            compression="lzf",
        )
        f.create_dataset(
            "edge_directions",
            shape=(0, max_edges, 3),
            maxshape=(total_samples, max_edges, 3),
            dtype=np.float32,
            chunks=(100, max_edges, 3),
            compression="lzf",
        )
        f.create_dataset(
            "edge_lengths",
            shape=(0, max_edges),
            maxshape=(total_samples, max_edges),
            dtype=np.float32,
            chunks=(100, max_edges),
            compression="lzf",
        )
        f.create_dataset(
            "edge_bboxes",
            shape=(0, max_edges, 6),
            maxshape=(total_samples, max_edges, 6),
            dtype=np.float32,
            chunks=(100, max_edges, 6),
            compression="lzf",
        )

        # Semantic types
        f.create_dataset(
            "face_types",
            shape=(0, max_faces),
            maxshape=(total_samples, max_faces),
            dtype=np.int8,
            chunks=(100, max_faces),
            compression="lzf",
        )
        f.create_dataset(
            "edge_types",
            shape=(0, max_edges),
            maxshape=(total_samples, max_edges),
            dtype=np.int8,
            chunks=(100, max_edges),
            compression="lzf",
        )

        # Face adjacency (sparse)
        f.create_dataset(
            "face_adjacency",
            shape=(0, max_faces, max_faces),
            maxshape=(total_samples, max_faces, max_faces),
            dtype=np.int8,
            chunks=(1, max_faces, max_faces),
            compression="lzf",
        )

        # UIDs (for joining with main file)
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset(
            "uids",
            shape=(0,),
            maxshape=(total_samples,),
            dtype=dt,
        )

        # Metadata
        f.attrs["face_grid_size"] = face_grid_size
        f.attrs["edge_curve_size"] = edge_curve_size
        f.attrs["max_faces"] = max_faces
        f.attrs["max_edges"] = max_edges
        f.attrs["n_samples"] = 0


def append_to_main_hdf5(
    output_path: Path,
    data: Dict[str, np.ndarray],
    uids: List[str],
):
    """Append batch to main HDF5 file."""
    with h5py.File(output_path, "a") as f:
        current_size = f["face_features"].shape[0]
        batch_size = data["face_features"].shape[0]
        new_size = current_size + batch_size

        # Resize all datasets
        for key in ["face_features", "edge_features", "face_masks", "edge_masks",
                    "bfs_to_original_face", "bfs_to_original_edge", "bfs_level",
                    "bfs_parent_face", "bfs_parent_edge", "edge_to_faces",
                    "face_centroids", "num_faces", "num_edges", "uids"]:
            f[key].resize(new_size, axis=0)

        # Write data
        f["face_features"][current_size:new_size] = data["face_features"]
        f["edge_features"][current_size:new_size] = data["edge_features"]
        f["face_masks"][current_size:new_size] = data["face_masks"]
        f["edge_masks"][current_size:new_size] = data["edge_masks"]
        f["bfs_to_original_face"][current_size:new_size] = data["bfs_to_original_face"]
        f["bfs_to_original_edge"][current_size:new_size] = data["bfs_to_original_edge"]
        f["bfs_level"][current_size:new_size] = data["bfs_level"]
        f["bfs_parent_face"][current_size:new_size] = data["bfs_parent_face"]
        f["bfs_parent_edge"][current_size:new_size] = data["bfs_parent_edge"]
        f["edge_to_faces"][current_size:new_size] = data["edge_to_faces"]
        f["face_centroids"][current_size:new_size] = data["face_centroids"]
        f["num_faces"][current_size:new_size] = data["num_faces"]
        f["num_edges"][current_size:new_size] = data["num_edges"]

        for i, uid in enumerate(uids):
            f["uids"][current_size + i] = uid

        f.attrs["n_samples"] = new_size
        f.flush()


def append_to_raw_hdf5(
    output_path: Path,
    data: Dict[str, np.ndarray],
    uids: List[str],
):
    """Append batch to raw HDF5 file."""
    with h5py.File(output_path, "a") as f:
        current_size = f["face_point_grids"].shape[0]
        batch_size = data["face_point_grids"].shape[0]
        new_size = current_size + batch_size

        # Resize all datasets
        for key in ["face_point_grids", "edge_point_grids", "face_normals",
                    "face_areas", "face_bboxes", "edge_midpoints", "edge_directions",
                    "edge_lengths", "edge_bboxes", "face_types", "edge_types",
                    "face_adjacency", "uids"]:
            f[key].resize(new_size, axis=0)

        # Write data
        f["face_point_grids"][current_size:new_size] = data["face_point_grids"]
        f["edge_point_grids"][current_size:new_size] = data["edge_point_grids"]
        f["face_normals"][current_size:new_size] = data["face_normals"]
        f["face_areas"][current_size:new_size] = data["face_areas"]
        f["face_bboxes"][current_size:new_size] = data["face_bboxes"]
        f["edge_midpoints"][current_size:new_size] = data["edge_midpoints"]
        f["edge_directions"][current_size:new_size] = data["edge_directions"]
        f["edge_lengths"][current_size:new_size] = data["edge_lengths"]
        f["edge_bboxes"][current_size:new_size] = data["edge_bboxes"]
        f["face_types"][current_size:new_size] = data["face_types"]
        f["edge_types"][current_size:new_size] = data["edge_types"]
        f["face_adjacency"][current_size:new_size] = data["face_adjacency"]

        for i, uid in enumerate(uids):
            f["uids"][current_size + i] = uid

        f.attrs["n_samples"] = new_size
        f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute B-Rep features with AutoBrep BFS ordering"
    )
    parser.add_argument(
        "--step-dir",
        type=str,
        required=True,
        help="Directory containing STEP files",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file with uid column",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for features",
    )
    parser.add_argument(
        "--surface-checkpoint",
        type=str,
        default=None,
        help="Path to AutoBrep surface FSQ VAE checkpoint",
    )
    parser.add_argument(
        "--edge-checkpoint",
        type=str,
        default=None,
        help="Path to AutoBrep edge FSQ VAE checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GPU encoding",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = auto)",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=192,
        help="Maximum number of faces per sample",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=512,
        help="Maximum number of edges per sample",
    )
    parser.add_argument(
        "--face-grid-size",
        type=int,
        default=32,
        help="Face UV grid size",
    )
    parser.add_argument(
        "--edge-curve-size",
        type=int,
        default=32,
        help="Edge curve length",
    )
    parser.add_argument(
        "--face-dim",
        type=int,
        default=48,
        help="Face feature dimension",
    )
    parser.add_argument(
        "--edge-dim",
        type=int,
        default=12,
        help="Edge feature dimension",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for encoding",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N batches",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable auto-download of weights",
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        default=None,
        help="CSV file mapping modelnames to STEP UIDs",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip saving raw HDF5 file (faster but no visualization)",
    )

    args = parser.parse_args()

    # Determine number of workers
    import platform
    is_windows = platform.system() == "Windows"

    # On Windows: fewer workers due to spawn overhead, max 60 due to handle limits
    if is_windows:
        max_workers = 60
        default_workers = min(4, cpu_count())  # Lower default on Windows
    else:
        max_workers = cpu_count()
        default_workers = cpu_count()

    num_workers = min(args.num_workers if args.num_workers > 0 else default_workers, max_workers)

    step_dir = Path(args.step_dir)
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    main_output_path = output_dir / "brep_autobrep.h5"
    raw_output_path = output_dir / "brep_autobrep_raw.h5"
    checkpoint_path = output_dir / "brep_autobrep.checkpoint.json"
    device = torch.device(args.device)

    print("=" * 60)
    print("Pre-computing B-Rep Features with AutoBrep BFS Ordering")
    print("=" * 60)
    auto_download = not args.no_auto_download
    print(f"STEP directory: {step_dir}")
    print(f"CSV file: {csv_path}")
    print(f"Output (main): {main_output_path}")
    print(f"Output (raw): {raw_output_path}" + (" [SKIPPED]" if args.skip_raw else ""))
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Parallel workers: {num_workers}")
    print(f"Max faces: {args.max_faces}, Max edges: {args.max_edges}")
    print()

    # Check for resume
    processed_uids = set()
    if args.resume:
        processed_uids = load_processed_uids_from_h5(main_output_path)
        if processed_uids:
            print(f"Resuming: {len(processed_uids)} UIDs already in HDF5 file")
        else:
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint:
                processed_uids = set(checkpoint.get("processed_uids", []))
                print(f"Resuming from checkpoint: {len(processed_uids)} UIDs")
    else:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if main_output_path.exists():
            main_output_path.unlink()
        if raw_output_path.exists():
            raw_output_path.unlink()
        print("Cleared previous output files")
    print()

    # Load UID list
    print("Loading UID list from CSV...")
    uids = load_uid_list(csv_path)
    print(f"Found {len(uids)} UIDs in CSV")

    # Load UID mapping
    uid_mapping = None
    if args.mapping_csv:
        mapping_path = Path(args.mapping_csv)
        print(f"Loading UID mapping from {mapping_path}...")
        uid_mapping = load_uid_mapping(mapping_path)
        print(f"Loaded {len(uid_mapping)} mappings")

    # Find STEP files
    print("Finding STEP files...")
    samples = find_step_files(step_dir, uids, uid_mapping)
    print(f"Found {len(samples)} STEP files")

    # Filter already processed
    if processed_uids:
        samples = [s for s in samples if s["uid"] not in processed_uids]
        print(f"Remaining after filtering: {len(samples)}")
    print()

    if len(samples) == 0:
        print("No samples to process!")
        return

    # Import autobrep utils
    from clip4cad.data.autobrep_utils import (
        bfs_order_faces_with_parents,
        bfs_order_edges,
        extract_spatial_properties,
        extract_geometry_types,
        build_face_adjacency_matrix,
    )

    # Create B-Rep encoder
    print("Creating B-Rep encoder...")
    from clip4cad.models.encoders.brep_encoder import BRepEncoder

    model = BRepEncoder(
        face_dim=args.face_dim,
        edge_dim=args.edge_dim,
        face_grid_size=args.face_grid_size,
        edge_curve_size=args.edge_curve_size,
        surface_checkpoint=args.surface_checkpoint,
        edge_checkpoint=args.edge_checkpoint,
        auto_download=auto_download,
        freeze=True,
    )
    model = model.to(device)
    model.eval()
    print(f"Model created (face_dim: {model.face_dim}, edge_dim: {model.edge_dim})")
    print()

    # Create HDF5 files if needed
    total_samples = len(samples) + len(processed_uids)
    is_first_write = not main_output_path.exists()

    if is_first_write:
        create_main_hdf5(main_output_path, args.max_faces, args.max_edges,
                        args.face_dim, args.edge_dim, total_samples)
        if not args.skip_raw:
            create_raw_hdf5(raw_output_path, args.max_faces, args.max_edges,
                          args.face_grid_size, args.edge_curve_size, total_samples)

    # Prepare worker arguments
    worker_args = [
        (sample, args.max_faces, args.max_edges, args.face_grid_size, args.edge_curve_size)
        for sample in samples
    ]

    # Process with multiprocessing
    print(f"Extracting geometry with {num_workers} workers...")

    all_processed_uids = list(processed_uids)

    # Geometry buffers (pre-GPU)
    geom_buffer = []

    # Feature buffers (post-GPU, post-BFS)
    main_buffer = {
        "face_features": [],
        "edge_features": [],
        "face_masks": [],
        "edge_masks": [],
        "bfs_to_original_face": [],
        "bfs_to_original_edge": [],
        "bfs_level": [],
        "bfs_parent_face": [],
        "bfs_parent_edge": [],
        "edge_to_faces": [],
        "face_centroids": [],
        "num_faces": [],
        "num_edges": [],
    }
    raw_buffer = {
        "face_point_grids": [],
        "edge_point_grids": [],
        "face_normals": [],
        "face_areas": [],
        "face_bboxes": [],
        "edge_midpoints": [],
        "edge_directions": [],
        "edge_lengths": [],
        "edge_bboxes": [],
        "face_types": [],
        "edge_types": [],
        "face_adjacency": [],
    }
    buffer_uids = []

    processed_count = 0
    failed_count = 0
    batch_count = 0
    gpu_batch_size = args.batch_size

    def process_geometry_batch():
        """Process geometry batch: normalize, encode, compute BFS ordering."""
        nonlocal geom_buffer, main_buffer, raw_buffer, buffer_uids

        if not geom_buffer:
            return

        # Stack geometry for GPU encoding
        batch_faces = []
        batch_edges = []
        batch_face_masks = []
        batch_edge_masks = []

        for result in geom_buffer:
            num_f = result["num_faces"]
            num_e = result["num_edges"]

            face_mask = np.zeros(args.max_faces, dtype=np.float32)
            face_mask[:num_f] = 1.0
            edge_mask = np.zeros(args.max_edges, dtype=np.float32)
            edge_mask[:num_e] = 1.0

            # Normalize geometry for encoding
            norm_faces, norm_edges, norm_face_mask, norm_edge_mask = normalize_geometry(
                result["faces"], result["edges"], face_mask, edge_mask
            )

            batch_faces.append(norm_faces)
            batch_edges.append(norm_edges)
            batch_face_masks.append(norm_face_mask)
            batch_edge_masks.append(norm_edge_mask)

        # GPU encoding
        faces_tensor = torch.from_numpy(np.stack(batch_faces, axis=0)).to(device)
        edges_tensor = torch.from_numpy(np.stack(batch_edges, axis=0)).to(device)
        face_mask_tensor = torch.from_numpy(np.stack(batch_face_masks, axis=0)).to(device)
        edge_mask_tensor = torch.from_numpy(np.stack(batch_edge_masks, axis=0)).to(device)

        with torch.no_grad():
            face_feats, edge_feats = model(faces_tensor, edges_tensor, face_mask_tensor, edge_mask_tensor)

        face_feats = face_feats.cpu().numpy()
        edge_feats = edge_feats.cpu().numpy()
        batch_face_masks = np.stack(batch_face_masks, axis=0)
        batch_edge_masks = np.stack(batch_edge_masks, axis=0)

        # Process each sample: BFS ordering and spatial properties
        for i, result in enumerate(geom_buffer):
            num_f = result["num_faces"]
            num_e = result["num_edges"]

            # Get face-edge incidence for this sample
            face_edge_inc = result["face_edge_incidence"][:num_f, :num_e]

            # Compute BFS ordering for faces
            face_bfs_info = bfs_order_faces_with_parents(
                face_edge_inc,
                result["face_bboxes"][:num_f]
            )

            # Compute BFS ordering for edges
            edge_bfs_info = bfs_order_edges(
                face_edge_inc,
                face_bfs_info["bfs_to_original_face"],
                result["edge_bboxes"][:num_e]
            )

            # Extract spatial properties from raw geometry
            spatial = extract_spatial_properties(
                result["faces"][:num_f],
                result["edges"][:num_e]
            )

            # Classify geometry types
            face_types, edge_types = extract_geometry_types(
                result["faces"][:num_f],
                result["edges"][:num_e]
            )

            # Build face adjacency matrix (in BFS order)
            face_adj = build_face_adjacency_matrix(
                face_edge_inc,
                face_bfs_info["bfs_to_original_face"]
            )

            # Reorder features and geometry to BFS order
            bfs_face_order = face_bfs_info["bfs_to_original_face"]
            bfs_edge_order = edge_bfs_info["bfs_to_original_edge"]

            # Pad BFS indices to max size
            bfs_to_orig_face_padded = np.zeros(args.max_faces, dtype=np.int32)
            bfs_to_orig_edge_padded = np.zeros(args.max_edges, dtype=np.int32)
            bfs_level_padded = np.zeros(args.max_faces, dtype=np.int32)
            bfs_parent_face_padded = np.full(args.max_faces, -1, dtype=np.int32)
            bfs_parent_edge_padded = np.full(args.max_faces, -1, dtype=np.int32)
            edge_to_faces_padded = np.full((args.max_edges, 2), -1, dtype=np.int32)
            face_adj_padded = np.zeros((args.max_faces, args.max_faces), dtype=np.int8)

            bfs_to_orig_face_padded[:num_f] = bfs_face_order
            bfs_to_orig_edge_padded[:num_e] = bfs_edge_order
            bfs_level_padded[:num_f] = face_bfs_info["bfs_level"]
            bfs_parent_face_padded[:num_f] = face_bfs_info["bfs_parent_face"]
            bfs_parent_edge_padded[:num_f] = face_bfs_info["bfs_parent_edge"]
            edge_to_faces_padded[:num_e] = edge_bfs_info["edge_to_faces"]
            face_adj_padded[:num_f, :num_f] = face_adj

            # Reorder features to BFS order
            face_feats_bfs = np.zeros((args.max_faces, args.face_dim), dtype=np.float32)
            edge_feats_bfs = np.zeros((args.max_edges, args.edge_dim), dtype=np.float32)
            face_feats_bfs[:num_f] = face_feats[i, bfs_face_order]
            edge_feats_bfs[:num_e] = edge_feats[i, bfs_edge_order]

            # Reorder masks to BFS order
            face_mask_bfs = np.zeros(args.max_faces, dtype=np.float32)
            edge_mask_bfs = np.zeros(args.max_edges, dtype=np.float32)
            face_mask_bfs[:num_f] = batch_face_masks[i, bfs_face_order]
            edge_mask_bfs[:num_e] = batch_edge_masks[i, bfs_edge_order]

            # Reorder spatial properties to BFS order
            face_centroids_bfs = np.zeros((args.max_faces, 3), dtype=np.float32)
            face_normals_bfs = np.zeros((args.max_faces, 3), dtype=np.float32)
            face_areas_bfs = np.zeros(args.max_faces, dtype=np.float32)
            face_bboxes_bfs = np.zeros((args.max_faces, 6), dtype=np.float32)
            edge_midpoints_bfs = np.zeros((args.max_edges, 3), dtype=np.float32)
            edge_directions_bfs = np.zeros((args.max_edges, 3), dtype=np.float32)
            edge_lengths_bfs = np.zeros(args.max_edges, dtype=np.float32)
            edge_bboxes_bfs = np.zeros((args.max_edges, 6), dtype=np.float32)
            face_types_bfs = np.zeros(args.max_faces, dtype=np.int8)
            edge_types_bfs = np.zeros(args.max_edges, dtype=np.int8)

            face_centroids_bfs[:num_f] = spatial["face_centroids"][bfs_face_order]
            face_normals_bfs[:num_f] = spatial["face_normals"][bfs_face_order]
            face_areas_bfs[:num_f] = spatial["face_areas"][bfs_face_order]
            face_bboxes_bfs[:num_f] = spatial["face_bboxes"][bfs_face_order]
            edge_midpoints_bfs[:num_e] = spatial["edge_midpoints"][bfs_edge_order]
            edge_directions_bfs[:num_e] = spatial["edge_directions"][bfs_edge_order]
            edge_lengths_bfs[:num_e] = spatial["edge_lengths"][bfs_edge_order]
            edge_bboxes_bfs[:num_e] = spatial["edge_bboxes"][bfs_edge_order]
            face_types_bfs[:num_f] = face_types[bfs_face_order]
            edge_types_bfs[:num_e] = edge_types[bfs_edge_order]

            # Reorder raw point grids to BFS order
            face_grids_bfs = np.zeros((args.max_faces, args.face_grid_size, args.face_grid_size, 3), dtype=np.float32)
            edge_grids_bfs = np.zeros((args.max_edges, args.edge_curve_size, 3), dtype=np.float32)
            face_grids_bfs[:num_f] = result["faces"][:num_f][bfs_face_order]
            edge_grids_bfs[:num_e] = result["edges"][:num_e][bfs_edge_order]

            # Add to buffers
            main_buffer["face_features"].append(face_feats_bfs)
            main_buffer["edge_features"].append(edge_feats_bfs)
            main_buffer["face_masks"].append(face_mask_bfs)
            main_buffer["edge_masks"].append(edge_mask_bfs)
            main_buffer["bfs_to_original_face"].append(bfs_to_orig_face_padded)
            main_buffer["bfs_to_original_edge"].append(bfs_to_orig_edge_padded)
            main_buffer["bfs_level"].append(bfs_level_padded)
            main_buffer["bfs_parent_face"].append(bfs_parent_face_padded)
            main_buffer["bfs_parent_edge"].append(bfs_parent_edge_padded)
            main_buffer["edge_to_faces"].append(edge_to_faces_padded)
            main_buffer["face_centroids"].append(face_centroids_bfs)
            main_buffer["num_faces"].append(num_f)
            main_buffer["num_edges"].append(num_e)

            raw_buffer["face_point_grids"].append(face_grids_bfs)
            raw_buffer["edge_point_grids"].append(edge_grids_bfs)
            raw_buffer["face_normals"].append(face_normals_bfs)
            raw_buffer["face_areas"].append(face_areas_bfs)
            raw_buffer["face_bboxes"].append(face_bboxes_bfs)
            raw_buffer["edge_midpoints"].append(edge_midpoints_bfs)
            raw_buffer["edge_directions"].append(edge_directions_bfs)
            raw_buffer["edge_lengths"].append(edge_lengths_bfs)
            raw_buffer["edge_bboxes"].append(edge_bboxes_bfs)
            raw_buffer["face_types"].append(face_types_bfs)
            raw_buffer["edge_types"].append(edge_types_bfs)
            raw_buffer["face_adjacency"].append(face_adj_padded)

            buffer_uids.append(result["uid"])

        # Clear geometry buffer
        geom_buffer = []

    def save_buffers():
        """Save buffered data to HDF5 files."""
        nonlocal main_buffer, raw_buffer, buffer_uids, batch_count

        if not buffer_uids:
            return

        # Stack main buffer arrays
        main_data = {k: np.stack(v, axis=0) for k, v in main_buffer.items()}
        append_to_main_hdf5(main_output_path, main_data, buffer_uids)

        # Stack and save raw buffer
        if not args.skip_raw:
            raw_data = {k: np.stack(v, axis=0) for k, v in raw_buffer.items()}
            append_to_raw_hdf5(raw_output_path, raw_data, buffer_uids)

        # Clear buffers
        for k in main_buffer:
            main_buffer[k] = []
        for k in raw_buffer:
            raw_buffer[k] = []
        buffer_uids.clear()

        batch_count += 1

    # Process with multiprocessing pool
    # Use maxtasksperchild to prevent memory leaks in workers
    # Use larger chunksize on Windows for less IPC overhead
    chunksize = 20 if is_windows else 10

    try:
        with Pool(num_workers, maxtasksperchild=100) as pool:
            pbar = tqdm(
                pool.imap_unordered(process_single_file, worker_args, chunksize=chunksize),
                total=len(worker_args),
                desc="Processing STEP files"
            )

            for result in pbar:
                if result is None:
                    failed_count += 1
                    continue

                geom_buffer.append(result)
                all_processed_uids.append(result["uid"])
                processed_count += 1

                # Process when we have a full batch
                if len(geom_buffer) >= gpu_batch_size:
                    process_geometry_batch()
                    pbar.set_postfix({"encoded": len(buffer_uids), "failed": failed_count})

                # Checkpoint (save to disk)
                if len(buffer_uids) >= gpu_batch_size * args.checkpoint_every:
                    pbar.set_postfix({"status": "saving..."})

                    # Process any remaining geometry
                    process_geometry_batch()

                    # Save to HDF5
                    save_buffers()
                    save_checkpoint(checkpoint_path, processed_count, all_processed_uids)

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    pbar.set_postfix({"saved": batch_count, "failed": failed_count})

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        process_geometry_batch()
        save_buffers()
        save_checkpoint(checkpoint_path, processed_count, all_processed_uids)
        print(f"Checkpoint saved. Processed {processed_count} samples.")
        print("Run with --resume to continue.")
        return
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        print("Saving checkpoint...")
        process_geometry_batch()
        save_buffers()
        save_checkpoint(checkpoint_path, processed_count, all_processed_uids)
        print(f"Checkpoint saved. Processed {processed_count} samples.")
        print("Run with --resume to continue.")
        raise

    # Process remaining geometry
    process_geometry_batch()

    # Save remaining features
    if buffer_uids:
        print("Saving final batch...")
        save_buffers()

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Cleaned up checkpoint file")

    # Print stats
    print("\n" + "=" * 60)
    print("Pre-computation complete!")
    print("=" * 60)
    print(f"Processed: {processed_count}")
    print(f"Failed: {failed_count}")

    if main_output_path.exists():
        with h5py.File(main_output_path, "r") as f:
            print(f"\nMain output: {main_output_path}")
            print(f"  - face_features: {f['face_features'].shape}")
            print(f"  - edge_features: {f['edge_features'].shape}")
            print(f"  - bfs_to_original_face: {f['bfs_to_original_face'].shape}")
            print(f"  - n_samples: {f.attrs['n_samples']}")

    if raw_output_path.exists() and not args.skip_raw:
        with h5py.File(raw_output_path, "r") as f:
            print(f"\nRaw output: {raw_output_path}")
            print(f"  - face_point_grids: {f['face_point_grids'].shape}")
            print(f"  - edge_point_grids: {f['edge_point_grids'].shape}")
            print(f"  - n_samples: {f.attrs['n_samples']}")


if __name__ == "__main__":
    main()
