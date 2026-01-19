#!/usr/bin/env python3
"""
Pre-compute B-Rep features from STEP files using AutoBrep encoder.

This script reads STEP files and extracts B-Rep features (face grids and edge curves),
then encodes them using the AutoBrep-style encoder.

Features:
- Auto-download pretrained weights from HuggingFace (default behavior)
- Multiprocessing for parallel geometry extraction (8-16x speedup)
- Checkpointing every N batches (default 100) to allow resuming
- Resume from checkpoint with --resume flag
- Robust error handling for invalid geometry
- Fast LZF compression for quick saves (full float32 precision)

Requirements:
    - pythonOCC-core (OpenCASCADE bindings)
    Install with: conda install -c conda-forge pythonocc-core

Usage:
    # Fast parallel mode with auto-download (recommended)
    python scripts/precompute_brep_features_step.py --step-dir ../data/extracted_step_files --csv ../data/169k.csv --output-dir ../data/embeddings --num-workers 8

    # Resume from checkpoint
    python scripts/precompute_brep_features_step.py --step-dir ../data/extracted_step_files --csv ../data/169k.csv --output-dir ../data/embeddings --resume

    # Without auto-download (random init or provide checkpoint)
    python scripts/precompute_brep_features_step.py --step-dir ../data/extracted_step_files --csv ../data/169k.csv --output-dir ../data/embeddings --no-auto-download

Output:
    Creates HDF5 file with structure:
    - brep_features.h5
        - face_features: [N, max_faces, face_dim]
        - edge_features: [N, max_edges, edge_dim]
        - face_masks: [N, max_faces]
        - edge_masks: [N, max_edges]
        - num_faces: [N]
        - num_edges: [N]
        - uids: [N] (string dataset)
"""

import argparse
import gc
import json
import os
import sys
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

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
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCC.Core.TopoDS import topods
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
        return {
            'STEPControl_Reader': STEPControl_Reader,
            'IFSelect_RetDone': IFSelect_RetDone,
            'TopExp_Explorer': TopExp_Explorer,
            'TopAbs_FACE': TopAbs_FACE,
            'TopAbs_EDGE': TopAbs_EDGE,
            'topods': topods,
            'BRepAdaptor_Surface': BRepAdaptor_Surface,
            'BRepAdaptor_Curve': BRepAdaptor_Curve,
        }
    except Exception as e:
        return None


def load_uid_list(csv_path: Path) -> list:
    """Load list of UIDs from CSV file."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["uid"])
    return [str(uid) for uid in df["uid"].tolist()]


def find_step_files(step_dir: Path, uids: list) -> list:
    """Find STEP files for given UIDs."""
    samples = []
    missing = 0

    for uid in uids:
        step_path = step_dir / f"{uid}.step"
        if not step_path.exists():
            step_path = step_dir / f"{uid}.stp"
        if step_path.exists():
            samples.append({
                "uid": uid,
                "path": str(step_path),
            })
        else:
            missing += 1

    if missing > 0:
        print(f"Warning: {missing} STEP files not found")

    return samples


def process_single_file(args):
    """
    Worker function: Extract geometry from a single STEP file.

    This runs in a separate process with its own pythonOCC import.

    Args:
        args: tuple of (sample_dict, max_faces, max_edges, face_grid_size, edge_curve_size)

    Returns:
        dict with extracted geometry or None if failed
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

        # Check for null shape
        if shape.IsNull():
            return None

        # Extract faces
        faces = []
        try:
            face_explorer = occ['TopExp_Explorer'](shape, occ['TopAbs_FACE'])
            while face_explorer.More():
                try:
                    face = occ['topods'].Face(face_explorer.Current())
                    grid = sample_face_grid(face, face_grid_size, occ)
                    if grid is not None:
                        faces.append(grid)
                except Exception:
                    pass  # Skip problematic faces
                face_explorer.Next()
        except Exception:
            pass  # Continue with whatever faces we got

        # Extract edges
        edges = []
        try:
            edge_explorer = occ['TopExp_Explorer'](shape, occ['TopAbs_EDGE'])
            while edge_explorer.More():
                try:
                    edge = occ['topods'].Edge(edge_explorer.Current())
                    curve = sample_edge_curve(edge, edge_curve_size, occ)
                    if curve is not None:
                        edges.append(curve)
                except Exception:
                    pass  # Skip problematic edges
                edge_explorer.Next()
        except Exception:
            pass  # Continue with whatever edges we got

        # Need at least some geometry
        if len(faces) == 0 and len(edges) == 0:
            return None

        # Convert to padded arrays
        num_faces = len(faces)
        num_edges = len(edges)

        face_array = np.zeros((max_faces, face_grid_size, face_grid_size, 3), dtype=np.float32)
        edge_array = np.zeros((max_edges, edge_curve_size, 3), dtype=np.float32)

        for i, face in enumerate(faces[:max_faces]):
            face_array[i] = face
        for i, edge in enumerate(edges[:max_edges]):
            edge_array[i] = edge

        # Create masks
        face_mask = np.zeros(max_faces, dtype=np.float32)
        face_mask[:min(num_faces, max_faces)] = 1.0

        edge_mask = np.zeros(max_edges, dtype=np.float32)
        edge_mask[:min(num_edges, max_edges)] = 1.0

        # Normalize each face/edge to [-1, 1] bounding box
        for i in range(min(num_faces, max_faces)):
            face = face_array[i]
            face_min = face.min(axis=(0, 1), keepdims=True)
            face_max = face.max(axis=(0, 1), keepdims=True)
            face_range = face_max - face_min
            # Skip degenerate faces (all zeros or very small range)
            if face_range.max() < 1e-8:
                face_mask[i] = 0.0  # Mark as invalid
                continue
            face_array[i] = 2.0 * (face - face_min) / np.maximum(face_range, 1e-8) - 1.0

        for i in range(min(num_edges, max_edges)):
            edge = edge_array[i]
            edge_min = edge.min(axis=0, keepdims=True)
            edge_max = edge.max(axis=0, keepdims=True)
            edge_range = edge_max - edge_min
            # Skip degenerate edges
            if edge_range.max() < 1e-8:
                edge_mask[i] = 0.0  # Mark as invalid
                continue
            edge_array[i] = 2.0 * (edge - edge_min) / np.maximum(edge_range, 1e-8) - 1.0

        return {
            "uid": uid,
            "faces": face_array,
            "edges": edge_array,
            "face_mask": face_mask,
            "edge_mask": edge_mask,
            "num_faces": min(num_faces, max_faces),
            "num_edges": min(num_edges, max_edges),
        }

    except Exception as e:
        return None


def sample_face_grid(face, grid_size, occ):
    """
    Sample a face as a UV grid of 3D points.

    Args:
        face: TopoDS_Face
        grid_size: Size of the UV grid
        occ: Dict of OCC modules

    Returns:
        grid: [grid_size, grid_size, 3] array of 3D points or None if failed
    """
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
    """
    Sample an edge as a curve of 3D points.

    Args:
        edge: TopoDS_Edge
        num_points: Number of points to sample
        occ: Dict of OCC modules

    Returns:
        curve: [num_points, 3] array of 3D points or None if failed
    """
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


def create_batches(items, batch_size):
    """Create batches from items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def save_checkpoint(checkpoint_path: Path, processed_count: int, processed_uids: list):
    """Save checkpoint information."""
    checkpoint = {
        "processed_count": processed_count,
        "processed_uids": processed_uids[-10000:],  # Keep last 10K UIDs for dedup
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint(checkpoint_path: Path):
    """Load checkpoint information."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None


def append_to_hdf5(
    output_path: Path,
    face_features: np.ndarray,
    edge_features: np.ndarray,
    face_masks: np.ndarray,
    edge_masks: np.ndarray,
    num_faces: np.ndarray,
    num_edges: np.ndarray,
    uids: list,
    model_config: str,
    face_dim: int,
    edge_dim: int,
    max_faces: int,
    max_edges: int,
    total_samples: int,
    is_first_write: bool = False,
):
    """Append features to HDF5 file (or create if first write)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_first_write:
        with h5py.File(output_path, "w") as f:
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
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset(
                "uids",
                shape=(0,),
                maxshape=(total_samples,),
                dtype=dt,
            )
            f.attrs["model_config"] = model_config
            f.attrs["face_dim"] = face_dim
            f.attrs["edge_dim"] = edge_dim
            f.attrs["max_faces"] = max_faces
            f.attrs["max_edges"] = max_edges
            f.attrs["total_samples"] = total_samples
            f.attrs["n_samples"] = 0

    with h5py.File(output_path, "a") as f:
        current_size = f["face_features"].shape[0]
        new_size = current_size + face_features.shape[0]

        f["face_features"].resize(new_size, axis=0)
        f["edge_features"].resize(new_size, axis=0)
        f["face_masks"].resize(new_size, axis=0)
        f["edge_masks"].resize(new_size, axis=0)
        f["num_faces"].resize(new_size, axis=0)
        f["num_edges"].resize(new_size, axis=0)
        f["uids"].resize(new_size, axis=0)

        f["face_features"][current_size:new_size] = face_features
        f["edge_features"][current_size:new_size] = edge_features
        f["face_masks"][current_size:new_size] = face_masks
        f["edge_masks"][current_size:new_size] = edge_masks
        f["num_faces"][current_size:new_size] = num_faces
        f["num_edges"][current_size:new_size] = num_edges
        for i, uid in enumerate(uids):
            f["uids"][current_size + i] = uid

        f.attrs["n_samples"] = new_size
        f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute B-Rep features from STEP files"
    )
    parser.add_argument(
        "--step-dir",
        type=str,
        required=True,
        help="Directory containing STEP files named by UID",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file with uid column to determine which files to process",
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
        help="Number of parallel workers for geometry extraction (0 = auto)",
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
        help="Face UV grid size (must match AutoBrep: 32)",
    )
    parser.add_argument(
        "--edge-curve-size",
        type=int,
        default=32,
        help="Edge curve length (must match AutoBrep: 32)",
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
        help="Device to use for encoding",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N batches (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="brep_features.h5",
        help="Output filename",
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable auto-download of pretrained weights from HuggingFace",
    )

    args = parser.parse_args()

    # Determine number of workers
    # Windows has a limit of 63 handles for WaitForMultipleObjects
    import platform
    max_workers = 60 if platform.system() == "Windows" else cpu_count()
    num_workers = min(args.num_workers if args.num_workers > 0 else cpu_count(), max_workers)

    step_dir = Path(args.step_dir)
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name
    checkpoint_path = output_dir / f"{args.output_name}.checkpoint.json"
    device = torch.device(args.device)

    print("=" * 60)
    print("Pre-computing B-Rep Features from STEP Files")
    print("=" * 60)
    auto_download = not args.no_auto_download
    print(f"STEP directory: {step_dir}")
    print(f"CSV file: {csv_path}")
    print(f"Output: {output_path}")
    if auto_download and not args.surface_checkpoint and not args.edge_checkpoint:
        print(f"Weights: auto-download from HuggingFace")
    else:
        print(f"Surface checkpoint: {args.surface_checkpoint or 'None (random init)'}")
        print(f"Edge checkpoint: {args.edge_checkpoint or 'None (random init)'}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Parallel workers: {num_workers}")
    print(f"Max faces: {args.max_faces}, Max edges: {args.max_edges}")
    print(f"Face dim: {args.face_dim}, Edge dim: {args.edge_dim}")
    print(f"Checkpoint every: {args.checkpoint_every} batches")
    print()

    # Check for resume
    processed_uids = set()
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            processed_uids = set(checkpoint.get("processed_uids", []))
            print(f"Resuming: {len(processed_uids)} UIDs already processed")
        else:
            print("No checkpoint found, starting from beginning")
    else:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if output_path.exists():
            output_path.unlink()
            print("Cleared previous output file")
    print()

    # Load UID list from CSV
    print("Loading UID list from CSV...")
    uids = load_uid_list(csv_path)
    print(f"Found {len(uids)} UIDs in CSV")

    # Find STEP files
    print("Finding STEP files...")
    samples = find_step_files(step_dir, uids)
    print(f"Found {len(samples)} STEP files")

    # Filter out already processed
    if processed_uids:
        samples = [s for s in samples if s["uid"] not in processed_uids]
        print(f"Remaining after filtering processed: {len(samples)}")
    print()

    if len(samples) == 0:
        print("No samples to process!")
        return

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

    print(f"Model created")
    print(f"  - face_dim: {model.face_dim}")
    print(f"  - edge_dim: {model.edge_dim}")
    print()

    # Prepare worker arguments
    worker_args = [
        (sample, args.max_faces, args.max_edges, args.face_grid_size, args.edge_curve_size)
        for sample in samples
    ]

    # Process with multiprocessing
    print(f"Extracting geometry with {num_workers} workers...")

    total_samples = len(samples)
    is_first_write = not output_path.exists()
    all_processed_uids = list(processed_uids)

    # Buffers for geometry (pre-GPU)
    geom_buffer_faces = []
    geom_buffer_edges = []
    geom_buffer_face_masks = []
    geom_buffer_edge_masks = []
    geom_buffer_num_faces = []
    geom_buffer_num_edges = []
    geom_buffer_uids = []

    # Buffers for encoded features (post-GPU)
    buffer_face_features = []
    buffer_edge_features = []
    buffer_face_masks = []
    buffer_edge_masks = []
    buffer_num_faces = []
    buffer_num_edges = []
    buffer_uids = []

    processed_count = 0
    failed_count = 0
    batch_count = 0
    gpu_batch_size = args.batch_size  # Batch size for GPU encoding

    def encode_geometry_batch():
        """Encode a batch of geometry through the GPU."""
        nonlocal geom_buffer_faces, geom_buffer_edges, geom_buffer_face_masks
        nonlocal geom_buffer_edge_masks, geom_buffer_num_faces, geom_buffer_num_edges
        nonlocal geom_buffer_uids, buffer_face_features, buffer_edge_features
        nonlocal buffer_face_masks, buffer_edge_masks, buffer_num_faces
        nonlocal buffer_num_edges, buffer_uids

        if not geom_buffer_faces:
            return

        # Stack geometry into batches
        faces_batch = torch.from_numpy(np.stack(geom_buffer_faces, axis=0)).to(device)
        edges_batch = torch.from_numpy(np.stack(geom_buffer_edges, axis=0)).to(device)
        face_mask_batch = torch.from_numpy(np.stack(geom_buffer_face_masks, axis=0)).to(device)
        edge_mask_batch = torch.from_numpy(np.stack(geom_buffer_edge_masks, axis=0)).to(device)

        # GPU encoding
        with torch.no_grad():
            face_feats, edge_feats = model(faces_batch, edges_batch, face_mask_batch, edge_mask_batch)

        # Move results to CPU and add to feature buffers
        buffer_face_features.append(face_feats.cpu().numpy())
        buffer_edge_features.append(edge_feats.cpu().numpy())
        buffer_face_masks.extend(geom_buffer_face_masks)
        buffer_edge_masks.extend(geom_buffer_edge_masks)
        buffer_num_faces.extend(geom_buffer_num_faces)
        buffer_num_edges.extend(geom_buffer_num_edges)
        buffer_uids.extend(geom_buffer_uids)

        # Clear geometry buffers
        geom_buffer_faces = []
        geom_buffer_edges = []
        geom_buffer_face_masks = []
        geom_buffer_edge_masks = []
        geom_buffer_num_faces = []
        geom_buffer_num_edges = []
        geom_buffer_uids = []

    # Use multiprocessing pool with imap_unordered for better throughput
    with Pool(num_workers) as pool:
        pbar = tqdm(
            pool.imap_unordered(process_single_file, worker_args, chunksize=10),
            total=len(worker_args),
            desc="Processing STEP files"
        )

        for result in pbar:
            if result is None:
                failed_count += 1
                continue

            # Add to geometry buffer
            geom_buffer_faces.append(result["faces"])
            geom_buffer_edges.append(result["edges"])
            geom_buffer_face_masks.append(result["face_mask"])
            geom_buffer_edge_masks.append(result["edge_mask"])
            geom_buffer_num_faces.append(result["num_faces"])
            geom_buffer_num_edges.append(result["num_edges"])
            geom_buffer_uids.append(result["uid"])
            all_processed_uids.append(result["uid"])

            processed_count += 1

            # Encode when we have a full batch
            if len(geom_buffer_faces) >= gpu_batch_size:
                encode_geometry_batch()
                pbar.set_postfix({"gpu_batches": len(buffer_face_features), "failed": failed_count})

            # Checkpoint (save to disk)
            if len(buffer_uids) >= gpu_batch_size * args.checkpoint_every:
                pbar.set_postfix({"status": "saving..."})

                # Encode any remaining geometry
                encode_geometry_batch()

                face_feat_array = np.concatenate(buffer_face_features, axis=0)
                edge_feat_array = np.concatenate(buffer_edge_features, axis=0)
                face_mask_array = np.stack(buffer_face_masks, axis=0)
                edge_mask_array = np.stack(buffer_edge_masks, axis=0)
                num_faces_array = np.array(buffer_num_faces)
                num_edges_array = np.array(buffer_num_edges)

                append_to_hdf5(
                    output_path=output_path,
                    face_features=face_feat_array,
                    edge_features=edge_feat_array,
                    face_masks=face_mask_array,
                    edge_masks=edge_mask_array,
                    num_faces=num_faces_array,
                    num_edges=num_edges_array,
                    uids=buffer_uids,
                    model_config="autobrep",
                    face_dim=args.face_dim,
                    edge_dim=args.edge_dim,
                    max_faces=args.max_faces,
                    max_edges=args.max_edges,
                    total_samples=total_samples,
                    is_first_write=is_first_write,
                )
                is_first_write = False

                save_checkpoint(checkpoint_path, processed_count, all_processed_uids)

                # Clear feature buffers
                buffer_face_features = []
                buffer_edge_features = []
                buffer_face_masks = []
                buffer_edge_masks = []
                buffer_num_faces = []
                buffer_num_edges = []
                buffer_uids = []

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_count += 1
                pbar.set_postfix({"saved": batch_count, "failed": failed_count})

    # Encode any remaining geometry
    encode_geometry_batch()

    # Save remaining features
    if buffer_uids:
        print("Saving final batch...")
        face_feat_array = np.concatenate(buffer_face_features, axis=0)
        edge_feat_array = np.concatenate(buffer_edge_features, axis=0)
        face_mask_array = np.stack(buffer_face_masks, axis=0)
        edge_mask_array = np.stack(buffer_edge_masks, axis=0)
        num_faces_array = np.array(buffer_num_faces)
        num_edges_array = np.array(buffer_num_edges)

        append_to_hdf5(
            output_path=output_path,
            face_features=face_feat_array,
            edge_features=edge_feat_array,
            face_masks=face_mask_array,
            edge_masks=edge_mask_array,
            num_faces=num_faces_array,
            num_edges=num_edges_array,
            uids=buffer_uids,
            model_config="autobrep",
            face_dim=args.face_dim,
            edge_dim=args.edge_dim,
            max_faces=args.max_faces,
            max_edges=args.max_edges,
            total_samples=total_samples,
            is_first_write=is_first_write,
        )

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

    if output_path.exists():
        with h5py.File(output_path, "r") as f:
            print(f"Output: {output_path}")
            print(f"  - face_features: {f['face_features'].shape}")
            print(f"  - edge_features: {f['edge_features'].shape}")
            print(f"  - n_samples: {f.attrs['n_samples']}")


if __name__ == "__main__":
    main()
