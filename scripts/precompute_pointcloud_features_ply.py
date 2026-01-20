#!/usr/bin/env python3
"""
Pre-compute point cloud features from PLY files using Point-BERT encoder.

This script reads PLY files from a directory and extracts Point-BERT encoder
outputs, saving them as HDF5 files for training.

Features:
- Checkpointing every N batches (default 100) to allow resuming
- Resume from checkpoint with --resume flag
- Incremental saving to avoid memory issues
- Fast LZF compression for quick saves (full float32 precision)

Usage:
    python scripts/precompute_pointcloud_features_ply.py --ply-dir ../data/abc_ply_organized --csv ../data/169k.csv --output-dir ../data/embeddings
    python scripts/precompute_pointcloud_features_ply.py --ply-dir ../data/abc_ply_organized --csv ../data/169k.csv --batch-size 16
    python scripts/precompute_pointcloud_features_ply.py --ply-dir ../data/abc_ply_organized --csv ../data/169k.csv --resume  # Resume from checkpoint

Output:
    Creates HDF5 file with structure:
    - pointcloud_features.h5
        - features: [N, num_tokens, embed_dim]  # CLS + group tokens (float32)
        - uids: [N] (string dataset)
"""

import argparse
import gc
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from clip4cad.models.encoders.pointbert_encoder import (
    PointBertEncoder,
    load_ply,
    download_ulip2_weights,
)


def load_uid_list(csv_path: Path) -> list:
    """Load list of UIDs from CSV file."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["uid"])
    return [str(uid) for uid in df["uid"].tolist()]


def load_uid_mapping(mapping_csv: Path) -> dict:
    """
    Load UID mapping from combined_sorted_models.csv.

    Maps stripped modelname -> PLY file uid.
    E.g., "72164" -> 14397 means 72164 in 169k.csv should use 14397.ply
    """
    if not mapping_csv.exists():
        return {}

    df = pd.read_csv(mapping_csv)
    df['modelname_clean'] = df['modelname'].str.replace('"', '')
    df['modelname_stripped'] = df['modelname_clean'].str.lstrip('0')

    return dict(zip(df['modelname_stripped'], df['uid']))


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


def find_ply_files(ply_dir: Path, uids: list, uid_mapping: dict = None) -> list:
    """
    Find PLY files for given UIDs using combined lookup strategy.

    Strategy:
    1. First try: uid as stripped modelname -> mapping -> PLY uid
    2. Fallback: uid directly as PLY filename
    """
    samples = []
    via_mapping = 0
    via_direct = 0
    missing = 0

    # Pre-scan directory for faster lookup
    ply_uids = set(f.stem for f in ply_dir.glob("*.ply"))

    for uid in uids:
        ply_path = None
        ply_uid = uid  # Default to direct

        # Method 1: Try mapping (uid is stripped modelname)
        if uid_mapping and uid in uid_mapping:
            mapped_uid = str(uid_mapping[uid])
            if mapped_uid in ply_uids:
                ply_path = ply_dir / f"{mapped_uid}.ply"
                ply_uid = mapped_uid
                via_mapping += 1

        # Method 2: Try direct (uid is PLY filename)
        if ply_path is None and uid in ply_uids:
            ply_path = ply_dir / f"{uid}.ply"
            ply_uid = uid
            via_direct += 1

        if ply_path:
            samples.append({
                "uid": uid,  # Original 169k uid (for output)
                "ply_uid": ply_uid,  # Actual PLY file uid
                "path": str(ply_path),
            })
        else:
            missing += 1

    print(f"Found via mapping: {via_mapping}")
    print(f"Found via direct: {via_direct}")
    if missing > 0:
        print(f"Warning: {missing} PLY files not found")

    return samples


def create_batches(samples, batch_size):
    """Create batches from samples."""
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


def save_checkpoint(checkpoint_path: Path, batch_idx: int, processed_samples: int):
    """Save checkpoint information."""
    checkpoint = {
        "batch_idx": batch_idx,
        "processed_samples": processed_samples,
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
    features: np.ndarray,
    uids: list,
    model_config: str,
    embed_dim: int,
    num_tokens: int,
    total_samples: int,
    is_first_write: bool = False,
):
    """Append features to HDF5 file (or create if first write)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_first_write:
        # Create new file with resizable datasets
        # Use LZF compression - much faster than gzip
        with h5py.File(output_path, "w") as f:
            # Create resizable datasets with small chunks for faster appends
            f.create_dataset(
                "features",
                shape=(0, num_tokens, embed_dim),
                maxshape=(total_samples, num_tokens, embed_dim),
                dtype=np.float32,
                chunks=(10, num_tokens, embed_dim),  # Small chunks for fast appends
                compression="lzf",  # Fast compression
            )
            # UIDs as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset(
                "uids",
                shape=(0,),
                maxshape=(total_samples,),
                dtype=dt,
            )
            # Metadata
            f.attrs["model_config"] = model_config
            f.attrs["embed_dim"] = embed_dim
            f.attrs["num_tokens"] = num_tokens
            f.attrs["total_samples"] = total_samples
            f.attrs["n_samples"] = 0

    # Append data
    with h5py.File(output_path, "a") as f:
        current_size = f["features"].shape[0]
        new_size = current_size + features.shape[0]

        # Resize datasets
        f["features"].resize(new_size, axis=0)
        f["uids"].resize(new_size, axis=0)

        # Write new data
        f["features"][current_size:new_size] = features
        for i, uid in enumerate(uids):
            f["uids"][current_size + i] = uid

        # Update count
        f.attrs["n_samples"] = new_size

        # Flush to disk
        f.flush()


@torch.no_grad()
def extract_features_with_checkpoints(
    model: PointBertEncoder,
    samples: list,
    batch_size: int,
    num_points: int,
    device: torch.device,
    output_path: Path,
    checkpoint_path: Path,
    checkpoint_every: int = 100,
    start_batch: int = 0,
):
    """
    Extract Point-BERT features with periodic checkpointing.
    """
    model.eval()

    embed_dim = model.embed_dim
    num_tokens = model.num_tokens
    total_samples = len(samples)

    batches = list(create_batches(samples, batch_size))
    total_batches = len(batches)

    # Buffers for accumulating before checkpoint
    buffer_features = []
    buffer_uids = []

    is_first_write = (start_batch == 0)

    pbar = tqdm(
        enumerate(batches),
        total=total_batches,
        desc="Extracting features",
        initial=start_batch
    )

    for batch_idx, batch in pbar:
        # Skip already processed batches
        if batch_idx < start_batch:
            continue

        # Load point clouds
        points_list = []
        valid_samples = []

        for sample in batch:
            try:
                points = load_ply(sample["path"], num_points=num_points)

                # Normalize to unit sphere
                xyz = points[:, :3]
                centroid = xyz.mean(axis=0)
                xyz = xyz - centroid
                max_dist = np.max(np.linalg.norm(xyz, axis=1))
                if max_dist > 0:
                    xyz = xyz / max_dist
                points[:, :3] = xyz

                points_list.append(points)
                valid_samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed to load {sample['uid']}: {e}")
                continue

        if not points_list:
            continue

        # Stack into batch
        points = np.stack(points_list, axis=0)  # [B, N, 6]
        points_tensor = torch.from_numpy(points).to(device)

        # Extract features
        features = model(points_tensor, return_all_tokens=True)  # [B, num_tokens, embed_dim]

        # Add to buffer (move to CPU immediately)
        buffer_features.append(features.cpu().float().numpy())
        buffer_uids.extend([s["uid"] for s in valid_samples])

        # Clean up GPU tensors
        del points_tensor, features

        # Checkpoint every N batches
        if (batch_idx + 1) % checkpoint_every == 0 or (batch_idx + 1) == total_batches:
            # Concatenate buffer
            if buffer_features:
                pbar.set_postfix({"status": "saving..."})

                feat_array = np.concatenate(buffer_features, axis=0)

                # Append to HDF5
                append_to_hdf5(
                    output_path=output_path,
                    features=feat_array,
                    uids=buffer_uids,
                    model_config=model.config_name if hasattr(model, 'config_name') else "ulip2-pointbert",
                    embed_dim=embed_dim,
                    num_tokens=num_tokens,
                    total_samples=total_samples,
                    is_first_write=is_first_write,
                )
                is_first_write = False

                # Save checkpoint
                processed = (batch_idx + 1) * batch_size
                save_checkpoint(checkpoint_path, batch_idx + 1, min(processed, total_samples))

                # Clear buffer and free memory
                del buffer_features, feat_array
                buffer_features = []
                buffer_uids = []

                # Force garbage collection and GPU cache clear
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Update progress bar
                pbar.set_postfix({"saved": f"{batch_idx + 1}/{total_batches}"})

    print(f"\nCompleted! Processed {total_batches} batches")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute point cloud features from PLY files"
    )
    parser.add_argument(
        "--ply-dir",
        type=str,
        required=True,
        help="Directory containing PLY files named by UID",
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
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Point-BERT checkpoint (default: auto-download ULIP-2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ulip2-pointbert",
        choices=["ulip2-pointbert", "ulip2-pointbert-small"],
        help="Point-BERT configuration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=10000,
        help="Number of points per sample",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
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
        default="pointcloud_features.h5",
        help="Output filename",
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download ULIP-2 weights if not present",
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        default=None,
        help="CSV file mapping modelnames to PLY UIDs (combined_sorted_models.csv)",
    )

    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name
    checkpoint_path = output_dir / f"{args.output_name}.checkpoint.json"
    device = torch.device(args.device)

    print("=" * 60)
    print("Pre-computing Point Cloud Features from PLY Files")
    print("=" * 60)
    print(f"PLY directory: {ply_dir}")
    print(f"CSV file: {csv_path}")
    if args.mapping_csv:
        print(f"Mapping file: {args.mapping_csv}")
    print(f"Output: {output_path}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint or 'auto-download'}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num points: {args.num_points}")
    print(f"Checkpoint every: {args.checkpoint_every} batches")
    print()

    # Check for resume
    start_batch = 0
    processed_uids = set()
    if args.resume:
        # First, try to load UIDs from the HDF5 file itself (most reliable)
        processed_uids = load_processed_uids_from_h5(output_path)
        if processed_uids:
            print(f"Resuming: {len(processed_uids)} UIDs already in HDF5 file")
        else:
            # Fall back to checkpoint
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint:
                start_batch = checkpoint["batch_idx"]
                print(f"Resuming from batch {start_batch} ({checkpoint['processed_samples']} samples)")
            else:
                print("No checkpoint or HDF5 found, starting from beginning")
    else:
        # Clear old checkpoint and output if not resuming
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

    # Load UID mapping if provided
    uid_mapping = None
    if args.mapping_csv:
        mapping_path = Path(args.mapping_csv)
        print(f"Loading UID mapping from {mapping_path}...")
        uid_mapping = load_uid_mapping(mapping_path)
        print(f"Loaded {len(uid_mapping)} mappings")

    # Find PLY files
    print("Finding PLY files...")
    samples = find_ply_files(ply_dir, uids, uid_mapping)
    print(f"Found {len(samples)} PLY files")

    # Filter out already processed
    if processed_uids:
        samples = [s for s in samples if s["uid"] not in processed_uids]
        print(f"Remaining after filtering processed: {len(samples)}")
    print()

    # Handle checkpoint
    model_checkpoint_path = args.checkpoint
    if model_checkpoint_path is None or args.download_weights:
        model_checkpoint_path = download_ulip2_weights()
        if model_checkpoint_path is None:
            print("Warning: Could not download ULIP-2 weights")
            print("Using randomly initialized encoder")

    # Create Point-BERT encoder
    print("Creating Point-BERT encoder...")
    model = PointBertEncoder.from_config(
        args.config,
        num_points=args.num_points,
        in_channels=6,  # xyz + normals
        checkpoint_path=model_checkpoint_path,
        freeze=True,
    )
    model = model.to(device)
    model.eval()

    print(f"Model created: {args.config}")
    print(f"  - embed_dim: {model.embed_dim}")
    print(f"  - num_tokens: {model.num_tokens}")
    print(f"  - num_groups: {model.num_groups}")
    print()

    # Extract features with checkpointing
    extract_features_with_checkpoints(
        model=model,
        samples=samples,
        batch_size=args.batch_size,
        num_points=args.num_points,
        device=device,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        start_batch=start_batch,
    )

    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Cleaned up checkpoint file")

    # Print final stats
    with h5py.File(output_path, "r") as f:
        print("\n" + "=" * 60)
        print("Pre-computation complete!")
        print("=" * 60)
        print(f"Output: {output_path}")
        print(f"  - features: {f['features'].shape}")
        print(f"  - n_samples: {f.attrs['n_samples']}")


if __name__ == "__main__":
    main()
