#!/usr/bin/env python3
"""
Pre-compute point cloud features using Point-BERT encoder.

This script extracts Point-BERT encoder outputs for all point cloud data and
saves them as HDF5 files. During training, these cached features are loaded
instead of running the encoder, which significantly speeds up training when
the encoder is frozen.

Usage:
    python scripts/precompute_pointcloud_features.py --data-root data/mmcad
    python scripts/precompute_pointcloud_features.py --data-root data/mmcad --checkpoint pretrained/pointbert/ulip2_pointbert.pt
    python scripts/precompute_pointcloud_features.py --data-root data/mmcad --batch-size 32 --output-dir embeddings/

Output:
    Creates HDF5 files with structure:
    - {split}_pointcloud_features.h5
        - features: [N, num_tokens, embed_dim]  # CLS + group tokens
        - sample_ids: [N] (string dataset)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from clip4cad.models.encoders.pointbert_encoder import (
    PointBertEncoder,
    load_ply,
    download_ulip2_weights,
)


def load_pointcloud_paths(data_root: Path, split: str):
    """Load point cloud paths for a split."""
    pc_dir = data_root / "pointcloud"
    split_file = data_root / "splits" / f"{split}.txt"

    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        return []

    with open(split_file, "r") as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    samples = []
    for sample_id in sample_ids:
        # Prefer PLY files
        ply_path = pc_dir / f"{sample_id}.ply"
        npy_path = pc_dir / f"{sample_id}.npy"

        if ply_path.exists():
            samples.append({
                "sample_id": sample_id,
                "path": str(ply_path),
                "format": "ply",
            })
        elif npy_path.exists():
            samples.append({
                "sample_id": sample_id,
                "path": str(npy_path),
                "format": "npy",
            })
        else:
            print(f"Warning: Point cloud not found for {sample_id}")

    return samples


def load_pointcloud(sample: dict, num_points: int = 10000) -> np.ndarray:
    """Load a single point cloud."""
    if sample["format"] == "ply":
        points = load_ply(sample["path"], num_points=num_points)
    else:  # npy
        points = np.load(sample["path"]).astype(np.float32)

        # Handle different shapes
        N = points.shape[0]
        C = points.shape[1] if len(points.shape) > 1 else 1

        if C >= 6:
            # Has normals
            points = points[:, :6]
        elif C >= 3:
            # Just xyz, add zero normals
            xyz = points[:, :3]
            normals = np.zeros_like(xyz)
            points = np.concatenate([xyz, normals], axis=1)
        else:
            raise ValueError(f"Invalid point cloud shape: {points.shape}")

        # Subsample or pad
        if N > num_points:
            idx = np.random.choice(N, num_points, replace=False)
            points = points[idx]
        elif N < num_points:
            pad_idx = np.random.choice(N, num_points - N, replace=True)
            points = np.concatenate([points, points[pad_idx]], axis=0)

    # Normalize to unit sphere (using positions only)
    xyz = points[:, :3]
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    max_dist = np.max(np.linalg.norm(xyz, axis=1))
    if max_dist > 0:
        xyz = xyz / max_dist

    # Update positions and keep normals
    points[:, :3] = xyz

    return points


def create_batches(samples, batch_size):
    """Create batches from samples."""
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


@torch.no_grad()
def extract_features(
    model: PointBertEncoder,
    samples: list,
    batch_size: int,
    num_points: int,
    device: torch.device,
):
    """
    Extract Point-BERT features for all samples.

    Returns:
        features: [N, num_tokens, embed_dim]
        sample_ids: [N]
    """
    model.eval()

    embed_dim = model.embed_dim
    num_tokens = model.num_tokens

    all_features = []
    all_sample_ids = []

    batches = list(create_batches(samples, batch_size))

    for batch in tqdm(batches, desc="Extracting features"):
        # Load point clouds
        points_list = []
        valid_samples = []

        for sample in batch:
            try:
                points = load_pointcloud(sample, num_points=num_points)
                points_list.append(points)
                valid_samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed to load {sample['sample_id']}: {e}")
                continue

        if not points_list:
            continue

        # Stack into batch
        points = np.stack(points_list, axis=0)  # [B, N, 6]
        points_tensor = torch.from_numpy(points).to(device)

        # Extract features
        features = model(points_tensor, return_all_tokens=True)  # [B, num_tokens, embed_dim]

        # Store results
        all_features.append(features.cpu().float().numpy())
        all_sample_ids.extend([s["sample_id"] for s in valid_samples])

    # Concatenate all batches
    features = np.concatenate(all_features, axis=0)

    return features, all_sample_ids


def save_features_hdf5(
    output_path: Path,
    features: np.ndarray,
    sample_ids: list,
    model_config: str,
):
    """Save features to HDF5 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Save features with compression
        f.create_dataset(
            "features",
            data=features,
            compression="gzip",
            compression_opts=4,
        )

        # Save sample IDs as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        sample_ids_ds = f.create_dataset("sample_ids", (len(sample_ids),), dtype=dt)
        for i, sid in enumerate(sample_ids):
            sample_ids_ds[i] = sid

        # Save metadata
        f.attrs["model_config"] = model_config
        f.attrs["embed_dim"] = features.shape[2]
        f.attrs["num_tokens"] = features.shape[1]
        f.attrs["n_samples"] = len(sample_ids)

    print(f"Saved features to {output_path}")
    print(f"  - features: {features.shape}")
    print(f"  - n_samples: {len(sample_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute point cloud features using Point-BERT"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to MMCAD data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for features (default: {data_root}/embeddings)",
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
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=10000,
        help="Number of points per sample",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download ULIP-2 weights if not present",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "embeddings"
    device = torch.device(args.device)

    print("=" * 60)
    print("Pre-computing Point Cloud Features")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint or 'auto-download'}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num points: {args.num_points}")
    print()

    # Handle checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None or args.download_weights:
        # Try to download ULIP-2 weights
        checkpoint_path = download_ulip2_weights()
        if checkpoint_path is None:
            print("Warning: Could not download ULIP-2 weights")
            print("Using randomly initialized encoder")

    # Create Point-BERT encoder
    print("Creating Point-BERT encoder...")
    model = PointBertEncoder.from_config(
        args.config,
        num_points=args.num_points,
        in_channels=6,  # xyz + normals
        checkpoint_path=checkpoint_path,
        freeze=True,
    )
    model = model.to(device)
    model.eval()

    print(f"Model created: {args.config}")
    print(f"  - embed_dim: {model.embed_dim}")
    print(f"  - num_tokens: {model.num_tokens}")
    print(f"  - num_groups: {model.num_groups}")
    print()

    # Process each split
    for split in args.splits:
        print(f"\n{'=' * 60}")
        print(f"Processing {split} split")
        print("=" * 60)

        # Load point cloud paths
        samples = load_pointcloud_paths(data_root, split)
        if not samples:
            print(f"No samples found for {split}")
            continue

        print(f"Found {len(samples)} samples")

        # Extract features
        features, sample_ids = extract_features(
            model=model,
            samples=samples,
            batch_size=args.batch_size,
            num_points=args.num_points,
            device=device,
        )

        # Save to HDF5
        output_path = output_dir / f"{split}_pointcloud_features.h5"
        save_features_hdf5(
            output_path=output_path,
            features=features,
            sample_ids=sample_ids,
            model_config=args.config,
        )

    print("\n" + "=" * 60)
    print("Pre-computation complete!")
    print("=" * 60)
    print(f"\nFeatures saved to: {output_dir}")
    print("\nTo use during training, update your config:")
    print("  data:")
    print(f"    embeddings_dir: {output_dir}")
    print("    use_cached_pc_features: true")


if __name__ == "__main__":
    main()
