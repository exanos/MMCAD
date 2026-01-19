#!/usr/bin/env python3
"""
Pre-compute B-Rep features using AutoBrep encoder.

This script extracts AutoBrep encoder outputs for all B-Rep data and
saves them as HDF5 files. During training, these cached features are loaded
instead of running the encoder, which speeds up training when the encoder
is frozen.

Usage:
    python scripts/precompute_brep_features.py --data-root data/mmcad
    python scripts/precompute_brep_features.py --data-root data/mmcad --surface-checkpoint pretrained/autobrep/surface_fsq_vae.pt
    python scripts/precompute_brep_features.py --data-root data/mmcad --batch-size 64 --output-dir embeddings/

Output:
    Creates HDF5 files with structure:
    - {split}_brep_features.h5
        - face_features: [N, max_faces, face_dim]
        - edge_features: [N, max_edges, edge_dim]
        - face_masks: [N, max_faces]
        - edge_masks: [N, max_edges]
        - adjacency: [N, max_faces, max_edges]
        - num_faces: [N]
        - num_edges: [N]
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

from clip4cad.models.encoders.brep_encoder import BRepEncoder


def load_brep_paths(data_root: Path, split: str):
    """Load B-Rep paths for a split."""
    brep_dir = data_root / "brep"
    split_file = data_root / "splits" / f"{split}.txt"

    if not split_file.exists():
        print(f"Split file not found: {split_file}")
        return []

    with open(split_file, "r") as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    samples = []
    for sample_id in sample_ids:
        face_path = brep_dir / f"{sample_id}_faces.npy"
        edge_path = brep_dir / f"{sample_id}_edges.npy"
        adj_path = brep_dir / f"{sample_id}_adjacency.npy"

        if face_path.exists():
            samples.append({
                "sample_id": sample_id,
                "face_path": str(face_path),
                "edge_path": str(edge_path) if edge_path.exists() else None,
                "adj_path": str(adj_path) if adj_path.exists() else None,
            })
        else:
            print(f"Warning: B-Rep not found for {sample_id}")

    return samples


def load_brep_data(
    sample: dict,
    max_faces: int = 192,
    max_edges: int = 512,
    face_grid_size: int = 32,
    edge_curve_size: int = 32,
) -> dict:
    """Load and preprocess a single B-Rep sample."""
    # Load faces
    faces = np.load(sample["face_path"]).astype(np.float32)  # [F, H, W, 3]
    F = faces.shape[0]

    # Load edges
    if sample["edge_path"]:
        edges = np.load(sample["edge_path"]).astype(np.float32)  # [E, L, 3]
    else:
        edges = np.zeros((0, edge_curve_size, 3), dtype=np.float32)
    E = edges.shape[0]

    # Load adjacency
    if sample["adj_path"]:
        adjacency = np.load(sample["adj_path"]).astype(np.float32)  # [F, E]
    else:
        adjacency = np.zeros((F, E), dtype=np.float32)

    # Create masks
    face_mask = np.zeros(max_faces, dtype=np.float32)
    face_mask[:min(F, max_faces)] = 1.0

    edge_mask = np.zeros(max_edges, dtype=np.float32)
    edge_mask[:min(E, max_edges)] = 1.0

    # Pad faces
    if F < max_faces:
        pad_faces = np.zeros((max_faces - F, face_grid_size, face_grid_size, 3), dtype=np.float32)
        faces = np.concatenate([faces, pad_faces], axis=0)
    else:
        faces = faces[:max_faces]
        F = max_faces

    # Pad edges
    if E < max_edges:
        pad_edges = np.zeros((max_edges - E, edge_curve_size, 3), dtype=np.float32)
        edges = np.concatenate([edges, pad_edges], axis=0)
    else:
        edges = edges[:max_edges]
        E = max_edges

    # Pad adjacency
    padded_adj = np.zeros((max_faces, max_edges), dtype=np.float32)
    f_end = min(adjacency.shape[0], max_faces)
    e_end = min(adjacency.shape[1], max_edges) if adjacency.shape[1] > 0 else 0
    if f_end > 0 and e_end > 0:
        padded_adj[:f_end, :e_end] = adjacency[:f_end, :e_end]

    # Normalize each face/edge to [-1, 1] bounding box
    for i in range(min(int(face_mask.sum()), max_faces)):
        face = faces[i]
        face_min = face.min(axis=(0, 1), keepdims=True)
        face_max = face.max(axis=(0, 1), keepdims=True)
        face_range = np.maximum(face_max - face_min, 1e-8)
        faces[i] = 2.0 * (face - face_min) / face_range - 1.0

    for i in range(min(int(edge_mask.sum()), max_edges)):
        edge = edges[i]
        edge_min = edge.min(axis=0, keepdims=True)
        edge_max = edge.max(axis=0, keepdims=True)
        edge_range = np.maximum(edge_max - edge_min, 1e-8)
        edges[i] = 2.0 * (edge - edge_min) / edge_range - 1.0

    return {
        "faces": faces,
        "edges": edges,
        "face_mask": face_mask,
        "edge_mask": edge_mask,
        "adjacency": padded_adj,
        "num_faces": int(face_mask.sum()),
        "num_edges": int(edge_mask.sum()),
    }


def create_batches(samples, batch_size):
    """Create batches from samples."""
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


@torch.no_grad()
def extract_features(
    model: BRepEncoder,
    samples: list,
    batch_size: int,
    max_faces: int,
    max_edges: int,
    face_grid_size: int,
    edge_curve_size: int,
    device: torch.device,
):
    """
    Extract B-Rep features for all samples.

    Returns:
        face_features: [N, max_faces, face_dim]
        edge_features: [N, max_edges, edge_dim]
        face_masks: [N, max_faces]
        edge_masks: [N, max_edges]
        adjacency: [N, max_faces, max_edges]
        num_faces: [N]
        num_edges: [N]
        sample_ids: [N]
    """
    model.eval()

    face_dim = model.face_dim
    edge_dim = model.edge_dim

    all_face_features = []
    all_edge_features = []
    all_face_masks = []
    all_edge_masks = []
    all_adjacency = []
    all_num_faces = []
    all_num_edges = []
    all_sample_ids = []

    batches = list(create_batches(samples, batch_size))

    for batch in tqdm(batches, desc="Extracting features"):
        # Load B-Rep data
        batch_data = []
        valid_samples = []

        for sample in batch:
            try:
                data = load_brep_data(
                    sample,
                    max_faces=max_faces,
                    max_edges=max_edges,
                    face_grid_size=face_grid_size,
                    edge_curve_size=edge_curve_size,
                )
                batch_data.append(data)
                valid_samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed to load {sample['sample_id']}: {e}")
                continue

        if not batch_data:
            continue

        # Stack into batch tensors
        faces = np.stack([d["faces"] for d in batch_data], axis=0)
        edges = np.stack([d["edges"] for d in batch_data], axis=0)
        face_mask = np.stack([d["face_mask"] for d in batch_data], axis=0)
        edge_mask = np.stack([d["edge_mask"] for d in batch_data], axis=0)
        adjacency = np.stack([d["adjacency"] for d in batch_data], axis=0)
        num_faces = np.array([d["num_faces"] for d in batch_data])
        num_edges = np.array([d["num_edges"] for d in batch_data])

        # Convert to tensors
        faces_tensor = torch.from_numpy(faces).to(device)
        edges_tensor = torch.from_numpy(edges).to(device)
        face_mask_tensor = torch.from_numpy(face_mask).to(device)
        edge_mask_tensor = torch.from_numpy(edge_mask).to(device)

        # Extract features
        face_feats, edge_feats = model(
            faces_tensor, edges_tensor, face_mask_tensor, edge_mask_tensor
        )

        # Store results
        all_face_features.append(face_feats.cpu().float().numpy())
        all_edge_features.append(edge_feats.cpu().float().numpy())
        all_face_masks.append(face_mask)
        all_edge_masks.append(edge_mask)
        all_adjacency.append(adjacency)
        all_num_faces.append(num_faces)
        all_num_edges.append(num_edges)
        all_sample_ids.extend([s["sample_id"] for s in valid_samples])

    # Concatenate all batches
    face_features = np.concatenate(all_face_features, axis=0)
    edge_features = np.concatenate(all_edge_features, axis=0)
    face_masks = np.concatenate(all_face_masks, axis=0)
    edge_masks = np.concatenate(all_edge_masks, axis=0)
    adjacency = np.concatenate(all_adjacency, axis=0)
    num_faces = np.concatenate(all_num_faces, axis=0)
    num_edges = np.concatenate(all_num_edges, axis=0)

    return (
        face_features, edge_features, face_masks, edge_masks,
        adjacency, num_faces, num_edges, all_sample_ids
    )


def save_features_hdf5(
    output_path: Path,
    face_features: np.ndarray,
    edge_features: np.ndarray,
    face_masks: np.ndarray,
    edge_masks: np.ndarray,
    adjacency: np.ndarray,
    num_faces: np.ndarray,
    num_edges: np.ndarray,
    sample_ids: list,
    model_config: str,
):
    """Save features to HDF5 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Save features with compression
        f.create_dataset(
            "face_features",
            data=face_features,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "edge_features",
            data=edge_features,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "face_masks",
            data=face_masks,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "edge_masks",
            data=edge_masks,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "adjacency",
            data=adjacency,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset("num_faces", data=num_faces)
        f.create_dataset("num_edges", data=num_edges)

        # Save sample IDs as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        sample_ids_ds = f.create_dataset("sample_ids", (len(sample_ids),), dtype=dt)
        for i, sid in enumerate(sample_ids):
            sample_ids_ds[i] = sid

        # Save metadata
        f.attrs["model_config"] = model_config
        f.attrs["face_dim"] = face_features.shape[2]
        f.attrs["edge_dim"] = edge_features.shape[2]
        f.attrs["max_faces"] = face_features.shape[1]
        f.attrs["max_edges"] = edge_features.shape[1]
        f.attrs["n_samples"] = len(sample_ids)

    print(f"Saved features to {output_path}")
    print(f"  - face_features: {face_features.shape}")
    print(f"  - edge_features: {edge_features.shape}")
    print(f"  - n_samples: {len(sample_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute B-Rep features using AutoBrep encoder"
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
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=64,
        help="Maximum number of faces per sample",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=128,
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
        help="Face feature dimension (AutoBrep default: 48)",
    )
    parser.add_argument(
        "--edge-dim",
        type=int,
        default=12,
        help="Edge feature dimension (AutoBrep default: 12)",
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

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "embeddings"
    device = torch.device(args.device)

    print("=" * 60)
    print("Pre-computing B-Rep Features")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Surface checkpoint: {args.surface_checkpoint or 'None (random init)'}")
    print(f"Edge checkpoint: {args.edge_checkpoint or 'None (random init)'}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max faces: {args.max_faces}, Max edges: {args.max_edges}")
    print(f"Face dim: {args.face_dim}, Edge dim: {args.edge_dim}")
    print()

    # Create B-Rep encoder
    print("Creating B-Rep encoder...")
    model = BRepEncoder(
        face_dim=args.face_dim,
        edge_dim=args.edge_dim,
        face_grid_size=args.face_grid_size,
        edge_curve_size=args.edge_curve_size,
        surface_checkpoint=args.surface_checkpoint,
        edge_checkpoint=args.edge_checkpoint,
        freeze=True,
    )
    model = model.to(device)
    model.eval()

    print(f"Model created")
    print(f"  - face_dim: {model.face_dim}")
    print(f"  - edge_dim: {model.edge_dim}")
    print()

    # Process each split
    for split in args.splits:
        print(f"\n{'=' * 60}")
        print(f"Processing {split} split")
        print("=" * 60)

        # Load B-Rep paths
        samples = load_brep_paths(data_root, split)
        if not samples:
            print(f"No samples found for {split}")
            continue

        print(f"Found {len(samples)} samples")

        # Extract features
        results = extract_features(
            model=model,
            samples=samples,
            batch_size=args.batch_size,
            max_faces=args.max_faces,
            max_edges=args.max_edges,
            face_grid_size=args.face_grid_size,
            edge_curve_size=args.edge_curve_size,
            device=device,
        )

        (face_features, edge_features, face_masks, edge_masks,
         adjacency, num_faces, num_edges, sample_ids) = results

        # Save to HDF5
        output_path = output_dir / f"{split}_brep_features.h5"
        save_features_hdf5(
            output_path=output_path,
            face_features=face_features,
            edge_features=edge_features,
            face_masks=face_masks,
            edge_masks=edge_masks,
            adjacency=adjacency,
            num_faces=num_faces,
            num_edges=num_edges,
            sample_ids=sample_ids,
            model_config="autobrep",
        )

    print("\n" + "=" * 60)
    print("Pre-computation complete!")
    print("=" * 60)
    print(f"\nFeatures saved to: {output_dir}")
    print("\nTo use during training, update your config:")
    print("  data:")
    print(f"    embeddings_dir: {output_dir}")
    print("    use_cached_brep_features: true")


if __name__ == "__main__":
    main()
