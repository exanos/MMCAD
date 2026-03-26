#!/usr/bin/env python
"""
Encode Database Script for CLIP4CAD-GFA

Extracts embeddings for all samples (train, val, test splits) given a checkpoint.
Outputs HDF5 files containing normalized embeddings for retrieval tasks.

Usage:
    python scripts/encode_database.py \
        --checkpoint outputs/gfa/checkpoint_best.pt \
        --config configs/model/clip4cad_gfa.yaml \
        --data-root data/mmcad \
        --output-dir outputs/embeddings

Output Structure:
    {output_dir}/
    ├── train_embeddings.h5
    ├── val_embeddings.h5
    └── test_embeddings.h5

    Each file contains:
    ├── text_embeddings    (N, 128) - normalized
    ├── brep_embeddings    (N, 128) - normalized
    ├── pc_embeddings      (N, 128) - normalized
    └── sample_ids         (N,) - string sample IDs
"""

import argparse
from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from clip4cad.models.clip4cad_gfa import CLIP4CAD_GFA
from clip4cad.data.gfa_dataset import GFAMappedDataset, gfa_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode database with CLIP4CAD-GFA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config YAML (if not embedded in checkpoint)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/embeddings",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated list of splits to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--pc-file",
        type=str,
        default=None,
        help="Path to PC HDF5 file",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to text embeddings HDF5 file",
    )
    parser.add_argument(
        "--brep-file",
        type=str,
        default=None,
        help="Path to B-Rep HDF5 file",
    )
    parser.add_argument(
        "--mapping-dir",
        type=str,
        default=None,
        help="Directory containing uid_mapping.json",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--load-to-memory",
        action="store_true",
        help="Load all data to RAM for fast encoding (recommended if you have 64GB+ RAM)",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, config_path: str = None, device: str = "cuda"):
    """
    Load CLIP4CAD-GFA model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Optional path to config YAML
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
        config: Model configuration
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to get config from checkpoint or load from file
    if "config" in checkpoint and config_path is None:
        config = OmegaConf.create(checkpoint["config"])
        print("  Using config from checkpoint")
    elif config_path is not None:
        config = OmegaConf.load(config_path)
        print(f"  Using config from: {config_path}")
    else:
        # Try default config location
        default_config = project_root / "configs" / "model" / "clip4cad_gfa.yaml"
        if default_config.exists():
            config = OmegaConf.load(default_config)
            print(f"  Using default config: {default_config}")
        else:
            raise ValueError("No config found in checkpoint and no config path provided")

    # Create model
    model = CLIP4CAD_GFA(config)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"  Model loaded: {model.count_parameters():,} parameters")

    return model, config


@torch.no_grad()
def encode_split(
    model: CLIP4CAD_GFA,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """
    Encode all samples in a split.

    Args:
        model: CLIP4CAD-GFA model
        dataloader: DataLoader for the split
        device: Device to run on

    Returns:
        Dictionary with embeddings and sample IDs
    """
    text_embeddings = []
    brep_embeddings = []
    pc_embeddings = []
    sample_ids = []

    for batch in tqdm(dataloader, desc="Encoding"):
        # Get sample IDs
        batch_ids = batch["sample_id"]
        sample_ids.extend(batch_ids)

        # Forward pass
        outputs = model(batch)

        # Extract and normalize global embeddings
        z_text = outputs["z_text"]
        z_text = F.normalize(z_text, p=2, dim=-1)
        text_embeddings.append(z_text.cpu().float())

        if "z_brep" in outputs:
            z_brep = outputs["z_brep"]
            z_brep = F.normalize(z_brep, p=2, dim=-1)
            brep_embeddings.append(z_brep.cpu().float())

        if "z_pc" in outputs:
            z_pc = outputs["z_pc"]
            z_pc = F.normalize(z_pc, p=2, dim=-1)
            pc_embeddings.append(z_pc.cpu().float())

    # Concatenate all batches
    result = {
        "text_embeddings": torch.cat(text_embeddings, dim=0).numpy(),
        "sample_ids": sample_ids,
    }

    if brep_embeddings:
        result["brep_embeddings"] = torch.cat(brep_embeddings, dim=0).numpy()

    if pc_embeddings:
        result["pc_embeddings"] = torch.cat(pc_embeddings, dim=0).numpy()

    return result


def save_embeddings(embeddings: dict, output_path: str):
    """
    Save embeddings to HDF5 file.

    Args:
        embeddings: Dictionary with embeddings and sample IDs
        output_path: Path to output HDF5 file
    """
    with h5py.File(output_path, "w") as f:
        # Save embeddings
        f.create_dataset(
            "text_embeddings",
            data=embeddings["text_embeddings"],
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )

        if "brep_embeddings" in embeddings:
            f.create_dataset(
                "brep_embeddings",
                data=embeddings["brep_embeddings"],
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )

        if "pc_embeddings" in embeddings:
            f.create_dataset(
                "pc_embeddings",
                data=embeddings["pc_embeddings"],
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )

        # Save sample IDs as variable-length strings
        sample_ids = embeddings["sample_ids"]
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset(
            "sample_ids",
            data=[str(sid) for sid in sample_ids],
            dtype=dt,
        )

        # Save metadata
        f.attrs["num_samples"] = len(sample_ids)
        f.attrs["embedding_dim"] = embeddings["text_embeddings"].shape[1]

    print(f"  Saved: {output_path}")
    print(f"    Samples: {len(sample_ids)}")
    print(f"    Embedding dim: {embeddings['text_embeddings'].shape[1]}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model(args.checkpoint, args.config, args.device)

    # Parse splits
    splits = [s.strip() for s in args.splits.split(",")]

    # Data paths
    data_root = Path(args.data_root)
    mapping_dir = Path(args.mapping_dir) if args.mapping_dir else data_root / "aligned"

    print("\n" + "=" * 60)
    print("CLIP4CAD-GFA Database Encoding")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print(f"Splits: {splits}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("=" * 60 + "\n")

    # Process each split
    for split in splits:
        print(f"\n{'='*40}")
        print(f"Processing: {split}")
        print(f"{'='*40}")

        # Check if split exists
        split_file = mapping_dir / "splits" / f"{split}_uids.txt"
        if not split_file.exists():
            split_file = mapping_dir / "splits" / f"{split}.txt"

        if not split_file.exists():
            print(f"  Warning: Split file not found: {split_file}")
            print(f"  Skipping {split} split")
            continue

        # Create dataset
        try:
            dataset = GFAMappedDataset(
                data_root=str(data_root),
                split=split,
                pc_file=args.pc_file,
                text_file=args.text_file,
                brep_file=args.brep_file,
                mapping_dir=str(mapping_dir),
                num_rotations=1,
                load_to_memory=args.load_to_memory,
            )
        except FileNotFoundError as e:
            print(f"  Error loading dataset: {e}")
            print(f"  Skipping {split} split")
            continue

        print(f"  Samples: {len(dataset)}")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False,
            collate_fn=gfa_collate_fn,
        )

        # Encode
        embeddings = encode_split(model, dataloader, args.device)

        # Save
        output_path = output_dir / f"{split}_embeddings.h5"
        save_embeddings(embeddings, str(output_path))

        # Cleanup
        dataset.close()

    # Save encoding metadata
    metadata = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config) if args.config else "from_checkpoint",
        "data_root": str(args.data_root),
        "splits": splits,
        "batch_size": args.batch_size,
        "embedding_dim": 128,  # d_proj from config
    }

    with open(output_dir / "encoding_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Encoding Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved: {output_dir / 'encoding_metadata.json'}")


if __name__ == "__main__":
    main()
