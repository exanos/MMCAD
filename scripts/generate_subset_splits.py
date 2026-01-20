#!/usr/bin/env python3
"""
Generate train/val/test splits from intersection of all modalities.

Only includes samples that have:
1. ShapeLLM point cloud features
2. B-Rep features
3. Text embeddings

Usage:
    python scripts/generate_subset_splits.py \
        --pc-h5 ../data/embeddings/shapellm_pc_features.h5 \
        --brep-h5 ../data/embeddings/brep_features.h5 \
        --text-h5 ../data/embeddings/text_embeddings.h5 \
        --output-dir ../data/splits_10pct \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import argparse
import random
from pathlib import Path

import h5py


def get_uids_from_h5(h5_path: Path, uid_key: str = "sample_ids") -> set:
    """Extract UIDs from H5 file."""
    uids = set()

    with h5py.File(h5_path, "r") as f:
        if uid_key not in f:
            # Try alternative key names
            for key in ["uids", "sample_ids", "ids"]:
                if key in f:
                    uid_key = key
                    break
            else:
                raise KeyError(f"No UID dataset found in {h5_path}. Keys: {list(f.keys())}")

        uid_data = f[uid_key][:]
        for uid in uid_data:
            if isinstance(uid, bytes):
                uid = uid.decode("utf-8")
            uids.add(str(uid))

    return uids


def generate_splits(
    pc_h5: Path,
    brep_h5: Path,
    text_h5: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Generate train/val/test splits from intersection of modalities.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    print("Loading UIDs from H5 files...")

    # Get UIDs from each modality
    pc_uids = get_uids_from_h5(pc_h5)
    print(f"  Point cloud (ShapeLLM): {len(pc_uids)} samples")

    brep_uids = get_uids_from_h5(brep_h5, uid_key="uids")
    print(f"  B-Rep: {len(brep_uids)} samples")

    text_uids = get_uids_from_h5(text_h5, uid_key="sample_ids")
    print(f"  Text: {len(text_uids)} samples")

    # Find intersection
    valid_uids = pc_uids & brep_uids & text_uids
    print(f"\nIntersection (all modalities): {len(valid_uids)} samples")

    if len(valid_uids) == 0:
        print("\nERROR: No samples have all three modalities!")
        print("\nDebug info:")
        print(f"  PC ∩ B-Rep: {len(pc_uids & brep_uids)}")
        print(f"  PC ∩ Text: {len(pc_uids & text_uids)}")
        print(f"  B-Rep ∩ Text: {len(brep_uids & text_uids)}")
        print(f"  Sample PC UIDs: {list(pc_uids)[:5]}")
        print(f"  Sample B-Rep UIDs: {list(brep_uids)[:5]}")
        print(f"  Sample Text UIDs: {list(text_uids)[:5]}")
        return

    # Sort for reproducibility
    valid_uids = sorted(list(valid_uids))

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(valid_uids)

    # Split
    n_total = len(valid_uids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_uids = valid_uids[:n_train]
    val_uids = valid_uids[n_train:n_train + n_val]
    test_uids = valid_uids[n_train + n_val:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_uids)} ({len(train_uids)/n_total:.1%})")
    print(f"  Val:   {len(val_uids)} ({len(val_uids)/n_total:.1%})")
    print(f"  Test:  {len(test_uids)} ({len(test_uids)/n_total:.1%})")

    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_uids,
        "val": val_uids,
        "test": test_uids,
    }

    for split_name, uids in splits.items():
        split_file = output_dir / f"{split_name}.txt"
        with open(split_file, "w") as f:
            f.write("\n".join(uids))
        print(f"Saved {split_file}")

    # Save summary
    summary_file = output_dir / "split_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Generated splits from intersection of modalities\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"\nSource files:\n")
        f.write(f"  PC: {pc_h5} ({len(pc_uids)} samples)\n")
        f.write(f"  B-Rep: {brep_h5} ({len(brep_uids)} samples)\n")
        f.write(f"  Text: {text_h5} ({len(text_uids)} samples)\n")
        f.write(f"\nIntersection: {n_total} samples\n")
        f.write(f"\nSplits:\n")
        f.write(f"  Train: {len(train_uids)} ({train_ratio:.0%})\n")
        f.write(f"  Val:   {len(val_uids)} ({val_ratio:.0%})\n")
        f.write(f"  Test:  {len(test_uids)} ({test_ratio:.0%})\n")

    print(f"\nSaved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test splits")
    parser.add_argument(
        "--pc-h5",
        type=Path,
        required=True,
        help="Path to point cloud features H5",
    )
    parser.add_argument(
        "--brep-h5",
        type=Path,
        required=True,
        help="Path to B-Rep features H5",
    )
    parser.add_argument(
        "--text-h5",
        type=Path,
        required=True,
        help="Path to text embeddings H5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for split files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    generate_splits(
        pc_h5=args.pc_h5,
        brep_h5=args.brep_h5,
        text_h5=args.text_h5,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
