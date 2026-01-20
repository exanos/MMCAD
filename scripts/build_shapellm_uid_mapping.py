#!/usr/bin/env python3
"""
Build UID mapping from ShapeLLM H5 filenames to 169k.csv UIDs.

The mapping chain is:
    H5 filename (PLY) -> STL name (replace .ply -> .stl)
    -> combined_sorted_models.csv -> UID -> 169k.csv

Usage:
    python scripts/build_shapellm_uid_mapping.py \
        --h5 ../data/embeddings/selected_embeddings_10pct_sample.h5 \
        --combined-csv ../data/combined_sorted_models.csv \
        --text-csv ../data/169k.csv \
        --output ../data/shapellm_uid_mapping.json
"""

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm


def build_shapellm_uid_mapping(
    h5_path: Path,
    combined_csv: Path,
    text_csv: Path,
    output_path: Path,
) -> dict:
    """
    Build mapping from ShapeLLM H5 indices to 169k UIDs.

    Returns:
        Dictionary with:
        - h5_filename_to_uid: {filename: uid}
        - uid_to_h5_idx: {uid: h5_index}
        - valid_uids: list of UIDs that have text
        - stats: coverage statistics
    """
    print(f"Loading ShapeLLM H5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        # Get filenames - handle both bytes and string encoding
        filenames_raw = f["filenames"][:]
        filenames = []
        for fn in filenames_raw:
            if isinstance(fn, bytes):
                fn = fn.decode("utf-8")
            filenames.append(fn)

        total_h5 = len(filenames)
        print(f"  Total H5 samples: {total_h5}")

        # Check shapes
        print(f"  local_features shape: {f['local_features'].shape}")
        print(f"  global_token shape: {f['global_token'].shape}")

    print(f"\nLoading combined_sorted_models.csv: {combined_csv}")
    combined_df = pd.read_csv(combined_csv)
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Columns: {list(combined_df.columns)}")

    # Build stlname -> uid lookup
    stl_to_uid = dict(zip(combined_df["stlname"], combined_df["uid"]))
    print(f"  STL to UID mappings: {len(stl_to_uid)}")

    print(f"\nLoading 169k.csv: {text_csv}")
    text_df = pd.read_csv(text_csv)
    print(f"  Total rows: {len(text_df)}")

    # Get UIDs that have text - convert to strings for consistency
    text_uids = set(str(uid) for uid in text_df["uid"].tolist())
    print(f"  UIDs with text: {len(text_uids)}")

    # Build mappings
    print("\nBuilding mappings...")
    h5_filename_to_uid = {}
    uid_to_h5_idx = {}
    valid_uids = []

    mapped_to_combined = 0
    with_text = 0

    for h5_idx, filename in enumerate(tqdm(filenames, desc="Processing")):
        # Convert .ply to .stl
        stl_name = filename.replace(".ply", ".stl")

        # Look up in combined_sorted_models.csv
        if stl_name in stl_to_uid:
            uid = str(stl_to_uid[stl_name])
            mapped_to_combined += 1

            h5_filename_to_uid[filename] = uid
            uid_to_h5_idx[uid] = h5_idx

            # Check if has text
            if uid in text_uids:
                with_text += 1
                valid_uids.append(uid)

    # Stats
    stats = {
        "total_h5_samples": total_h5,
        "mapped_to_combined": mapped_to_combined,
        "with_text": with_text,
        "coverage_combined": mapped_to_combined / total_h5 if total_h5 > 0 else 0,
        "coverage_text": with_text / total_h5 if total_h5 > 0 else 0,
    }

    print(f"\n=== Mapping Statistics ===")
    print(f"Total H5 samples:      {stats['total_h5_samples']:,}")
    print(f"Mapped to combined:    {stats['mapped_to_combined']:,} ({stats['coverage_combined']:.1%})")
    print(f"With text (usable):    {stats['with_text']:,} ({stats['coverage_text']:.1%})")

    result = {
        "h5_filename_to_uid": h5_filename_to_uid,
        "uid_to_h5_idx": uid_to_h5_idx,
        "valid_uids": valid_uids,
        "stats": stats,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved mapping to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Build ShapeLLM UID mapping")
    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Path to ShapeLLM H5 file",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        required=True,
        help="Path to combined_sorted_models.csv",
    )
    parser.add_argument(
        "--text-csv",
        type=Path,
        required=True,
        help="Path to 169k.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path",
    )

    args = parser.parse_args()

    build_shapellm_uid_mapping(
        h5_path=args.h5,
        combined_csv=args.combined_csv,
        text_csv=args.text_csv,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
