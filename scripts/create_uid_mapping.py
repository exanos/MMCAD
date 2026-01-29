#!/usr/bin/env python3
"""
Create UID mapping file for aligned training.
This creates a JSON mapping file that the dataset can use at runtime.
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='d:/Defect_Det/MMCAD/data')
    parser.add_argument('--pc-file', type=str, default='c:/Users/User/Desktop/selected_embeddings.h5')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    pc_file = Path(args.pc_file)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / 'aligned'
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Build ABC -> Canonical mapping ===
    print("Loading combined_sorted_models.csv...")
    df = pd.read_csv(data_root / 'combined_sorted_models.csv')
    df['modelname'] = df['modelname'].astype(str).str.strip('"').str.zfill(8).astype(int)
    df['uid'] = df['uid'].astype(str).str.strip('"').astype(int)
    abc_to_canonical = dict(zip(df['modelname'], df['uid']))
    print(f"  {len(abc_to_canonical)} ABC->Canonical mappings")

    # === Step 2: Get UIDs and build index maps ===
    print("\nLoading B-Rep...")
    with h5py.File(data_root / 'embeddings' / 'brep_features.h5', 'r') as f:
        brep_abc_uids = f['uids'][:].astype(int)
    brep_abc_to_idx = {int(u): i for i, u in enumerate(brep_abc_uids)}
    print(f"  {len(brep_abc_uids)} samples")

    print("Loading Text...")
    with h5py.File(data_root / 'embeddings' / 'text_embeddings.h5', 'r') as f:
        raw_uids = f['uids'][:]
    text_abc_to_idx = {}
    for i, u in enumerate(raw_uids):
        try:
            abc = int(u.decode() if isinstance(u, bytes) else u)
            text_abc_to_idx[abc] = i
        except:
            pass
    print(f"  {len(text_abc_to_idx)} samples")

    print("Loading PC...")
    with h5py.File(pc_file, 'r') as f:
        filenames = f['filenames'][:]
    pc_abc_to_idx = {}
    for i, fn in enumerate(filenames):
        abc = int(fn.decode().split('_')[0] if isinstance(fn, bytes) else fn.split('_')[0])
        pc_abc_to_idx[abc] = i
    print(f"  {len(pc_abc_to_idx)} samples")

    # === Step 3: Find intersection and build mapping ===
    print("\nFinding intersection...")
    brep_abc = set(brep_abc_to_idx.keys())
    text_abc = set(text_abc_to_idx.keys())
    pc_abc = set(pc_abc_to_idx.keys())

    # ABC UIDs present in all 3
    common_abc = brep_abc & text_abc & pc_abc
    # Filter to those with canonical mapping
    valid_abc = [abc for abc in common_abc if abc in abc_to_canonical]
    print(f"  {len(valid_abc)} samples with all 3 modalities")

    # Build the mapping: canonical_uid -> {brep_idx, text_idx, pc_idx}
    mapping = {}
    for abc_uid in valid_abc:
        canonical = abc_to_canonical[abc_uid]
        mapping[canonical] = {
            'brep_idx': brep_abc_to_idx[abc_uid],
            'text_idx': text_abc_to_idx[abc_uid],
            'pc_idx': pc_abc_to_idx[abc_uid],
            'abc_uid': abc_uid
        }

    # === Step 4: Generate splits ===
    print("\nGenerating splits...")
    np.random.seed(args.seed)
    canonical_uids = list(mapping.keys())
    np.random.shuffle(canonical_uids)

    n_val = int(len(canonical_uids) * args.val_ratio)
    val_uids = canonical_uids[:n_val]
    train_uids = canonical_uids[n_val:]

    # === Step 5: Save everything ===
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)

    # Save mapping
    with open(output_dir / 'uid_mapping.json', 'w') as f:
        json.dump(mapping, f)
    print(f"  Saved {output_dir / 'uid_mapping.json'}")

    # Save splits
    np.savetxt(splits_dir / 'train_uids.txt', train_uids, fmt='%d')
    np.savetxt(splits_dir / 'val_uids.txt', val_uids, fmt='%d')

    with open(splits_dir / 'split_info.json', 'w') as f:
        json.dump({
            'train': len(train_uids),
            'val': len(val_uids),
            'total': len(canonical_uids),
            'pc_file': str(pc_file)
        }, f, indent=2)

    print(f"\nDone!")
    print(f"  Total: {len(canonical_uids)}")
    print(f"  Train: {len(train_uids)}, Val: {len(val_uids)}")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
