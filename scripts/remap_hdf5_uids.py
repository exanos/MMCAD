#!/usr/bin/env python3
"""
Remap HDF5 files to use canonical UIDs and create aligned datasets.
Optimized version using vectorized operations.
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description='Remap HDF5 files to canonical UIDs')
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

    # === Step 1: Build ABC -> Canonical mapping (vectorized) ===
    print("Loading combined_sorted_models.csv...")
    df = pd.read_csv(data_root / 'combined_sorted_models.csv')
    df['modelname'] = df['modelname'].astype(str).str.strip('"').str.zfill(8).astype(int)
    df['uid'] = df['uid'].astype(str).str.strip('"').astype(int)
    abc_to_canonical = dict(zip(df['modelname'], df['uid']))
    print(f"  {len(abc_to_canonical)} ABC->Canonical mappings")

    # === Step 2: Get UIDs from each modality ===
    print("\nLoading B-Rep UIDs...")
    with h5py.File(data_root / 'embeddings' / 'brep_features.h5', 'r') as f:
        brep_abc_uids = f['uids'][:].astype(int)
    print(f"  {len(brep_abc_uids)} B-Rep samples")

    print("Loading Text UIDs...")
    with h5py.File(data_root / 'embeddings' / 'text_embeddings.h5', 'r') as f:
        raw_uids = f['uids'][:]
        text_abc_uids = []
        for u in raw_uids:
            try:
                text_abc_uids.append(int(u.decode() if isinstance(u, bytes) else u))
            except:
                text_abc_uids.append(-1)
        text_abc_uids = np.array(text_abc_uids)
    print(f"  {len(text_abc_uids)} Text samples")

    print("Loading PC filenames...")
    with h5py.File(pc_file, 'r') as f:
        filenames = f['filenames'][:]
        pc_abc_uids = np.array([int(fn.decode().split('_')[0]) if isinstance(fn, bytes)
                                else int(fn.split('_')[0]) for fn in filenames])
    print(f"  {len(pc_abc_uids)} PC samples")

    # === Step 3: Find intersection ===
    print("\nFinding intersection...")
    # Convert to canonical UIDs
    brep_canonical = set(abc_to_canonical.get(u, -1) for u in brep_abc_uids)
    text_canonical = set(abc_to_canonical.get(u, -1) for u in text_abc_uids)
    pc_canonical = set(abc_to_canonical.get(u, -1) for u in pc_abc_uids)

    brep_canonical.discard(-1)
    text_canonical.discard(-1)
    pc_canonical.discard(-1)

    valid_canonical = sorted(brep_canonical & text_canonical & pc_canonical)
    print(f"  B-Rep: {len(brep_canonical)}, Text: {len(text_canonical)}, PC: {len(pc_canonical)}")
    print(f"  Intersection: {len(valid_canonical)} samples")

    # === Step 4: Build index lookups ===
    valid_set = set(valid_canonical)

    # B-Rep: abc_uid -> index
    brep_abc_to_idx = {u: i for i, u in enumerate(brep_abc_uids)}
    # Text: abc_uid -> index
    text_abc_to_idx = {u: i for i, u in enumerate(text_abc_uids) if u != -1}
    # PC: abc_uid -> index
    pc_abc_to_idx = {u: i for i, u in enumerate(pc_abc_uids)}

    # Canonical -> ABC (reverse mapping)
    canonical_to_abc = {v: k for k, v in abc_to_canonical.items()}

    # === Step 5: Write aligned B-Rep ===
    print("\nWriting aligned B-Rep features...")
    with h5py.File(data_root / 'embeddings' / 'brep_features.h5', 'r') as f_in:
        face_feat = f_in['face_features']
        edge_feat = f_in['edge_features']
        face_masks = f_in['face_masks']
        edge_masks = f_in['edge_masks']

        n = len(valid_canonical)
        with h5py.File(output_dir / 'brep_features.h5', 'w') as f_out:
            out_uids = f_out.create_dataset('uids', (n,), dtype='i8')
            out_face = f_out.create_dataset('face_features', (n,) + face_feat.shape[1:], dtype=face_feat.dtype)
            out_edge = f_out.create_dataset('edge_features', (n,) + edge_feat.shape[1:], dtype=edge_feat.dtype)
            out_fmask = f_out.create_dataset('face_masks', (n,) + face_masks.shape[1:], dtype=face_masks.dtype)
            out_emask = f_out.create_dataset('edge_masks', (n,) + edge_masks.shape[1:], dtype=edge_masks.dtype)

            for out_idx, canonical_uid in enumerate(valid_canonical):
                abc_uid = canonical_to_abc[canonical_uid]
                in_idx = brep_abc_to_idx[abc_uid]
                out_uids[out_idx] = canonical_uid
                out_face[out_idx] = face_feat[in_idx]
                out_edge[out_idx] = edge_feat[in_idx]
                out_fmask[out_idx] = face_masks[in_idx]
                out_emask[out_idx] = edge_masks[in_idx]
                if out_idx % 10000 == 0:
                    print(f"  {out_idx}/{n}")
    print(f"  Saved {output_dir / 'brep_features.h5'}")

    # === Step 6: Write aligned Text ===
    print("\nWriting aligned Text embeddings...")
    with h5py.File(data_root / 'embeddings' / 'text_embeddings.h5', 'r') as f_in:
        desc_emb = f_in['desc_embeddings']
        desc_masks = f_in['desc_masks']
        has_title = 'title_embeddings' in f_in

        n = len(valid_canonical)
        with h5py.File(output_dir / 'text_embeddings.h5', 'w') as f_out:
            out_uids = f_out.create_dataset('uids', (n,), dtype='i8')
            out_desc = f_out.create_dataset('desc_embeddings', (n,) + desc_emb.shape[1:], dtype=desc_emb.dtype)
            out_dmask = f_out.create_dataset('desc_masks', (n,) + desc_masks.shape[1:], dtype=desc_masks.dtype)

            if has_title:
                title_emb = f_in['title_embeddings']
                title_masks = f_in['title_masks']
                out_title = f_out.create_dataset('title_embeddings', (n,) + title_emb.shape[1:], dtype=title_emb.dtype)
                out_tmask = f_out.create_dataset('title_masks', (n,) + title_masks.shape[1:], dtype=title_masks.dtype)

            for out_idx, canonical_uid in enumerate(valid_canonical):
                abc_uid = canonical_to_abc[canonical_uid]
                in_idx = text_abc_to_idx[abc_uid]
                out_uids[out_idx] = canonical_uid
                out_desc[out_idx] = desc_emb[in_idx]
                out_dmask[out_idx] = desc_masks[in_idx]
                if has_title:
                    out_title[out_idx] = title_emb[in_idx]
                    out_tmask[out_idx] = title_masks[in_idx]
                if out_idx % 10000 == 0:
                    print(f"  {out_idx}/{n}")
    print(f"  Saved {output_dir / 'text_embeddings.h5'}")

    # === Step 7: Write aligned PC (concatenate local + global) ===
    print("\nWriting aligned PC features...")
    with h5py.File(pc_file, 'r') as f_in:
        local_feat = f_in['local_features']  # [N, 32, 1024]
        global_tok = f_in['global_token']    # [N, 16, 1024]

        n = len(valid_canonical)
        with h5py.File(output_dir / 'pc_features.h5', 'w') as f_out:
            out_uids = f_out.create_dataset('uids', (n,), dtype='i8')
            out_feat = f_out.create_dataset('features', (n, 48, 1024), dtype=local_feat.dtype)

            for out_idx, canonical_uid in enumerate(valid_canonical):
                abc_uid = canonical_to_abc[canonical_uid]
                in_idx = pc_abc_to_idx[abc_uid]
                out_uids[out_idx] = canonical_uid
                out_feat[out_idx, :32, :] = local_feat[in_idx]
                out_feat[out_idx, 32:, :] = global_tok[in_idx]
                if out_idx % 10000 == 0:
                    print(f"  {out_idx}/{n}")
    print(f"  Saved {output_dir / 'pc_features.h5'}")

    # === Step 8: Generate splits ===
    print("\nGenerating train/val splits...")
    np.random.seed(args.seed)
    uids = np.array(valid_canonical)
    np.random.shuffle(uids)

    n_val = int(len(uids) * args.val_ratio)
    val_uids = uids[:n_val]
    train_uids = uids[n_val:]

    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    np.savetxt(splits_dir / 'train_uids.txt', train_uids, fmt='%d')
    np.savetxt(splits_dir / 'val_uids.txt', val_uids, fmt='%d')

    with open(splits_dir / 'split_info.json', 'w') as f:
        json.dump({'train': len(train_uids), 'val': len(val_uids), 'total': len(uids)}, f)

    print(f"  Train: {len(train_uids)}, Val: {len(val_uids)}")
    print(f"\nDone! Output: {output_dir}")


if __name__ == '__main__':
    main()
