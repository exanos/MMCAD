#!/usr/bin/env python3
"""
Preprocess text embeddings: extract only needed samples for each split.
Run once, then training loads instantly.

Usage:
    python scripts/preprocess_text_splits.py \
        --text-file "c:/Users/User/Desktop/text_embeddings.h5" \
        --data-root "d:/Defect_Det/MMCAD/data" \
        --output-dir "c:/Users/User/Desktop/text_splits"
"""

import argparse
import json
import time
from pathlib import Path
import h5py
import numpy as np


def extract_split(text_h5, uid_mapping, split_uids, split_name, output_dir, use_fp16=True, max_samples=None):
    """Extract text embeddings for one split."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"{'='*60}")

    # Limit samples for train split if requested
    original_n = len(split_uids)
    if max_samples and split_name == 'train' and original_n > max_samples:
        print(f"  Reducing from {original_n} to {max_samples} samples (fits in RAM)")
        split_uids = split_uids[:max_samples]

    n = len(split_uids)
    text_indices = np.array([uid_mapping[uid]['text_idx'] for uid in split_uids])

    print(f"  Samples: {n}")
    print(f"  Text indices range: {text_indices.min()} - {text_indices.max()}")

    # Get dimensions
    seq_len = text_h5['desc_embeddings'].shape[1]  # 256
    emb_dim = text_h5['desc_embeddings'].shape[2]  # 3072
    total_samples = text_h5['desc_embeddings'].shape[0]  # 169k

    # Pre-allocate
    dtype = np.float16 if use_fp16 else np.float32
    text_embs = np.empty((n, seq_len, emb_dim), dtype=dtype)
    text_masks = np.empty((n, seq_len), dtype=dtype)

    print(f"  Output dtype: {dtype}")
    print(f"  Expected size: {text_embs.nbytes / 1e9:.1f} GB")

    # Sequential scan and extract
    start_time = time.time()
    chunk_size = 10000  # 10k chunks to avoid OOM (50k × FP32 = 157GB!)
    loaded_count = 0

    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)

        # Find which needed indices are in this chunk
        mask = (text_indices >= chunk_start) & (text_indices < chunk_end)

        if mask.any():
            # Sequential read (fast!)
            chunk_embs = text_h5['desc_embeddings'][chunk_start:chunk_end]
            chunk_masks = text_h5['desc_masks'][chunk_start:chunk_end]

            # Extract needed samples FIRST, then convert (avoids converting full chunk)
            local_indices = text_indices[mask] - chunk_start
            needed_embs = chunk_embs[local_indices]  # Still FP32, but smaller
            needed_masks = chunk_masks[local_indices]

            # Now convert to FP16
            text_embs[mask] = needed_embs.astype(dtype)
            text_masks[mask] = needed_masks.astype(dtype)

            loaded_count += mask.sum()
            del chunk_embs, chunk_masks, needed_embs, needed_masks

        # Progress
        if chunk_end % 50000 == 0 or chunk_end == total_samples:
            elapsed = time.time() - start_time
            pct = chunk_end / total_samples * 100
            print(f"    {chunk_end}/{total_samples} ({pct:.1f}%) | {loaded_count}/{n} found | {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"  ✓ Extracted {loaded_count} samples in {elapsed:.1f}s")

    # Save to new HDF5 file
    output_path = output_dir / f"{split_name}_text_embeddings.h5"
    print(f"  Saving to: {output_path} (no compression for speed!)")

    save_start = time.time()
    with h5py.File(output_path, 'w') as f:
        # NO compression - saves in seconds instead of hours!
        print(f"    Writing desc_embeddings...")
        f.create_dataset('desc_embeddings', data=text_embs)
        print(f"    Writing desc_masks...")
        f.create_dataset('desc_masks', data=text_masks)

        # Save metadata
        f.attrs['num_samples'] = n
        f.attrs['seq_len'] = seq_len
        f.attrs['emb_dim'] = emb_dim
        f.attrs['dtype'] = str(dtype)

    save_time = time.time() - save_start
    print(f"  ✓ Saved in {save_time:.1f}s: {output_path.stat().st_size / 1e9:.1f} GB")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-file', required=True, help='Original text_embeddings.h5')
    parser.add_argument('--data-root', required=True, help='Data root with aligned/')
    parser.add_argument('--output-dir', required=True, help='Output directory for split files')
    parser.add_argument('--fp16', action='store_true', help='Save as FP16 (saves 50% space)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit train samples to fit in RAM (e.g., 111000 for 256GB)')
    parser.add_argument('--train-output-dir', default=None,
                       help='Separate output dir for train (e.g., C:/Desktop, val/test stay in --output-dir)')
    args = parser.parse_args()

    text_file = Path(args.text_file)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate output dir for train (large) vs val/test (small)
    train_output_dir = Path(args.train_output_dir) if args.train_output_dir else output_dir
    train_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Text Embeddings Split Preprocessor")
    print(f"{'='*60}")
    print(f"Input: {text_file}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print(f"FP16: {args.fp16}")

    # Load mapping
    mapping_file = data_root / 'aligned' / 'uid_mapping.json'
    print(f"\nLoading UID mapping: {mapping_file}")
    with open(mapping_file) as f:
        uid_mapping = json.load(f)
    print(f"  ✓ {len(uid_mapping)} samples")

    # Load splits
    splits_dir = data_root / 'aligned' / 'splits'
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f'{split_name}_uids.txt'
        print(f"Loading {split_name} split: {split_file}")
        with open(split_file) as f:
            uids = [line.strip() for line in f]
        splits[split_name] = uids
        print(f"  ✓ {len(uids)} samples")

    # Open text file
    print(f"\nOpening text embeddings: {text_file}")
    with h5py.File(text_file, 'r') as text_h5:
        print(f"  Shape: {text_h5['desc_embeddings'].shape}")
        print(f"  Size: {text_h5['desc_embeddings'].nbytes / 1e9:.1f} GB")

        # Process each split
        output_files = {}
        for split_name, split_uids in splits.items():
            # Train goes to train_output_dir (C:), val/test go to output_dir (D:)
            split_output_dir = train_output_dir if split_name == 'train' else output_dir
            output_path = extract_split(
                text_h5, uid_mapping, split_uids, split_name,
                split_output_dir, use_fp16=args.fp16, max_samples=args.max_samples
            )
            output_files[split_name] = str(output_path)

    # Save metadata
    meta_file = output_dir / 'split_files.json'
    metadata = {
        'output_files': output_files,
        'original_file': str(text_file),
        'fp16': args.fp16,
        'num_samples': {k: len(v) for k, v in splits.items()},
    }
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Preprocessing complete!")
    print(f"{'='*60}")
    print(f"Metadata saved to: {meta_file}")
    print(f"\nOutput files:")
    for split_name, output_path in output_files.items():
        print(f"  {split_name}: {output_path}")

    print(f"\nTo use in training:")
    print(f"  --text-file {output_files['train']}")

    if args.train_output_dir:
        print(f"\n⚠️  To save disk space on C: drive:")
        print(f"  After verifying training works, you can delete:")
        print(f"    {text_file}")
        print(f"  This will free up ~390GB!")


if __name__ == '__main__':
    main()
