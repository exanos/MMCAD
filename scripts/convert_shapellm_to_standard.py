#!/usr/bin/env python3
"""
Convert ShapeLLM H5 to standard CLIP4CAD format.

This script:
1. Loads ShapeLLM H5 with local_features (32, 1024) and global_token (16, 1024)
2. Concatenates to get (48, 1024) per sample
3. Filters to only samples with valid UIDs (have text)
4. Saves in standard format with sample_ids and features datasets

Usage:
    python scripts/convert_shapellm_to_standard.py \
        --h5 ../data/embeddings/selected_embeddings_10pct_sample.h5 \
        --mapping ../data/shapellm_uid_mapping.json \
        --output ../data/embeddings/shapellm_pc_features.h5
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def convert_shapellm_to_standard(
    h5_path: Path,
    mapping_path: Path,
    output_path: Path,
    batch_size: int = 1000,
) -> None:
    """
    Convert ShapeLLM H5 to standard CLIP4CAD format.

    Output HDF5 structure:
        sample_ids: [N] - UIDs as strings
        features: [N, 48, 1024] - concatenated local + global features
    """
    print(f"Loading mapping: {mapping_path}")
    with open(mapping_path) as f:
        mapping = json.load(f)

    valid_uids = mapping["valid_uids"]
    uid_to_h5_idx = mapping["uid_to_h5_idx"]

    print(f"Valid UIDs (with text): {len(valid_uids)}")

    print(f"\nOpening ShapeLLM H5: {h5_path}")
    with h5py.File(h5_path, "r") as src:
        local_features = src["local_features"]
        global_token = src["global_token"]

        n_samples = len(valid_uids)
        local_dim = local_features.shape[1:]  # (32, 1024)
        global_dim = global_token.shape[1:]   # (16, 1024)

        print(f"local_features shape: {local_features.shape}")
        print(f"global_token shape: {global_token.shape}")

        # Output dimensions
        total_tokens = local_dim[0] + global_dim[0]  # 32 + 16 = 48
        embed_dim = local_dim[1]  # 1024
        print(f"\nOutput shape per sample: ({total_tokens}, {embed_dim})")
        print(f"Total samples to convert: {n_samples}")

        # Create output H5
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as dst:
            # Create datasets
            dt = h5py.special_dtype(vlen=str)
            sample_ids_ds = dst.create_dataset(
                "sample_ids",
                shape=(n_samples,),
                dtype=dt,
            )

            features_ds = dst.create_dataset(
                "features",
                shape=(n_samples, total_tokens, embed_dim),
                dtype=np.float32,
                chunks=(min(100, n_samples), total_tokens, embed_dim),
                compression="lzf",
            )

            # Store metadata
            dst.attrs["model_config"] = "shapellm-recon++"
            dst.attrs["embed_dim"] = embed_dim
            dst.attrs["num_tokens"] = total_tokens
            dst.attrs["num_local_tokens"] = int(local_dim[0])
            dst.attrs["num_global_tokens"] = int(global_dim[0])
            dst.attrs["source_h5"] = str(h5_path)

            # Process in batches
            print("\nConverting features...")
            for batch_start in tqdm(range(0, n_samples, batch_size), desc="Batches"):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_uids = valid_uids[batch_start:batch_end]

                batch_features = []
                for uid in batch_uids:
                    h5_idx = uid_to_h5_idx[uid]

                    # Load and concatenate
                    local = local_features[h5_idx]      # (32, 1024)
                    global_ = global_token[h5_idx]      # (16, 1024)
                    combined = np.concatenate([local, global_], axis=0)  # (48, 1024)
                    batch_features.append(combined)

                # Write batch
                batch_features = np.array(batch_features, dtype=np.float32)
                features_ds[batch_start:batch_end] = batch_features
                sample_ids_ds[batch_start:batch_end] = batch_uids

            dst.attrs["n_samples"] = n_samples

    print(f"\nSaved converted features to: {output_path}")

    # Verify
    print("\nVerifying output...")
    with h5py.File(output_path, "r") as f:
        print(f"  sample_ids: {f['sample_ids'].shape}")
        print(f"  features: {f['features'].shape}")
        print(f"  Metadata: embed_dim={f.attrs['embed_dim']}, num_tokens={f.attrs['num_tokens']}")

        # Sample check
        print(f"  First 5 UIDs: {[f['sample_ids'][i] for i in range(5)]}")
        print(f"  Sample feature stats: mean={f['features'][0].mean():.4f}, std={f['features'][0].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Convert ShapeLLM H5 to standard format")
    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Path to ShapeLLM H5 file",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="Path to UID mapping JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output H5 path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Processing batch size",
    )

    args = parser.parse_args()

    convert_shapellm_to_standard(
        h5_path=args.h5,
        mapping_path=args.mapping,
        output_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
