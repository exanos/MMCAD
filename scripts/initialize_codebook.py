#!/usr/bin/env python
"""
Initialize Hierarchical Codebook from Text Features using Model's Text Encoder

This script pre-computes codebook initialization using the ACTUAL model's text encoder
to ensure codes are in the correct feature space.

IMPORTANT: Run this AFTER Stage 0 training to use the trained model weights.

Usage:
    python scripts/initialize_codebook.py \
        --text-h5 c:/Users/User/Desktop/text_splits/train_text_embeddings.h5 \
        --checkpoint outputs/gfa_v4_8_2/checkpoint_stage0.pt \
        --output outputs/gfa_v4_8_2/codebook_init.pt \
        --max-samples 50000

Output:
    A .pt file containing the codebook state dict that can be loaded with:
    ```python
    model.load_codebook('codebook_init.pt')
    ```
"""

import argparse
import os
import sys
import gc
from pathlib import Path

# CRITICAL: Set OpenMP threads BEFORE importing numpy/sklearn
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Initialize codebook using model text encoder')
    parser.add_argument('--text-h5', type=str, required=True,
                        help='Path to text embeddings HDF5 file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Stage 0 checkpoint (or any checkpoint with trained text encoder)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for codebook state dict')
    parser.add_argument('--max-samples', type=int, default=50000,
                        help='Maximum samples to use for clustering (default: 50000)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for text encoder forward pass')

    # Model config (v4.8.2 defaults with increased capacity)
    parser.add_argument('--d-text', type=int, default=3072,
                        help='Text feature dimension (Phi-4-mini: 3072)')
    parser.add_argument('--d', type=int, default=320,
                        help='Internal model dimension (v4.8.2: 320, v4.8.1: 256)')
    parser.add_argument('--n-category', type=int, default=20,
                        help='Number of category codes (v4.8.2: 20, v4.8.1: 16)')
    parser.add_argument('--n-type-per-cat', type=int, default=10,
                        help='Number of type codes per category (v4.8.2: 10, v4.8.1: 8)')
    parser.add_argument('--n-variant-per-type', type=int, default=4,
                        help='Number of variant codes per type')
    parser.add_argument('--n-spatial', type=int, default=20,
                        help='Number of spatial codes (v4.8.2: 20, v4.8.1: 16)')

    # Clustering options
    parser.add_argument('--n-init', type=int, default=5,
                        help='Number of K-means initializations')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def load_text_features(h5_path: str, max_samples: int = None):
    """Load text features from HDF5 file."""
    print(f"Loading text features from {h5_path}...")

    with h5py.File(h5_path, 'r') as f:
        print(f"Available keys: {list(f.keys())}")

        # Find the text features dataset
        if 'desc_embeddings' in f:
            embeddings = f['desc_embeddings']
            masks = f.get('desc_masks', None)
        elif 'desc_embedding' in f:
            embeddings = f['desc_embedding']
            masks = f.get('desc_mask', None)
        else:
            raise KeyError(f"Could not find text embeddings. Available keys: {list(f.keys())}")

        n_total = embeddings.shape[0]
        print(f"Found {n_total} samples, shape: {embeddings.shape}")

        if max_samples and max_samples < n_total:
            np.random.seed(42)
            indices = np.random.choice(n_total, max_samples, replace=False)
            indices = np.sort(indices)

            print(f"Subsampling {max_samples} samples...")
            text_feats = embeddings[indices]
            text_masks = masks[indices] if masks is not None else None
        else:
            text_feats = embeddings[:]
            text_masks = masks[:] if masks is not None else None

        print(f"Loaded shape: {text_feats.shape}")
        return text_feats, text_masks


def project_with_model_encoder(
    text_feats: np.ndarray,
    text_masks: np.ndarray,
    checkpoint_path: str,
    batch_size: int,
    device: str = 'cuda'
) -> np.ndarray:
    """Project text features through the model's actual text encoder."""
    from clip4cad.models.clip4cad_gfa_v4_8_2 import CLIP4CAD_GFA_v482, GFAv482Config

    print(f"Loading model from checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Use config from checkpoint if available, otherwise use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"Using config from checkpoint: d={config.d}, n_category={config.n_category}")
    else:
        config = GFAv482Config()
        print("Using default config")

    model = CLIP4CAD_GFA_v482(config).to(device)

    # Check if checkpoint was saved with gradient checkpointing
    # by looking for '_checkpointed_msg_layers' keys
    state_dict = checkpoint['model_state_dict']
    has_checkpointing = any('_checkpointed_msg_layers' in k for k in state_dict.keys())

    if has_checkpointing:
        print("Checkpoint has gradient checkpointing enabled, enabling on model...")
        model.enable_gradient_checkpointing()

    # Load weights (use strict=False as fallback for any remaining mismatches)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed, trying non-strict: {e}")
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("Projecting text features through model's text encoder...")

    all_projected = []
    n_samples = len(text_feats)

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Projecting"):
            batch_feats = torch.tensor(
                text_feats[i:i+batch_size], device=device, dtype=torch.float32
            )

            if text_masks is not None:
                batch_masks = torch.tensor(
                    text_masks[i:i+batch_size], device=device
                )
            else:
                batch_masks = None

            # Use model's text encoder (same path as during training!)
            X_text, mask = model.text_encoder(batch_feats, batch_masks)

            # Pool: masked mean
            if mask is not None:
                mask_float = mask.float().unsqueeze(-1)
                pooled = (X_text * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)
            else:
                pooled = X_text.mean(dim=1)

            all_projected.append(pooled.cpu().numpy())

            # Clear GPU memory periodically
            if i % (batch_size * 20) == 0:
                torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return np.concatenate(all_projected, axis=0).astype(np.float32)


def run_hierarchical_kmeans(
    features: np.ndarray,
    n_category: int,
    n_type_per_cat: int,
    n_variant_per_type: int,
    n_spatial: int,
    n_init: int = 5,
    random_state: int = 42,
) -> dict:
    """Run hierarchical K-means to initialize all codebook levels."""
    d = features.shape[1]

    print(f"\nRunning hierarchical K-means on {len(features)} samples, d={d}")
    print("=" * 60)

    # Level 0: Category codes
    print(f"Level 0: {n_category} category codes...")
    km_cat = MiniBatchKMeans(
        n_clusters=n_category, random_state=random_state, n_init=n_init, batch_size=1024
    )
    km_cat.fit(features)
    category_codes = km_cat.cluster_centers_.astype(np.float32)
    print(f"  Inertia: {km_cat.inertia_:.4f}")

    # Get category assignments
    cat_labels = pairwise_distances_argmin(features, category_codes)
    gc.collect()

    # Level 1: Type codes (hierarchical within category)
    print(f"Level 1: {n_category * n_type_per_cat} type codes...")
    type_codes = np.zeros((n_category, n_type_per_cat, d), dtype=np.float32)

    for cat_idx in tqdm(range(n_category), desc="Type clustering"):
        cat_mask = cat_labels == cat_idx
        cat_feats = features[cat_mask]

        if len(cat_feats) < n_type_per_cat:
            # Not enough samples, use category center with noise
            for j in range(n_type_per_cat):
                noise = np.random.randn(d).astype(np.float32) * 0.02
                type_codes[cat_idx, j] = category_codes[cat_idx] + noise
        else:
            km = MiniBatchKMeans(
                n_clusters=n_type_per_cat, random_state=random_state + cat_idx, n_init=3
            )
            km.fit(cat_feats)
            type_codes[cat_idx] = km.cluster_centers_.astype(np.float32)

    gc.collect()

    # Get type assignments
    type_codes_flat = type_codes.reshape(-1, d)
    type_labels = pairwise_distances_argmin(features, type_codes_flat)
    gc.collect()

    # Level 2: Variant codes (hierarchical within type)
    n_types = n_category * n_type_per_cat
    print(f"Level 2: {n_types * n_variant_per_type} variant codes...")
    variant_codes = np.zeros(
        (n_category, n_type_per_cat, n_variant_per_type, d), dtype=np.float32
    )

    for type_idx in tqdm(range(n_types), desc="Variant clustering"):
        type_mask = type_labels == type_idx
        type_feats = features[type_mask]

        cat_idx = type_idx // n_type_per_cat
        type_in_cat = type_idx % n_type_per_cat

        if len(type_feats) < n_variant_per_type:
            # Not enough samples, use type center with noise
            for k in range(n_variant_per_type):
                noise = np.random.randn(d).astype(np.float32) * 0.02
                variant_codes[cat_idx, type_in_cat, k] = type_codes[cat_idx, type_in_cat] + noise
        else:
            km = MiniBatchKMeans(
                n_clusters=n_variant_per_type, random_state=random_state + type_idx, n_init=3
            )
            km.fit(type_feats)
            variant_codes[cat_idx, type_in_cat] = km.cluster_centers_.astype(np.float32)

        # Periodic cleanup
        if type_idx % 32 == 0:
            gc.collect()

    gc.collect()

    # Spatial codes (independent clustering)
    print(f"Spatial: {n_spatial} spatial codes...")
    km_spatial = MiniBatchKMeans(
        n_clusters=n_spatial, random_state=random_state + 1000, n_init=n_init, batch_size=1024
    )
    km_spatial.fit(features)
    spatial_codes = km_spatial.cluster_centers_.astype(np.float32)
    print(f"  Inertia: {km_spatial.inertia_:.4f}")

    gc.collect()

    return {
        'category_codes': category_codes,
        'type_codes': type_codes,
        'variant_codes': variant_codes,
        'spatial_codes': spatial_codes,
    }


def create_codebook_state_dict(codes: dict, d: int) -> dict:
    """Create codebook state dict matching HierarchicalCodebook."""
    state = {
        'category_codes': torch.tensor(codes['category_codes']),
        'type_codes': torch.tensor(codes['type_codes']),
        'variant_codes': torch.tensor(codes['variant_codes']),
        'spatial_codes': torch.tensor(codes['spatial_codes']),
        'log_tau': torch.zeros(1),
        'category_proj.weight': torch.eye(d),
        'category_proj.bias': torch.zeros(d),
        'type_proj.weight': torch.eye(d),
        'type_proj.bias': torch.zeros(d),
        'variant_proj.weight': torch.eye(d),
        'variant_proj.bias': torch.zeros(d),
        'spatial_proj.weight': torch.eye(d),
        'spatial_proj.bias': torch.zeros(d),
    }
    return state


def main():
    args = parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"OpenMP threads limited to 4 for stability")

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load text features
    text_feats, text_masks = load_text_features(args.text_h5, args.max_samples)

    # Load checkpoint to check for config
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Use config from checkpoint if available, override with CLI args if provided
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        # Use checkpoint values as defaults, but allow CLI override
        d = args.d if args.d != 320 else ckpt_config.d  # 320 is the new default
        n_category = args.n_category if args.n_category != 20 else ckpt_config.n_category
        n_type_per_cat = args.n_type_per_cat if args.n_type_per_cat != 10 else ckpt_config.n_type_per_cat
        n_variant_per_type = args.n_variant_per_type if args.n_variant_per_type != 4 else ckpt_config.n_variant_per_type
        n_spatial = args.n_spatial if args.n_spatial != 20 else ckpt_config.n_spatial
        print(f"Using config from checkpoint: d={d}, n_category={n_category}, n_type_per_cat={n_type_per_cat}")
    else:
        d = args.d
        n_category = args.n_category
        n_type_per_cat = args.n_type_per_cat
        n_variant_per_type = args.n_variant_per_type
        n_spatial = args.n_spatial
        print(f"Using CLI args: d={d}, n_category={n_category}")

    del checkpoint
    gc.collect()

    # Project through MODEL'S text encoder (not a random one!)
    projected = project_with_model_encoder(
        text_feats, text_masks,
        args.checkpoint,
        args.batch_size, device
    )
    print(f"Projected features shape: {projected.shape}")

    # Free memory
    del text_feats, text_masks
    gc.collect()

    # Run hierarchical K-means
    codes = run_hierarchical_kmeans(
        projected,
        n_category,
        n_type_per_cat,
        n_variant_per_type,
        n_spatial,
        args.n_init,
        args.random_seed,
    )

    # Create state dict
    print("\nCreating codebook state dict...")
    state = create_codebook_state_dict(codes, d)

    # Save
    print(f"Saving to {args.output}...")
    torch.save(state, args.output)

    # Summary
    total_codes = (
        n_category +
        n_category * n_type_per_cat +
        n_category * n_type_per_cat * n_variant_per_type +
        n_spatial
    )
    print(f"\n" + "=" * 60)
    print(f"Codebook initialized successfully!")
    print(f"  Category codes: {n_category}")
    print(f"  Type codes: {n_category * n_type_per_cat}")
    print(f"  Variant codes: {n_category * n_type_per_cat * n_variant_per_type}")
    print(f"  Spatial codes: {n_spatial}")
    print(f"  Total: {total_codes}")
    print(f"  Dimension: {d}")
    print(f"\nTo load in training notebook:")
    print(f"  model.load_codebook('{args.output}')")


if __name__ == '__main__':
    main()
