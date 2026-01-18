#!/usr/bin/env python3
"""
Feature extraction script for CLIP4CAD-H.

Extracts embeddings from trained model for downstream tasks.

Usage:
    python scripts/extract_features.py checkpoint=outputs/checkpoints/best.pt
    python scripts/extract_features.py checkpoint=path/to/checkpoint.pt output_dir=features/
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from tqdm import tqdm

from clip4cad.models.clip4cad_h import CLIP4CAD_H
from clip4cad.data.dataset import MMCADDataset, create_dataloader
from clip4cad.utils.misc import get_device


@torch.no_grad()
def extract_all_features(
    model,
    dataloader,
    device: torch.device,
    output_dir: Path,
    save_detail_features: bool = False,
):
    """
    Extract all features from dataset and save to disk.

    Args:
        model: CLIP4CAD model
        dataloader: Data loader
        device: Device
        output_dir: Output directory
        save_detail_features: Whether to save detail-level features
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Storage
    all_features = {
        "brep_global": [],
        "point_global": [],
        "text_global": [],
        "unified_global": [],
        "sample_ids": [],
    }

    if save_detail_features:
        all_features["brep_detail"] = []
        all_features["point_detail"] = []
        all_features["text_local"] = []

    sample_idx = 0

    for batch in tqdm(dataloader, desc="Extracting features"):
        outputs = model(batch)
        batch_size = outputs.get("brep_global", outputs.get("point_global")).shape[0]

        # Global features
        if "brep_global" in outputs:
            all_features["brep_global"].append(outputs["brep_global"].cpu())

        if "point_global" in outputs:
            all_features["point_global"].append(outputs["point_global"].cpu())

        if "text_global" in outputs:
            all_features["text_global"].append(outputs["text_global"].cpu())

        # Unified representation
        if "unified_global" in outputs:
            all_features["unified_global"].append(outputs["unified_global"].cpu())

        # Sample IDs
        all_features["sample_ids"].extend(range(sample_idx, sample_idx + batch_size))
        sample_idx += batch_size

        # Detail features (optional, can be large)
        if save_detail_features:
            if "brep_detail" in outputs:
                all_features["brep_detail"].append(outputs["brep_detail"].cpu())
            if "point_detail" in outputs:
                all_features["point_detail"].append(outputs["point_detail"].cpu())
            if "text_local" in outputs:
                all_features["text_local"].append(outputs["text_local"].cpu())

    # Concatenate and save
    for key in all_features:
        if key == "sample_ids":
            data = np.array(all_features[key])
        elif all_features[key]:
            data = torch.cat(all_features[key], dim=0).numpy()
        else:
            continue

        save_path = output_dir / f"{key}.npy"
        np.save(save_path, data)
        print(f"Saved {key}: {data.shape} -> {save_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(config: DictConfig):
    """Main feature extraction entry point."""
    print("=" * 60)
    print("CLIP4CAD-H Feature Extraction")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    # ============================================================
    # Load Model
    # ============================================================

    print("\nLoading model...")
    model = CLIP4CAD_H(config.model)

    # Load checkpoint
    checkpoint_path = config.get("checkpoint", None)
    if checkpoint_path is None:
        output_dir = Path(config.experiment.output_dir)
        checkpoint_path = output_dir / "checkpoints" / "best.pt"

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    # Load LLM
    print("Loading LLM for text encoding...")
    model.load_llm(device)

    model = model.to(device)
    model.eval()

    # ============================================================
    # Process Splits
    # ============================================================

    tokenizer = model.get_tokenizer()
    feature_output_dir = Path(config.get("output_dir", "features"))

    for split in ["train", "val", "test"]:
        split_file = Path(config.data.data_root) / "splits" / f"{split}.txt"
        if not split_file.exists():
            print(f"Skipping {split} split (not found)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {split} split")
        print("=" * 60)

        dataset = MMCADDataset(
            data_root=config.data.data_root,
            split=split,
            tokenizer=tokenizer,
            max_faces=config.data.max_faces,
            max_edges=config.data.max_edges,
            num_points=config.data.num_points,
            max_title_len=config.data.max_title_len,
            max_desc_len=config.data.max_desc_len,
            rotation_augment=False,
            point_jitter=0.0,
        )

        dataloader = create_dataloader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        print(f"Samples: {len(dataset)}")

        split_output_dir = feature_output_dir / split
        extract_all_features(
            model=model,
            dataloader=dataloader,
            device=device,
            output_dir=split_output_dir,
            save_detail_features=config.get("save_detail_features", False),
        )

    print("\nFeature extraction complete!")
    print(f"Features saved to: {feature_output_dir}")


if __name__ == "__main__":
    main()
