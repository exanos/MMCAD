#!/usr/bin/env python3
"""
Evaluation script for CLIP4CAD-H.

Usage:
    python scripts/evaluate.py checkpoint=outputs/checkpoints/best.pt
    python scripts/evaluate.py checkpoint=outputs/checkpoints/best.pt eval.rotation=true
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import json

from clip4cad.models.clip4cad_h import CLIP4CAD_H
from clip4cad.data.dataset import MMCADDataset, create_dataloader
from clip4cad.evaluation.retrieval import (
    extract_embeddings,
    cross_modal_retrieval,
    text_to_geometry_retrieval,
    geometry_to_text_retrieval,
)
from clip4cad.evaluation.rotation_robustness import evaluate_rotation_robustness
from clip4cad.utils.misc import get_device


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(config: DictConfig):
    """Main evaluation entry point."""
    print("=" * 60)
    print("CLIP4CAD-H Evaluation")
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
        # Try to find best checkpoint
        output_dir = Path(config.experiment.output_dir)
        checkpoint_path = output_dir / "checkpoints" / "best.pt"
        if not checkpoint_path.exists():
            checkpoint_path = output_dir / "checkpoints" / "latest.pt"

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: No checkpoint found, using random weights")

    # Load LLM for text encoding
    print("Loading LLM for text encoding...")
    model.load_llm(device)

    model = model.to(device)
    model.eval()

    # ============================================================
    # Load Data
    # ============================================================

    print("\nLoading evaluation data...")
    tokenizer = model.get_tokenizer()

    # Test split
    test_dataset = MMCADDataset(
        data_root=config.data.data_root,
        split="test",
        tokenizer=tokenizer,
        max_faces=config.data.max_faces,
        max_edges=config.data.max_edges,
        num_points=config.data.num_points,
        max_title_len=config.data.max_title_len,
        max_desc_len=config.data.max_desc_len,
        rotation_augment=False,
        point_jitter=0.0,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    print(f"Test samples: {len(test_dataset)}")

    # ============================================================
    # Extract Embeddings
    # ============================================================

    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(
        model=model,
        dataloader=test_loader,
        device=device,
        modalities=["brep", "point", "text"],
    )

    print(f"Extracted embeddings:")
    for mod, emb in embeddings.items():
        print(f"  {mod}: {emb.shape}")

    # ============================================================
    # Retrieval Evaluation
    # ============================================================

    print("\n" + "=" * 60)
    print("Retrieval Evaluation")
    print("=" * 60)

    all_metrics = {}

    # Cross-modal retrieval
    retrieval_metrics = cross_modal_retrieval(
        embeddings,
        k_values=[1, 5, 10],
    )

    print("\nCross-modal Retrieval:")
    for key, value in sorted(retrieval_metrics.items()):
        print(f"  {key}: {value:.2f}")
        all_metrics[key] = value

    # Text-to-geometry (using combined geometry embedding)
    if "brep" in embeddings and "text" in embeddings:
        t2g_metrics = text_to_geometry_retrieval(
            embeddings["text"],
            embeddings["brep"],
            k_values=[1, 5, 10],
        )
        print("\nText-to-BRep Retrieval:")
        for key, value in sorted(t2g_metrics.items()):
            print(f"  {key}: {value:.2f}")
            all_metrics[key] = value

    if "point" in embeddings and "text" in embeddings:
        t2p_metrics = text_to_geometry_retrieval(
            embeddings["text"],
            embeddings["point"],
            k_values=[1, 5, 10],
        )
        print("\nText-to-Point Retrieval:")
        for key, value in sorted(t2p_metrics.items()):
            print(f"  {key}: {value:.2f}")
            all_metrics[f"t2p_{key.replace('t2g_', '')}"] = value

    # ============================================================
    # Rotation Robustness (optional)
    # ============================================================

    eval_config = config.get("eval", {})
    if eval_config.get("rotation", False):
        print("\n" + "=" * 60)
        print("Rotation Robustness Evaluation")
        print("=" * 60)

        rotation_metrics = evaluate_rotation_robustness(
            model=model,
            dataloader=test_loader,
            device=device,
            n_rotations=eval_config.get("n_rotations", 12),
        )

        print("\nRotation Robustness:")
        for key, value in sorted(rotation_metrics.items()):
            print(f"  {key}: {value:.4f}")
            all_metrics[key] = value

    # ============================================================
    # Save Results
    # ============================================================

    output_dir = Path(config.experiment.output_dir)
    results_path = output_dir / "evaluation_results.json"

    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
