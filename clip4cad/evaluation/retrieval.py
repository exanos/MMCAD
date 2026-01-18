"""
Retrieval Evaluation Metrics

Computes retrieval performance metrics for cross-modal alignment.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute retrieval metrics from similarity matrix.

    Args:
        similarity_matrix: [N, M] similarity scores (higher = more similar)
        k_values: K values for Recall@K

    Returns:
        Dictionary with metrics (R@K, MRR, median rank)
    """
    N, M = similarity_matrix.shape
    device = similarity_matrix.device

    # Ground truth: diagonal matches (assumes N == M for paired data)
    # For unpaired, need separate ground truth labels

    metrics = {}

    # Query -> Gallery direction
    sorted_indices = similarity_matrix.argsort(dim=1, descending=True)

    # Find rank of correct match (diagonal)
    ranks = torch.zeros(N, device=device)
    for i in range(N):
        gt_idx = i if i < M else i % M  # Handle size mismatch
        rank = (sorted_indices[i] == gt_idx).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            ranks[i] = rank[0].float() + 1  # 1-indexed

    # Recall@K
    for k in k_values:
        recall_at_k = (ranks <= k).float().mean().item()
        metrics[f"R@{k}"] = recall_at_k * 100

    # Mean Reciprocal Rank
    mrr = (1.0 / ranks).mean().item()
    metrics["MRR"] = mrr * 100

    # Median Rank
    median_rank = ranks.median().item()
    metrics["MedR"] = median_rank

    return metrics


def text_to_geometry_retrieval(
    text_embeddings: torch.Tensor,
    geometry_embeddings: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Text-to-geometry retrieval evaluation.

    Args:
        text_embeddings: [N, D] text embeddings
        geometry_embeddings: [N, D] geometry embeddings
        k_values: K values for Recall@K

    Returns:
        Retrieval metrics
    """
    # Normalize
    text_norm = F.normalize(text_embeddings, dim=-1)
    geom_norm = F.normalize(geometry_embeddings, dim=-1)

    # Similarity matrix
    similarity = text_norm @ geom_norm.T

    metrics = compute_retrieval_metrics(similarity, k_values)

    # Add prefix
    return {f"t2g_{k}": v for k, v in metrics.items()}


def geometry_to_text_retrieval(
    geometry_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Geometry-to-text retrieval evaluation.

    Args:
        geometry_embeddings: [N, D] geometry embeddings
        text_embeddings: [N, D] text embeddings
        k_values: K values for Recall@K

    Returns:
        Retrieval metrics
    """
    # Normalize
    geom_norm = F.normalize(geometry_embeddings, dim=-1)
    text_norm = F.normalize(text_embeddings, dim=-1)

    # Similarity matrix
    similarity = geom_norm @ text_norm.T

    metrics = compute_retrieval_metrics(similarity, k_values)

    # Add prefix
    return {f"g2t_{k}": v for k, v in metrics.items()}


def cross_modal_retrieval(
    embeddings: Dict[str, torch.Tensor],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Comprehensive cross-modal retrieval evaluation.

    Args:
        embeddings: Dict with keys like 'brep', 'point', 'text'
        k_values: K values for Recall@K

    Returns:
        All retrieval metrics
    """
    all_metrics = {}

    modalities = list(embeddings.keys())

    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            if i >= j:
                continue

            e1 = F.normalize(embeddings[mod1], dim=-1)
            e2 = F.normalize(embeddings[mod2], dim=-1)

            # mod1 -> mod2
            sim_12 = e1 @ e2.T
            metrics_12 = compute_retrieval_metrics(sim_12, k_values)
            for k, v in metrics_12.items():
                all_metrics[f"{mod1}2{mod2}_{k}"] = v

            # mod2 -> mod1
            sim_21 = e2 @ e1.T
            metrics_21 = compute_retrieval_metrics(sim_21, k_values)
            for k, v in metrics_21.items():
                all_metrics[f"{mod2}2{mod1}_{k}"] = v

    return all_metrics


@torch.no_grad()
def extract_embeddings(
    model,
    dataloader,
    device: torch.device,
    modalities: List[str] = ["brep", "point", "text"],
) -> Dict[str, torch.Tensor]:
    """
    Extract embeddings from dataset.

    Args:
        model: CLIP4CAD model
        dataloader: Data loader
        device: Device
        modalities: Which modalities to extract

    Returns:
        Dict of embeddings per modality
    """
    model.eval()

    all_embeddings = {mod: [] for mod in modalities}

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        outputs = model(batch)

        if "brep" in modalities and "brep_global" in outputs:
            all_embeddings["brep"].append(outputs["brep_global"].cpu())

        if "point" in modalities and "point_global" in outputs:
            all_embeddings["point"].append(outputs["point_global"].cpu())

        if "text" in modalities and "text_global" in outputs:
            all_embeddings["text"].append(outputs["text_global"].cpu())

    # Concatenate
    result = {}
    for mod in modalities:
        if all_embeddings[mod]:
            result[mod] = torch.cat(all_embeddings[mod], dim=0)

    return result


def compute_category_retrieval(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute retrieval metrics with category labels.

    Args:
        embeddings: [N, D] embeddings
        labels: [N] category labels
        k_values: K values for Recall@K

    Returns:
        Category-aware retrieval metrics
    """
    N = embeddings.shape[0]
    device = embeddings.device

    # Normalize
    embeddings = F.normalize(embeddings, dim=-1)

    # Similarity matrix
    similarity = embeddings @ embeddings.T

    # Mask out self-similarity
    mask = torch.eye(N, device=device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))

    metrics = {}

    # For each query, check if retrieved items have same label
    sorted_indices = similarity.argsort(dim=1, descending=True)

    for k in k_values:
        correct = 0
        for i in range(N):
            retrieved_labels = labels[sorted_indices[i, :k]]
            if (retrieved_labels == labels[i]).any():
                correct += 1
        metrics[f"cat_R@{k}"] = (correct / N) * 100

    return metrics
