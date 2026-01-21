"""
Hard Negative Mining for CLIP4CAD-GFA

Offline hard negative mining identifies geometrically similar but
semantically different samples for improved contrastive learning.

Following insights from OpenSHAPE (Liu et al., 2023), we:
1. Extract embeddings for all training samples
2. Find k-nearest neighbors by geometric embedding similarity
3. Filter out neighbors with high text similarity (true positives)
4. Remaining neighbors are hard negatives
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract geometric and text embeddings from all samples.

    Args:
        model: CLIP4CAD-GFA model
        dataloader: DataLoader for the training set
        device: Device to run on
        show_progress: Whether to show progress bar

    Returns:
        geo_embeddings: (N, d) geometric embeddings
        text_embeddings: (N, d) text embeddings
        sample_ids: List of sample IDs
    """
    model.eval()

    geo_embeddings = []
    text_embeddings = []
    sample_ids = []

    iterator = tqdm(dataloader, desc="Extracting embeddings") if show_progress else dataloader

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):  # Use autocast for FP16 consistency
        for batch in iterator:
            outputs = model(batch)

            # Get global embeddings
            z_brep = outputs.get("z_brep")
            z_pc = outputs.get("z_pc")
            z_text = outputs.get("z_text")

            # Average geometric embeddings if both available
            if z_brep is not None and z_pc is not None:
                z_geo = (z_brep + z_pc) / 2
            elif z_brep is not None:
                z_geo = z_brep
            else:
                z_geo = z_pc

            # Convert to float32 to avoid dtype issues in downstream operations
            geo_embeddings.append(z_geo.float().cpu().numpy())
            text_embeddings.append(z_text.float().cpu().numpy())
            sample_ids.extend(batch["sample_id"])

    geo_embeddings = np.vstack(geo_embeddings)
    text_embeddings = np.vstack(text_embeddings)

    return geo_embeddings, text_embeddings, sample_ids


def compute_similarity_matrix(
    embeddings: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: (N, d) embeddings
        normalize: Whether to L2 normalize before computing similarity

    Returns:
        similarity: (N, N) similarity matrix
    """
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

    return embeddings @ embeddings.T


def find_knn_indices(
    embeddings: np.ndarray,
    k: int = 20,
    use_faiss: bool = True
) -> np.ndarray:
    """
    Find k nearest neighbors for each sample.

    Args:
        embeddings: (N, d) embeddings
        k: Number of neighbors to find
        use_faiss: Whether to use FAISS (faster for large datasets)

    Returns:
        neighbors: (N, k+1) indices of k nearest neighbors (includes self)
    """
    N, d = embeddings.shape

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings_norm = (embeddings / norms).astype(np.float32)

    if use_faiss:
        try:
            import faiss

            # Build FAISS index for cosine similarity (inner product after normalization)
            index = faiss.IndexFlatIP(d)
            index.add(embeddings_norm)

            # Search
            _, neighbors = index.search(embeddings_norm, k + 1)
            return neighbors

        except ImportError:
            print("FAISS not available, using numpy (slower)")
            use_faiss = False

    if not use_faiss:
        # Brute force with numpy
        similarity = embeddings_norm @ embeddings_norm.T
        neighbors = np.argsort(-similarity, axis=1)[:, :k+1]
        return neighbors


def mine_hard_negatives(
    geo_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    k: int = 20,
    text_sim_threshold: float = 0.8,
    min_negatives: int = 1,
    max_negatives: int = 10,
    use_faiss: bool = True,
    show_progress: bool = True
) -> Dict[int, List[int]]:
    """
    Mine hard negatives: samples that are geometrically similar
    but semantically different.

    Args:
        geo_embeddings: (N, d) geometric embeddings
        text_embeddings: (N, d) text embeddings
        k: Number of nearest neighbors to consider
        text_sim_threshold: Filter neighbors with text similarity above this
        min_negatives: Minimum negatives per sample to include
        max_negatives: Maximum negatives per sample
        use_faiss: Whether to use FAISS for kNN
        show_progress: Whether to show progress bar

    Returns:
        hard_neg_dict: Mapping from sample index to list of hard negative indices
    """
    N = geo_embeddings.shape[0]

    print(f"Mining hard negatives for {N} samples...")

    # Find k nearest neighbors by geometric similarity
    print("  Computing geometric kNN...")
    geo_neighbors = find_knn_indices(geo_embeddings, k, use_faiss)

    # Normalize text embeddings for similarity computation
    text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8
    text_norm = text_embeddings / text_norms

    # Filter based on text similarity
    print("  Filtering by text similarity...")
    hard_neg_dict = {}

    iterator = range(N)
    if show_progress:
        iterator = tqdm(iterator, desc="  Filtering")

    for i in iterator:
        hard_negs = []

        for j in geo_neighbors[i][1:]:  # Skip self (index 0)
            # Compute text similarity
            text_sim = np.dot(text_norm[i], text_norm[j])

            # If text similarity is low, this is a hard negative
            if text_sim < text_sim_threshold:
                hard_negs.append(int(j))

            if len(hard_negs) >= max_negatives:
                break

        if len(hard_negs) >= min_negatives:
            hard_neg_dict[i] = hard_negs

    print(f"  Found hard negatives for {len(hard_neg_dict)} samples")
    print(f"  Average negatives per sample: {np.mean([len(v) for v in hard_neg_dict.values()]):.1f}")

    return hard_neg_dict


def save_hard_negatives(
    hard_neg_dict: Dict[int, List[int]],
    save_path: str,
    sample_ids: Optional[List[str]] = None
):
    """
    Save hard negative dictionary to JSON.

    Args:
        hard_neg_dict: Mapping from sample index to hard negative indices
        save_path: Path to save JSON file
        sample_ids: Optional list of sample IDs (for readability)
    """
    # Convert keys to strings for JSON
    serializable = {str(k): v for k, v in hard_neg_dict.items()}

    save_data = {
        "hard_negatives": serializable,
        "num_samples_with_negatives": len(hard_neg_dict),
        "total_negatives": sum(len(v) for v in hard_neg_dict.values()),
    }

    if sample_ids is not None:
        # Add sample ID mapping
        save_data["sample_ids"] = sample_ids

    with open(save_path, "w") as f:
        json.dump(save_data, f)

    print(f"Saved hard negatives to {save_path}")


def load_hard_negatives(load_path: str) -> Dict[int, List[int]]:
    """
    Load hard negative dictionary from JSON.

    Args:
        load_path: Path to JSON file

    Returns:
        hard_neg_dict: Mapping from sample index to hard negative indices
    """
    with open(load_path, "r") as f:
        data = json.load(f)

    # Convert string keys back to integers
    hard_neg_dict = {int(k): v for k, v in data["hard_negatives"].items()}

    print(f"Loaded hard negatives: {len(hard_neg_dict)} samples")

    return hard_neg_dict


class HardNegativeMiner:
    """
    Manager for hard negative mining in GFA training.

    Handles:
    - Initial mining after stage 1
    - Optional periodic re-mining during stage 2
    - Caching and loading of mined negatives
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        cache_dir: str,
        k: int = 20,
        text_sim_threshold: float = 0.8,
        min_negatives: int = 1,
        max_negatives: int = 10,
        use_faiss: bool = True,
        device: str = "cuda",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.k = k
        self.text_sim_threshold = text_sim_threshold
        self.min_negatives = min_negatives
        self.max_negatives = max_negatives
        self.use_faiss = use_faiss
        self.device = device

        self.hard_neg_dict = None
        self.sample_ids = None

    def mine(self, epoch: Optional[int] = None) -> Dict[int, List[int]]:
        """
        Perform hard negative mining.

        Args:
            epoch: Optional epoch number for cache naming

        Returns:
            hard_neg_dict: Mined hard negatives
        """
        print("\n" + "=" * 60)
        print("Mining Hard Negatives")
        print("=" * 60)

        # Extract embeddings
        geo_embeddings, text_embeddings, self.sample_ids = extract_embeddings(
            self.model,
            self.train_dataloader,
            device=self.device,
        )

        # Mine hard negatives
        self.hard_neg_dict = mine_hard_negatives(
            geo_embeddings,
            text_embeddings,
            k=self.k,
            text_sim_threshold=self.text_sim_threshold,
            min_negatives=self.min_negatives,
            max_negatives=self.max_negatives,
            use_faiss=self.use_faiss,
        )

        # Save to cache
        if epoch is not None:
            cache_path = self.cache_dir / f"hard_negatives_epoch{epoch}.json"
        else:
            cache_path = self.cache_dir / "hard_negatives.json"

        save_hard_negatives(self.hard_neg_dict, str(cache_path), self.sample_ids)

        print("=" * 60 + "\n")

        return self.hard_neg_dict

    def load_cached(self, epoch: Optional[int] = None) -> Optional[Dict[int, List[int]]]:
        """
        Load cached hard negatives if available.

        Args:
            epoch: Optional epoch number

        Returns:
            hard_neg_dict or None if not cached
        """
        if epoch is not None:
            cache_path = self.cache_dir / f"hard_negatives_epoch{epoch}.json"
        else:
            cache_path = self.cache_dir / "hard_negatives.json"

        if cache_path.exists():
            self.hard_neg_dict = load_hard_negatives(str(cache_path))
            return self.hard_neg_dict

        return None

    def get_hard_negatives(self) -> Optional[Dict[int, List[int]]]:
        """Get current hard negatives."""
        return self.hard_neg_dict


def construct_hard_negative_batch(
    all_indices: np.ndarray,
    hard_neg_dict: Dict[int, List[int]],
    batch_size: int = 64,
    num_seeds: int = 16,
    negs_per_seed: int = 3
) -> List[int]:
    """
    Construct a training batch that includes hard negatives.

    Strategy:
    1. Sample seed indices uniformly
    2. For each seed, add some of its hard negatives
    3. Fill remaining slots with random samples

    Args:
        all_indices: Array of all sample indices
        hard_neg_dict: Mapping from sample index to hard negative indices
        batch_size: Target batch size
        num_seeds: Number of seed samples
        negs_per_seed: Hard negatives to add per seed

    Returns:
        batch_indices: List of sample indices for the batch
    """
    seeds = np.random.choice(all_indices, num_seeds, replace=False).tolist()
    batch_indices = seeds.copy()

    for seed in seeds:
        if seed in hard_neg_dict:
            available = [n for n in hard_neg_dict[seed] if n not in batch_indices]
            if available:
                num_add = min(negs_per_seed, len(available))
                negs = np.random.choice(available, num_add, replace=False)
                batch_indices.extend(negs.tolist())

    # Fill remaining
    remaining = batch_size - len(batch_indices)
    if remaining > 0:
        available = [i for i in all_indices if i not in batch_indices]
        if len(available) >= remaining:
            fill = np.random.choice(available, remaining, replace=False)
        else:
            fill = np.random.choice(all_indices, remaining, replace=True)
        batch_indices.extend(fill.tolist())

    return batch_indices[:batch_size]
