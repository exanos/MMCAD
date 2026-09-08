#!/usr/bin/env python3
"""
=============================================================================
Recursive DPGMM Application Tree Builder
MM-CAD:B Dataset (192k CAD models, ~76k unique application keywords)

Method: Bayesian Nonparametric Hierarchical Clustering
  1. Sentence-transformer embeddings (semantic)
  2. Co-occurrence PPMI embeddings (distributional)
  3. Weighted fusion + UMAP dimensionality reduction
  4. Recursive Dirichlet Process Gaussian Mixture Model
     - Variable branching factor at each node (data-driven)
     - BIC-based split validation
     - Automatic depth/granularity control

Designed for Google Colab with T4/A100/RTX Pro 6000 GPU.
=============================================================================
"""

# %% [markdown]
# # Setup

# %%
# === CELL 1: Install dependencies ===
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'umap-learn', 'sentence-transformers', 'tqdm'])

# %%
# === CELL 2: Mount Google Drive ===
import os
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = "/content/drive/MyDrive/MMCAD"
os.makedirs(DRIVE_DIR, exist_ok=True)
print(f"Drive mounted. Working dir: {DRIVE_DIR}")

# %%
# === CELL 3: Imports + CUDA setup ===
import numpy as np
import pandas as pd
import json
import time
import gc
import os
import logging
import torch
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any

from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import svds
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import umap
from tqdm import tqdm

# --- CUDA hardening ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.zeros(1, device=device)
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    device = torch.device('cpu')
    print("WARNING: No GPU detected, falling back to CPU")


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# %%
# === CELL 4: Configuration ===

CONFIG = {
    # --- Paths (all on Google Drive) ---
    'dataset_csv': f'{DRIVE_DIR}/abc_dataset_clean.csv',
    'keyword_metadata_json': f'{DRIVE_DIR}/keyword_metadata.json',
    'embeddings_npy': f'{DRIVE_DIR}/embeddings.npy',
    'output_tree_json': f'{DRIVE_DIR}/application_tree_dpgmm.json',
    'output_stats_json': f'{DRIVE_DIR}/tree_statistics.json',

    # --- Embedding computation ---
    'embedding_model': 'BAAI/bge-large-en-v1.5',
    'embedding_dim': 1024,
    'embedding_batch_size': 256,
    'recompute_embeddings': True,

    # --- Co-occurrence embeddings ---
    'cooccurrence_svd_dims': 128,
    'cooccurrence_min_count': 2,
    'ppmi_shift': 1.0,
    'semantic_weight': 0.7,
    'cooccurrence_weight': 0.3,

    # --- UMAP ---
    'umap_n_components': 50,
    'umap_n_neighbors': 30,
    'umap_min_dist': 0.0,
    'umap_metric': 'cosine',

    # --- Recursive DPGMM ---
    'dpgmm_covariance_type': 'diag',
    'dpgmm_max_iter': 500,
    'dpgmm_n_init': 3,
    'dpgmm_weight_threshold': 0.01,
    'dpgmm_max_comp_schedule': [20, 30, 40, 40, 30, 25, 20, 15],
    'dpgmm_alpha_schedule':    [0.01, 0.05, 0.1, 0.2, 0.5, 0.5, 1.0, 1.0],
    'max_children_per_node': 25,

    # --- Tree building ---
    'min_split_size': 12,
    'max_depth': 7,
    'bic_improvement_threshold': 0.005,
    'min_component_size': 3,

    # --- LLM labeling ---
    'openrouter_api_key': os.environ.get('OPENROUTER_API_KEY', ''),  # export OPENROUTER_API_KEY before running
    'llm_model': 'google/gemini-2.5-flash',
    'llm_temperature': 0.3,
    'llm_max_tokens': 30,
    'llm_parallel_workers': 16,

    # --- Reproducibility ---
    'random_state': 42,
}

# %% [markdown]
# # Step 1: Load Data

# %%
# === CELL 5: Load keyword metadata and dataset ===

def load_keyword_metadata(path: str) -> Tuple[List[str], Dict[str, int]]:
    with open(path, 'r') as f:
        meta = json.load(f)
    keywords = meta['keywords']
    if isinstance(meta['keyword_counts'], dict):
        freq = meta['keyword_counts']
    else:
        freq = dict(zip(keywords, meta['keyword_counts']))
    log.info(f"Loaded {len(keywords)} keywords, {meta.get('total_instances', sum(freq.values()))} total instances")
    return keywords, freq


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=['uid', 'title', 'applications'])
    log.info(f"Loaded dataset: {len(df)} models")
    return df


def parse_applications(apps_str: str) -> List[str]:
    if pd.isna(apps_str):
        return []
    parts = apps_str.split('", "')
    cleaned = []
    for p in parts:
        p = p.strip().strip('"').strip()
        if p:
            cleaned.append(p)
    return cleaned


keywords, keyword_freq = load_keyword_metadata(CONFIG['keyword_metadata_json'])
keyword_to_idx = {kw: i for i, kw in enumerate(keywords)}
n_keywords = len(keywords)

dataset = load_dataset(CONFIG['dataset_csv'])

sample_apps = parse_applications(dataset.iloc[0]['applications'])
log.info(f"Sample parsed applications: {sample_apps}")

# %% [markdown]
# # Step 2: Compute / Load Embeddings

# %%
# === CELL 6: Embeddings ===

def load_or_compute_embeddings(
    keywords: List[str],
    embeddings_path: str,
    model_name: str,
    batch_size: int,
    recompute: bool = False
) -> np.ndarray:
    if not recompute and os.path.exists(embeddings_path):
        try:
            existing = np.load(embeddings_path)
            if existing.shape[0] >= len(keywords):
                log.info(f"Loaded cached embeddings: {existing.shape}")
                return existing[:len(keywords)]
            else:
                log.warning(f"Cached embeddings have {existing.shape[0]} rows, "
                            f"need {len(keywords)}. Recomputing all.")
        except Exception as e:
            log.warning(f"Failed to load cached embeddings: {e}. Recomputing.")

    log.info(f"Computing embeddings for {len(keywords)} keywords using {model_name}")
    from sentence_transformers import SentenceTransformer

    cleanup_memory()

    st_model = SentenceTransformer(model_name, device='cpu')
    st_model = st_model.to(torch.float32)
    if torch.cuda.is_available():
        st_model = st_model.to('cuda')
    log.info(f"Embedding model on: {next(st_model.parameters()).device}")

    embeddings = st_model.encode(
        keywords,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    del st_model
    cleanup_memory()

    output_path = embeddings_path.replace('.npy', f'_{model_name.split("/")[-1]}.npy')
    np.save(output_path, embeddings)
    log.info(f"Saved embeddings to {output_path}")
    log.info(f"Final embeddings shape: {embeddings.shape}")
    return embeddings


embeddings_semantic = load_or_compute_embeddings(
    keywords,
    CONFIG['embeddings_npy'],
    CONFIG['embedding_model'],
    CONFIG['embedding_batch_size'],
    CONFIG['recompute_embeddings']
)

norms = np.linalg.norm(embeddings_semantic, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-10)
embeddings_semantic = embeddings_semantic / norms

# %% [markdown]
# # Step 3: Co-occurrence PPMI Embeddings

# %%
# === CELL 7: Build co-occurrence matrix ===

def build_cooccurrence_matrix(
    dataset: pd.DataFrame,
    keyword_to_idx: Dict[str, int],
    n_keywords: int,
    min_count: int = 2
) -> csr_matrix:
    log.info("Building co-occurrence matrix...")
    rows, cols, data = [], [], []
    n_models = len(dataset)

    for _, row in tqdm(dataset.iterrows(), total=n_models, desc="Co-occurrence"):
        apps = parse_applications(row['applications'])
        indices = []
        for app in apps:
            if app in keyword_to_idx:
                indices.append(keyword_to_idx[app])

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                rows.append(indices[i])
                cols.append(indices[j])
                data.append(1.0)
                rows.append(indices[j])
                cols.append(indices[i])
                data.append(1.0)

    cooc = coo_matrix(
        (data, (rows, cols)),
        shape=(n_keywords, n_keywords),
        dtype=np.float32
    ).tocsr()

    log.info(f"Co-occurrence matrix: {cooc.nnz} nonzero entries")

    if min_count > 1:
        cooc.data[cooc.data < min_count] = 0
        cooc.eliminate_zeros()
        log.info(f"After min_count={min_count} filter: {cooc.nnz} nonzero entries")

    return cooc


def compute_ppmi_matrix(
    cooc: csr_matrix,
    keyword_freq: Dict[str, int],
    keywords: List[str],
    shift: float = 1.0
) -> csr_matrix:
    log.info("Computing shifted PPMI matrix...")
    total_cooc = cooc.sum()

    freq_array = np.array([keyword_freq.get(kw, 1) for kw in keywords], dtype=np.float64)
    total_freq = freq_array.sum()
    p_marginal = freq_array / total_freq

    cooc_coo = cooc.tocoo()
    rows, cols, vals = cooc_coo.row, cooc_coo.col, cooc_coo.data.astype(np.float64)

    p_joint = vals / total_cooc
    p_marginal_product = p_marginal[rows] * p_marginal[cols]
    pmi_vals = np.log2(p_joint / (p_marginal_product + 1e-15))
    ppmi_vals = np.maximum(0, pmi_vals - shift).astype(np.float32)

    nonzero_mask = ppmi_vals > 0
    ppmi = csr_matrix(
        (ppmi_vals[nonzero_mask], (rows[nonzero_mask], cols[nonzero_mask])),
        shape=cooc.shape
    )
    log.info(f"PPMI matrix: {ppmi.nnz} nonzero entries, "
             f"density={ppmi.nnz / (ppmi.shape[0]**2):.6f}")
    return ppmi


def embed_ppmi(ppmi: csr_matrix, n_dims: int) -> np.ndarray:
    log.info(f"Computing SVD of PPMI matrix (k={n_dims})...")
    actual_dims = min(n_dims, min(ppmi.shape) - 1, ppmi.nnz - 1)
    if actual_dims < n_dims:
        log.warning(f"Reducing SVD dims from {n_dims} to {actual_dims}")

    U, S, Vt = svds(ppmi.astype(np.float64), k=actual_dims)

    order = np.argsort(-S)
    U = U[:, order]
    S = S[order]

    embeddings = U * np.sqrt(S)[np.newaxis, :]
    log.info(f"Co-occurrence embeddings: {embeddings.shape}, "
             f"top singular values: {S[:5]}")
    return embeddings.astype(np.float32)


cooc_matrix = build_cooccurrence_matrix(
    dataset, keyword_to_idx, n_keywords, CONFIG['cooccurrence_min_count']
)
ppmi_matrix = compute_ppmi_matrix(
    cooc_matrix, keyword_freq, keywords, CONFIG['ppmi_shift']
)
embeddings_cooccurrence = embed_ppmi(ppmi_matrix, CONFIG['cooccurrence_svd_dims'])

del cooc_matrix, ppmi_matrix
cleanup_memory()

# %% [markdown]
# # Step 4: Embedding Fusion + UMAP

# %%
# === CELL 8: Fuse embeddings and reduce dimensionality ===

def fuse_embeddings(
    semantic: np.ndarray,
    cooccurrence: np.ndarray,
    w_semantic: float,
    w_cooccurrence: float
) -> np.ndarray:
    log.info(f"Fusing embeddings: semantic {semantic.shape} (w={w_semantic}) + "
             f"cooccurrence {cooccurrence.shape} (w={w_cooccurrence})")

    sem_norm = normalize(semantic, norm='l2', axis=1)
    cooc_norm = normalize(cooccurrence, norm='l2', axis=1)

    fused = np.hstack([
        sem_norm * w_semantic,
        cooc_norm * w_cooccurrence
    ])

    log.info(f"Fused embedding shape: {fused.shape}")
    return fused


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int
) -> np.ndarray:
    log.info(f"UMAP reduction: {embeddings.shape[1]}d -> {n_components}d "
             f"(n_neighbors={n_neighbors}, metric={metric})")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=True,
        verbose=True
    )
    reduced = reducer.fit_transform(embeddings)
    log.info(f"UMAP output shape: {reduced.shape}")
    return reduced


fused_embeddings = fuse_embeddings(
    embeddings_semantic,
    embeddings_cooccurrence,
    CONFIG['semantic_weight'],
    CONFIG['cooccurrence_weight']
)

embeddings_reduced = reduce_umap(
    fused_embeddings,
    CONFIG['umap_n_components'],
    CONFIG['umap_n_neighbors'],
    CONFIG['umap_min_dist'],
    CONFIG['umap_metric'],
    CONFIG['random_state']
)

del fused_embeddings, embeddings_semantic, embeddings_cooccurrence
cleanup_memory()

# %% [markdown]
# # Step 5: Recursive DPGMM Tree Construction

# %%
# === CELL 9: Core tree builder ===

def agglomerative_regroup(
    embeddings: np.ndarray,
    child_clusters: List[np.ndarray],
    max_children: int
) -> List[np.ndarray]:
    from sklearn.metrics.pairwise import euclidean_distances

    clusters = list(child_clusters)
    while len(clusters) > max_children:
        centroids = np.array([embeddings[c].mean(axis=0) for c in clusters])
        dists = euclidean_distances(centroids)
        np.fill_diagonal(dists, np.inf)
        i, j = np.unravel_index(dists.argmin(), dists.shape)
        if i > j:
            i, j = j, i
        clusters[i] = np.concatenate([clusters[i], clusters[j]])
        clusters.pop(j)

    return clusters


class TreeNode:
    __slots__ = ['name', 'keyword_indices', 'children', 'depth',
                 'n_keywords', 'total_freq', 'metadata']

    def __init__(self, keyword_indices: np.ndarray, depth: int):
        self.keyword_indices = keyword_indices
        self.depth = depth
        self.children: List[TreeNode] = []
        self.n_keywords = len(keyword_indices)
        self.total_freq = 0
        self.name = ''
        self.metadata: Dict[str, Any] = {}


def compute_bic_single_gaussian(X: np.ndarray, cov_type: str = 'diag') -> float:
    n, d = X.shape
    if n <= d + 1:
        return np.inf
    gm = GaussianMixture(n_components=1, covariance_type=cov_type, random_state=42)
    gm.fit(X)
    return gm.bic(X)


def fit_dpgmm(
    X: np.ndarray,
    max_components: int,
    concentration: float,
    cov_type: str,
    max_iter: int,
    n_init: int,
    weight_threshold: float,
    random_state: int,
    depth: int = 0,
    config: Optional[Dict] = None
) -> Tuple[Optional[np.ndarray], int, Dict]:
    n_samples, n_features = X.shape

    if config is not None:
        schedule_mc = config.get('dpgmm_max_comp_schedule', [30, 40, 50, 40, 30])
        schedule_alpha = config.get('dpgmm_alpha_schedule', [0.01, 0.05, 0.1, 0.5, 1.0])
        idx = min(depth, len(schedule_mc) - 1)
        max_components = schedule_mc[idx]
        concentration = schedule_alpha[idx]

    effective_max = min(max_components, n_samples // 3)
    if effective_max < 2:
        return None, 0, {'reason': 'too_few_samples'}

    dpgmm = BayesianGaussianMixture(
        n_components=effective_max,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=concentration,
        covariance_type=cov_type,
        max_iter=max_iter,
        n_init=n_init,
        reg_covar=1e-4,
        random_state=random_state,
        warm_start=False,
        verbose=0
    )

    try:
        dpgmm.fit(X)
    except Exception as e:
        return None, 0, {'reason': f'fit_failed: {e}'}

    weights = dpgmm.weights_
    active_mask = weights > weight_threshold
    n_active = active_mask.sum()

    if n_active < 2:
        return None, n_active, {
            'reason': 'single_component',
            'weights': weights[active_mask].tolist(),
            'converged': dpgmm.converged_,
            'n_iter': dpgmm.n_iter_
        }

    labels = dpgmm.predict(X)

    active_components = np.where(active_mask)[0]
    label_map = {old: new for new, old in enumerate(active_components)}

    remapped = np.full_like(labels, -1)
    for old_label, new_label in label_map.items():
        remapped[labels == old_label] = new_label

    orphan_mask = remapped == -1
    if orphan_mask.any():
        active_means = dpgmm.means_[active_components]
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(X[orphan_mask], active_means)
        remapped[orphan_mask] = dists.argmin(axis=1)

    info = {
        'n_active': int(n_active),
        'weights': weights[active_mask].tolist(),
        'converged': bool(dpgmm.converged_),
        'n_iter': int(dpgmm.n_iter_),
        'cov_type': cov_type,
        'concentration': concentration,
        'max_components_used': effective_max,
        'bic_mixture': float(dpgmm.lower_bound_)
    }

    return remapped, n_active, info


def should_split(
    X: np.ndarray,
    n_active: int,
    dpgmm_info: Dict,
    min_split_size: int,
    bic_threshold: float
) -> bool:
    n = X.shape[0]
    if n < min_split_size:
        return False
    if n_active < 2:
        return False

    cov_type = dpgmm_info.get('cov_type', 'diag')
    bic_single = compute_bic_single_gaussian(X, cov_type)

    try:
        gmm = GaussianMixture(
            n_components=n_active,
            covariance_type=cov_type,
            reg_covar=1e-4,
            random_state=42,
            max_iter=100
        )
        gmm.fit(X)
        bic_mixture = gmm.bic(X)
    except Exception:
        return True

    improvement = (bic_single - bic_mixture) / abs(bic_single)
    return improvement > bic_threshold


def build_tree_recursive(
    embeddings: np.ndarray,
    all_indices: np.ndarray,
    keywords: List[str],
    keyword_freq: Dict[str, int],
    config: Dict,
    depth: int = 0,
    pbar: Optional[tqdm] = None
) -> TreeNode:
    node = TreeNode(all_indices, depth)
    node.total_freq = sum(keyword_freq.get(keywords[i], 0) for i in all_indices)
    n = len(all_indices)

    if pbar:
        pbar.set_postfix(depth=depth, n=n, refresh=True)

    if n <= config['min_split_size']:
        node.metadata['leaf_reason'] = 'below_min_split_size'
        if pbar:
            pbar.update(n)
        return node

    if depth >= config['max_depth']:
        node.metadata['leaf_reason'] = 'max_depth_reached'
        if pbar:
            pbar.update(n)
        return node

    X = embeddings[all_indices]

    labels, n_active, dpgmm_info = fit_dpgmm(
        X,
        config['dpgmm_max_comp_schedule'][0],
        config['dpgmm_alpha_schedule'][0],
        config['dpgmm_covariance_type'],
        config['dpgmm_max_iter'],
        config['dpgmm_n_init'],
        config['dpgmm_weight_threshold'],
        config['random_state'],
        depth=depth,
        config=config
    )

    node.metadata['dpgmm_info'] = dpgmm_info

    if labels is None or n_active < 2:
        node.metadata['leaf_reason'] = dpgmm_info.get('reason', 'no_split')
        if pbar:
            pbar.update(n)
        return node

    if not should_split(X, n_active, dpgmm_info, config['min_split_size'],
                        config['bic_improvement_threshold']):
        node.metadata['leaf_reason'] = 'bic_no_improvement'
        if pbar:
            pbar.update(n)
        return node

    unique_labels = np.unique(labels)
    child_clusters = []

    for lbl in sorted(unique_labels):
        mask = labels == lbl
        child_indices = all_indices[mask]
        if len(child_indices) >= config['min_component_size']:
            child_clusters.append(child_indices)

    tiny_indices = []
    for lbl in sorted(unique_labels):
        mask = labels == lbl
        child_indices = all_indices[mask]
        if len(child_indices) < config['min_component_size']:
            tiny_indices.extend(child_indices.tolist())

    if tiny_indices and child_clusters:
        tiny_emb = embeddings[tiny_indices]
        centroids = np.array([embeddings[c].mean(axis=0) for c in child_clusters])
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(tiny_emb, centroids)
        nearest = dists.argmin(axis=1)
        for idx, target in zip(tiny_indices, nearest):
            child_clusters[target] = np.append(child_clusters[target], idx)

    if len(child_clusters) < 2:
        node.metadata['leaf_reason'] = 'insufficient_children_after_merge'
        if pbar:
            pbar.update(n)
        return node

    max_children = config.get('max_children_per_node')
    if max_children and len(child_clusters) > max_children:
        child_clusters = agglomerative_regroup(
            embeddings, child_clusters, max_children
        )
        node.metadata['regrouped_from'] = node.metadata.get('n_children', len(child_clusters))

    node.metadata['n_children'] = len(child_clusters)

    for child_indices in child_clusters:
        child_node = build_tree_recursive(
            embeddings, child_indices, keywords, keyword_freq,
            config, depth + 1, pbar
        )
        node.children.append(child_node)

    node.children.sort(key=lambda c: c.total_freq, reverse=True)

    return node


def build_application_tree(
    embeddings: np.ndarray,
    keywords: List[str],
    keyword_freq: Dict[str, int],
    config: Dict
) -> TreeNode:
    log.info(f"Building tree: {len(keywords)} keywords, max_depth={config['max_depth']}")
    all_indices = np.arange(len(keywords))

    with tqdm(total=len(keywords), desc="Tree construction") as pbar:
        root = build_tree_recursive(
            embeddings, all_indices, keywords, keyword_freq,
            config, depth=0, pbar=pbar
        )

    return root


log.info("=" * 60)
log.info("STARTING TREE CONSTRUCTION")
log.info("=" * 60)

t0 = time.time()
root = build_application_tree(embeddings_reduced, keywords, keyword_freq, CONFIG)
elapsed = time.time() - t0
log.info(f"Tree construction completed in {elapsed:.1f}s")

# %% [markdown]
# # Step 6: Parallel LLM Labeling via OpenRouter

# %%
# === CELL 10: Parallel LLM node labeling ===
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

_rate_lock = threading.Lock()
_last_call_time = [0.0]


def get_all_leaf_indices(node):
    if not node.children:
        return node.keyword_indices.tolist()
    result = []
    for child in node.children:
        result.extend(get_all_leaf_indices(child))
    return result


def llm_label_single(keyword_samples: list, child_labels: list, depth: int,
                     config: Dict, max_retries: int = 3) -> Optional[str]:
    prompt = (
        "You are labeling nodes in a hierarchical taxonomy of CAD model applications.\n"
        f"This node is at depth {depth} in the tree.\n"
    )
    if child_labels:
        prompt += f"Its children are labeled: {', '.join(child_labels[:15])}\n"
    prompt += (
        f"The top keywords in this subtree (by frequency): {', '.join(keyword_samples[:20])}\n\n"
        "Return ONLY a short category name (2-5 words, no quotes, no explanation). "
        "It should be a precise engineering/manufacturing domain label, not generic."
    )

    payload = {
        "model": config['llm_model'],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config['llm_temperature'],
        "max_tokens": config['llm_max_tokens'],
    }
    headers = {
        "Authorization": f"Bearer {config['openrouter_api_key']}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                name = r.json()['choices'][0]['message']['content'].strip()
                name = name.replace('**', '').replace('*', '').replace(':', '')
                name = name.strip('"\'.,: ')
                if '\n' in name:
                    name = name.split('\n')[0].strip()
                return name[:60]
            elif r.status_code == 429:
                time.sleep(3 * (attempt + 1))
            else:
                log.warning(f"LLM error {r.status_code}: {r.text[:200]}")
                break
        except Exception as e:
            log.warning(f"LLM request failed: {e}")
            time.sleep(2)
    return None


def heuristic_label(kw_sorted):
    if len(kw_sorted) <= 5:
        return ' / '.join(kw for kw, _ in kw_sorted)
    else:
        return f"{' / '.join(kw for kw, _ in kw_sorted[:3])} (+{len(kw_sorted)-3} more)"


def label_tree_parallel(root, keywords, keyword_freq, config):
    """
    Two-pass labeling:
      Pass 1: Heuristic labels for all nodes (instant, no API calls).
      Pass 2: Parallel LLM relabeling for internal nodes only.
    Bottom-up order so child labels are available as LLM context.
    """
    # --- Pass 1: heuristic labels everywhere ---
    def heuristic_pass(node):
        for child in node.children:
            heuristic_pass(child)
        kw_sorted = sorted(
            [(keywords[i], keyword_freq.get(keywords[i], 0))
             for i in get_all_leaf_indices(node)],
            key=lambda x: x[1], reverse=True
        )
        if not node.children:
            node.name = heuristic_label(kw_sorted)
        else:
            node.name = ' / '.join(kw for kw, _ in kw_sorted[:3])
        node.metadata['_kw_sorted'] = kw_sorted  # stash for pass 2

    heuristic_pass(root)
    log.info("Pass 1 (heuristic labels) complete.")

    # --- Pass 2: collect internal nodes bottom-up, LLM label in parallel ---
    internal_nodes = []

    def collect_bottom_up(node, depth_order):
        for child in node.children:
            collect_bottom_up(child, depth_order)
        if node.children:
            depth_order.append(node)

    collect_bottom_up(root, internal_nodes)
    # internal_nodes is now in bottom-up order (deepest first)

    log.info(f"Pass 2: LLM labeling {len(internal_nodes)} internal nodes "
             f"with {config['llm_parallel_workers']} workers...")

    # Group by depth so we process each depth level fully before moving up.
    # Within a depth level, all calls are independent and can be parallel.
    from itertools import groupby

    depth_groups = defaultdict(list)
    for node in internal_nodes:
        depth_groups[node.depth].append(node)

    labeled = 0
    failed = 0

    for depth in sorted(depth_groups.keys(), reverse=True):
        nodes_at_depth = depth_groups[depth]

        def label_one_node(node):
            kw_sorted = node.metadata.pop('_kw_sorted', [])
            top_kws = [kw for kw, _ in kw_sorted[:20]]
            child_labels = [c.name for c in node.children if c.name]
            return node, llm_label_single(top_kws, child_labels, node.depth, config)

        with ThreadPoolExecutor(max_workers=config['llm_parallel_workers']) as pool:
            futures = {pool.submit(label_one_node, n): n for n in nodes_at_depth}
            for future in as_completed(futures):
                node, llm_name = future.result()
                if llm_name:
                    node.name = llm_name
                    labeled += 1
                else:
                    failed += 1

        log.info(f"  Depth {depth}: {len(nodes_at_depth)} nodes done")

    # Clean up stashed metadata from leaves
    def cleanup_meta(node):
        node.metadata.pop('_kw_sorted', None)
        for child in node.children:
            cleanup_meta(child)

    cleanup_meta(root)

    log.info(f"LLM labeling complete: {labeled} labeled, {failed} fell back to heuristic")


label_tree_parallel(root, keywords, keyword_freq, CONFIG)

# %% [markdown]
# # Step 7: Tree Statistics and Validation

# %%
# === CELL 11: Compute tree statistics ===

def compute_tree_stats(node: TreeNode, stats: Optional[Dict] = None) -> Dict:
    if stats is None:
        stats = {
            'total_nodes': 0,
            'internal_nodes': 0,
            'leaf_nodes': 0,
            'max_depth': 0,
            'branching_factors': [],
            'leaf_sizes': [],
            'depth_distribution': defaultdict(int),
            'leaf_reasons': Counter(),
            'keywords_per_depth': defaultdict(int),
        }

    stats['total_nodes'] += 1
    stats['max_depth'] = max(stats['max_depth'], node.depth)
    stats['depth_distribution'][node.depth] += 1

    if node.children:
        stats['internal_nodes'] += 1
        stats['branching_factors'].append(len(node.children))
        for child in node.children:
            compute_tree_stats(child, stats)
    else:
        stats['leaf_nodes'] += 1
        stats['leaf_sizes'].append(node.n_keywords)
        stats['keywords_per_depth'][node.depth] += node.n_keywords
        reason = node.metadata.get('leaf_reason', 'unknown')
        stats['leaf_reasons'][reason] += 1

    return stats


tree_stats = compute_tree_stats(root)

bf = np.array(tree_stats['branching_factors'])
ls = np.array(tree_stats['leaf_sizes'])

summary = {
    'total_nodes': tree_stats['total_nodes'],
    'internal_nodes': tree_stats['internal_nodes'],
    'leaf_nodes': tree_stats['leaf_nodes'],
    'max_depth': tree_stats['max_depth'],
    'branching_factor': {
        'mean': float(bf.mean()) if len(bf) > 0 else 0,
        'median': float(np.median(bf)) if len(bf) > 0 else 0,
        'min': int(bf.min()) if len(bf) > 0 else 0,
        'max': int(bf.max()) if len(bf) > 0 else 0,
        'std': float(bf.std()) if len(bf) > 0 else 0,
    },
    'leaf_sizes': {
        'mean': float(ls.mean()) if len(ls) > 0 else 0,
        'median': float(np.median(ls)) if len(ls) > 0 else 0,
        'min': int(ls.min()) if len(ls) > 0 else 0,
        'max': int(ls.max()) if len(ls) > 0 else 0,
    },
    'depth_distribution': dict(tree_stats['depth_distribution']),
    'leaf_reasons': dict(tree_stats['leaf_reasons']),
    'construction_time_seconds': elapsed,
}

print("\n" + "=" * 60)
print("TREE STATISTICS")
print("=" * 60)
for key, val in summary.items():
    if isinstance(val, dict):
        print(f"\n{key}:")
        for k, v in val.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {val}")

# %% [markdown]
# # Step 8: Export

# %%
# === CELL 12: Export tree to JSON ===

def tree_to_dict(node: TreeNode, keywords: List[str],
                 keyword_freq: Dict[str, int]) -> Dict:
    result = {
        'name': node.name,
        'depth': node.depth,
        'n_keywords': node.n_keywords,
        'total_frequency': node.total_freq,
        'type': 'internal' if node.children else 'leaf',
    }

    if node.children:
        result['n_children'] = len(node.children)
        result['children'] = [
            tree_to_dict(child, keywords, keyword_freq)
            for child in node.children
        ]
    else:
        kw_list = []
        for idx in node.keyword_indices:
            kw = keywords[idx]
            kw_list.append({
                'keyword': kw,
                'frequency': keyword_freq.get(kw, 0),
                'index': int(idx)
            })
        kw_list.sort(key=lambda x: x['frequency'], reverse=True)
        result['keywords'] = kw_list
        if 'leaf_reason' in node.metadata:
            result['leaf_reason'] = node.metadata['leaf_reason']

    return result


tree_dict = tree_to_dict(root, keywords, keyword_freq)

with open(CONFIG['output_tree_json'], 'w') as f:
    json.dump(tree_dict, f, indent=2)
log.info(f"Tree saved to {CONFIG['output_tree_json']}")

with open(CONFIG['output_stats_json'], 'w') as f:
    json.dump(summary, f, indent=2)
log.info(f"Statistics saved to {CONFIG['output_stats_json']}")

# %% [markdown]
# # Step 9: Validation

# %%
# === CELL 13: Clustering quality metrics ===

def validate_tree(
    embeddings: np.ndarray,
    root: TreeNode,
    keywords: List[str],
    sample_size: int = 10000
) -> Dict:
    log.info("Computing validation metrics...")
    metrics = {}

    for target_depth in range(1, root.depth + 1):
        clusters = collect_clusters_at_depth(root, target_depth)
        if len(clusters) < 2:
            continue

        all_indices = []
        all_labels = []
        for cluster_id, node in enumerate(clusters):
            indices = get_all_leaf_indices(node)
            all_indices.extend(indices)
            all_labels.extend([cluster_id] * len(indices))

        all_indices = np.array(all_indices)
        all_labels = np.array(all_labels)

        if len(all_indices) > sample_size:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(len(all_indices), sample_size, replace=False)
            sample_emb = embeddings[all_indices[sample_idx]]
            sample_labels = all_labels[sample_idx]
        else:
            sample_emb = embeddings[all_indices]
            sample_labels = all_labels

        n_unique = len(np.unique(sample_labels))
        if n_unique < 2:
            continue

        try:
            sil = silhouette_score(sample_emb, sample_labels, sample_size=min(5000, len(sample_labels)))
            ch = calinski_harabasz_score(sample_emb, sample_labels)
            metrics[f'depth_{target_depth}'] = {
                'n_clusters': len(clusters),
                'silhouette': round(float(sil), 4),
                'calinski_harabasz': round(float(ch), 2),
                'n_samples': len(all_indices)
            }
            log.info(f"Depth {target_depth}: {len(clusters)} clusters, "
                     f"silhouette={sil:.4f}, CH={ch:.1f}")
        except Exception as e:
            log.warning(f"Metrics failed at depth {target_depth}: {e}")

    return metrics


def collect_clusters_at_depth(node: TreeNode, target_depth: int) -> List[TreeNode]:
    if node.depth == target_depth:
        return [node]
    if node.depth > target_depth:
        return []
    result = []
    for child in node.children:
        result.extend(collect_clusters_at_depth(child, target_depth))
    if not node.children and node.depth < target_depth:
        result.append(node)
    return result


validation_metrics = validate_tree(embeddings_reduced, root, keywords)

print("\n" + "=" * 60)
print("VALIDATION METRICS")
print("=" * 60)
for depth, m in sorted(validation_metrics.items()):
    print(f"{depth}: {m}")

# %% [markdown]
# # Step 10: Visualization (Top Levels)

# %%
# === CELL 14: Print tree summary ===

def print_tree(node: TreeNode, keywords: List[str],
               keyword_freq: Dict[str, int],
               max_depth: int = 3, indent: int = 0):
    prefix = "  " * indent
    freq_str = f"[{node.total_freq:,} instances, {node.n_keywords} keywords]"

    if node.children:
        n_children = len(node.children)
        print(f"{prefix}+ {node.name} {freq_str} ({n_children} children)")
        if indent < max_depth:
            for child in node.children:
                print_tree(child, keywords, keyword_freq, max_depth, indent + 1)
        else:
            print(f"{prefix}  ...")
    else:
        print(f"{prefix}- {node.name} {freq_str}")


print("\n" + "=" * 60)
print("APPLICATION TREE (Top 3 Levels)")
print("=" * 60)
print_tree(root, keywords, keyword_freq, max_depth=3)


# %%
# === CELL 15: Generate HTML visualization (optional) ===

def generate_d3_visualization(tree_dict: Dict, output_path: str = 'tree_visualization.html'):
    truncated = truncate_tree_for_viz(tree_dict, max_nodes=500)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MM-CAD Application Tree (DPGMM)</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #fafafa; }}
.node circle {{ stroke: #555; stroke-width: 1.5px; cursor: pointer; }}
.node text {{ font-size: 11px; fill: #333; }}
.link {{ fill: none; stroke: #ccc; stroke-width: 1.5px; }}
#info {{ position: fixed; top: 10px; right: 10px; background: white;
         padding: 15px; border: 1px solid #ddd; border-radius: 4px;
         max-width: 300px; font-size: 13px; }}
</style>
</head>
<body>
<div id="info">
  <h3>MM-CAD Application Tree</h3>
  <p>Method: Recursive DPGMM</p>
  <p>Click nodes to expand/collapse</p>
  <div id="details"></div>
</div>
<svg id="tree"></svg>
<script>
const treeData = {json.dumps(truncated)};

const width = window.innerWidth - 350;
const margin = {{top: 20, right: 120, bottom: 20, left: 180}};

const svg = d3.select("#tree")
  .attr("width", width)
  .attr("height", 2000)
  .append("g")
  .attr("transform", `translate(${{margin.left}}, ${{margin.top}})`);

const tree = d3.tree().size([1960, width - margin.left - margin.right]);
const root = d3.hierarchy(treeData);
root.x0 = 980; root.y0 = 0;

root.descendants().forEach(d => {{
  if (d.depth > 1 && d.children) {{
    d._children = d.children;
    d.children = null;
  }}
}});

function update(source) {{
  const treeLayout = tree(root);
  const nodes = root.descendants();
  const links = root.links();

  nodes.forEach(d => d.y = d.depth * 220);

  const node = svg.selectAll(".node").data(nodes, d => d.data.name);
  const nodeEnter = node.enter().append("g")
    .attr("class", "node")
    .attr("transform", d => `translate(${{source.y0 || 0}}, ${{source.x0 || 0}})`)
    .on("click", (e, d) => {{
      if (d.children) {{ d._children = d.children; d.children = null; }}
      else if (d._children) {{ d.children = d._children; d._children = null; }}
      update(d);
    }})
    .on("mouseover", (e, d) => {{
      d3.select("#details").html(
        `<b>${{d.data.name}}</b><br>` +
        `Keywords: ${{d.data.n_keywords}}<br>` +
        `Frequency: ${{(d.data.total_frequency || 0).toLocaleString()}}<br>` +
        `Depth: ${{d.data.depth}}<br>` +
        `Children: ${{d.data.n_children || 0}}`
      );
    }});

  nodeEnter.append("circle")
    .attr("r", d => Math.max(3, Math.min(15, Math.sqrt(d.data.n_keywords))))
    .style("fill", d => d._children ? "#4a90d9" : d.children ? "#7cb5ec" : "#f0f0f0");

  nodeEnter.append("text")
    .attr("dy", ".35em")
    .attr("x", d => (d.children || d._children) ? -15 : 15)
    .attr("text-anchor", d => (d.children || d._children) ? "end" : "start")
    .text(d => d.data.name.length > 40 ? d.data.name.slice(0, 37) + "..." : d.data.name);

  const nodeUpdate = nodeEnter.merge(node);
  nodeUpdate.transition().duration(300)
    .attr("transform", d => `translate(${{d.y}}, ${{d.x}})`);

  nodeUpdate.select("circle")
    .style("fill", d => d._children ? "#4a90d9" : d.children ? "#7cb5ec" : "#f0f0f0");

  node.exit().transition().duration(300)
    .attr("transform", d => `translate(${{source.y}}, ${{source.x}})`)
    .remove();

  const link = svg.selectAll(".link").data(links, d => d.target.data.name);
  link.enter().insert("path", "g")
    .attr("class", "link")
    .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));

  link.transition().duration(300)
    .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));

  link.exit().transition().duration(300).remove();

  nodes.forEach(d => {{ d.x0 = d.x; d.y0 = d.y; }});
}}

update(root);
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    log.info(f"Visualization saved to {output_path}")


def truncate_tree_for_viz(tree_dict: Dict, max_nodes: int = 500,
                          current_count: List = None) -> Dict:
    if current_count is None:
        current_count = [0]

    result = {k: v for k, v in tree_dict.items() if k != 'children' and k != 'keywords'}
    current_count[0] += 1

    if 'children' in tree_dict and current_count[0] < max_nodes:
        result['children'] = []
        for child in tree_dict['children']:
            if current_count[0] < max_nodes:
                result['children'].append(
                    truncate_tree_for_viz(child, max_nodes, current_count)
                )

    return result


generate_d3_visualization(tree_dict, CONFIG['output_tree_json'].replace('.json', '_viz.html'))

print("\nDone. Output files:")
print(f"  Tree JSON: {CONFIG['output_tree_json']}")
print(f"  Statistics: {CONFIG['output_stats_json']}")
print(f"  Visualization: {CONFIG['output_tree_json'].replace('.json', '_viz.html')}")
