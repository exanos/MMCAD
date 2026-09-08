# MM-CAD:B Application Taxonomy Pipeline

End-to-end pipeline that turns ~76k raw application keywords from 182k CAD models into a
hierarchical taxonomy with per-node UID lookups and a publication figure.

## 1. Pipeline overview

```
abc_dataset_clean.csv
        |
        |-- (1) keyword extraction + frequency counting
        v
keyword_metadata.json  (list of unique keywords + freq)
        |
        |-- (2) semantic embeddings (BGE) + PPMI co-occurrence embeddings
        |-- (3) weighted fusion + UMAP (50d)
        |-- (4) recursive DPGMM clustering (BIC-validated splits)
        |-- (5) bottom-up parallel LLM labeling (Gemini 2.5 Flash)
        v
application_tree_dpgmm.json        (tree, 4862 nodes, max_depth 4)
tree_statistics.json               (shape metrics)
        |
        |-- (6) UID enrichment (build_uid_tree.py)
        v
keyword_uid_index.json             (keyword -> [uid, ...])
application_tree_dpgmm_uids.json   (tree + per-leaf uid lists + aggregate counts)
        |
        |-- (7) publication figure (create_dpgmm_figure.py)
        v
dpgmm_taxonomy_figure.{png,pdf,svg}
```

Stages 1 through 5 are executed in `recursive_dpgmm_tree.py` / `.ipynb` (Colab, GPU).
Stages 6 and 7 are local Python scripts.

## 2. Input dataset

`abc_dataset_clean.csv` (182,166 rows). Relevant columns:

- `uid`: unique model identifier used throughout the pipeline
- `abc_uid`: original ABC dataset id
- `title`, `description`: free text metadata
- `applications`: LLM-generated application keywords per model, formatted as
  `'Keyword1", "Keyword2", "Keyword3'` (note: leading quote is often missing because
  of CSV quoting of the outer field). Parsed by splitting on the `", "` delimiter and
  stripping residual quotes.
- `iso1_path`, `iso2_path`, `top_path`: ISO/top-level ontology paths (not used by
  this pipeline).

`keyword_metadata.json` holds the flattened vocabulary: `keywords` (list, ~76,593
entries) and `keyword_counts` (dict or aligned list of per-keyword frequencies across
all models). `total_instances` is the sum over the corpus.

## 3. Stage 1: embeddings

Two independent embedding spaces are computed, then fused.

### Semantic embeddings

`BAAI/bge-large-en-v1.5`, 1024 dimensions, L2-normalized. One vector per keyword string,
computed in batches of 256 on GPU (fp32). Cached to `embeddings_<model>.npy` and reused
if dimensions match.

### Co-occurrence PPMI embeddings

Keywords are considered to co-occur when they appear on the same model's `applications`
list.

1. Build a symmetric `(n_keywords, n_keywords)` sparse co-occurrence count matrix by
   iterating each model's keyword list and incrementing every ordered pair.
2. Drop entries with count below `cooccurrence_min_count` (default 2).
3. Compute shifted PPMI: `PPMI(i,j) = max(0, log2(p(i,j) / (p(i) * p(j))) - shift)`
   where marginals come from `keyword_freq` and joints from the co-occurrence matrix.
   `ppmi_shift = 1.0` suppresses weak/noisy associations.
4. Truncated SVD to `cooccurrence_svd_dims = 128`, dimensions scaled by `sqrt(singular_values)`.

### Fusion + UMAP

Each embedding is L2-normalized per-row, concatenated horizontally with weights
`semantic_weight = 0.7` and `cooccurrence_weight = 0.3`. Fused vectors are reduced to
50 dimensions with UMAP (`n_neighbors=30`, `min_dist=0.0`, `metric='cosine'`,
`random_state=42`). The cosine metric + L2 normalization keeps angular structure intact
while collapsing redundant axes.

## 4. Stage 2: recursive DPGMM tree

Each node owns a subset of keyword indices. Splitting decision per node:

1. Fit a Bayesian Gaussian Mixture with a Dirichlet-process weight prior
   (`weight_concentration_prior_type='dirichlet_process'`). `covariance_type='diag'`,
   `reg_covar=1e-4`, `n_init=3`, `max_iter=500`.
2. Depth-adaptive priors, indexed by node depth:
   - `dpgmm_max_comp_schedule = [20, 30, 40, 40, 30, 25, 20, 15]`
   - `dpgmm_alpha_schedule    = [0.01, 0.05, 0.1, 0.2, 0.5, 0.5, 1.0, 1.0]`
   Higher `alpha` at deeper levels discourages superfluous sub-splits; higher
   `max_components` at mid-depth gives the model room to find fine-grained structure in
   large mid-level sectors.
3. Active components = those with posterior weight above `dpgmm_weight_threshold = 0.01`.
   If only one active component survives, the node becomes a leaf.
4. BIC validation: fit a single Gaussian and an `n_active`-component Gaussian (same
   covariance type). Split only if
   `(bic_single - bic_mixture) / |bic_single| > 0.005`.
5. Post-processing:
   - Components smaller than `min_component_size = 3` are dissolved and their members
     reassigned to the nearest surviving cluster by Euclidean distance to centroid.
   - If the node still yields more than `max_children_per_node = 25` clusters, agglomerative
     centroid merging reduces to 25.
6. Recurse into each surviving child. Stop conditions: `n <= min_split_size = 12`, or
   `depth >= max_depth = 7`.

Orphan samples (predicted into a suppressed component) are reassigned by nearest active
centroid.

Resulting tree: **4,862 nodes, 271 internal, 4,591 leaves, max_depth 4, branching factor
mean 17.9 / median 20 / max 25.** Leaf reasons: `bic_no_improvement` (1,557),
`below_min_split_size` (3,033), `single_component` (1). Construction time ~482s on Colab.

## 5. Stage 3: LLM labeling

Two passes in `label_tree_parallel`.

### Pass 1: heuristic

Every node gets a fallback name of the form
`"Top1 / Top2 / Top3"` (internal) or `"Top1 / Top2 / Top3 (+N more)"` (leaf), where
ranking is by keyword frequency within the subtree. Guarantees every node has *some* name
before any network call.

### Pass 2: parallel LLM relabel (internal nodes only)

Bottom-up by depth. Within each depth, nodes are labeled concurrently with a
`ThreadPoolExecutor` of `llm_parallel_workers = 16`. Each call:

- Model: `google/gemini-2.5-flash` via OpenRouter
- `temperature=0.3`, `max_tokens=30`
- Prompt contains: node depth, already-labeled child names (<=15), and top 20 keywords
  in the subtree by frequency
- Response stripped of markdown, quotes, trailing punctuation; truncated to 60 chars
- Retries on HTTP 429 with linear backoff, 3 attempts max; heuristic fallback on failure

Processing deeper depths first means child labels are available as context when labeling
their parents.

## 6. Stage 4: tree JSON schema

`application_tree_dpgmm.json` (~13 MB), produced by `tree_to_dict`.

Internal node:
```json
{
  "name": "Robotics / End Effector / Servo Mount",
  "depth": 1,
  "n_keywords": 1843,
  "total_frequency": 12480,
  "type": "internal",
  "n_children": 14,
  "children": [ ... ]
}
```

Leaf node:
```json
{
  "name": "Actuator Mount / Linear Actuator (+3 more)",
  "depth": 3,
  "n_keywords": 6,
  "total_frequency": 212,
  "type": "leaf",
  "keywords": [
    {"keyword": "Actuator Mount", "frequency": 87, "index": 4123},
    ...
  ],
  "leaf_reason": "bic_no_improvement"
}
```

`tree_statistics.json` carries shape metrics: total/internal/leaf counts, max depth,
branching factor stats, leaf size distribution, per-depth node counts, leaf reasons
histogram, wall-clock construction time.

## 7. Stage 5: UID enrichment (`build_uid_tree.py`)

Attaches the actual model UIDs from `abc_dataset_clean.csv` to the keyword-tree.
Run from `application_graph/`:

```
python build_uid_tree.py
```

Pipeline:

1. `build_keyword_uid_index(dataset_path)` streams the CSV once with `csv.DictReader`,
   parses each row's `applications` field, and builds `kw_to_uids: defaultdict(set)`.
   Progress logged every 50k rows.
2. Flat index `keyword_uid_index.json` (~10 MB) is written first:
   `{ "Actuator Mount": ["uid1","uid2",...], ... }`. One key per keyword.
3. `load_tree(path)` parses the existing tree JSON. It brace-walks until the outer
   object closes to tolerate any trailing bytes in the file.
4. `enrich_tree(node, kw_to_uids)` recurses through the tree:
   - **Leaf**: for each keyword in `node.keywords`, attach the sorted UID list and count.
     Aggregate the deduplicated union as `model_uids` and set `n_models = len(model_uids)`.
   - **Internal**: recurse, aggregate the union of descendant UIDs, store `n_models` and
     `n_children`. Internal nodes do NOT store the full UID list (too large, redundant);
     `collect_uids(node)` can rebuild it on demand.
5. Output: `application_tree_dpgmm_uids.json` (~69 MB).

### Enriched tree schema

Leaf:
```json
{
  "name": "Actuator Mount / Linear Actuator (+3 more)",
  "depth": 3,
  "type": "leaf",
  "n_keywords": 6,
  "n_models": 184,
  "model_uids": ["<uid>", ...],
  "keywords": [
    {
      "keyword": "Actuator Mount",
      "frequency": 87,
      "n_models": 81,
      "uids": ["<uid>", ...]
    },
    ...
  ]
}
```

Internal:
```json
{
  "name": "Robotics & Motion Systems",
  "depth": 1,
  "type": "internal",
  "n_keywords": 1843,
  "n_children": 14,
  "n_models": 9871,
  "children": [ ... ]
}
```

Sanity check observed on the current output: 182,166 rows scanned, 76,593 keywords
indexed, 182,147 unique UIDs at the root (19 models lost to empty/malformed
`applications` fields).

## 8. Stage 6: publication figure (`create_dpgmm_figure.py`)

Matplotlib sunburst + rectangular inset. Designed for a quarter-page, full-width slot
in the paper. Run from `application_graph/`:

```
python create_dpgmm_figure.py
```

Reads `application_tree_dpgmm.json`, writes `dpgmm_taxonomy_figure.{png,pdf,svg}`.

### Layout

- `figsize=(14.0, 5.8)`, serif (Times New Roman), base font 8pt, 300 DPI.
- Two fixed axes:
  - Sunburst on the left: `[0.02, 0.0, 0.56, 1.0]`
  - Inset panel on the right: `[0.59, 0.03, 0.40, 0.94]`
- Radii: inner `ri=0.85`, outer `ro=1.55`, subcategory ring `r_sub=2.45`,
  leaf ring `r_leaf=3.25`.
- Sectors are proportional to `n_keywords` with a `0.6 deg` gap between sectors.
  The Robotics sector (`ROBOTICS_IDX=4`) is rotated to sit centered at `0 deg`
  so the inset callout line can attach cleanly on the right.

### Sector labels (inside the ring)

Tangential, multi-line, size-adaptive. For each sector:

- Font size `fs = 2.0 + t * 2.2` where `t` is the normalized sector span. Range ~2.0 to
  4.2pt.
- Characters per line from arc length: `arc_len / (fs * 0.009)`.
- `max_lines = 2` for large-font sectors, `3` for small. Prevents radial overflow.
- Name cleaned by `clean_name`: strips ` / ` variants, `(+N more)` suffixes, and any
  list past the second comma. `textwrap.wrap` with `break_long_words=True`.

### Subcategory + leaf rings

Per sector, `n_show` subcategories drawn based on sector span:
`3 if span>16, 2 if span>10, 1 if span>5, else 0`. Within each subcategory, `n_leaves`
leaves drawn: `2 if span>12, 1 if span>6, else 0`. Both rings use radial text
(`radial_text_params` flips rotation 180 deg on the left half so all labels read
left-to-right) with thin connector lines.

### Inset

Hardcoded actuator-focused subtree (actuators/motor-mounts/end-of-arm-tooling) drawn as
a simple left-to-right tree in `ax_r`. Two dashed `ConnectionPatch` lines tie the top and
bottom of the Robotics sector to the top and bottom of the inset bounding box (a
`FancyBboxPatch`).

### Why this figure, not `create_paper_figure.py`

`create_paper_figure.py` was the earlier generic radial renderer. It is kept as a
reference for the radial text rotation logic (`if 90 < angle_deg < 270: angle += 180;
ha='right'`), but the published figure is produced by `create_dpgmm_figure.py`.

## 9. Configuration surface

All pipeline knobs live in `CONFIG` at the top of `recursive_dpgmm_tree.py`.
Practical tuning notes:

- **Tree depth and granularity**: `max_depth`, `min_split_size`, `min_component_size`,
  `bic_improvement_threshold`. Raise `bic_improvement_threshold` to get a coarser tree,
  lower it to allow weaker splits.
- **Fanout**: `dpgmm_max_comp_schedule` bounds the initial mixture size per depth;
  `max_children_per_node` is a hard post-merge cap.
- **Semantic vs. distributional**: shift weight between `semantic_weight` and
  `cooccurrence_weight`. Pure semantic (1.0/0.0) gives taxonomy closer to vocabulary
  meaning; pure co-occurrence surfaces dataset-specific affinities.
- **UMAP**: larger `n_neighbors` gives more global structure, fewer islands; `min_dist=0`
  is intentional to keep tight clusters tight.
- **LLM throughput**: `llm_parallel_workers`. Bound by OpenRouter rate limits; 16 is safe
  for Gemini 2.5 Flash. Swap `llm_model` for alternative providers.

## 10. Reproduction

```
# Stage 1-5 (Colab):
#   Open recursive_dpgmm_tree.ipynb
#   Set CONFIG['openrouter_api_key']
#   Run all cells. Outputs land in /content/drive/MyDrive/MMCAD/.

# Stage 6 (local):
cd application_graph
python build_uid_tree.py

# Stage 7 (local):
python create_dpgmm_figure.py
```

All downstream scripts assume the tree JSON, UIDs JSON, and CSV are sitting in the
paths hardcoded at the top of each script (relative to `application_graph/`).

## 11. File inventory

Pipeline scripts:
- `recursive_dpgmm_tree.py` / `.ipynb`: stages 1-5
- `build_uid_tree.py`: stage 6
- `create_dpgmm_figure.py`: stage 7 (publication figure)
- `create_paper_figure.py`: reference only

Data inputs:
- `../abc_dataset_clean.csv`: raw dataset
- `keyword_metadata.json`: vocabulary + counts

Data outputs:
- `application_tree_dpgmm.json` (~13 MB): taxonomy
- `tree_statistics.json`: shape metrics
- `keyword_uid_index.json` (~10 MB): flat keyword -> UIDs
- `application_tree_dpgmm_uids.json` (~69 MB): enriched taxonomy
- `dpgmm_taxonomy_figure.{png,pdf,svg}`: publication figure

Legacy / exploratory scripts in this directory (`analyze_categories.py`,
`create_enhanced_visualization.py`, `create_mcb_style_figure.py`, `fix_labels.py`,
`generate_category_labels.py`, `hierarchical_clustering.py`, `relabel_all_nodes.py`,
`smart_relabel.py`, etc.) are not part of the current pipeline.
