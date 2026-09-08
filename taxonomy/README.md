# MM-CAD:B application taxonomy: recursive DPGMM

Code that builds the hierarchical application taxonomy over MM-CAD:B: **4,862 nodes, 4,591 leaves, max depth 4**, discovered from ~76K application keywords mined from the grounded captions.

The method exists because **the number of clusters is never declared**. Nobody knows in advance how many kinds of "mount" or "bracket" live in a million-model corpus, so a Dirichlet Process Gaussian Mixture is fit recursively at every node, the branching factor comes from the data, and BIC decides whether each split is worth keeping.

## Method

1. **Keyword extraction.** Application keywords per model from the construction-grounded captions, about 76K unique.
2. **Dual embedding.** BGE-large semantic embeddings together with PPMI co-occurrence embeddings over the keyword graph. Semantic embeddings miss which keywords actually appear together on real parts; co-occurrence embeddings conflate synonyms. Fusing both at 0.7 / 0.3 covers each one's blind spot.
3. **UMAP** to 50 dimensions (cosine).
4. **Recursive DPGMM.** A `BayesianGaussianMixture` per node with a depth-scheduled concentration prior. Components below a weight threshold are dropped, and a split is kept only if it improves BIC over a single Gaussian.
5. **Bottom-up LLM naming.** Each node is named from its children's labels plus its top keywords, so a parent is named after its subtree rather than the other way round.
6. **UID enrichment.** `build_uid_tree.py` attaches member model UIDs to every leaf and aggregates counts up the tree.

## Files

| File | Purpose |
|------|---------|
| `recursive_dpgmm_tree.py` | Stages 1 to 5. Colab/GPU; writes `application_tree_dpgmm.json` and `tree_statistics.json` |
| `build_uid_tree.py` | Stage 6. Attaches model UIDs per leaf, aggregates counts |
| `APPLICATION_TREE_PIPELINE.md` | Full pipeline documentation, inputs, outputs, config reference |
| `tree_statistics.json` | Shape metrics of the released tree |

## Running

```bash
pip install umap-learn sentence-transformers scikit-learn scipy tqdm
export OPENROUTER_API_KEY=...     # only needed for stage 5 (node naming)
python recursive_dpgmm_tree.py
```

Inputs (`abc_dataset_clean.csv`, `keyword_metadata.json`) and the built tree are distributed with the dataset under [`mmcad_b/taxonomy/`](https://huggingface.co/datasets/exanos/MMCAD/tree/main/mmcad_b/taxonomy) on Hugging Face. Key knobs live in the `CONFIG` dict at the top of `recursive_dpgmm_tree.py`; `random_state=42` reproduces the released tree.

## Released tree

```
total nodes        4,862
internal nodes       271
leaves             4,591
max depth              4
branching factor    17.9 mean / 20 median / 25 max
leaf size           16.7 mean / 9 median / 456 max
build time            482 s
```

Leaf termination reasons: 3,033 below minimum split size, 1,557 no BIC improvement, 1 single component.
