# MM-CAD: A Multi-Modal CAD Dataset and Benchmark for Cross-Modal Geometric Learning

**Anush Bharathi, Ananthakrishnan A, Ramanathan Muthuganapathy**
Indian Institute of Technology Madras
*Symposium on Geometry Processing (SGP) 2026 — Computer Graphics Forum*

[![Paper](https://img.shields.io/badge/Paper-10.1111%2Fcgf.70523-b31b1b)](https://doi.org/10.1111/cgf.70523)
[![Project Page](https://img.shields.io/badge/Project%20Page-exanos.github.io%2FMMCAD-blue)](https://exanos.github.io/MMCAD)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/exanos/MMCAD)
[![Talk](https://img.shields.io/badge/Talk-YouTube-red)](https://www.youtube.com/watch?v=3-a2MjjOJT0)
[![License: Data](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: Code](https://img.shields.io/badge/Code-MIT-lightgrey)](LICENSE)

**Project page:** https://exanos.github.io/MMCAD · **Talk:** https://www.youtube.com/watch?v=3-a2MjjOJT0

---

## Overview

MM-CAD is a large-scale multi-modal CAD dataset built for retrieval and retrieval-augmented generation over engineering geometry — *retrieval and identification, not generation*. It consists of two complementary parts, and the design principle is that **A is built by hand so that A can build B**.

**MM-CAD:A** — 33,816 unique CAD models consolidated from eleven benchmarks (MCB, DeepCAD, Thingi10K, ShapeNetV2, Fusion360 Gallery, PSB, IFCNet, CADParser, ModelNet40, CADNET, ESB), with isometric renders, 10K-point clouds with oriented normals, 4,069 real human sketches, and human-validated multi-level captions. Split 27,048 / 3,376 / 3,392.

**MM-CAD:B** — 192,626 models curated from the 1M-model ABC corpus through a seven-stage pipeline centered on **Manifold-Aware Adaptive Sampling (MAAS)**, which organizes models into semantically coherent neighborhoods rather than merely removing duplicates — directly supplying the hard negatives contrastive retrieval training needs. Every survivor is annotated across five aligned modalities. Split 173,363 train / 19,263 validation.

The distinguishing choice is **construction-sequence grounding**: captions are conditioned on each model's parsed FeatureScript history rather than on rendered views alone, so the annotation knows what a render cannot show — that a hole is blind rather than through, that a taper is a declared 5° draft rather than a loft, that a circular pattern has exactly 54 instances. Blind human raters scored the grounded captions Very/Extremely Accurate 85.7% of the time.

A joint retrieval architecture aligning text, sketch, image, B-Rep, and point cloud encoders in a shared Matryoshka space (d ∈ {128, 256, 512, 768}) is trained on MM-CAD:B and released as a reference benchmark. Trimodal (text + sketch + image) → B-Rep reaches **45.91% R@1** on the validation gallery.

---

## Dataset Access

Both corpora are fully released on Hugging Face: **[huggingface.co/datasets/exanos/MMCAD](https://huggingface.co/datasets/exanos/MMCAD)**

```python
from datasets import load_dataset

a = load_dataset("exanos/MMCAD", "mmcad_a", split="train")
b = load_dataset("exanos/MMCAD", "mmcad_b", split="train")
```

| MM-CAD:A | Records | | MM-CAD:B | Records |
|---|---:|---|---|---:|
| Renders (per view) | 33,816 | | STEP B-Rep | 192,625 |
| Point clouds | 32,001 | | Point clouds (10K + normals) | 192,625 |
| Meshes | 31,616 | | Text annotations (3-level) | 192,625 |
| Contour sketches | 25,026 | | Renders (ISO1/ISO2/top) | 192,541 / 192,112 / 188,566 |
| Canny sketches | 1,814 | | Contour sketches (ISO1/ISO2) | 189,908 / 189,861 |
| Human sketches (drawn/traced) | 2,996 / 1,073 | | Photorealistic images | 129,679 |
| Human text annotations | 22,684 | | Application taxonomy | 4,862 nodes |

Everything is keyed by a global `uid`. Assets are located through the `*_archive` and `*_member` columns in `metadata.parquet`; an empty path means that modality is unavailable for that record. Large MM-CAD:B modalities ship as uncompressed tar shards with per-shard SHA-256 checksums and complete manifests.

ShapeNetV2 *meshes* are excluded because the upstream license does not permit mesh redistribution; its derived modalities (point clouds, renders, sketches, captions) are included.

---

## Code

### `taxonomy/` — application taxonomy pipeline

The recursive DPGMM pipeline that discovers the 4,862-node application hierarchy from ~76K caption-mined keywords, with the cluster count inferred at every level rather than declared. Dual semantic + PPMI co-occurrence embeddings, UMAP, Dirichlet-Process mixtures with BIC-validated splits, bottom-up LLM naming. See [`taxonomy/README.md`](taxonomy/README.md).

### `notebooks/` — training, synthesis, inference

Colab-ready, outputs stripped.

| Notebook | Purpose |
|----------|---------|
| `mmcad_training_colab.ipynb` | Baseline multi-modal retrieval training on MM-CAD:A (sketch/text → point cloud) |
| `mmcad_v_trimodal_c.ipynb` | Joint tri-modal training (EmbeddingGemma + BRepFormer + DGCNN, Matryoshka InfoNCE) |
| `mmcad_sketch_encoder.ipynb` | ViT-Base sketch encoder, B-Rep-anchored alignment |
| `mmcad_render_encoder.ipynb` | SigLIP-Base photorealistic-image encoder, B-Rep-anchored alignment |
| `mmcad_inference.ipynb` | Retrieval inference + full Matryoshka evaluation matrix |

Notebooks expect the dataset archives mounted from your own storage; paths are set in the first cells. Additional pipeline code (FLUX.2 synthesis, motif tokenizer) will follow.

---

## Pretrained Models

Three checkpoints, all sharing one d=768 Matryoshka space over {128, 256, 512, 768}, hosted with the dataset under [`checkpoints/`](https://huggingface.co/datasets/exanos/MMCAD/tree/main/checkpoints):

| Checkpoint | Towers | Role |
|---|---|---|
| `baseline_trimodal_v4.pth` | text (EmbeddingGemma-300M) · B-Rep (BRepFormer) · point cloud (DGCNN) | Stage 1, trained jointly with symmetric InfoNCE at all four scales |
| `sketch_encoder_v1.pth` | sketch (ViT-Base) | Stage 2, aligned to frozen B-Rep anchors |
| `render_encoder_v1.pth` | photorealistic image (SigLIP-Base) | Stage 2, aligned to frozen B-Rep anchors |

Run retrieval with [`inference.py`](inference.py):

```bash
python inference.py --query "servo mount with four bolt holes" --dim 128
python inference.py --query "bevel gear" --sketch sketch.png --image photo.jpg --top-k 10
```

Query vectors are summed and renormalized — no learned fusion head. Truncating to d=128 moves trimodal R@1 only from 45.91 to 45.50, so a d=128 FAISS index serves interactive queries at negligible cost.

Helper scripts in [`scripts/`](scripts/): `audit_checkpoints.py` reports what any `.pth` contains (towers, epoch, stored metrics) by reading only its pickle header, so multi-GB files are inspected instantly; `upload_checkpoints_to_hf.ipynb` republishes checkpoints from Drive to Hugging Face.

---

## Open problems

Two are released as benchmark tasks rather than hidden:

1. **Feature terms do not ground to geometry.** Retrieval matches silhouette, not feature: "herringbone gear · ten lightening holes" returns a water-bottle base at rank 1 (correct gear at #77); "symmetrical V-groove pulley" returns a toroidal wheel (correct part at #426).
2. **Geometric motif vocabulary.** Decomposing CAD models into maximal recurring units under chamfer-congruence — 120,794 motifs mined with 0.78 cross-model recurrence, but a median 0.23 residual and 44.7% of parts above 80% residual. The tail is the open problem.

---

## Citation

```bibtex
@article{bharathi2026mmcad,
  title     = {MM-CAD: A Multi-Modal CAD Dataset and Benchmark for Cross-Modal Geometric Learning},
  author    = {Bharathi, Anush and Ananthakrishnan, A and Muthuganapathy, Ramanathan},
  journal   = {Computer Graphics Forum},
  year      = {2026},
  publisher = {Wiley},
  volume    = {45},
  number    = {5},
  doi       = {10.1111/cgf.70523},
  note      = {Proc. SGP 2026}
}
```

---

## License

- **Project-created annotations and derived assets** (captions, renders, sketches, point clouds, photorealistic images, taxonomy): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- **Code:** [MIT](LICENSE)
- **Underlying geometry** remains subject to each source dataset's own terms, including ABC/Onshape terms for MM-CAD:B. See [`LICENSES.md`](https://huggingface.co/datasets/exanos/MMCAD/blob/main/LICENSES.md) on the dataset repository for per-benchmark attribution and redistribution notes. Verify current terms at each source before redistributing.
