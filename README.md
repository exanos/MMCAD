# MM-CAD: A Multi-Modal CAD Dataset and Benchmark for Cross-Modal Geometric Learning

**Anush Bharathi, Ananthakrishnan A, Ramanathan Muthuganapathy**
Indian Institute of Technology Madras
*Symposium on Geometry Processing (SGP) 2026 — Computer Graphics Forum*

[![Project Page](https://img.shields.io/badge/Project%20Page-exanos.github.io%2FMMCAD-blue)](https://exanos.github.io/MMCAD)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/exanos/MMCAD)
[![License: Data](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![License: Code](https://img.shields.io/badge/Code-MIT-lightgrey)](LICENSE)

Project page: **https://exanos.github.io/MMCAD**

---

## Overview

MM-CAD is a large-scale multi-modal CAD dataset designed for retrieval and retrieval-augmented generation over engineering geometry. It consists of two complementary parts:

**MM-CAD:A** — 33,816 unique CAD models consolidated from eleven widely used benchmarks (MCB, CADNET, ESB, PSB, ShapeNet, ModelNet, DeepCAD, Fusion360 Gallery, CADParser, IFCNet, Thingi10K), with:
- Isometric renders (3 viewpoints) and meshes
- 10K-point clouds with oriented normals
- Multi-level sketches including 3,276 real hand-drawn user sketches and 1,100 traced training sketches
- Human-validated multi-level text captions (Title, Description, Class)
- 80:10:10 split — 27,048 / 3,376 / 3,392

**MM-CAD:B** — 192,626 semantically organized models from the 1M-model ABC corpus, curated through a 7-stage pipeline centered on **Manifold-Aware Adaptive Sampling (MAAS)**, with:
- Construction-sequence-grounded text captions from parsed FeatureScript metadata
- Photorealistic in-context images conditioned on source CAD geometry (~131K models)
- Multi-level synthetic contour sketches
- Hierarchical application taxonomy (4,862 nodes, recursive DPGMM)
- FAISS nearest-neighbor graph for hard-negative contrastive training

A joint retrieval architecture aligning text, sketch, image, B-Rep, and point cloud encoders in a shared Matryoshka embedding space (d ∈ {128, 256, 512, 768}) is trained on MM-CAD:B and released as a reference benchmark. Trimodal (text+sketch+image) → B-Rep retrieval reaches **45.9% R@1** on the validation gallery.

---

## Dataset Access

| Part | Models | Status |
|------|--------|--------|
| MM-CAD:A | 33,816 | **Live** on [Hugging Face](https://huggingface.co/datasets/exanos/MMCAD) |
| MM-CAD:B | 192,626 | Hugging Face upload **in progress** — interim access via Google Drive below |

Google Drive (raw archives):
- Photorealistic images: [Drive](https://drive.google.com/drive/folders/18-0zTkMVf5h7KF8RsZdBD1rCHZ6t-Qrw)
- Point clouds, text, checkpoints, compressed modalities: [Drive](https://drive.google.com/drive/folders/1AajlWNhzbKkjdCK1cx9zw6G33UXw_Fz7)

Everything is keyed by a global `uid`; `metadata.csv` / `metadata.parquet` join all modalities, source benchmark, category, split, and captions.

---

## Code

Training, synthesis, and inference notebooks (tri-modal and 5-modal retrieval, BRep/sketch encoders, FLUX.2 photorealistic synthesis, DPGMM taxonomy, geometric motif tokenizer) are being cleaned for release and will be uploaded here shortly.

---

## Pretrained Models

Tri-modal and 5-modal checkpoints (all four Matryoshka dimensions) are available in the [Drive weights folder](https://drive.google.com/drive/folders/1AajlWNhzbKkjdCK1cx9zw6G33UXw_Fz7). Hugging Face model hub release coming soon!

---

## Citation

```bibtex
@article{bharathi2026mmcad,
  title     = {MM-CAD: A Multi-Modal CAD Dataset and Benchmark for Cross-Modal Geometric Learning},
  author    = {Bharathi, Anush and Ananthakrishnan, A and Muthuganapathy, Ramanathan},
  journal   = {Computer Graphics Forum},
  volume    = {},
  year      = {2026},
  publisher = {Wiley},
  note      = {Proc. SGP 2026}
}
```

---

## License

- **Dataset:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- **Code:** [MIT](LICENSE)
- Models derived from the ABC dataset are additionally subject to the [ABC dataset terms](https://deep-geometry.github.io/abc-dataset/). Per-source-benchmark attribution and redistribution notes for MM-CAD:A will be provided in `LICENSES.md`.
