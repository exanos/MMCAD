# MM-CAD: A Multi-Modal CAD Dataset and Benchmark for Cross-Modal Geometric Learning

**Anush Bharathi, Ananthakrishnan A, Ramanathan Muthuganapathy**  
Indian Institute of Technology Madras  
*Symposium on Geometry Processing (SGP) 2026 — Computer Graphics Forum*

[![Paper](https://img.shields.io/badge/Paper-SGP%202026-blue)](https://github.com/exanos/MMCAD)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/exanos/MMCAD)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Overview

MM-CAD is a large-scale multi-modal CAD dataset designed for retrieval and retrieval-augmented generation over engineering geometry. It consists of two complementary parts:

**MM-CAD:A** — 33,816 unique CAD models consolidated from eleven widely used benchmarks (MCB, CADNET, ESB, PSB, ShapeNet, ModelNet, DeepCAD, Fusion360, CADParser, IFCNet, Thingi10K), with:
- Isometric renders (3 viewpoints)
- 10K-point clouds with oriented normals
- 3,276 real hand-drawn user sketches + CGAN contour sketches
- Human-validated multi-level text captions (Title, Description, Class)

**MM-CAD:B** — 192,626 semantically organized models from the ABC corpus, curated through a 7-stage pipeline centered on **Manifold-Aware Adaptive Sampling (MAAS)**, with:
- Metadata-grounded text captions from parsed FeatureScript construction sequences (Gemini 3 Pro/Flash)
- Photorealistic images conditioned on source CAD geometry (FLUX.2-Klein, ~131K models)
- Synthetic hand-drawn contour sketches
- Hierarchical application taxonomy (4,862 nodes, recursive DPGMM)
- FAISS nearest-neighbor graph for hard-negative retrieval training

A joint multi-modal retrieval architecture aligning text, sketch, image, B-Rep, and point cloud encoders in a shared Matryoshka embedding space is trained on MM-CAD:B and released as a reference benchmark.

---

## Dataset

The full dataset is available on HuggingFace:  
👉 **[huggingface.co/datasets/exanos/MMCAD](https://huggingface.co/datasets/exanos/MMCAD)**

| Split | Models | Modalities |
|-------|--------|------------|
| MM-CAD:A | 33,816 | Renders, point clouds, real sketches, human captions |
| MM-CAD:B | 192,626 | Renders, point clouds, contour sketches, metadata captions, photorealistic images, taxonomy |

Google Drive (raw archives):
- Photorealistic images: [Drive](https://drive.google.com/drive/folders/18-0zTkMVf5h7KF8RsZdBD1rCHZ6t-Qrw)
- Point clouds, text, weights, compressed modalities: [Drive](https://drive.google.com/drive/folders/1AajlWNhzbKkjdCK1cx9zw6G33UXw_Fz7)

---

## Code Structure

```
├── application_graph/          # DPGMM hierarchical taxonomy pipeline
├── mmcad_baseline_trimodal*.ipynb   # Tri-modal retrieval training (text, BRep, PC)
├── mmcad_brep_training_colab*.ipynb # BRep encoder training (BRepFormer)
├── mmcad_sketch_encoder*.ipynb      # Sketch encoder training (ViT-Base)
├── mmcad_v_trimodal*.ipynb          # Full 5-modal training (+ image, render)
├── FLUX2_klein*.ipynb               # Photorealistic image synthesis pipeline
├── InternVL3_5*.ipynb               # View selection and prompt generation
├── mmcad_tokenizer.ipynb            # Geometric motif vocabulary baseline
└── mmcad_inference*.ipynb           # Inference and retrieval evaluation
```

> Full code cleanup and documented release in progress. Colab-ready notebooks with Drive integration will be uploaded shortly.

---

## Models

Pretrained checkpoints (tri-modal and 5-modal) available at:  
👉 [Drive — weights](https://drive.google.com/drive/folders/1AajlWNhzbKkjdCK1cx9zw6G33UXw_Fz7)

HuggingFace model hub release coming soon.

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

MM-CAD dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Code in this repository is MIT licensed. Models derived from the ABC dataset are subject to the [ABC dataset terms](https://deep-geometry.github.io/abc-dataset/).
