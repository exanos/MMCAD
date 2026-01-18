# MMCAD: A Unified Multimodal Scalable Dataset for 3D CAD Model Representation Learning

This repository contains the implementation of **CLIP4CAD-H**, a unified multimodal encoder for CAD models that aligns B-Rep geometry, point cloud, and hierarchical text representations in a shared latent space.

## Overview

CLIP4CAD-H is designed to learn representations that:
- Capture both global structure and local geometric details
- Are robust to rotation augmentations
- Align naturally with hierarchical text descriptions
- Support cross-modal retrieval between geometry and text

### Key Components

1. **B-Rep Encoder**: Encodes face point grids and edge curves using 2D/1D convolutions
2. **Point Cloud Encoder**: Point-BERT style encoder with FPS + KNN tokenization
3. **Text Encoder**: Frozen LLM (Qwen2.5-3B) with hierarchical encoding
4. **Hierarchical Compression Module**: GSC (Global Structure Compression) + ADM (Adaptive Detail Mining)
5. **Multi-task Losses**: InfoNCE contrastive + local matching + reconstruction

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MMCAD.git
cd MMCAD

# Create environment
conda create -n clip4cad python=3.10
conda activate clip4cad

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+
- ~24GB VRAM (RTX 4090 recommended)

## Project Structure

```
MMCAD/
├── clip4cad/
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── brep_encoder.py      # B-Rep face/edge encoder
│   │   │   ├── pointbert_encoder.py # Point-BERT encoder
│   │   │   └── unified_projection.py
│   │   ├── text_encoder.py          # Hierarchical text encoder
│   │   ├── hierarchical_compression.py  # GSC + ADM
│   │   ├── clip4cad_h.py            # Main model
│   │   └── projection_heads.py
│   ├── data/
│   │   ├── dataset.py               # MMCAD dataset
│   │   └── augmentation.py          # Rotation augmentation
│   ├── losses/
│   │   ├── infonce.py               # Symmetric InfoNCE
│   │   ├── local_matching.py        # Hungarian matching
│   │   ├── reconstruction.py        # L1 reconstruction
│   │   └── combined.py              # Multi-task loss
│   ├── training/
│   │   └── trainer.py               # Training pipeline
│   ├── evaluation/
│   │   ├── retrieval.py             # Retrieval metrics
│   │   └── rotation_robustness.py   # Rotation evaluation
│   └── utils/
├── configs/
│   ├── default.yaml
│   ├── model/
│   ├── data/
│   └── training/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── extract_features.py
└── pretrained/
```

## Data Preparation

The MMCAD:B dataset should be organized as follows:

```
data/mmcad/
├── brep/
│   ├── {sample_id}.npz          # Contains face_grids, edge_curves
├── pointcloud/
│   ├── {sample_id}.npy          # Point cloud [N, 3]
├── text/
│   ├── {sample_id}.json         # Contains title, description
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Training

### Two-Stage Training

CLIP4CAD-H uses two-stage training:

1. **Stage 1** (epochs 1-40): Global contrastive + reconstruction
2. **Stage 2** (epochs 41-100): Add local contrastive alignment

```bash
# Default training
python scripts/train.py

# Custom configuration
python scripts/train.py training.epochs=50 data.batch_size=16

# Resume from checkpoint
python scripts/train.py  # Automatically resumes if checkpoint exists
```

### Configuration

Override configuration via command line or create custom configs:

```bash
# Override parameters
python scripts/train.py \
    model.d_unified=256 \
    training.lr=1e-4 \
    data.batch_size=8

# Use custom config
python scripts/train.py --config-name=custom_config
```

### Logging

Training supports Weights & Biases logging:

```bash
python scripts/train.py logging.use_wandb=true logging.wandb_project=clip4cad
```

## Evaluation

```bash
# Evaluate checkpoint
python scripts/evaluate.py checkpoint=outputs/checkpoints/best.pt

# Include rotation robustness evaluation
python scripts/evaluate.py checkpoint=outputs/checkpoints/best.pt eval.rotation=true
```

### Metrics

- **Retrieval**: R@1, R@5, R@10, MRR, Median Rank
- **Rotation Robustness**: Mean/Min cosine similarity under rotations

## Feature Extraction

Extract embeddings for downstream tasks:

```bash
python scripts/extract_features.py \
    checkpoint=outputs/checkpoints/best.pt \
    output_dir=features/
```

## Architecture Details

### Hierarchical Compression Module

- **GSC (Global Structure Compression)**: Cross-attention with N_g=8 learnable queries
- **ADM (Adaptive Detail Mining)**: Coverage-based selection of N_d=32 detail tokens

### Loss Functions

| Loss | Weight (Stage 1) | Weight (Stage 2) |
|------|------------------|------------------|
| Global Contrastive | 1.0 | 1.0 |
| Local Matching | 0.0 | 0.5 |
| Reconstruction | 0.5 | 0.25 |

### Model Dimensions

| Component | Dimension |
|-----------|-----------|
| Unified Space | 256 |
| Projection Head | 128 |
| LLM Features | 3072 (Qwen2.5-3B) |
| B-Rep Face Features | 256 |
| Point Tokens | 384 |

## Citation

```bibtex
@article{mmcad2024,
  title={MMCAD: A Unified Multimodal Scalable Dataset for 3D CAD Model Representation Learning},
  author={},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This implementation builds upon ideas from:
- [HCC-CAD](https://github.com/HCC-CAD) for hierarchical compression concepts
- [OpenShape](https://github.com/Colin97/OpenShape) for contrastive learning patterns
- [AutoBrep](https://github.com/AutoBrep) for B-Rep encoding architecture
- [ShapeLLM](https://github.com/ShapeLLM) for multimodal alignment strategies
