# MMCAD: A Unified Multimodal Scalable Dataset for 3D CAD Model Representation Learning

This repository contains the implementation of **CLIP4CAD-H**, a unified multimodal encoder for CAD models that aligns B-Rep geometry, point cloud, and hierarchical text representations in a shared latent space.

## Overview

CLIP4CAD-H is designed to learn representations that:
- Capture both global structure and local geometric details
- Are robust to rotation augmentations
- Align naturally with hierarchical text descriptions
- Support cross-modal retrieval between geometry and text

### Key Components

1. **B-Rep Encoder**: AutoBrep-style FSQ VAE encoder for face grids and edge curves
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

## Pretrained Weights

### AutoBrep Weights (B-Rep Encoder)

The B-Rep encoder is based on [AutoBrep](https://github.com/AutodeskAILab/AutoBrep) and can be initialized with pretrained FSQ VAE weights from HuggingFace.

```bash
# Download pretrained AutoBrep weights
python scripts/download_autobrep_weights.py --output-dir pretrained/autobrep
```

This downloads the surface and edge FSQ VAE checkpoints from [SamGiantEagle/AutoBrep](https://huggingface.co/SamGiantEagle/AutoBrep).

Then update your config to use the pretrained weights:

```yaml
# In configs/model/clip4cad_h.yaml
encoders:
  brep:
    surface_checkpoint: pretrained/autobrep/surface_fsq_vae.pt
    edge_checkpoint: pretrained/autobrep/edge_fsq_vae.pt
```

### B-Rep Encoder Architecture

The B-Rep encoder follows AutoBrep's FSQ VAE architecture:

| Component | Details |
|-----------|---------|
| Surface Encoder | 2D CNN with channel mult [1,2,4,8], 16 latent channels |
| Edge Encoder | 1D CNN with channel mult [1,2,4], 4 latent channels |
| FSQ Levels | [8, 5, 5, 5] = 1000 codebook entries |
| Face Output | 48-dim (3 × 16, matching AutoBrep XAEncoder surfZ) |
| Edge Output | 12-dim (3 × 4, matching AutoBrep XAEncoder edgeZ) |

For CLIP4CAD, we use continuous features (pre-FSQ quantization) for contrastive learning.

## Project Structure

```
MMCAD/
├── clip4cad/
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── brep_encoder.py      # AutoBrep-style FSQ VAE encoder
│   │   │   ├── pointbert_encoder.py # Point-BERT encoder
│   │   │   └── unified_projection.py
│   │   ├── fsq.py                   # Finite Scalar Quantization
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
│   ├── extract_features.py
│   └── download_autobrep_weights.py
└── pretrained/
    └── autobrep/                    # AutoBrep FSQ VAE weights
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

### Pre-compute Text Embeddings (Recommended)

Since the text encoder (LLM) is frozen during training, you can pre-compute text embeddings once and reuse them. This significantly speeds up training and reduces GPU memory usage.

```bash
# Pre-compute embeddings using Phi-4-mini
python scripts/precompute_text_embeddings.py \
    --data-root data/mmcad \
    --model microsoft/Phi-4-mini-instruct \
    --batch-size 8

# This creates HDF5 files in data/mmcad/embeddings/
#   - train_text_embeddings.h5
#   - val_text_embeddings.h5
#   - test_text_embeddings.h5
```

Then enable cached embeddings in your training config:
```bash
python scripts/train.py data.use_cached_text_embeddings=true
```

### Two-Stage Training

CLIP4CAD-H uses two-stage training:

1. **Stage 1** (epochs 1-40): Global contrastive + reconstruction
2. **Stage 2** (epochs 41-100): Add local contrastive alignment

```bash
# Download pretrained weights first
python scripts/download_autobrep_weights.py

# Default training (with live LLM inference)
python scripts/train.py

# Training with pre-computed text embeddings (recommended)
python scripts/train.py data.use_cached_text_embeddings=true

# With pretrained AutoBrep weights
python scripts/train.py \
    model.encoders.brep.surface_checkpoint=pretrained/autobrep/surface_fsq_vae.pt \
    model.encoders.brep.edge_checkpoint=pretrained/autobrep/edge_fsq_vae.pt

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
| B-Rep Face Features | 48 (AutoBrep surfZ) |
| B-Rep Edge Features | 12 (AutoBrep edgeZ) |
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
- [AutoBrep](https://github.com/AutodeskAILab/AutoBrep) - B-Rep FSQ VAE encoder architecture and pretrained weights
- [HCC-CAD](https://github.com/HCC-CAD) - Hierarchical compression concepts
- [OpenShape](https://github.com/Colin97/OpenShape) - Contrastive learning patterns
- [ShapeLLM](https://github.com/ShapeLLM) - Multimodal alignment strategies
