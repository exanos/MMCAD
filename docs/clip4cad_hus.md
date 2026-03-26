# CLIP4CAD-HUS v2: Hierarchical Unified Space

A multimodal representation learning architecture for CAD models using a hierarchical query-based approach.

**Version**: 2.0 (Optimized for fast convergence with unified-dominant loss)

## Overview

CLIP4CAD-HUS is an alternative architecture to CLIP4CAD-GFA that uses a **hierarchical unified space** approach instead of text-grounded feature alignment. Key design principles:

1. **Two-Level Hierarchy**: G=8 global queries capture coarse shape information; M=24 detail queries extract fine-grained features
2. **Modality-Symmetric**: All modalities (B-Rep, Point Cloud, Text) processed through the same pipeline
3. **Learnable Fusion**: Gated combination of global and detail features
4. **Unified-Dominant Loss**: Primary focus on the unified embedding (what retrieval actually uses)
5. **Simplified Training**: 3-term InfoNCE instead of 8-term

---

## Architecture

```
Input Tokens (B-Rep/PC/Text)
        │
        ▼
┌──────────────────────┐
│  Input Projections   │  (face_proj, edge_proj, pc_proj, text_proj)
│  + Type Embeddings   │
└──────────────────────┘
        │
        ▼ X_tokens (N, d)
        │
┌──────────────────────┐
│  Global Attention    │  G=8 queries attend to tokens
│  (ModalityAwareQ)    │
└──────────────────────┘
        │
        ▼ z_global (G, d)
        │
┌──────────────────────┐
│  Detail Attention    │  M=24 queries conditioned on z_global
│  (GlobalConditioned) │
└──────────────────────┘
        │
        ▼ z_detail (M, d)
        │
┌──────────────────────┐
│  Gated Fusion        │  gate * pool(z_global) + (1-gate) * pool(z_detail)
└──────────────────────┘
        │
        ▼ z_unified (d)  → PRIMARY embedding for retrieval
```

---

## Key Components

### 1. ModalityAwareQueries

Shared learnable queries with modality-specific adapters:

```python
# Base queries (shared across modalities)
global_queries: (1, G, d)   # G=8 global queries
detail_queries: (1, M, d)   # M=24 detail queries

# Modality adapters (small residual adaptation)
adapter_brep, adapter_pc, adapter_text: Linear(d, d)

# Forward: query + tanh(adapter(query)) * 0.1
```

The Tanh activation bounds adaptations to ±0.1, preventing modality queries from diverging too far from the shared base.

**Why Modality-Aware?**
- B-Rep tokens are structured (faces/edges with type embeddings)
- Point Cloud tokens have spatial locality (local patches + global tokens)
- Text tokens are sequential (LLM hidden states)

Each modality needs slightly different query patterns to extract relevant features.

### 2. GlobalConditionedDetailAttention

Detail queries are modulated by global context before attending to tokens:

```python
# Pool global features
global_pooled = z_global.mean(dim=1)  # (B, d)

# Generate condition signal
condition = condition_proj(global_pooled)  # (B, d)

# Modulate detail queries (additive)
detail_queries_cond = detail_queries + condition.unsqueeze(1)

# Attend to tokens
z_detail = attention(detail_queries_cond, X_tokens, X_tokens)
```

This allows detail features to be guided by global shape understanding (e.g., "this is a gear, so focus on tooth details").

### 3. Gated Fusion

Learnable balance between global and detail per sample:

```python
# Pool each level
z_g_pool = z_global.mean(dim=1)   # (B, d)
z_d_pool = z_detail.mean(dim=1)   # (B, d)

# Predict gate per sample
gate = sigmoid(gate_proj(concat(z_g_pool, z_d_pool)))  # (B, 2)

# Fuse (weighted by 2-element softmax)
z_unified = gate[:, 0] * z_g_pool + gate[:, 1] * z_d_pool
```

Gate values provide interpretability:
- **High gate[0]**: More global influence (shape category matters)
- **High gate[1]**: More detail influence (fine-grained features matter)

---

## Comparison: GFA vs HUS

| Aspect | GFA | HUS |
|--------|-----|-----|
| **Architecture** | Text parsing → Grounding matrices | Global queries → Detail queries |
| **Text Role** | Creates grounding signals for geometry | Equal modality with same processing |
| **Levels** | Single (grounded features) | Two (Global + Detail) |
| **Queries** | K=12 text-parsed slots | G=8 global + M=24 detail |
| **Loss Terms** | 8 terms | 3 terms |
| **Geometry Inference** | Requires `self_ground_queries` | Direct (same pipeline works) |
| **Parameters** | ~3.5M | ~2.7M |
| **Training Time** | ~4 hours (35 epochs) | ~3.5 hours (35 epochs) |

---

## Loss Function

HUS uses a simplified **unified-dominant** 3-term loss:

```
L_total = λ_unified * L_unified + λ_global * L_global + λ_detail * L_detail
```

Where each term is a 3-way InfoNCE loss over pairs:
- B-Rep ↔ Text
- PC ↔ Text
- B-Rep ↔ PC

### InfoNCE Implementation

```python
def infonce_3way(z_a, z_b, z_c, tau):
    """3-way InfoNCE with numerical stability."""
    # Cast to FP32 (critical for AMP training)
    z_a = F.normalize(z_a.float(), dim=-1)
    z_b = F.normalize(z_b.float(), dim=-1)
    z_c = F.normalize(z_c.float(), dim=-1)
    tau = tau.float().clamp(min=0.02)  # Minimum temperature

    B = z_a.shape[0]
    labels = torch.arange(B, device=z_a.device)

    loss = 0
    # Text-centric pairs (2 pairs)
    for zi, zj in [(z_a, z_c), (z_b, z_c)]:
        logits = (zi @ zj.T / tau).clamp(-100, 100)  # Clamp for stability
        loss += F.cross_entropy(logits, labels, label_smoothing=0.05)
        loss += F.cross_entropy(logits.T, labels, label_smoothing=0.05)

    # Geometry-geometry pair
    logits_geo = (z_a @ z_b.T / tau).clamp(-100, 100)
    loss += F.cross_entropy(logits_geo, labels, label_smoothing=0.05)
    loss += F.cross_entropy(logits_geo.T, labels, label_smoothing=0.05)

    return loss / 6  # Average over 3 pairs × 2 directions
```

**Numerical Stability Features:**
- FP32 casting for normalization and similarity computation
- Temperature clamping (min 0.02) to prevent division by near-zero
- Logit clamping (-100, 100) to prevent overflow in exp()
- Label smoothing (0.05) for softer targets

### Loss Weights (v2 - Optimized)

**Key Insight:** The **unified** embedding is what retrieval actually uses. Global and detail are auxiliary losses that ensure the hierarchical structure exists, but should NOT dominate training.

| Weight | Stage 1 | Stage 2 | Rationale |
|--------|---------|---------|-----------|
| λ_unified | **1.0** | **1.0** | PRIMARY - this is the retrieval embedding |
| λ_global | 0.2 | 0.1 | Mild regularizer (ensures coarse structure) |
| λ_detail | 0.2 | 0.5 | Mild regularizer (increased in Stage 2 for hard negs) |
| label_smoothing | 0.05 | 0.05 | Light smoothing for stability |

**Previous Mistake (v1):** We tried global-dominant training (λ_global=1.0, λ_unified=0.5) to "establish hierarchy first". This was **slower** because the model was optimizing for auxiliary losses instead of the primary retrieval objective.

**v2 Fix:** Unified-dominant from the start. The hierarchy emerges naturally as global/detail regularizers guide the structure.

---

## Training Protocol

### Two-Stage Curriculum

**Stage 1 (Epochs 1-15): Unified Space Learning**
- Standard InfoNCE with all modality pairs
- Unified-dominant loss (λ_unified=1.0, others=0.2)
- High learning rate (3e-4) for fast convergence
- Focus on learning the primary retrieval embedding

**Stage 2 (Epochs 16-35): Hard Negative Mining**
- Mine hard negatives using **detail embeddings** (not unified)
- Increase detail weight to 0.5
- Reduce learning rate by 0.3× (→ ~1e-4)
- Fine-tune for difficult discriminations

### Why Mine with Detail Embeddings?

Detail embeddings capture fine-grained features that differentiate similar shapes (e.g., 32 vs 64 teeth on a gear). Mining hard negatives at this level improves discrimination where it matters most.

```python
# Stage 2 hard negative mining
embedding_key = 'z_brep_detail'  # Not 'z_brep' (unified)
```

### Training Stability

**NaN Handling:**
- Skip batches with NaN loss or gradients (don't update weights)
- Monitor NaN rate per epoch (warn if >10%)
- FP32 loss computation prevents most NaN issues

**Gradient Clipping:**
- Max gradient norm: 1.0
- Prevents explosion from hard batches

**Learning Rate Schedule:**
- 2-epoch linear warmup (critical for LR=3e-4)
- Cosine decay to min_lr=1e-6
- Stage 2 reduction by 0.3× for stability

---

## Configuration

```yaml
# Core dimensions
d_unified: 256      # Unified embedding dimension
d_proj: 128         # Projection dimension for contrastive loss
num_global: 8       # G=8 global queries
num_detail: 24      # M=24 detail queries
num_heads: 8        # Attention heads (must divide d_unified)
dropout: 0.1

# Temperatures (learnable, per-level)
tau_global_init: 0.07    # Global level (coarse discrimination)
tau_detail_init: 0.05    # Detail level (fine-grained discrimination)
tau_unified_init: 0.07   # Unified (final retrieval embedding)

# Training - OPTIMIZED FOR FAST CONVERGENCE
training:
  num_epochs_stage1: 15
  num_epochs_stage2: 20

  # Optimization
  batch_size: 64
  learning_rate: 3.0e-4       # HIGH LR for fast convergence
  weight_decay: 0.01          # Low to allow fast learning
  warmup_epochs: 2            # Longer warmup for stability with high LR
  min_lr: 1.0e-6
  max_grad_norm: 1.0

  # Stage 2
  stage2_lr_factor: 0.3       # Reduce to ~1e-4

  # Loss weights - UNIFIED-DOMINANT
  lambda_unified: 1.0         # PRIMARY
  lambda_global: 0.2          # Mild regularizer
  lambda_detail: 0.2          # Mild regularizer
  label_smoothing: 0.05       # Light smoothing (0.1 was too aggressive)

  # Stage 2 weights
  lambda_unified_stage2: 1.0  # Still primary
  lambda_global_stage2: 0.1   # Reduce further
  lambda_detail_stage2: 0.5   # Increase for hard negatives

  # Hard negative mining
  hard_neg_k: 20
  hard_neg_text_threshold: 0.8

  # Data loading
  num_workers: 0              # Use 0 for memory-mapped datasets
  pin_memory: true

  # Checkpointing
  save_every: 5
  validate_every: 5
  empty_cache_every_epoch: true
```

---

## Usage

### Training

**CLI:**
```bash
# Full training
python scripts/train_hus.py \
    --config configs/model/clip4cad_hus.yaml \
    --output-dir outputs/hus \
    --data-root d:/Defect_Det/MMCAD/data \
    --pc-file c:/Users/User/Desktop/pc_embeddings_full.h5 \
    --brep-file c:/Users/User/Desktop/brep_features.h5 \
    --text-file c:/Users/User/Desktop/text_splits/

# Resume training
python scripts/train_hus.py --resume outputs/hus/checkpoint_epoch15.pt
```

**Notebook:**
```python
# See: notebooks/train_hus.ipynb
# 1. Load datasets to RAM (run once)
# 2. Configure training
# 3. Run 2-stage training
# 4. Evaluate checkpoints
```

### Inference

```python
from clip4cad.models.clip4cad_hus import CLIP4CAD_HUS_v2
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

# Load model
config = OmegaConf.load("configs/model/clip4cad_hus.yaml")
model = CLIP4CAD_HUS_v2(config)
checkpoint = torch.load("outputs/hus/checkpoint_epoch35.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model = model.eval().cuda()

# Forward pass (with all modalities)
outputs = model(batch)

# Extract embeddings (ALWAYS USE UNIFIED FOR RETRIEVAL)
z_brep = outputs["z_brep"]      # (B, 128) - Unified embedding
z_text = outputs["z_text"]      # (B, 128)
z_pc = outputs["z_pc"]          # (B, 128)

# Normalize for retrieval
z_brep_norm = F.normalize(z_brep, p=2, dim=-1)
z_text_norm = F.normalize(z_text, p=2, dim=-1)

# Compute similarity
sim = torch.mm(z_text_norm, z_brep_norm.T)  # (B, B)

# Top-K retrieval
scores, indices = torch.topk(sim, k=10, dim=1)

# Interpretability: gate values
gate_brep = outputs["gate_brep"]  # (B, 2) - [global_weight, detail_weight]
print(f"Global importance: {gate_brep[:, 0].mean():.3f}")
print(f"Detail importance: {gate_brep[:, 1].mean():.3f}")
```

### Geometry-Only Inference (No Text!)

Unlike GFA, HUS can encode geometry without needing text:

```python
# Encode geometry only
geo_outputs = model.encode_geometry_inference(batch)
z_brep = geo_outputs["z_brep"]  # Same unified embedding
z_pc = geo_outputs["z_pc"]

# This works because queries are modality-aware, not text-dependent
```

### Retrieval Example

```python
def retrieve_top_k(query_emb, gallery_emb, k=10):
    """Retrieve top-k similar items."""
    # Normalize
    query_norm = F.normalize(query_emb, p=2, dim=-1)
    gallery_norm = F.normalize(gallery_emb, p=2, dim=-1)

    # Similarity
    sim = torch.mm(query_norm, gallery_norm.T)

    # Top-K
    scores, indices = torch.topk(sim, k=k, dim=1)
    return scores, indices

# Text-to-BRep retrieval
scores, indices = retrieve_top_k(z_text, z_brep, k=10)

# BRep-to-PC retrieval (cross-modal geometry)
scores, indices = retrieve_top_k(z_brep, z_pc, k=10)
```

---

## Output Keys

The model forward pass returns:

```python
{
    # ===== PRIMARY EMBEDDINGS (USE THESE FOR RETRIEVAL) =====
    "z_brep": (B, d_proj),       # Unified B-Rep embedding
    "z_pc": (B, d_proj),         # Unified Point Cloud embedding
    "z_text": (B, d_proj),       # Unified Text embedding

    # ===== HIERARCHICAL EMBEDDINGS (for analysis) =====
    # Global level (coarse shape)
    "z_brep_global": (B, d_proj),
    "z_pc_global": (B, d_proj),
    "z_text_global": (B, d_proj),

    # Detail level (fine-grained features)
    "z_brep_detail": (B, d_proj),
    "z_pc_detail": (B, d_proj),
    "z_text_detail": (B, d_proj),

    # ===== LEARNABLE PARAMETERS =====
    "tau_unified": float,        # Temperature for unified loss
    "tau_global": float,         # Temperature for global loss
    "tau_detail": float,         # Temperature for detail loss

    # ===== INTERPRETABILITY =====
    "gate_brep": (B, 2),         # [global_weight, detail_weight]
    "gate_pc": (B, 2),
    "gate_text": (B, 2),

    # Optional: attention maps (if return_attn=True)
    "attn_brep_global": (B, num_heads, G, N_tokens),
    "attn_brep_detail": (B, num_heads, M, N_tokens),
    # ... (same for pc, text)
}
```

---

## Data Compatibility

HUS uses the same pre-computed embeddings as GFA:

### File Formats

| Data | File | Format | Dimensions |
|------|------|--------|------------|
| B-Rep faces | `brep_features.h5` | HDF5 | (166400, 192, 48) |
| B-Rep edges | `brep_features.h5` | HDF5 | (166400, 512, 12) |
| B-Rep masks | `brep_features.h5` | HDF5 | (166400, 192), (166400, 512) |
| PC local | `pc_embeddings_full.h5` | HDF5 | (275820, 32, 1024) |
| PC global | `pc_embeddings_full.h5` | HDF5 | (275820, 16, 1024) |
| Text embeddings | `text_splits/*.h5` | HDF5 | (N, 256, 3072) |

### Encoder Pipeline

```python
# B-Rep: AutoBrep FSQ VAE features
face_features (192, 48) + edge_features (512, 12)
    ↓ face_proj, edge_proj
concat → brep_tokens (704, d)

# Point Cloud: ShapeLLM pre-computed
local_features (32, 1024) + global_token (16, 1024)
    ↓ proj_pc_local, proj_pc_global
concat → pc_tokens (48, d)

# Text: Phi-4-mini hidden states
desc_embeddings (256, 3072)
    ↓ proj_text
text_tokens (256, d)
```

**No re-encoding required** when switching from GFA to HUS.

---

## File Structure

```
clip4cad/
├── models/
│   ├── clip4cad_hus.py           # Main model (CLIP4CAD_HUS_v2)
│   └── encoders/                 # Reused from GFA
├── losses/
│   └── hus_losses.py             # HUSLoss (3-term)
├── training/
│   ├── hus_trainer.py            # HUSTrainer (2-stage)
│   └── hard_negative_mining.py   # Reused from GFA
├── data/
│   └── gfa_dataset.py            # Reused (GFAMappedDataset)
configs/
└── model/
    └── clip4cad_hus.yaml         # Configuration
scripts/
└── train_hus.py                  # CLI training entry point
notebooks/
├── train_hus.ipynb               # Training notebook
└── test_hus_ablations.ipynb      # Inference & evaluation
docs/
└── clip4cad_hus.md               # This file
```

---

## Expected Performance

| Metric | GFA (baseline) | HUS (expected) |
|--------|----------------|----------------|
| Text→BRep R@1 | 55-60% | 55-60% (maintained) |
| Text→BRep R@5 | 75-80% | 75-80% |
| Text→PC R@1 | 60-65% | 60-65% |
| BRep→PC R@1 | 70-75% | 70-75% |
| Training time | ~4 hours | ~3.5 hours (simpler loss) |
| Parameters | ~3.5M | ~2.7M (fewer projections) |
| Stage 1 convergence | 9.05 → 6.23 (15 epochs) | Target: <6.0 with new config |

HUS should match GFA performance with:
- ✅ Simpler architecture (no text parsing)
- ✅ Better interpretability (gate values)
- ✅ Faster training (fewer loss terms)
- ✅ Direct geometry inference (no self_ground_queries)

---

## Interpretability

### Gate Values

Gate values reveal what the model relies on for each sample:

```python
gate_brep = outputs["gate_brep"]  # (B, 2)

# Analyze distribution
global_weight = gate_brep[:, 0].mean()
detail_weight = gate_brep[:, 1].mean()

print(f"Global importance: {global_weight:.3f}")
print(f"Detail importance: {detail_weight:.3f}")
```

**Interpretation:**
- **High gate[0] (>0.6)**: Model uses more global features
  - Examples: Simple shapes, category-level retrieval
- **High gate[1] (>0.6)**: Model uses more detail features
  - Examples: Complex shapes with fine-grained differences (gears, threads)
- **Balanced (~0.5 each)**: Both levels contribute equally

**Use Cases:**
- **Debugging**: Why did this retrieval fail? Check if model used wrong level.
- **Dataset Analysis**: What kinds of shapes need more detail?
- **Model Improvement**: If gates are always extreme (0 or 1), fusion might be broken.

### Attention Visualization

```python
# Forward with attention maps
outputs = model(batch, return_attn=True)

# Extract attention for B-Rep
attn_global = outputs["attn_brep_global"]  # (B, num_heads, G, 704)
attn_detail = outputs["attn_brep_detail"]  # (B, num_heads, M, 704)

# Visualize which tokens each query attends to
import matplotlib.pyplot as plt

# Average over heads and batch
attn_avg = attn_global[0].mean(dim=0)  # (G, 704)

# Plot
plt.figure(figsize=(12, 4))
plt.imshow(attn_avg.cpu(), aspect='auto', cmap='hot')
plt.xlabel("Token Index (704 = 192 faces + 512 edges)")
plt.ylabel("Global Query Index (0-7)")
plt.colorbar(label="Attention Weight")
plt.title("Global Queries Attention to B-Rep Tokens")
plt.tight_layout()
plt.show()
```

---

## Training Convergence Notes

### Convergence Timeline (v2 Config)

**Expected Stage 1 (LR=3e-4):**
```
Epoch  1: loss ~9.5
Epoch  3: loss ~7.5  (converging fast)
Epoch  5: loss ~6.5
Epoch 10: loss ~5.5
Epoch 15: loss ~5.0  (ready for Stage 2)
```

**Expected Stage 2 (LR=1e-4, hard negatives):**
```
Epoch 16: loss ~5.2  (slight increase from hard negatives)
Epoch 20: loss ~4.8
Epoch 25: loss ~4.5
Epoch 30: loss ~4.3
Epoch 35: loss ~4.1  (final)
```

### Troubleshooting

**Problem: Loss not decreasing**
- Check learning rate (should be 3e-4 in Stage 1)
- Check loss weights (unified should be 1.0)
- Verify data loading (check batch contents)

**Problem: NaN loss**
- Reduce learning rate (try 1e-4 instead of 3e-4)
- Check temperature clamps (min 0.02)
- Verify label smoothing is enabled (0.05)
- Check for corrupted data samples

**Problem: Training too slow**
- Increase learning rate (try 5e-4)
- Reduce regularizers (try λ_global=0.1, λ_detail=0.1)
- Check batch size (larger = more stable gradients)

**Problem: Gate values stuck at 0.5**
- Fusion module might not be learning
- Try initializing gate_proj with small weights
- Check if global and detail embeddings are too similar

---

## Ablation Studies (Planned)

Future experiments to validate design choices:

1. **Query Count**: Compare G=4/8/16, M=12/24/48
2. **Loss Weights**: Unified-only vs. unified-dominant vs. balanced
3. **Conditioning**: Global conditioning vs. no conditioning for detail queries
4. **Fusion**: Gated vs. concatenation vs. learned weighted sum
5. **Hard Negative Mining**: Detail-level vs. unified-level mining

Results should be documented in `notebooks/hus_ablations/`.

---

## Citation

If you use CLIP4CAD-HUS in your research, please cite:

```bibtex
@article{clip4cad_hus2025,
  title={CLIP4CAD-HUS: Hierarchical Unified Space for Multimodal CAD Retrieval},
  author={Your Name},
  year={2025},
  note={Technical Report}
}
```

---

## Version History

### v2.0 (2025-01-28)
- **Optimized Training**: Unified-dominant loss (λ_unified=1.0, regularizers=0.2)
- **Faster Convergence**: LR=3e-4 with 2-epoch warmup
- **Numerical Stability**: Label smoothing, logit clamping, FP32 loss computation
- **Increased Capacity**: G=8 global queries, M=24 detail queries

### v1.0 (2025-01-27)
- Initial implementation
- G=4 global queries, M=16 detail queries
- Global-dominant loss (didn't work well)
- LR=3e-5 (too slow)

---

## Acknowledgments

- **GFA Architecture**: Foundation for data pipeline and hard negative mining
- **AutoBrep**: Pre-trained B-Rep feature extractor
- **ShapeLLM**: Point cloud encoder
- **Phi-4-mini**: Text encoder

---

**Document Version**: 2.0
**Last Updated**: 2025-01-28
**Model Checkpoint**: `outputs/hus/checkpoint_epoch35.pt`
