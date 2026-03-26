# CLIP4CAD-GFA v4.8.1: Anchored Staged Learning with Hierarchical Codebook

## Overview

GFA v4.8.1 addresses the **cold start problem** in v4.8 through staged training and a hierarchical codebook structure. The key insight is that we need to establish **anchors** before attempting cross-modal alignment.

## Problem: Why v4.8 Training Was Stuck

```
v4.8 Training Logs:
  Epoch 1: gap=2.40, cos=0.038, div=1.000
  Epoch 2: gap=2.40, cos=0.038, div=1.000  <- NO CHANGE
  Epoch 3: gap=2.45, cos=0.030, div=1.000  <- GETTING WORSE
```

**Root Cause:** Everything starts random with no anchor!

| Component | Status |
|-----------|--------|
| Text Encoder (Phi-4) | Pre-trained, meaningful |
| PC Encoder (ShapeLLM) | Pre-trained, meaningful |
| BRep Encoder | **Random noise** |
| Codebook | **Random noise** |

The contrastive loss says "make z_brep similar to z_text", but z_brep is noise! How can noise become meaningful without a stable target?

## Solution: Anchored Staged Learning

### Stage 0: Anchor BRep to PC (5-10 epochs)

**Goal:** Make BRep encoder produce meaningful features.

**Why PC as anchor:**
- ShapeLLM is pre-trained and produces meaningful features
- PC and BRep represent the **same 3D geometry**
- Stable target for BRep encoder to learn toward

```
    BRep --> BRep Encoder --> z_brep
                                 |
                            ALIGN TO PC
                                 |
    PC ----> PC Encoder ----> z_pc (ANCHOR - meaningful)
```

**Losses:**
- `L_contrastive`: InfoNCE(z_brep, z_pc)
- `L_align`: MSE(z_brep, z_pc.detach())
- `L_recon`: MSE(decoder(z_brep), face_features)

**Expected:** BRep-PC gap → 0, cosine → 0.7+

### Stage 1: Add Text + Codebook (15 epochs)

**Goal:** Learn codebook structure and text alignment.

**Process:**
1. Initialize codebook from text encoder (K-means)
2. Enable full forward pass with hierarchical grounding
3. Train with 3-way contrastive + code alignment

```
    Text --> Encoder --> Codebook Grounding --> z_text
    BRep --> Encoder --> Codebook Grounding --> z_brep (now meaningful!)
    PC ----> Encoder --> Codebook Grounding --> z_pc
```

**Losses:**
- `L_contrastive`: 3-way InfoNCE
- `L_code`: Hierarchical KL divergence
- `L_diversity`: Entropy-based utilization
- `L_recon`: Reduced weight (0.3×)

**No gap closing yet** - let contrastive establish relative positions.

**Expected:** Retrieval → 70%+

### Stage 2: Gap Closing + Hard Negatives (10 epochs)

**Goal:** Close absolute gap and improve discrimination.

**New losses:**
- `L_ATP`: Align True Pairs - pulls geometry toward text
- `L_CU`: Centroid Uniformity - prevents collapse
- `L_hard_neg`: Hard negative mining

**Lower LR:** 1e-5 (fine-tuning)

**Expected:** Gap → ~0, cosine → 0.8+

---

## Architecture

### Hierarchical Codebook (672 codes)

```
Category (16 codes) - "What type of feature?"
├── C0: Cylindrical features
├── C1: Planar features
├── C2: Freeform/curved features
├── C3: Edge treatments (fillet/chamfer)
├── C4: Pattern features (array/symmetry)
└── ...

    └── Type (8 per category = 128) - "What specific type?"
        ├── C0.0: Through-hole
        ├── C0.1: Blind-hole
        ├── C0.2: Bore
        ├── C0.3: Shaft
        └── ...

            └── Variant (4 per type = 512) - "What variant?"
                ├── C0.0.0: Small through-hole
                ├── C0.0.1: Large through-hole
                ├── C0.0.2: Threaded through-hole
                └── ...

Spatial (16 codes) - "Where is it?"
├── S0: Center/core
├── S1: Top
├── S2: Bottom
├── S3: Radial/peripheral
└── ...
```

**Total:** 672 codes, but sparse selection uses ~10-50 per sample.

### Sparse Code Selection

```python
# Dense selection (v4.8):
z = sum(w[i] * H[i] for all i)  # All codes contribute

# Sparse selection (v4.8.1):
z = sum(w[i] * H[i] for w[i] > threshold)  # Only active codes
```

This gives:
- Variable number of active codes per sample
- Simple models use fewer codes, complex use more
- Sparse, interpretable representation

### Position-Aware Aggregation

```python
# Without position awareness:
z = level_weights @ [z_cat, z_type, z_var, z_spatial]

# With position awareness:
pos_gate = sigmoid(MLP([z_cat, positions]))
z_cat = z_cat * pos_gate
z_type = z_type * pos_gate
z_var = z_var * pos_gate
z = level_weights @ [z_cat, z_type, z_var, z_spatial]
```

This captures spatial nuances like "chamfer at top" vs "chamfer at bottom".

---

## Model Components

### 1. HierarchicalCodebook

```python
class HierarchicalCodebook(nn.Module):
    category_codes: (16, d)           # Level 0
    type_codes: (16, 8, d)            # Level 1
    variant_codes: (16, 8, 4, d)      # Level 2
    spatial_codes: (16, d)            # Spatial
```

### 2. HierarchicalCodebookGrounding

```python
def forward(X, codebook, positions, mask):
    # Category attention
    G_cat = softmax(Q_cat @ K / sqrt(d))
    H_cat = G_cat @ X
    w_cat = sparse_select(norm(H_cat) / tau)

    # Type attention (gated by category)
    w_type = sparse_select(w_type_raw * w_cat)

    # Variant attention (gated by type)
    w_var = sparse_select(w_var_raw * w_type)

    # Spatial attention (independent)
    w_spatial = sparse_select(norm(H_spatial) / tau)

    # Aggregate with position gating
    z = level_weights @ [z_cat, z_type, z_var, z_spatial]

    return z, {w_cat, w_type, w_var, w_spatial}, G_cat
```

### 3. TopologyBRepEncoder (Enhanced)

Additions from v4.8:
- Gated residual connections
- Position encoding output for grounding
- Returns `(X, mask, positions)`

### 4. BRepDecoder

Lightweight decoder for reconstruction auxiliary loss:
```python
decoder = Linear(d, d*2) -> GELU -> Linear(d*2, d*2) -> GELU -> Linear(d*2, d_face)
```

---

## Loss Functions

### Stage 0 Loss

```python
L = L_contrastive(z_brep, z_pc)
  + lambda_align * MSE(z_brep, z_pc.detach())
  + lambda_recon * MSE(recon, face_features)
```

### Stage 1 Loss

```python
L = L_contrastive_3way(z_text, z_brep, z_pc)
  + lambda_code * KL_hierarchical(w_brep, w_text)
  + lambda_diversity * (1 - entropy(avg_usage))
  + 0.3 * lambda_recon * MSE(recon, face_features)
```

### Stage 2 Loss

```python
L = L_contrastive_3way(z_text, z_brep, z_pc)
  + lambda_align * MSE(z_brep + z_pc, z_text.detach())  # ATP
  + lambda_uniform * centroid_uniformity(centroids)    # CU
  + lambda_code * KL_hierarchical(w_brep, w_text)
  + lambda_diversity * (1 - entropy(avg_usage))
  + lambda_hard_neg * hard_negative_loss(z_brep, z_text)
```

---

## Configuration

```python
@dataclass
class GFAv481Config:
    # Dimensions
    d: int = 256
    d_proj: int = 128
    d_text: int = 3072
    d_face: int = 48
    d_edge: int = 12
    d_pc: int = 1024

    # Hierarchical Codebook
    n_category: int = 16
    n_type_per_cat: int = 8      # 128 type codes
    n_variant_per_type: int = 4  # 512 variant codes
    n_spatial: int = 16
    code_sparsity: float = 0.1   # Activation threshold

    # Architecture
    num_msg_layers: int = 3
    num_brep_tf_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
```

---

## Training Schedule

| Stage | Epochs | LR | Losses | Expected |
|-------|--------|-----|--------|----------|
| 0 | 5-10 | 5e-4 | contrastive, align, recon | BRep-PC cos → 0.7+ |
| 1 | 15 | 1e-4 | contrastive, code, diversity | Retrieval → 70%+ |
| 2 | 10 | 1e-5 | + ATP, CU, hard_neg | Gap → 0, cos → 0.8+ |

---

## Key Differences from v4.8

| Feature | v4.8 | v4.8.1 |
|---------|------|--------|
| Cold start | Everything random | PC anchor first |
| Codebook | 256 flat codes | 672 hierarchical |
| Selection | Dense | Sparse (threshold) |
| Position | Ignored | Position-gated |
| Stages | 2 | 3 |
| BRep aux | None | Reconstruction |
| Code align | Flat KL | Per-level KL |

---

## Files

| File | Description |
|------|-------------|
| `clip4cad/models/clip4cad_gfa_v4_8_1.py` | Model implementation |
| `clip4cad/losses/gfa_v4_8_1_losses.py` | Loss functions |
| `notebooks/train_gfa_v4_8_1.ipynb` | Training notebook |
| `docs/CLIP4CAD_GFA_v4_8_1_Architecture.md` | This documentation |

---

## Usage

```python
from clip4cad.models import CLIP4CAD_GFA_v481, GFAv481Config
from clip4cad.losses import GFAv481Loss

# Create model
config = GFAv481Config()
model = CLIP4CAD_GFA_v481(config).to(device)

# Create loss
criterion = GFAv481Loss()

# Stage 0: Anchor to PC
for epoch in range(10):
    outputs = model.forward_stage0(batch)
    loss, losses = criterion(outputs, stage=0)

# Initialize codebook after Stage 0
model.initialize_codebook(dataloader, device)

# Stage 1: Add text + codebook
for epoch in range(15):
    outputs = model(batch, stage=1)
    loss, losses = criterion(outputs, stage=1)

# Stage 2: Gap closing
criterion.lambda_hard_neg = 0.3
for epoch in range(10):
    outputs = model(batch, stage=2)
    loss, losses = criterion(outputs, stage=2, hard_negatives=hard_negs)
```
