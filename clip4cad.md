# CLIP4CAD-GFA v4.8.2: Simplified Topology-Aware Multimodal Alignment

## Technical Architecture Specification

---

## 1. Executive Summary

**CLIP4CAD-GFA v4.8.2** is a multimodal embedding model that aligns CAD B-Rep geometry, point clouds, and natural language text in a shared 160-dimensional space. It enables text-based retrieval of 3D CAD models.

### Key Innovations

1. **Simplified Topology Encoding**: Uses only `edge_to_faces` topology and `bfs_level` ordering - no spatial fields required
2. **Three-Stage Training**: Anchor → Align → Close Gap paradigm that solves the cold-start problem
3. **Hierarchical Codebook**: 1040 semantic codes organized as category → type → variant + spatial
4. **Smooth Curriculum Learning**: Gradual loss weight transitions to prevent training instability
5. **Gradient Checkpointing**: ~40% memory reduction for larger batch sizes

### Architecture Summary

| Component | Input Dim | Output Dim | Pre-trained |
|-----------|-----------|------------|-------------|
| BRep Encoder | 48 (face) + 12 (edge) | 320 | No |
| PC Encoder | 1024 (ShapeLLM) | 320 | Yes (frozen Stage 0) |
| Text Encoder | 3072 (Phi-4-mini) | 320 | Yes (frozen) |
| Hierarchical Codebook | 320 | 320 | K-means init |
| Output Projection | 320 | 160 | No |

### v4.8.2 Optimizations (vs v4.8.1)

| Parameter | v4.8.1 | v4.8.2 | Change |
|-----------|--------|--------|--------|
| d (internal dim) | 256 | 320 | +25% |
| d_proj (output dim) | 128 | 160 | +25% |
| n_category | 16 | 20 | +25% |
| n_type_per_cat | 8 | 10 | +25% |
| n_spatial | 16 | 20 | +25% |
| num_brep_tf_layers | 4 | 5 | +1 layer |
| num_heads | 8 | 10 | maintains head_dim=32 |
| Total codes | 672 | 1040 | +55% |
| Parameters | ~12M | ~15M | +25% |

---

## 2. Problem Statement

### 2.1 The Modality Gap Problem

Cross-modal contrastive learning produces embeddings that cluster by modality rather than by semantic content. Text embeddings form one cluster, geometry embeddings form another - even when they represent the same CAD model.

```
THE PROBLEM:
─────────────────────────────────────────────────────────────────────
Text embeddings:     [cluster A] ──────── gap ────────► [cluster B] :Geometry embeddings
                     "gear with teeth"                   same gear's BRep

RESULT: Text and geometry for the SAME model are far apart!
```

### 2.2 The Cold Start Problem

BRep encoders start with random weights while text encoders (Phi-4-mini) are pre-trained. Early training produces:
- Meaningless BRep features that don't correlate with geometry
- Text features that already capture semantics
- Gradients that don't provide useful learning signal

### 2.3 Why Spatial Fields Were Removed

Previous versions required:
- `face_centroids`, `face_normals`, `face_areas`
- `edge_midpoints`, `edge_lengths`

**v4.8.2 removes these because:**
1. AutoBrep FSQ features (48-dim faces, 12-dim edges) already encode surface/curve geometry
2. Spatial fields were redundant and added noise
3. Topology (`edge_to_faces`) + ordering (`bfs_level`) capture structural relationships
4. Simpler architecture = better gradients = faster training

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CLIP4CAD-GFA v4.8.2 Architecture                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUTS (Pre-computed)                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                    │
│  │ B-Rep Features │  │ Point Cloud    │  │ Text Features  │                    │
│  │ faces: (F,48)  │  │ local: (N,1024)│  │ (L, 3072)      │                    │
│  │ edges: (E,12)  │  │ global: (1024) │  │ Phi-4-mini     │                    │
│  │ edge_to_faces  │  │ ShapeLLM       │  │                │                    │
│  │ bfs_level      │  │                │  │                │                    │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                    │
│          │                   │                   │                              │
│          ▼                   ▼                   ▼                              │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │                         ENCODERS                                     │       │
│  │                                                                      │       │
│  │  SimplifiedTopologyBRepEncoder    PCEncoder       TextEncoder        │       │
│  │  ┌─────────────────────────┐     ┌──────────┐    ┌──────────────┐   │       │
│  │  │ • face_proj (48→320)    │     │ • proj   │    │ • proj       │   │       │
│  │  │ • edge_proj (12→320)    │     │   (1024  │    │   (3072→320) │   │       │
│  │  │ • bfs_level_emb (32,320)│     │   →320)  │    │ • 2-layer TF │   │       │
│  │  │ • 3× EdgeMessageLayer   │     │          │    │   (10 heads) │   │       │
│  │  │ • 5-layer Transformer   │     │          │    │              │   │       │
│  │  │   (10 heads)            │     │          │    │              │   │       │
│  │  └─────────────────────────┘     └──────────┘    └──────────────┘   │       │
│  │  → X_brep (N_f+N_e, 320)        → X_pc (49, 320) → X_text (L, 320)  │       │
│  │  → positions (N_f+N_e, 320)                                          │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│          │                   │                   │                              │
│          ▼                   ▼                   ▼                              │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │                  HIERARCHICAL CODEBOOK GROUNDING                     │       │
│  │                                                                      │       │
│  │  HierarchicalCodebook: 1040 codes total                              │       │
│  │  ┌─────────────────────────────────────────────────────────────┐    │       │
│  │  │ Level 0: 20 category codes                                   │    │       │
│  │  │ Level 1: 20×10 = 200 type codes (hierarchical)              │    │       │
│  │  │ Level 2: 20×10×4 = 800 variant codes (hierarchical)         │    │       │
│  │  │ Spatial: 20 spatial codes (independent)                      │    │       │
│  │  └─────────────────────────────────────────────────────────────┘    │       │
│  │                                                                      │       │
│  │  HierarchicalCodebookGrounding (×3: text, brep, pc)                 │       │
│  │  • Cross-attention: codes attend to tokens                          │       │
│  │  • Top-k sparse selection                                           │       │
│  │  • Hierarchical gating: category → type → variant                   │       │
│  │  • Position-gated aggregation                                       │       │
│  │                                                                      │       │
│  │  → z_text_raw (320)      → z_brep_raw (320)      → z_pc_raw (320)   │       │
│  │  → w_text (code weights) → w_brep (code weights) → w_pc             │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│          │                   │                   │                              │
│          ▼                   ▼                   ▼                              │
│  ┌─────────────────────────────────────────────────────────────────────┐       │
│  │                      OUTPUT PROJECTION                               │       │
│  │                                                                      │       │
│  │  proj_head: Linear(320) → GELU → Linear(160)                        │       │
│  │                                                                      │       │
│  │  → z_text (160)          → z_brep (160)          → z_pc (160)       │       │
│  │                                                                      │       │
│  │  [For contrastive learning and retrieval]                           │       │
│  └─────────────────────────────────────────────────────────────────────┘       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Specifications

### 4.1 SimplifiedTopologyBRepEncoder

The core innovation of v4.8.2: a BRep encoder that uses **only topology and ordering**.

```python
class SimplifiedTopologyBRepEncoder(nn.Module):
    """
    Encodes B-Rep using ONLY:
    - face_features (48-dim from AutoBrep FSQ)
    - edge_features (12-dim from AutoBrep FSQ)
    - edge_to_faces (topology connections)
    - bfs_level (structural ordering)

    Does NOT use:
    - face_centroids, face_normals, face_areas
    - edge_midpoints, edge_lengths
    """
```

**Architecture:**

| Layer | Input → Output | Description |
|-------|----------------|-------------|
| face_proj | 48 → 320 | Linear + LayerNorm + GELU + Dropout |
| edge_proj | 12 → 320 | Linear + LayerNorm + GELU + Dropout |
| face_type | - | Learned type embedding (1, 1, 320) |
| edge_type | - | Learned type embedding (1, 1, 320) |
| level_emb | 32 → 320 | BFS level embedding |
| msg_layers | 320 → 320 | 3× EdgeMessageLayer |
| transformer | 320 → 320 | 5-layer Pre-LN Transformer (10 heads) |
| position_encoder | 320 → 320 | 2-layer MLP for positions |

**Forward Pass:**
```
1. Project faces: F = face_proj(face_feats) + face_type + level_emb(bfs_level)
2. Project edges: E = edge_proj(edge_feats) + edge_type
3. Message passing: F, E = EdgeMessageLayer(F, E, edge_to_faces) × 3
4. Concatenate: X = [F; E]
5. Transform: X = Transformer(X)
6. Positions: pos = position_encoder(level_emb)
7. Return: X, mask, positions
```

### 4.2 EdgeMessageLayer

Bidirectional message passing through B-Rep topology.

```python
class EdgeMessageLayer(nn.Module):
    """Message passing between faces and edges using edge_to_faces topology."""
```

**Face → Edge:**
```
For each edge e connected to faces (f1, f2):
  msg_e = MLP([E[e], F[f1], F[f2]])
  E[e] = LayerNorm(E[e] + gate_e * msg_e)
```

**Edge → Face:**
```
For each face f:
  incident_edges = {e : edge_to_faces[e] contains f}
  msg_f = mean(E[incident_edges])
  F[f] = LayerNorm(F[f] + gate_f * MLP([F[f], msg_f]))
```

**Gated Residual:**
```python
gate = sigmoid(Linear([X, msg]))
X_new = X + gate * msg
```

### 4.3 HierarchicalCodebook

Semantic codebook with 1040 codes organized hierarchically.

```python
class HierarchicalCodebook(nn.Module):
    """Three-level semantic codebook + spatial codes."""

    def __init__(self, config):
        # Level 0: Category (coarse semantic categories)
        self.category_codes = Parameter(20, 320)

        # Level 1: Type (sub-categories, hierarchical)
        self.type_codes = Parameter(20, 10, 320)  # 200 total

        # Level 2: Variant (fine-grained, hierarchical)
        self.variant_codes = Parameter(20, 10, 4, 320)  # 800 total

        # Spatial: Independent position codes
        self.spatial_codes = Parameter(20, 320)

        # Learnable temperature
        self.log_tau = Parameter(0)  # tau = exp(log_tau) + 0.1
```

**Code Counts:**

| Level | Count | Hierarchical? | Purpose |
|-------|-------|---------------|---------|
| Category | 20 | No | Coarse semantic type (gear, shaft, housing...) |
| Type | 200 | Yes (under category) | Sub-type (spur gear, helical gear...) |
| Variant | 800 | Yes (under type) | Fine-grained variant |
| Spatial | 20 | No | Position/orientation encoding |
| **Total** | **1040** | - | - |

**K-means Initialization:**
```python
def initialize_from_text(self, text_features):
    """Initialize ALL codebook levels from text encoder outputs using K-means."""
    # Level 0: Category codes
    kmeans_cat = KMeans(n_clusters=20)
    kmeans_cat.fit(text_features)
    self.category_codes.data = kmeans_cat.cluster_centers_

    # Level 1: Type codes (hierarchical within category)
    for cat_idx in range(20):
        cat_mask = cat_labels == cat_idx
        kmeans_type = KMeans(n_clusters=10)
        kmeans_type.fit(text_features[cat_mask])
        self.type_codes.data[cat_idx] = kmeans_type.cluster_centers_

    # Level 2: Variant codes (hierarchical within type)
    # ... similar hierarchical clustering

    # Spatial codes (independent)
    kmeans_spatial = KMeans(n_clusters=20)
    kmeans_spatial.fit(text_features)
    self.spatial_codes.data = kmeans_spatial.cluster_centers_
```

### 4.4 HierarchicalCodebookGrounding

Grounds tokens to hierarchical codebook with sparse selection.

```python
class HierarchicalCodebookGrounding(nn.Module):
    """Ground tokens to hierarchical codebook with sparse selection."""
```

**Forward Pass:**

```
1. Project tokens: K = k_proj(X)

2. Category attention:
   attn_cat = softmax(Q_cat @ K^T / sqrt(d))
   H_cat = attn_cat @ X
   w_cat = topk_sparse(||H_cat|| / tau, k=5)

3. Type attention (gated by category):
   attn_type = softmax(Q_type @ K^T / sqrt(d))
   H_type = attn_type @ X
   w_type = topk_sparse(||H_type|| / tau * w_cat, k=10)

4. Variant attention (gated by type):
   attn_var = softmax(Q_var @ K^T / sqrt(d))
   H_var = attn_var @ X
   w_var = topk_sparse(||H_var|| / tau * w_type, k=8)

5. Spatial attention (independent):
   attn_spatial = softmax(Q_spatial @ K^T / sqrt(d))
   H_spatial = attn_spatial @ X
   w_spatial = topk_sparse(||H_spatial|| / tau, k=5)

6. Aggregate:
   z_cat = w_cat @ H_cat
   z_type = w_type @ H_type
   z_var = w_var @ H_var
   z_spatial = w_spatial @ H_spatial

7. Position gating (if positions provided):
   gate = position_gate([z_cat, pos_pooled])
   z_cat, z_type, z_var *= gate

8. Level fusion:
   level_w = softmax(level_weights)  # Learned
   z = level_w[0]*z_cat + level_w[1]*z_type + level_w[2]*z_var + level_w[3]*z_spatial

9. Output projection:
   z = out_proj(z)
```

### 4.5 TextEncoder

Simple projection + transformer for text features.

```python
class TextEncoder(nn.Module):
    def __init__(self, config):
        self.proj = Linear(3072, 320) + LayerNorm(320)
        self.encoder = TransformerEncoder(
            num_layers=2,
            d_model=320,
            nhead=10,
            dim_feedforward=1280,
            norm_first=True  # Pre-LN
        )
        self.norm = LayerNorm(320)
```

### 4.6 PCEncoder

Two-layer MLP for point cloud features.

```python
class PCEncoder(nn.Module):
    def __init__(self, config):
        self.proj = Sequential(
            Linear(1024, 320),
            LayerNorm(320),
            GELU(),
            Dropout(0.1),
            Linear(320, 320),
            LayerNorm(320)
        )

    def forward(self, pc_local, pc_global):
        X = cat([pc_local, pc_global.unsqueeze(1)], dim=1)  # (B, 49, 1024)
        return self.proj(X)  # (B, 49, 320)
```

### 4.7 BRepDecoder

Reconstruction head for auxiliary loss.

```python
class BRepDecoder(nn.Module):
    """Reconstructs pooled face features from grounded embedding."""

    def __init__(self, d=320, d_face=48):
        self.decoder = Sequential(
            Linear(320, 640),
            GELU(),
            Linear(640, 640),
            GELU(),
            Linear(640, 48)
        )
```

### 4.8 Gradient Checkpointing

v4.8.2 supports gradient checkpointing for memory efficiency (~40% reduction):

```python
# Enable gradient checkpointing
model.enable_gradient_checkpointing()

# Checkpoints:
# - BRep encoder transformer layers
# - BRep encoder message passing layers
# - Text encoder transformer layers
```

---

## 5. Three-Stage Training with Smooth Curriculum

### 5.1 Training Paradigm

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  THREE-STAGE TRAINING (37 epochs total)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 0: ANCHOR (12 epochs)           STAGE 1: ALIGN (15 epochs)       │
│  ────────────────────────              ────────────────────────         │
│  BRep ──────► PC (frozen)              Text ◄────► BRep ◄────► PC       │
│        ↓                                      3-way contrastive         │
│  "Learn geometry from                         + codebook                │
│   pre-trained anchor"                         + diversity               │
│                                                                         │
│  • LR: 3e-4 (1 epoch warmup)           • LR: 1e-4 (2 epoch warmup)      │
│  • NO codebook                         • Codebook initialized           │
│  • Cosine: 1.0 → 0.5                   • Code warmup: 3 epochs          │
│  • Contrastive: 0.5 → 1.0              • KL blend: 0 → 0.4 (capped)     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE 2: CLOSE GAP (10 epochs)                                         │
│  ──────────────────────────────                                         │
│  Text ◄════► BRep ◄════► PC                                            │
│       ATP         ATP                                                   │
│                                                                         │
│  "Close absolute modality gap"                                          │
│  + hard negative mining                                                 │
│                                                                         │
│  • LR: 2e-5 (1 epoch warmup)                                            │
│  • ATP/CU/hard_neg warmup: 2 epochs                                     │
│  • Hard negatives re-mined every 3 epochs                               │
│  • Adaptive boost: 1.1-1.5x based on similarity                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Smooth Curriculum Features

v4.8.2 introduces `GFAv482LossSmooth` with:

1. **Dynamic Loss Weights**: Weights interpolate smoothly within each stage
2. **Temperature Annealing**: tau: 0.07 → 0.06 (starts after Stage 0, sqrt decay)
3. **Label Smoothing**: 0.1 in all stages
4. **Conservative KL Blend**: Capped at 0.4 to prevent instability
5. **Warmup Schedulers**: Linear warmup + cosine decay per stage

```python
class GFAv482LossSmooth(nn.Module):
    def get_stage_weights(self, epoch_in_stage, stage):
        """Returns smoothly interpolated loss weights."""

        if stage == 0:
            # Cosine: 1.0 → 0.5, Contrastive: 0.5 → 1.0
            progress = epoch_in_stage / 12
            weights['cosine'] = 1.0 - 0.5 * progress
            weights['contrastive'] = 0.5 + 0.5 * progress

        elif stage == 1:
            # Code warmup over 3 epochs
            code_progress = min(1.0, epoch_in_stage / 3)
            weights['code'] = 0.3 * code_progress

            # KL blend: 0 → 0.4 (capped for stability)
            stage_progress = epoch_in_stage / 15
            weights['use_kl_blend'] = min(0.4, stage_progress * 0.5)

            # Cosine: 0.5 → 0.35 (keep more for stability)
            weights['cosine'] = 0.5 - 0.15 * stage_progress

        elif stage == 2:
            # ATP/CU/hard_neg warmup over 2 epochs
            atp_progress = min(1.0, epoch_in_stage / 2)
            weights['atp'] = 0.5 * atp_progress
            weights['uniform'] = 0.3 * atp_progress
            weights['hard_neg'] = 0.3 * atp_progress
```

### 5.3 Stage 0: Anchor BRep to PC

**Goal:** Give BRep encoder a stable learning target by aligning to pre-trained ShapeLLM.

**Why it works:**
- ShapeLLM is pre-trained and produces meaningful point cloud features
- BRep and PC represent the **same** 3D geometry
- PC features provide a stable anchor for BRep to learn toward

**Key:** PC encoder is **frozen** - only BRep learns.

```python
# Stage 0 forward (no codebook)
def forward_stage0(self, batch):
    X_brep, mask, _ = self.brep_encoder(...)
    X_pc = self.pc_encoder(...)  # Frozen

    z_brep_raw = self.brep_direct_proj(pool(X_brep))
    z_pc_raw = self.pc_direct_proj(pool(X_pc))  # Frozen

    z_brep = self.proj_head(z_brep_raw)
    z_pc = self.proj_head(z_pc_raw)

    return {z_brep, z_pc, z_brep_raw, z_pc_raw, recon, ...}
```

### 5.4 Stage 1: Add Text + Codebook

**Goal:** Learn codebook structure and establish 3-way alignment.

**Steps:**
1. Unfreeze PC encoder
2. Load pre-initialized codebook (K-means from text features)
3. Reset grounding layers to identity
4. Train with 3-way contrastive + code alignment (gradual ramp-up)

**Key:** Code loss ramps up over 3 epochs; KL blend capped at 0.4.

### 5.5 Stage 2: Gap Closing + Hard Negatives

**Goal:** Close the absolute modality gap and improve fine-grained discrimination.

**New losses (with warmup):**
- **L_ATP (Align True Pairs):** Pull BRep/PC toward text (closes gap) - ramps over 2 epochs
- **L_CU (Centroid Uniformity):** Prevent collapse - ramps over 2 epochs
- **L_hard_neg:** Adaptive hard negative mining (1.1-1.5x boost) - ramps over 2 epochs

---

## 6. Loss Functions

### 6.1 Stage 0 Losses (with smooth weights)

```python
def stage0_loss(outputs, weights, tau):
    z_brep = normalize(outputs['z_brep'])
    z_pc = normalize(outputs['z_pc'])  # Detached

    # Contrastive: BRep finds its matching PC (label_smoothing=0.1)
    L_contrastive = CrossEntropy(z_brep @ z_pc.detach().T / tau, labels, label_smoothing=0.1)

    # Alignment: Pull BRep toward PC
    L_align = MSE(z_brep_raw, z_pc_raw.detach())

    # Cosine: Direction matching
    L_cosine = (1 - CosineSim(z_brep_raw, z_pc_raw.detach())).mean()

    # Reconstruction: Auxiliary task
    L_recon = MSE(recon, face_features_pooled)

    # Dynamic weights: cosine 1.0→0.5, contrastive 0.5→1.0
    return weights['contrastive']*L_contrastive + weights['align']*L_align +
           weights['cosine']*L_cosine + weights['recon']*L_recon
```

### 6.2 Stage 1 Losses (with code warmup and KL blend)

```python
def stage1_loss(outputs, weights, tau):
    z_text, z_brep, z_pc = normalize(outputs['z_text', 'z_brep', 'z_pc'])

    # 3-way contrastive (label_smoothing=0.1)
    L_contrastive = InfoNCE_3way(z_text, z_brep, z_pc, tau, label_smoothing=0.1)

    # Cosine alignment (direct)
    L_cosine = (3 - cos(z_text, z_brep) - cos(z_text, z_pc) - cos(z_brep, z_pc)) / 3

    # Code alignment: blend cosine → KL (capped at 0.4)
    kl_blend = weights['use_kl_blend']  # 0 → 0.4 over stage

    L_code_cosine = cosine_code_loss(w_text, w_brep, w_pc)  # Soft
    L_code_kl = kl_code_loss(w_text, w_brep, w_pc).clamp(max=2.0)  # Hard, clamped
    L_code = (1 - kl_blend) * L_code_cosine + kl_blend * L_code_kl
    L_code = L_code.clamp(max=1.5)  # Overall clamp

    # Diversity: Entropy of average code usage
    L_diversity = 1 - entropy(avg_usage) / log(20)

    # Reconstruction (reduced weight)
    L_recon = MSE(recon, face_features_pooled)

    return L_contrastive + weights['cosine']*L_cosine + weights['code']*L_code +
           weights['diversity']*L_diversity + 0.1*L_recon
```

### 6.3 Stage 2 Losses (with ATP warmup and adaptive hard negatives)

```python
def stage2_loss(outputs, weights, tau, hard_negatives):
    # Contrastive
    L_contrastive = InfoNCE_3way(z_text, z_brep, z_pc, tau, label_smoothing=0.1)

    # ATP: Align True Pairs (closes the gap!)
    L_align = (MSE(z_brep_raw, z_text_raw.detach()) + MSE(z_pc_raw, z_text_raw.detach())) / 2

    # CU: Centroid Uniformity (prevents collapse)
    centroids = (z_text_raw + z_brep_raw + z_pc_raw) / 3
    L_uniform = log(exp(-2 * pairwise_dist).sum() / (B*(B-1)))

    # Code alignment (full KL)
    L_code = kl_code_loss(w_text, w_brep, w_pc).clamp(max=10.0)

    # Adaptive hard negatives: 1.1-1.5x boost based on similarity
    L_hard_neg = adaptive_hard_negative_loss(z_brep, z_text, hard_negatives, tau)

    # Weights ramp up over 2 epochs
    return L_contrastive + weights['atp']*L_align + weights['uniform']*L_uniform +
           weights['code']*L_code + weights['diversity']*L_diversity +
           weights['hard_neg']*L_hard_neg
```

### 6.4 Loss Weights Summary (Dynamic)

| Loss | Stage 0 Start | Stage 0 End | Stage 1 | Stage 2 |
|------|---------------|-------------|---------|---------|
| L_contrastive | 0.5 | 1.0 | 1.0 | 1.0 |
| L_align | 0.5 | 0.5 | - | 0→0.5 |
| L_cosine | 1.0 | 0.5 | 0.5→0.35 | 0.1 |
| L_code | - | - | 0→0.3 | 0.3 |
| L_diversity | - | - | 0.1 | 0.1 |
| L_uniform | - | - | - | 0→0.3 |
| L_hard_neg | - | - | - | 0→0.3 |
| L_recon | 0.5 | 0.5 | 0.05 | - |

---

## 7. Configuration

```python
@dataclass
class GFAv482Config:
    """Configuration for CLIP4CAD-GFA v4.8.2 (optimized)."""

    # ═══════════════════════════════════════════════════════════════════
    # Input Dimensions (from pre-trained encoders)
    # ═══════════════════════════════════════════════════════════════════
    d_face: int = 48           # AutoBrep face FSQ features
    d_edge: int = 12           # AutoBrep edge FSQ features
    d_pc: int = 1024           # ShapeLLM point cloud features
    d_text: int = 3072         # Phi-4-mini text features

    # ═══════════════════════════════════════════════════════════════════
    # Model Dimensions (v4.8.2: +25% capacity)
    # ═══════════════════════════════════════════════════════════════════
    d: int = 320               # Internal unified dimension (was 256)
    d_proj: int = 160          # Final embedding dimension (was 128)

    # ═══════════════════════════════════════════════════════════════════
    # Hierarchical Codebook (v4.8.2: more codes)
    # ═══════════════════════════════════════════════════════════════════
    n_category: int = 20               # Level 0: coarse categories (was 16)
    n_type_per_cat: int = 10           # Level 1: 20 * 10 = 200 types (was 8)
    n_variant_per_type: int = 4        # Level 2: 200 * 4 = 800 variants
    n_spatial: int = 20                # Spatial position codes (was 16)
    code_sparsity: float = 0.1         # Activation threshold

    # ═══════════════════════════════════════════════════════════════════
    # Architecture (v4.8.2: deeper)
    # ═══════════════════════════════════════════════════════════════════
    num_heads: int = 10                # Attention heads (was 8, keeps head_dim=32)
    dropout: float = 0.1               # Dropout rate

    # BRep Encoder
    num_msg_layers: int = 3            # EdgeMessageLayer count
    num_brep_tf_layers: int = 5        # BRep transformer layers (was 4)
    max_bfs_levels: int = 32           # Max BFS level for embedding

    # Text Encoder
    num_text_tf_layers: int = 2        # Text transformer layers

    # PC Encoder
    num_pc_tokens: int = 48            # Expected local PC tokens (+ 1 global)

    @property
    def total_codes(self) -> int:
        """Total number of codes across all levels."""
        return (
            self.n_category +                                    # 20
            self.n_category * self.n_type_per_cat +              # 200
            self.n_category * self.n_type_per_cat * self.n_variant_per_type +  # 800
            self.n_spatial                                        # 20
        )  # = 1040
```

---

## 8. Training Hyperparameters

### 8.1 Per-Stage Configuration

| Parameter | Stage 0 | Stage 1 | Stage 2 |
|-----------|---------|---------|---------|
| Epochs | 12 | 15 | 10 |
| Learning Rate | 3e-4 | 1e-4 | 2e-5 |
| Warmup Epochs | 1 | 2 | 1 |
| Batch Size | 256 | 256 | 256 |
| Weight Decay | 0.01 | 0.01 | 0.01 |
| Max Grad Norm | 1.0 | 1.0 | 1.0 |

### 8.2 Loss Function Configuration

```python
criterion = GFAv482LossSmooth(
    lambda_recon=0.5,
    lambda_align=0.5,
    lambda_uniform=0.3,
    lambda_code=0.3,
    lambda_diversity=0.1,
    lambda_hard_neg=0.3,
    label_smoothing=0.1,   # All stages
    tau_start=0.07,        # Initial temperature
    tau_end=0.05,          # Target (actual: 0.06 with conservative annealing)
)
```

### 8.3 Warmup + Cosine Scheduler

```python
from clip4cad.losses.gfa_v4_8_2_losses import get_warmup_cosine_scheduler

scheduler = get_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs=1,       # Linear warmup
    total_epochs=12,       # Per stage
    steps_per_epoch=len(train_loader),
    min_lr_ratio=0.01,     # Final LR = 1% of peak
)

# Step per batch (not per epoch)
for batch in train_loader:
    ...
    scheduler.step()
```

---

## 9. Data Pipeline

### 9.1 Required Inputs

| Field | Shape | Source | Description |
|-------|-------|--------|-------------|
| `face_features` | (B, 192, 48) | AutoBrep FSQ | Encoded face UV grids |
| `edge_features` | (B, 512, 12) | AutoBrep FSQ | Encoded edge curves |
| `face_mask` | (B, 192) | Computed | Valid face indicators |
| `edge_mask` | (B, 512) | Computed | Valid edge indicators |
| `edge_to_faces` | (B, 512, 2) | Topology | Edge-face connectivity |
| `bfs_level` | (B, 192) | BFS traversal | Face ordering depth |
| `pc_local_features` | (B, N, 1024) | ShapeLLM | Local point tokens |
| `pc_global_features` | (B, 1024) | ShapeLLM | Global shape token |
| `text_features` | (B, L, 3072) | Phi-4-mini | Token embeddings |
| `text_mask` | (B, L) | Tokenizer | Valid token indicators |

### 9.2 NOT Required (v4.8.2 Simplification)

| Field | Reason Removed |
|-------|----------------|
| `face_centroids` | Redundant - FSQ encodes geometry |
| `face_normals` | Redundant - FSQ encodes geometry |
| `face_areas` | Redundant - FSQ encodes geometry |
| `edge_midpoints` | Redundant - FSQ encodes geometry |
| `edge_lengths` | Redundant - FSQ encodes geometry |

---

## 10. File Structure

```
clip4cad/
├── models/
│   └── clip4cad_gfa_v4_8_2.py       # Main model file
│       ├── GFAv482Config             # Configuration dataclass
│       ├── CLIP4CAD_GFA_v482         # Main model class
│       ├── SimplifiedTopologyBRepEncoder  # v4.8.2 BRep encoder
│       ├── HierarchicalCodebook      # 1040-code codebook
│       ├── HierarchicalCodebookGrounding  # Sparse grounding
│       ├── EdgeMessageLayer          # Topology message passing
│       ├── TextEncoder               # Text projection + transformer
│       ├── PCEncoder                 # PC projection
│       └── BRepDecoder               # Reconstruction head
│
├── losses/
│   └── gfa_v4_8_2_losses.py         # Loss functions
│       ├── GFAv482LossSmooth         # Smooth curriculum loss (NEW)
│       ├── GFAv482Loss               # Legacy alias to v4.8.1
│       ├── get_warmup_cosine_scheduler  # LR scheduler (NEW)
│       ├── compute_modality_gap      # Gap metric
│       ├── compute_true_pair_cosine  # Cosine metric
│       ├── compute_brep_pc_metrics   # Stage 0 metrics
│       ├── compute_code_diversity    # Code usage metric
│       ├── compute_active_codes      # Active code counts
│       └── mine_hard_negatives_by_code  # Hard negative mining
│
├── data/
│   └── gfa_dataset.py               # Dataset class
│       └── GFAMappedDataset          # Memory-mapped dataset
│
└── training/
    └── gfa_trainer.py               # Trainer class

scripts/
└── initialize_codebook.py           # Offline codebook initialization
    # Default args: --d 320 --n-category 20 --n-type-per-cat 10 --n-spatial 20

notebooks/
└── train_gfa_v4_8_2.ipynb           # Training notebook

outputs/
└── gfa_v4_8_2/                      # Training outputs
    ├── checkpoint_stage0.pt          # After Stage 0
    ├── codebook_initial.pt           # Initial codebook
    ├── checkpoint_stage1.pt          # After Stage 1
    ├── checkpoint_stage2.pt          # After Stage 2
    ├── checkpoint_best.pt            # Best gap model
    └── gfa_v4_8_2_final.pt          # Final model
```

---

## 11. Usage Examples

### 11.1 Model Creation

```python
from clip4cad.models.clip4cad_gfa_v4_8_2 import CLIP4CAD_GFA_v482, GFAv482Config
from clip4cad.losses.gfa_v4_8_2_losses import GFAv482LossSmooth, get_warmup_cosine_scheduler

# Default configuration (v4.8.2 optimized)
config = GFAv482Config()
model = CLIP4CAD_GFA_v482(config).cuda()

# Enable gradient checkpointing for memory efficiency
model.enable_gradient_checkpointing()

print(f"Parameters: {model.count_parameters():,}")  # ~15M
print(f"Total codes: {model.codebook.total_codes}")  # 1040
```

### 11.2 Stage 0 Training

```python
# Freeze PC encoder (anchor mode)
model.freeze_pc_encoder()

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=1, total_epochs=12, steps_per_epoch=len(train_loader))
criterion = GFAv482LossSmooth()

for epoch in range(1, 13):
    for batch in train_loader:
        batch = remap_batch(batch)

        outputs = model.forward_stage0(batch)
        loss, losses = criterion(outputs, stage=0, epoch_in_stage=epoch, global_epoch=epoch, total_epochs=37)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Monitor: BRep-PC cosine should reach 0.85+ by epoch 12
        gap, cos = compute_brep_pc_metrics(outputs['z_brep_raw'], outputs['z_pc_raw'])
```

### 11.3 Pre-compute Codebook

```bash
# Run AFTER Stage 0, generates codebook_initial.pt
python scripts/initialize_codebook.py \
    --text-h5 path/to/text_embeddings.h5 \
    --checkpoint outputs/gfa_v4_8_2/checkpoint_stage0.pt \
    --output outputs/gfa_v4_8_2/codebook_initial.pt \
    --max-samples 50000 \
    --d 320 --n-category 20 --n-type-per-cat 10 --n-spatial 20
```

### 11.4 Stage 1 Training

```python
# Unfreeze and load pre-computed codebook
model.unfreeze_pc_encoder()
model.load_codebook('outputs/gfa_v4_8_2/codebook_initial.pt')
model.reset_grounding_to_identity()

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=2, total_epochs=15, steps_per_epoch=len(train_loader))

for epoch in range(1, 16):
    global_epoch = 12 + epoch  # Continue from Stage 0

    for batch in train_loader:
        batch = remap_batch(batch)

        outputs = model(batch, stage=1)
        loss, losses = criterion(outputs, stage=1, epoch_in_stage=epoch, global_epoch=global_epoch, total_epochs=37)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Monitor: Diversity > 0.3, text-BRep cosine positive within 5 epochs
```

### 11.5 Stage 2 Training

```python
# Mine hard negatives
hard_negatives = mine_hard_negatives_by_code(model, train_loader, device, remap_fn=remap_batch)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=1, total_epochs=10, steps_per_epoch=len(train_loader))

for epoch in range(1, 11):
    global_epoch = 27 + epoch  # Continue from Stage 1

    # Re-mine every 3 epochs
    if epoch > 1 and epoch % 3 == 0:
        hard_negatives = mine_hard_negatives_by_code(model, train_loader, device, remap_fn=remap_batch)

    for batch_idx, batch in enumerate(train_loader):
        batch = remap_batch(batch)
        batch_hard_negs = [hard_negatives.get(batch_idx * B + i) for i in range(B)]

        outputs = model(batch, stage=2)
        loss, losses = criterion(outputs, stage=2, epoch_in_stage=epoch, global_epoch=global_epoch,
                                 total_epochs=37, hard_negatives=batch_hard_negs)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 11.6 Inference

```python
model.eval()
with torch.no_grad():
    # Encode text query
    z_text = model.encode_text({'text_features': text_feats, 'text_mask': text_mask})

    # Encode BRep database
    z_brep = model.encode_brep({
        'face_features': face_feats,
        'edge_features': edge_feats,
        'face_mask': face_mask,
        'edge_mask': edge_mask,
        'edge_to_faces': edge_to_faces,
        'bfs_level': bfs_level,
    })

    # Retrieval
    z_text = F.normalize(z_text, dim=-1)
    z_brep = F.normalize(z_brep, dim=-1)
    similarities = z_text @ z_brep.T
    top_k = similarities.topk(10)
```

---

## 12. Metrics and Targets

### 12.1 Training Metrics

| Metric | Stage 0 Target | Stage 1 Target | Stage 2 Target |
|--------|----------------|----------------|----------------|
| BRep-PC Cosine | ≥ 0.85 | - | - |
| BRep-PC Gap | ↓ | - | - |
| Modality Gap (BRep-Text) | - | decreasing | → 0 |
| True-pair Cosine | - | increasing, positive | → 0.8+ |
| Code Diversity | - | > 0.3 | > 0.5 |
| Active Codes (category) | - | 4-8 | 4-8 |

### 12.2 Evaluation Metrics

| Task | Metric | Expected Range |
|------|--------|----------------|
| Text → BRep | R@1 | 10-30% |
| Text → BRep | R@5 | 30-50% |
| Text → BRep | R@10 | 40-60% |
| Text → PC | R@1 | 15-35% |
| BRep → PC | R@1 | 70-90% |

---

## 13. Version History

| Version | Key Changes |
|---------|-------------|
| v4.4 | Topology-aware encoder + BFS hierarchy |
| v4.8.1 | Three-stage training + PC anchoring + hierarchical codebook (672 codes) |
| **v4.8.2** | **+25% capacity (d=320, 1040 codes), smooth curriculum, gradient checkpointing** |

### v4.8.2 Specific Changes

- **Model**: d: 256→320, d_proj: 128→160, 5 TF layers, 10 heads, 1040 codes
- **Loss**: `GFAv482LossSmooth` with dynamic weights, temperature annealing, label smoothing
- **Training**: Warmup+cosine schedulers, conservative KL blend (max 0.4), adaptive hard negatives
- **Memory**: Gradient checkpointing (~40% reduction)

---

*Document Version: 4.8.2*
*Last Updated: February 2026*
*Architecture: CLIP4CAD-GFA v4.8.2*
