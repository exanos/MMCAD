# CLIP4CAD-GFA v4.8: Gap-Closing Codebook Architecture with Shared Fusion

## Overview

GFA v4.8 represents a major architectural simplification that **directly closes the modality gap** instead of working around it with self-grounding paths. The key innovation is the **SharedFusionNetwork** - a single network with the same weights for all modalities that ensures structural alignment.

### Key Insight

InfoNCE (the standard contrastive loss) only optimizes **relative ranking** - it ensures correct samples rank higher than incorrect ones, but it doesn't minimize the absolute distance between matched pairs. This leaves a persistent gap between modality centroids.

Previous versions (v4.0-v4.4) tried to work around this gap with self-grounding paths. v4.8 **closes the gap directly** with:
1. **L_ATP (Align True Pairs)** - Direct MSE alignment on final embeddings
2. **SharedFusionNetwork** - Same weights ensure same code activations → same embeddings

## Architecture Comparison: v4.4 vs v4.8

| Aspect | v4.4 | v4.8 |
|--------|------|------|
| Self-grounding path | Yes (complex) | **Eliminated** |
| Curriculum learning | Yes (hints 90%→0%) | **No** |
| Number of paths | 2 (guided + self) | **1** (single path) |
| Loss terms | 7 | **6** (well-motivated) |
| Codebook | No | **Yes** (256 shared codes) |
| Fusion Network | Separate proj_heads | **SharedFusionNetwork** |
| Gap handling | Work around it | **Close it directly** |
| Model parameters | ~25M | ~15-20M |

## Architecture Diagram

```
                              CLIP4CAD-GFA v4.8
                         Gap-Closing Codebook Architecture
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │
    │   │    TEXT     │   │    BREP     │   │    PC       │                   │
    │   │  (Phi-4)    │   │ (AutoBrep)  │   │ (ShapeLLM)  │                   │
    │   │  3072-d     │   │  48+12-d    │   │  1024-d     │                   │
    │   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                   │
    │          │                 │                 │                           │
    │          ▼                 ▼                 ▼                           │
    │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │
    │   │   Text      │   │  Topology   │   │    PC       │                   │
    │   │  Encoder    │   │   BRep      │   │  Encoder    │                   │
    │   │ (2 TF)      │   │  Encoder    │   │ (MLP)       │                   │
    │   │             │   │ (MSG+3 TF)  │   │             │                   │
    │   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                   │
    │          │                 │                 │                           │
    │          │      X_text     │      X_brep     │      X_pc                │
    │          │      (B,T,256)  │    (B,N_f+N_e,  │    (B,N+1,256)           │
    │          │                 │        256)     │                           │
    │          ▼                 ▼                 ▼                           │
    │   ┌─────────────────────────────────────────────────────────────┐       │
    │   │                  SHARED SEMANTIC CODEBOOK                    │       │
    │   │                       (256 codes, d=256)                     │       │
    │   │                                                              │       │
    │   │   C = [c_1, c_2, ..., c_256]   ← Learnable code vectors      │       │
    │   │                                                              │       │
    │   │   Codes learn semantic primitives:                           │       │
    │   │   [cylindrical] [planar] [teeth] [hole] [fillet] [thread]   │       │
    │   └─────────────────────────────────────────────────────────────┘       │
    │          │                 │                 │                           │
    │          ▼                 ▼                 ▼                           │
    │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                   │
    │   │   Text      │   │   BRep      │   │    PC       │                   │
    │   │  Grounding  │   │  Grounding  │   │  Grounding  │                   │
    │   │             │   │             │   │             │                   │
    │   │  ┌───────┐  │   │  ┌───────┐  │   │  ┌───────┐  │                   │
    │   │  │Attn(C,│  │   │  │Attn(C,│  │   │  │Attn(C,│  │                   │
    │   │  │X_text)│  │   │  │X_brep)│  │   │  │X_pc)  │  │                   │
    │   │  └───┬───┘  │   │  └───┬───┘  │   │  └───┬───┘  │                   │
    │   │      │      │   │      │      │   │      │      │                   │
    │   │      ▼      │   │      ▼      │   │      ▼      │                   │
    │   │   H_text    │   │   H_brep    │   │    H_pc     │                   │
    │   │   (B,M,d)   │   │   (B,M,d)   │   │   (B,M,d)   │                   │
    │   │   w_text    │   │   w_brep    │   │    w_pc     │                   │
    │   │   G_text    │   │   G_brep    │   │    G_pc     │                   │
    │   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                   │
    │          │                 │                 │                           │
    │          │    ← L_code: KL(w_*, w_text.detach()) →                      │
    │          │                 │                 │                           │
    │          └─────────────────┼─────────────────┘                           │
    │                            │                                             │
    │                            ▼                                             │
    │   ┌─────────────────────────────────────────────────────────────┐       │
    │   │              SHARED FUSION NETWORK (SAME WEIGHTS!)           │       │
    │   │                                                              │       │
    │   │   z = f(H, w) = proj(weighted_sum(H,w) + attn(H))           │       │
    │   │                                                              │       │
    │   │   CRITICAL: If w_text ≈ w_brep, then z_text ≈ z_brep!       │       │
    │   └─────────────────────────────────────────────────────────────┘       │
    │          │                 │                 │                           │
    │          ▼                 ▼                 ▼                           │
    │       z_text            z_brep            z_pc                          │
    │          │                 │                 │                           │
    │          │    ← L_ATP: MSE(z_*, z_text.detach()) →                      │
    │          │    ← L_CU: Centroid Uniformity →                             │
    │          │                 │                 │                           │
    │          ▼                 ▼                 ▼                           │
    │   ┌─────────────────────────────────────────────────────────────┐       │
    │   │                    SHARED PROJECTION HEAD                    │       │
    │   │                   Linear → GELU → Linear                     │       │
    │   │                      (256 → 128)                             │       │
    │   └─────────────────────────────────────────────────────────────┘       │
    │          │                 │                 │                           │
    │          ▼                 ▼                 ▼                           │
    │       z_text            z_brep            z_pc                          │
    │      (B,128)           (B,128)           (B,128)                         │
    │          │                 │                 │                           │
    │          └────────────────┴─────────────────┘                           │
    │                           │                                             │
    │                           ▼                                             │
    │                 ┌─────────────────────┐                                 │
    │                 │  L_contrastive      │                                 │
    │                 │  3-way InfoNCE      │                                 │
    │                 │  (T↔B, T↔P, B↔P)    │                                 │
    │                 └─────────────────────┘                                 │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Semantic Codebook

The shared codebook is the central component that ensures structural alignment across modalities.

```python
class SemanticCodebook(nn.Module):
    def __init__(self, num_codes: int = 256, d: int = 256):
        self.codes = nn.Parameter(torch.randn(num_codes, d) * 0.02)
        self.log_tau = nn.Parameter(torch.zeros(1))  # Temperature
```

**Key features:**
- 256 learnable code vectors
- K-means initialization from text encoder outputs
- Shared across all modalities

### 2. Codebook Grounding

Grounds tokens from any modality to the shared codebook via attention. **Returns per-code features H instead of aggregated z.**

```python
class CodebookGrounding(nn.Module):
    def forward(self, X, codebook, mask):
        # 1. Code-token attention: G = softmax(C @ X.T / sqrt(d))
        # 2. Aggregate per code: H = G @ X
        # 3. Code activation weights: w = softmax(||H|| / tau)
        return H, w, G  # per-code features, weights, grounding matrix
```

**Returns:**
- `H`: (B, M, d) - Per-code aggregated features
- `w`: (B, M) - Code activation weights (used for L_code)
- `G`: (B, M, N) - Grounding matrix (for interpretability)

### 3. SharedFusionNetwork (NEW in v4.8!)

**CRITICAL COMPONENT:** Same weights for all modalities ensures structural alignment.

```python
class SharedFusionNetwork(nn.Module):
    def __init__(self, d: int, d_out: int, num_heads: int = 8):
        # Attention-based aggregation over codes
        self.agg_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.agg_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        # Final projection
        self.proj = nn.Sequential(LayerNorm(d), Linear(d, d), GELU, Linear(d, d_out))

    def forward(self, H: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Weighted sum: z_weighted = sum(w * H)
        # Attention: z_attn = attn(query, H, H)
        # Combine and project
        return self.proj(z_weighted + z_attn)
```

**Why this matters:**
- If `w_text ≈ w_brep` (from L_code alignment)
- And same network weights
- Then `z_text ≈ z_brep` (structural guarantee!)

### 4. Topology-Aware BRep Encoder

Preserves the topology-aware encoding from v4.4.

```python
class TopologyBRepEncoder(nn.Module):
    # Face↔Edge message passing
    # Spatial embeddings (centroids, normals, BFS level)
    # Transformer for global context
```

### 5. Text Encoder

Simple projection + transformer for text tokens.

### 6. PC Encoder

Simple MLP projection for ShapeLLM features.

## Loss Functions

### 6 Well-Motivated Loss Terms

| Loss | Formula | Purpose | Default Weight |
|------|---------|---------|----------------|
| **L_contrastive** | 3-way InfoNCE(z_text, z_brep, z_pc) | Preserve retrieval ranking | 1.0 |
| **L_ATP** | MSE(z_brep, z_text.detach) + MSE(z_pc, z_text.detach) | **Close the gap!** | 0.5 |
| **L_CU** | log(sum exp(-2‖μ_i - μ_j‖²)) | Prevent collapse | 0.3 |
| **L_code** | KL(w_brep ‖ w_text.detach) + KL(w_pc ‖ w_text.detach) | Code alignment | 0.3 |
| **L_diversity** | 1 - entropy(avg_usage) / max_entropy | Use all codes | 0.1 |
| **L_hard_neg** | Boosted negatives (Stage 2) | Fine-grained | 0.3 (S2 only) |

### L_ATP (Align True Pairs): The Key Innovation

Applied to **final embeddings** after SharedFusionNetwork (not raw embeddings).

```python
# Text is anchor - geometry is pulled toward it
# Applied to final z_* (after SharedFusionNetwork)
atp_brep = (z_brep - z_text.detach()).pow(2).sum(-1).mean()
atp_pc = (z_pc - z_text.detach()).pow(2).sum(-1).mean()
losses['atp'] = (atp_brep + atp_pc) / 2
```

This directly minimizes the distance between matched pairs, closing the modality gap.

### L_CU (Centroid Uniformity)

```python
# Push sample centroids apart to prevent collapse
centroids = (z_text + z_brep + z_pc) / 3  # (B, d)
dists_sq = torch.cdist(centroids, centroids).pow(2)
rbf = torch.exp(-2 * dists_sq)
losses['cu'] = torch.log(rbf[mask].sum() / (B * (B - 1)))
```

### L_code: Code Activation Alignment

```python
# Matched samples should activate the same codes
kl_brep = F.kl_div((w_brep + 1e-8).log(), w_text.detach(), reduction='batchmean')
kl_pc = F.kl_div((w_pc + 1e-8).log(), w_text.detach(), reduction='batchmean')
losses['code'] = (kl_brep + kl_pc) / 2
```

## Training

### Stage 1: Gap Closing (Epochs 1-20)

Focus on closing the modality gap and learning good code representations.

```python
STAGE1_WEIGHTS = {
    'lambda_atp': 0.5,
    'lambda_cu': 0.3,
    'lambda_code': 0.3,
    'lambda_diversity': 0.1,
    'lambda_hard_neg': 0.0,  # Disabled in Stage 1
}
```

### Stage 2: Hard Negative Refinement (Epochs 21-35)

Enable hard negative mining for fine-grained discrimination.

```python
STAGE2_WEIGHTS = {
    'lambda_atp': 0.3,
    'lambda_cu': 0.2,
    'lambda_code': 0.2,
    'lambda_diversity': 0.1,
    'lambda_hard_neg': 0.3,  # Enable hard negatives
}
```

### Hard Negative Mining

Hard negatives are mined based on code activation similarity:

```python
hard_negatives = mine_hard_negatives_by_code(model, train_loader, device, top_k=10)
```

Samples that activate similar codes but are different instances are good hard negatives.

## Key Metrics

Track these during training:

| Metric | Target | Description |
|--------|--------|-------------|
| **Modality Gap** | → 0 | Distance between modality centroids |
| **True-Pair Cosine** | → 1.0 | Cosine similarity of matched pairs |
| **Code Diversity** | > 0.8 | Entropy of average code usage |

```python
from clip4cad.losses.gfa_v4_8_losses import (
    compute_modality_gap,
    compute_true_pair_cosine,
    compute_code_diversity,
)

gap_brep, gap_pc = compute_modality_gap(z_text_raw, z_brep_raw, z_pc_raw)
cos_brep, cos_pc = compute_true_pair_cosine(z_text, z_brep, z_pc)
diversity = compute_code_diversity(w_text, w_brep, w_pc)
```

## Usage

### Model Instantiation

```python
from clip4cad.models import CLIP4CAD_GFA_v48, GFAv48Config

config = GFAv48Config(
    d_face=48,
    d_edge=12,
    d_pc=1024,
    d_text=3072,
    d_unified=256,
    d_proj=128,
    num_codes=256,
)

model = CLIP4CAD_GFA_v48(config).to(device)
```

### Codebook Initialization

```python
# Initialize codebook via K-means before training
model.initialize_codebook(train_loader, device, remap_fn=remap_batch)
```

### Training

```python
from clip4cad.losses import GFAv48Loss

criterion = GFAv48Loss(
    lambda_align=0.5,
    lambda_uniform=0.3,
    lambda_code=0.3,
    lambda_diversity=0.1,
    lambda_hard_neg=0.0,
)

outputs = model(batch)
loss, loss_dict = criterion(outputs)
```

## Expected Results

After training:
- Modality gap should decrease to near 0
- True-pair cosine should approach 1.0
- Code diversity should remain > 0.8
- Retrieval metrics comparable to v4.4 with simpler architecture

## Benefits

1. **Simpler architecture** - One path instead of two
2. **No curriculum tuning** - Removes hyperparameter complexity
3. **Faster training** - No self-path computation
4. **Better interpretability** - Codebook shows which features are activated
5. **Direct gap closing** - z_brep ≈ z_text by design
6. **Geometry-only retrieval** - Works naturally when gap is closed

## Files

| File | Description |
|------|-------------|
| `clip4cad/models/clip4cad_gfa_v4_8.py` | Model implementation |
| `clip4cad/losses/gfa_v4_8_losses.py` | Loss functions |
| `notebooks/train_gfa_v4_8.ipynb` | Training notebook |
