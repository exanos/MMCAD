# CLIP4CAD-GFA v4: Slot Attention with Query Distillation

## Overview

GFA v4 addresses the self-grounding collapse problem by **directly supervising Q_self to match T_feat at the query level**, not just the embedding level.

## Problem with GFA v2.4

In v2.4, even with shared grounding and aggregation, self-grounding collapsed:

```
v2.4 Architecture (STILL BROKEN):
─────────────────────────────────────────────────────────────────────────────
Text → [Text Parser] → T_feat  ──→ [SHARED Grounding] ──→ z_guided
                        (encodes: "serrated teeth", "cylindrical bore")

Geo → [SelfQueryGen] → Q_self  ──→ [SHARED Grounding] ──→ z_self
                        (encodes: ???)

PROBLEM: T_feat and Q_self are in DIFFERENT semantic spaces!
Even shared grounding can't fix this - different inputs → different outputs.
```

**Evidence**:
- UMAP shows self embeddings (orange) isolated from guided+text (green+blue)
- Self-cos BRep: 0.08 (should be ~0.85)
- Self R@1: 0.05% (should be ~60%)

## Solution: Query-Level Supervision

**Key insight**: Directly supervise Q_self to match T_feat **before** grounding.

```
v4 Architecture (FIXED):
─────────────────────────────────────────────────────────────────────────────

T_feat ────────────────────────┬──→ [SHARED Grounding] ──→ z_guided
        ↑                      │
        │  L_query = cosine    │
        │  (THE KEY FIX!)      │
        ↓                      │
Q_self ────────────────────────┴──→ [SHARED Grounding] ──→ z_self
   ↑
[Slot Attention]  ←── iterative refinement, competition between slots
```

**Why this works:**
- L_query forces Q_self to match T_feat semantically
- Same queries + shared grounding = same embeddings
- No more divergence!

## Architecture Components

### 1. SlotAttentionQueryGenerator (Replaces SelfQueryGenerator)

Based on "Object-Centric Learning with Slot Attention" (Locatello et al., NeurIPS 2020).

```python
class SlotAttentionQueryGenerator(nn.Module):
    """
    Advantages over Transformer decoder:
    1. Iterative refinement is more stable
    2. Competition between slots prevents collapse
    3. Attention is explicitly computed (easier to supervise)
    """

    def __init__(self, d, num_slots, num_iterations=3, hidden_dim=None):
        # Learnable slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, d) * 0.02)
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, d))

        # Key, Query, Value projections
        self.to_q, self.to_k, self.to_v = Linear projections

        # GRU for iterative refinement
        self.gru = nn.GRUCell(d, d)

        # MLP for slot update
        self.mlp = 2-layer MLP

        # Confidence prediction
        self.confidence_head = ...

    def forward(self, X_geo, geo_mask=None):
        # Initialize slots (with noise for diversity during training)
        slots = self.slots_mu.expand(B, -1, -1)
        if self.training:
            slots = slots + noise * self.slots_log_sigma.exp()

        # Iterative refinement
        for _ in range(self.num_iterations):
            # Attention: softmax over SLOTS (competition!)
            attn = softmax(q @ k.T, dim=1)  # Key: dim=1, not dim=-1
            updates = attn @ v
            slots = self.gru(updates, slots)
            slots = slots + self.mlp(slots)

        # Final attention for distillation
        attn_weights = softmax(q @ k.T, dim=-1)

        return Q_self, confidence, attn_weights
```

### 2. GFAv4Loss (Query + Attention Distillation)

```python
class GFAv4Loss(nn.Module):
    def __init__(
        self,
        lambda_self=0.05,       # Very low early
        lambda_query=1.5,       # HIGH - THE KEY FIX!
        lambda_attn=0.5,        # Attention pattern distillation
        lambda_embed=0.3,       # Reduced (query distill handles alignment)
        lambda_detail=0.0,
    ):
        ...

    def forward(self, outputs, ...):
        # 1. GUIDED contrastive (InfoNCE 3-way)
        # 2. SELF contrastive (InfoNCE 3-way)

        # 3. QUERY DISTILLATION (THE KEY FIX!)
        T_feat_norm = F.normalize(T_feat.detach(), dim=-1)
        Q_self_norm = F.normalize(Q_self, dim=-1)
        cos_sim = (T_feat_norm * Q_self_norm).sum(dim=-1)  # (B, K)
        L_query = (1 - cos_sim).mean()  # Weighted by confidence

        # 4. ATTENTION DISTILLATION
        L_attn = KL_div(A_self, G_guided.detach())

        # 5. EMBEDDING DISTILLATION (reduced weight)
        # 6. DETAIL contrastive (Stage 2)
```

## Loss Weight Summary

| Loss | Description | Stage 1 | Stage 2 | Notes |
|------|-------------|---------|---------|-------|
| **guided** | InfoNCE 3-way | 1.0 | 1.0 | Primary objective |
| **self** | Self-path InfoNCE | 0.05 | 0.3 | Increased in Stage 2 |
| **query** | Cosine(Q_self, T_feat) | **1.5** | **0.8** | THE KEY FIX! |
| **attn** | KL(A_self, G_guided) | 0.5 | 0.3 | Attention patterns |
| **embed** | Cosine embedding | 0.3 | 0.3 | Reduced (query handles it) |
| **detail** | Hard negative mining | 0.0 | 0.3 | Stage 2 only |

## Training Configuration

### Stage 1: Heavy Query Distillation (Epochs 1-15)

```yaml
stage1_lr: 3.0e-5
stage1_lambda_self: 0.05        # Very low
stage1_lambda_query: 1.5        # HIGH!
stage1_lambda_attn: 0.5
stage1_lambda_embed: 0.3
stage1_lambda_detail: 0.0
```

### Stage 2: Balanced Training (Epochs 16-35)

```yaml
stage2_lr: 1.0e-5
stage2_lambda_self: 0.3
stage2_lambda_query: 0.8        # Still important
stage2_lambda_attn: 0.3
stage2_lambda_embed: 0.3
stage2_lambda_detail: 0.3       # Hard negatives
```

## Expected Results

| Metric | v2.4 (Broken) | v4 (Expected) |
|--------|---------------|---------------|
| Query alignment | N/A | **≥0.70** |
| Self-cos BRep | 0.08 | **0.80-0.90** |
| Self-cos PC | 0.07 | **0.80-0.90** |
| Text→BRep R@1 (guided) | 71.5% | ~70% |
| Text→BRep R@1 (self) | 0.05% | **55-65%** |
| Gap (guided - self) | 71.5% | **<10%** |

## Key Metrics

### 1. Query Alignment (NEW - Most Important!)

```python
def compute_query_alignment(T_feat, Q_self, confidence=None):
    T_feat_norm = F.normalize(T_feat, dim=-1)
    Q_self_norm = F.normalize(Q_self, dim=-1)
    cos_sim = (T_feat_norm * Q_self_norm).sum(dim=-1)  # (B, K)

    if confidence is not None:
        # Weighted by text confidence
        return (cos_sim * confidence).sum() / confidence.sum()
    return cos_sim.mean()
```

**Target**: ≥0.70 after training

### 2. Self-Grounding Quality

```python
def compute_self_grounding_quality(z_guided, z_self):
    z_guided = F.normalize(z_guided, dim=-1)
    z_self = F.normalize(z_self, dim=-1)
    return (z_guided * z_self).sum(dim=-1).mean()
```

**Target**: ≥0.80 after training

## Verification Checklist

1. **Architecture Check**:
   - [ ] Slot attention produces (B, K, d) queries
   - [ ] Attention weights A_self have shape (B, K, N)
   - [ ] Model outputs include T_feat, Q_brep_self, Q_pc_self

2. **Training Metrics**:
   - [ ] `query` loss decreases steadily (target < 0.2 by epoch 15)
   - [ ] Query alignment > 0.5 by epoch 10
   - [ ] Self-cos stays > 0.7 throughout training

3. **Evaluation**:
   - [ ] Query alignment ≥ 0.70
   - [ ] Self-path R@1 within 10% of guided-path R@1
   - [ ] UMAP shows overlapping guided and self embeddings

## File Structure

```
clip4cad/
├── models/
│   ├── __init__.py              # Exports CLIP4CAD_GFA_v4, GFAv4Config
│   └── clip4cad_gfa_v4.py       # Model with Slot Attention
├── losses/
│   ├── __init__.py              # Exports GFAv4Loss
│   └── gfa_v4_losses.py         # Query + Attention distillation
configs/
└── model/
    └── clip4cad_gfa_v4.yaml     # Configuration
notebooks/
├── train_gfa_v4.ipynb           # Training notebook
└── eval_gfa_v4.ipynb            # Evaluation notebook
docs/
└── CLIP4CAD_GFA_v4_Architecture.md  # This file
```

## Usage

```python
from clip4cad.models import CLIP4CAD_GFA_v4, GFAv4Config
from clip4cad.losses import GFAv4Loss
from clip4cad.losses.gfa_v4_losses import compute_query_alignment

# Create config
config = GFAv4Config(
    d_unified=256,
    d_proj=128,
    num_slots=12,
    num_slot_iterations=3,
    slot_hidden_dim=512,
)

# Create model and loss
model = CLIP4CAD_GFA_v4(config).cuda()
criterion = GFAv4Loss(
    lambda_query=1.5,  # THE KEY!
    lambda_attn=0.5,
)

# Forward pass
outputs = model(batch)
loss, loss_dict = criterion(outputs)

# Monitor query alignment (KEY METRIC!)
q_align = compute_query_alignment(
    outputs['T_feat'],
    outputs['Q_brep_self'],
    outputs['confidence']
)
print(f"Query alignment: {q_align:.4f}")  # Should be > 0.70 after training
```

## Why Query Distillation Works

```
BEFORE (v2.4):
─────────────────────────────────────────────────────────────────────────────
Q_self learns to minimize InfoNCE by finding ANY pattern that works.
Even with shared grounding, if Q_self ≠ T_feat semantically,
the embeddings will be in different spaces.

AFTER (v4):
─────────────────────────────────────────────────────────────────────────────
L_query = cosine_distance(Q_self, T_feat.detach())

Q_self is FORCED to produce the same semantic queries as text parser:
  - "serrated teeth" slot in T_feat → similar slot in Q_self
  - "cylindrical bore" slot in T_feat → similar slot in Q_self

Same queries + shared grounding = same embeddings!
```

## Comparison: v2.4 vs v4

| Aspect | v2.4 | v4 |
|--------|------|-----|
| **Query Generator** | Transformer Decoder | **Slot Attention** |
| **Query Supervision** | None | **L_query (λ=1.5)** |
| **Attention Supervision** | Grounding KL only | **Slot attention KL** |
| **Key Loss** | Embedding distill | **Query distillation** |
| **Why it fails/works** | Q_self learns different space | Q_self forced to match T_feat |

## References

- Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020
- Carion et al., "End-to-End Object Detection with Transformers" (DETR), ECCV 2020
- Jaegle et al., "Perceiver: General Perception with Iterative Attention", ICML 2021
