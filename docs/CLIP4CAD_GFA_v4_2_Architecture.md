# CLIP4CAD-GFA v4.2: Conditional Self-Query Generation with Curriculum Learning

## Overview

GFA v4.2 addresses the self-grounding collapse problem by **teaching the model what T_feat looks like during training**, then gradually removing the hints.

## Problem with v4

In v4, even with query distillation, the self-query generator struggled because:
- It was trying to produce text-like features from geometry **without ever seeing text features**
- The decoder didn't know what the target distribution looked like
- Query distillation loss alone wasn't enough signal

**Evidence**:
```
v2:   PC self-cos = 0.34,  BRep self-cos = 0.15  (PC is 2.3x better)
v2.4: PC self-cos = 0.32,  BRep self-cos = 0.08  (PC is 4x better)
```
PC is better because ShapeLLM features are already multimodal-aligned. BRep has no text alignment.

## Solution: Conditional Hints + Curriculum Learning

**Key insight**: During training, occasionally SHOW the model what T_feat looks like by adding it to the queries. Then gradually remove the hints via curriculum learning.

```
v4 (BROKEN):                              v4.2 (FIXED):
─────────────────────                     ─────────────────────
Q_self ← Decoder(X_geo)                   Q_self ← Decoder(X_geo, T_feat_hint)
         ↓                                         ↓
L_query = cos(Q_self, T_feat)             Curriculum: hint_rate 90% → 0%
         ↓                                         ↓
Model doesn't know target distribution    Model LEARNS target distribution
```

## Architecture Components

### 1. ConditionalSelfQueryGenerator

```python
class ConditionalSelfQueryGenerator(nn.Module):
    """
    Self-query generator with optional text conditioning.

    During training:
    - Some samples get T_feat added to queries (hints)
    - Per-sample dropout controls which samples get hints
    - Curriculum gradually reduces hint rate

    Inference:
    - No hints (fully independent)
    """

    def __init__(self, d, K, num_encoder_layers, num_decoder_layers, num_heads):
        # Geometry encoder (deeper for BRep)
        self.geo_encoder = TransformerEncoder(num_encoder_layers)

        # Base queries (learned)
        self.base_queries = nn.Parameter(torch.randn(K, d) * 0.02)

        # Conditioning projection
        self.cond_proj = nn.Sequential(Linear(d, d), LayerNorm(d))

        # Decoder
        self.decoder = TransformerDecoder(num_decoder_layers)

        # Output projection
        self.output_proj = MLP(d → d*2 → d)

        # Curriculum state
        self.cond_drop_rate = 0.1  # 90% get hints initially

    def forward(self, X_geo, geo_mask=None, T_feat=None):
        # Encode geometry
        Z_geo = self.geo_encoder(X_geo)

        # Start with base queries
        queries = self.base_queries.expand(B, -1, -1)

        # CONDITIONAL HINT (training only)
        if self.training and T_feat is not None:
            keep_cond = (torch.rand(B) > self.cond_drop_rate)
            cond = self.cond_proj(T_feat.detach())  # DETACH!
            queries = queries + cond * keep_cond

        # Decode
        Q = self.decoder(queries, Z_geo)
        Q = self.output_proj(Q)

        return Q, confidence
```

### 2. Curriculum Schedule

```python
def get_cond_dropout(epoch, stage):
    """
    Stage 1 (epochs 1-15):
      Epoch 1-3:   0.1 (90% hints) - learn output distribution
      Epoch 4-7:   0.3 (70% hints) - start independence
      Epoch 8-11:  0.5 (50% hints) - balanced
      Epoch 12-15: 0.7 (30% hints) - mostly independent

    Stage 2 (epochs 16+):
      Always 1.0 (0% hints) - fully independent
    """
```

### 3. Distribution Matching Loss (NEW)

```python
def distribution_matching_loss(Q_brep, Q_pc, T_feat):
    """
    Match batch statistics of Q_self to T_feat.

    This regularizes the feature space during curriculum transition,
    helping Q_self maintain the right distribution even as hints decrease.
    """
    # Per-slot mean across batch
    Q_mean = Q_brep.mean(dim=0)  # (K, d)
    T_mean = T_feat.mean(dim=0)

    # Per-slot std across batch
    Q_std = Q_brep.std(dim=0)
    T_std = T_feat.std(dim=0)

    # Match mean + 0.5 * std
    loss = mse(Q_mean, T_mean) + 0.5 * mse(Q_std, T_std)

    return loss
```

## Loss Weight Summary

| Loss | Description | Stage 1 | Stage 2 |
|------|-------------|---------|---------|
| **guided** | InfoNCE 3-way | 1.0 | 1.0 |
| **self** | Self-path InfoNCE | 0.1 | 0.3 |
| **query** | Cosine(Q_self, T_feat) | **1.5** | 1.0 |
| **embed** | Cosine embedding | 0.3 | 0.3 |
| **dist** | Distribution matching (NEW) | 0.3 | 0.2 |
| **detail** | Hard negatives | 0.0 | 0.3 |

## Model Architecture Differences (BRep vs PC)

| Component | BRep | PC | Reason |
|-----------|------|-----|--------|
| Encoder layers | **4** | 2 | BRep needs more capacity (no multimodal alignment) |
| Decoder layers | **4** | 2 | BRep needs more capacity |
| Initial cond_drop | 0.1 | 0.1 | Same curriculum |

## Expected Behavior During Training

```
Early training (cond=0.1, 90% hints):
  - Model sees T_feat most of the time
  - Learns what kind of output is expected
  - Self-cos should stay HIGH (0.7+)

Mid training (cond=0.5, 50% hints):
  - Model sees T_feat half the time
  - Must learn to generate similar features independently
  - Self-cos may dip slightly but should recover

Late Stage 1 (cond=0.7, 30% hints):
  - Model rarely sees T_feat
  - Must be mostly independent
  - Self-cos should stabilize (0.6-0.8)

Stage 2 (cond=1.0, 0% hints):
  - Fully independent
  - Hard negative mining for discrimination
  - Self-cos should be 0.7-0.9
```

## Expected Results

| Metric | v4 (Broken) | v4.2 (Expected) |
|--------|-------------|-----------------|
| Self-cos BRep | 0.1-0.2 | **0.75-0.90** |
| Self-cos PC | 0.1-0.3 | **0.80-0.90** |
| Query alignment BRep | 0.1-0.2 | **0.65-0.80** |
| Query alignment PC | 0.1-0.3 | **0.70-0.85** |
| Text→BRep R@1 (guided) | 60-70% | ~65-70% |
| Text→BRep R@1 (self) | 0-5% | **55-65%** |
| Gap (guided - self) | 60-70% | **<10%** |

## Why Curriculum Learning Works

```
WITHOUT CURRICULUM (v4):
─────────────────────────────────────────────────────────────────────
Q_self must match T_feat, but never sees examples of T_feat.
It's like asking someone to draw a cat without showing them any cats.
Even with a "match the cat" loss, they don't know what to aim for.

WITH CURRICULUM (v4.2):
─────────────────────────────────────────────────────────────────────
Early: "Here's what T_feat looks like" (90% hints)
       Q_self learns the target distribution

Middle: "Can you do it yourself sometimes?" (50% hints)
        Q_self starts generating independently

Late: "Now do it alone" (0% hints)
      Q_self can produce T_feat-like features independently
```

## File Structure

```
clip4cad/
├── models/
│   ├── __init__.py              # Exports CLIP4CAD_GFA_v4_2, GFAv4_2Config
│   └── clip4cad_gfa_v4_2.py     # Model with ConditionalSelfQueryGenerator
├── losses/
│   ├── __init__.py              # Exports GFAv4_2Loss
│   └── gfa_v4_2_losses.py       # Loss with distribution matching
configs/
└── model/
    └── clip4cad_gfa_v4_2.yaml   # Configuration
notebooks/
├── train_gfa_v4_2.ipynb         # Training with curriculum
└── eval_gfa_v4_2.ipynb          # Evaluation
docs/
└── CLIP4CAD_GFA_v4_2_Architecture.md  # This file
```

## Usage

```python
from clip4cad.models import CLIP4CAD_GFA_v4_2, GFAv4_2Config, get_cond_dropout
from clip4cad.losses import GFAv4_2Loss

# Create config
config = GFAv4_2Config(
    d_unified=256,
    d_proj=128,
    num_slots=12,
    brep_encoder_layers=4,  # Deeper for BRep
    brep_decoder_layers=4,
    pc_encoder_layers=2,    # Lighter for PC
    pc_decoder_layers=2,
)

# Create model and loss
model = CLIP4CAD_GFA_v4_2(config).cuda()
criterion = GFAv4_2Loss(
    lambda_query=1.5,  # High for Stage 1
    lambda_dist=0.3,   # Distribution matching
)

# Training loop with curriculum
for epoch in range(1, total_epochs + 1):
    # Update curriculum
    cond_drop = get_cond_dropout(epoch, stage)
    model.set_cond_dropout(cond_drop)

    for batch in train_loader:
        outputs = model(batch)
        loss, loss_dict = criterion(outputs)
        # ... backward, step ...
```

## Key Differences from v4

| Aspect | v4 | v4.2 |
|--------|-----|------|
| **Query Generator** | Transformer Decoder | Conditional + Curriculum |
| **Conditioning** | None | T_feat hints during training |
| **Curriculum** | None | 90% → 0% hints |
| **Distribution Loss** | None | **λ_dist = 0.3** |
| **BRep Encoder** | 2 layers | **4 layers** |
| **Why it works** | Doesn't work | Model learns target distribution |

## References

- Curriculum Learning: Bengio et al., "Curriculum Learning", ICML 2009
- DETR: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020
- Knowledge Distillation: Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2015 Workshop
