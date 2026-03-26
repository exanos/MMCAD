# CLIP4CAD-GFA v2.4: Shared Encoder Architecture

## Overview

GFA v2.4 addresses the self-grounding collapse problem observed in v2 by **sharing the encoder** (grounding and aggregation modules) between guided and self paths. Only query generation remains separate.

## Problem with GFA v2

In GFA v2, self-grounding had a completely separate encoder:

```
v2 Architecture (BROKEN):
─────────────────────────────────────────────────
Text → [Text Parser] → T_feat → [Grounding A] → [Aggregation A] → z_guided
                                      ↑
                              (DIFFERENT modules)
                                      ↓
Geometry → [Self Encoder] → Q_self → [Grounding B] → [Aggregation B] → z_self
```

**Result**: Self-grounding learned a **different embedding space** that minimized loss but didn't align with the guided space.

**Evidence**:
- Guided Text→BRep R@1: 67.72% (excellent)
- Self Text→BRep R@1: 0.05% (collapsed)
- Self-cos similarity: 0.15 (should be ~0.9)

## Solution: Shared Encoder

The key insight is that **only query generation should be separate**. Grounding and aggregation must be SHARED to guarantee the same embedding space.

```
v2.4 Architecture (FIXED):
─────────────────────────────────────────────────
Text → [Text Parser] → T_feat ───────┐
                                     │
                                     ├→ [SHARED Grounding] → [SHARED Aggregation] → z
                                     │
Geometry → [Self Query Gen] → Q_self ┘
```

## Architecture Components

### 1. SelfQueryGenerator (Only Separate Component)

Minimal capacity module that generates queries for self-grounding:

```python
class SelfQueryGenerator(nn.Module):
    """
    Generates queries from geometry features for self-grounding.

    KEY DESIGN: Minimal capacity to prevent learning different space.
    - Learnable base queries (initialized small: 0.02 std)
    - Lightweight transformer decoder (FFN: d*2 instead of d*4)
    - Confidence head for slot activation
    """

    def __init__(self, d, num_slots, num_layers=2, num_heads=8, dropout=0.1):
        self.base_queries = nn.Parameter(torch.randn(num_slots, d) * 0.02)
        self.query_adapter = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, num_heads, d*2, dropout)  # Smaller FFN!
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.ReLU(),
            nn.Linear(d // 4, 1),
            nn.Sigmoid()
        )
```

### 2. UnifiedGrounding (SHARED)

Single grounding module used by both guided and self paths:

```python
class UnifiedGrounding(nn.Module):
    """
    SHARED grounding for both text-guided and self paths.

    Query projection is SHARED - this is the key to forcing same space!
    Only geometry projections are modality-specific (BRep vs PC).
    """

    def __init__(self, d, d_ground=128):
        self.proj_query = nn.Linear(d, d_ground)  # SHARED for T_feat and Q_self!
        self.proj_brep = nn.Sequential(...)        # Modality-specific
        self.proj_pc = nn.Sequential(...)          # Modality-specific
```

### 3. HierarchicalAggregator (SHARED)

Single aggregation module for both paths:

```python
class HierarchicalAggregator(nn.Module):
    """
    SHARED aggregation producing global + detail embeddings.

    Processes grounded features identically for both paths.
    """

    def forward(self, X_geo, G, confidence, geo_mask=None):
        # Same processing for guided and self paths
        return z_global, z_detail, z_unified
```

### 4. CLIP4CAD_GFA_v2_4 (Main Model)

```python
class CLIP4CAD_GFA_v2_4(nn.Module):
    def __init__(self, config):
        # Input projections
        self.brep_proj = BRepProjection(...)
        self.pc_proj = PCProjection(...)
        self.text_proj = nn.Sequential(...)

        # Text parser (for guided path)
        self.text_parser = TextFeatureParser(...)

        # Self query generator (ONLY SEPARATE COMPONENT)
        self.self_query_gen = SelfQueryGenerator(...)

        # SHARED modules
        self.grounding = UnifiedGrounding(...)
        self.hierarchical_agg = HierarchicalAggregator(...)
        self.proj_head = nn.Sequential(...)

    def encode_geometry(self, X_geo, queries, confidence, modality, geo_mask=None):
        """SHARED encoding for both guided and self paths."""
        G = self.grounding.compute_grounding(queries, X_geo, modality, geo_mask)
        z_global, z_detail, z_unified = self.hierarchical_agg(X_geo, G, confidence, geo_mask)
        z = self.proj_head(z_unified)
        return {'z': z, 'z_detail': z_detail, 'G': G}
```

## Loss Function: GFAv2_4Loss

### Loss Components

| Loss | Description | Weight (Stage 1) | Weight (Stage 2) |
|------|-------------|------------------|------------------|
| **Guided** | InfoNCE 3-way (text, brep, pc) | 1.0 | 1.0 |
| **Self** | InfoNCE 3-way with self-grounded embeddings | 0.05 | 0.2 |
| **Distill** | KL divergence on grounding matrices | 0.5 | 0.3 |
| **Embed Distill** | **MSE** on embeddings (stronger than cosine!) | 1.0 | 0.5 |
| **Conf Align** | MSE on confidence patterns (NEW) | 0.2 | 0.1 |
| **Detail** | Fine-grained contrastive with hard negatives | 0.0 | 0.3 |

### Key Changes from v2

1. **MSE instead of Cosine** for embedding distillation:
   ```python
   # v2 (weak gradient):
   loss = 1 - F.cosine_similarity(z_self, z_guided.detach())

   # v2.4 (strong gradient):
   loss = F.mse_loss(z_self, z_guided.detach())
   ```

2. **Confidence Alignment** (new):
   ```python
   loss_conf = F.mse_loss(conf_self, conf_text.detach())
   ```

3. **Lower λ_self Early**: Prevents self from competing before alignment

## Training Configuration

### Stage 1: Heavy Distillation (Epochs 1-15)

Focus on forcing self path to align with guided path:

```yaml
stage1_lr: 3.0e-5
stage1_lambda_self: 0.05        # Very low - don't let self compete
stage1_lambda_distill: 0.5      # High - grounding alignment
stage1_lambda_embed_distill: 1.0 # Very high - MSE embedding alignment
stage1_lambda_conf_align: 0.2   # Match activation patterns
stage1_lambda_detail: 0.0       # Disabled
```

### Stage 2: Balanced Training (Epochs 16-35)

Increase self weight, add detail loss with hard negatives:

```yaml
stage2_lr: 1.0e-5
stage2_lambda_self: 0.2
stage2_lambda_distill: 0.3
stage2_lambda_embed_distill: 0.5
stage2_lambda_conf_align: 0.1
stage2_lambda_detail: 0.3
```

## Expected Results

| Metric | v2 (Broken) | v2.4 (Expected) |
|--------|-------------|-----------------|
| Self-cos BRep | 0.15 | **0.85-0.95** |
| Self-cos PC | 0.12 | **0.85-0.95** |
| Text→BRep R@1 (guided) | 67.72% | ~67% (maintained) |
| Text→BRep R@1 (self) | 0.05% | **60-65%** |
| Text→PC R@1 (self) | 0.04% | **55-60%** |

## Verification Checklist

1. **Architecture Verification**:
   - [ ] `self.grounding` is same instance for both paths
   - [ ] `self.hierarchical_agg` is same instance for both paths
   - [ ] Only `self.self_query_gen` is separate

2. **Training Metrics**:
   - [ ] `conf_align` loss appears and decreases
   - [ ] `embed_distill` decreases faster than v2 cosine version
   - [ ] Self-cos stays >0.7 throughout training

3. **Retrieval Evaluation**:
   - [ ] Self-path R@1 is close to guided-path R@1
   - [ ] UMAP shows overlapping embeddings (not separate clusters)

## File Structure

```
clip4cad/
├── models/
│   ├── __init__.py              # Exports CLIP4CAD_GFA_v2_4, GFAv2_4Config
│   └── clip4cad_gfa_v2_4.py     # Main model implementation
├── losses/
│   ├── __init__.py              # Exports GFAv2_4Loss
│   └── gfa_v2_4_losses.py       # Loss function implementation
configs/
└── model/
    └── clip4cad_gfa_v2_4.yaml   # Configuration file
notebooks/
└── train_gfa_v2_4.ipynb         # Training notebook
docs/
└── CLIP4CAD_GFA_v2_4_Architecture.md  # This file
```

## Usage

```python
from clip4cad.models import CLIP4CAD_GFA_v2_4, GFAv2_4Config
from clip4cad.losses import GFAv2_4Loss

# Create config
config = GFAv2_4Config(
    d_unified=256,
    d_proj=128,
    num_slots=12,
    num_self_query_layers=2  # Minimal capacity
)

# Create model and loss
model = CLIP4CAD_GFA_v2_4(config).cuda()
criterion = GFAv2_4Loss(
    lambda_self=0.05,
    lambda_embed_distill=1.0,
    lambda_conf_align=0.2
)

# Forward pass
outputs = model(batch)
loss, loss_dict = criterion(outputs)

# Monitor self-grounding quality
from clip4cad.losses import compute_self_grounding_quality_v2_4
self_cos = compute_self_grounding_quality_v2_4(outputs['z_brep'], outputs['z_brep_self'])
print(f"Self-cos: {self_cos:.4f}")  # Should be > 0.85 after training
```

## Comparison Table

| Aspect | v2 | v2.4 |
|--------|-----|------|
| **Grounding Module** | Separate | **SHARED** |
| **Aggregation Module** | Separate | **SHARED** |
| **Query Generator** | Separate | Separate (minimal) |
| **Embedding Distill** | Cosine | **MSE** |
| **Confidence Alignment** | No | **Yes** |
| **Self Embed Space** | Different | **Same** |
| **Expected Self R@1** | 0.05% | **60%+** |
