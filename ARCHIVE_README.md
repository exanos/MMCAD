# CLIP4CAD-GFA Project Archive

**Date Archived:** 2026-03-26
**Purpose:** Complete documentation for porting to new system

---

## Quick Start

### What Works ✅
- **GFA v2.4:** 54.8% Text→BRep R@1 (baseline)
- **Architecture:** Text-guided grounding + self-grounding
- **Files:** `clip4cad/models/clip4cad_gfa_v2_4.py`
- **Training:** `notebooks/train_gfa_v2_4.ipynb`

### What Failed ❌
- **GFA v4.8.2:** 0% retrieval (codebook bottleneck)
- **Root cause:** Discrete codes destroy instance discrimination
- **Evidence:** Cos=0.985 for ALL pairs, not just true pairs
- **Lesson:** Don't use codebook for retrieval

### What's Next 🔄
- **GFA v4.9:** Designed but not implemented
- **Plan:** Remove codebook, pure contrastive (CLIP-style)
- **Expected:** 50-60% R@1
- **Files:** See `C:\Users\User\.claude\plans\atomic-herding-adleman.md`

---

## Essential Documents

### Primary Reference
📄 **`docs/EXPERIMENTAL_RESULTS_MASTER.md`**
- Complete experiment history
- All architecture versions (v2.4 → v4.8.2)
- Training logs with results
- What works and what doesn't
- Hardware configuration
- **READ THIS FIRST**

### Architecture Specs
1. `docs/CLIP4CAD_GFA_v2_4_Architecture.md` - Working baseline ✅
2. `docs/CLIP4CAD_GFA_v4_Architecture.md` - Codebook intro
3. `docs/CLIP4CAD_GFA_v4_8_Architecture.md` - Expanded codebook
4. `docs/CLIP4CAD_GFA_v4_9_Architecture.md` - Proposed fix (NO codebook) 🔄

### Other Important Docs
- `docs/autobrep_features.md` - B-Rep feature extraction
- `docs/TOPOLOGY_FIX_REFERENCE.md` - Topology encoding
- `clip4cad.md` - Project overview
- `modalitygap.pdf` - Key paper on gap-closing

---

## Key Findings (TL;DR)

### ✅ What Works
1. **Direct contrastive alignment** (InfoNCE) - foundational
2. **Modality-specific projections** - B-Rep ≠ PC semantics
3. **Joint self-grounding training** - direct loss, not just distillation
4. **FP16 + memory-mapped data** - essential for 137K dataset
5. **BFS-level topology encoding** - helps encoder quality

### ❌ What Doesn't Work
1. **Codebook for retrieval** - information bottleneck → 0% R@1
2. **Gap-closing for retrieval** - helps clustering, not discrimination
3. **Strong grounding constraints** - over-constrains, hurts performance
4. **Self-grounding via distillation only** - too weak, need contrastive

### 📊 Performance Summary

| Version | Approach | Text→BRep R@1 | Status |
|---------|----------|---------------|--------|
| v2.4 | Grounding + contrastive | 54.8% | ✅ Working |
| v4.0 | + Small codebook (144) | ~45% | ⚠️ Declining |
| v4.8 | + Large codebook (232) | ~35% | ⚠️ Worse |
| v4.8.2 | + Huge codebook (1040) | **0.0%** | ❌ Failed |
| v4.9 | No codebook (proposed) | 50-60% (expected) | 🔄 Planned |

**Trend:** Larger codebook → WORSE retrieval

---

## File Structure

```
MMCAD/
├── clip4cad/
│   ├── models/
│   │   ├── clip4cad_gfa_v2_4.py       ✅ Working baseline
│   │   ├── clip4cad_gfa_v4_8_2.py     ❌ Failed (codebook)
│   │   └── clip4cad_gfa_v4_9.py       🔄 To implement
│   ├── losses/
│   │   ├── gfa_v2_4_losses.py         ✅ Working
│   │   ├── gfa_v4_8_2_losses.py       ❌ Failed
│   │   └── gfa_v4_9_losses.py         🔄 To create
│   └── data/
│       └── gfa_dataset.py             ✅ Optimized dataloading
├── notebooks/
│   ├── train_gfa_v2_4.ipynb           ✅ Working training
│   ├── train_gfa_v4_8_2.ipynb         ❌ Failed experiment
│   └── train_gfa_v4_9.ipynb           🔄 To create
├── docs/
│   ├── EXPERIMENTAL_RESULTS_MASTER.md 📋 READ THIS FIRST
│   ├── CLIP4CAD_GFA_v2_4_Architecture.md
│   ├── CLIP4CAD_GFA_v4_9_Architecture.md
│   └── autobrep_features.md
└── pretrained/
    └── autobrep/                      Pre-trained AutoBrep encoders
        ├── surf-fsq.ckpt
        └── edge-fsq.ckpt
```

---

## Training Configuration (v2.4 - Proven)

```python
# Hardware
GPU: RTX 4090 (24GB)
RAM: 256GB
Batch size: 512
Precision: FP16

# Dataset
Samples: 137K CAD models
Features: Pre-extracted (FP16 .npy)
Loading: Memory-mapped

# Model
d_unified: 256
d_proj: 128
Parameters: ~8M

# Training
Stage 1: 15 epochs, lr=3e-5
Stage 2: 20 epochs, lr=1e-5
Loss: InfoNCE (contrastive)
  λ_guided: 1.0
  λ_self: 0.3→0.5
  λ_distill: 0.1→0.2

# Results
Text→BRep R@1: 54.8%
Text→PC R@1: 59.7%
Self-grounding: Working (cos > 0.9)
```

---

## v4.8.2 Failure Details

### Training Logs
**Stage 1 (15 epochs):**
- Loss: 5.59 → 4.72 (minimal improvement)
- Cosine: 0.79 → 0.99 (misleadingly high!)
- **Retrieval: 0.0% across all metrics**

**Stage 2 (10 epochs):**
- Loss unstable (spikes to 73, 102, 147)
- Gap: 8.6 → 2.2 (closes successfully)
- **Retrieval: STILL 0.0%**

### Root Cause Diagnosis
```python
# What we observed:
cosine_similarity(z_brep, z_pc).mean() = 0.985  # Very high!
retrieval_r1(z_brep, z_text) = 0.0%            # Complete failure!

# The issue:
# ALL pairs have ~0.985 cosine, not just true pairs
# Model cannot distinguish instances

# Why:
# 137K models → 1040 codes (top-k=15) → weighted_sum → collapse
# Gear ≈ Shaft ≈ Housing → all activate similar codes
```

### Key Metrics for Debugging
```python
# ✅ Good metric: MARGIN
margin = mean_pos_sim - mean_neg_sim
# Should grow: 0.02 → 0.50+
# v4.8.2 had: ~0 (stuck)

# ❌ Bad metric: Average cosine
avg_cos = mean(all_similarities)
# Can be high even when model collapsed
# v4.8.2 had: 0.985 (misleading!)
```

---

## v4.9 Implementation Plan

### Design Principles
1. **No codebook** - the bottleneck that killed retrieval
2. **Attention pooling** - learnable queries, continuous
3. **Strong contrastive loss** - InfoNCE only
4. **Staged training** - keep Stage 0 PC-anchoring
5. **Minimal losses** - don't fight yourself

### Architecture Changes from v4.8.2
```diff
- Hierarchical codebook (1040 codes)
+ AttentionPooling (8 continuous queries)

- L_codebook + L_diversity + L_ATP + L_CU
+ InfoNCE only

- 15M parameters
+ ~10M parameters (simpler!)

- Complex curriculum with 7 loss terms
+ Simple 3-stage training
```

### Expected Timeline
```
Stage 0 (8 epochs): BRep-PC anchoring
  Target: margin > 0.3

Stage 1 (20 epochs): 3-way contrastive
  Epoch 5: margin > 0.1, R@1 ~ 5%
  Epoch 10: margin > 0.2, R@1 ~ 20%
  Epoch 20: margin > 0.4, R@1 ~ 40-60%

Stage 2 (5 epochs): Hard negatives (optional)
  Fine-tuning
```

### Files to Create
1. `clip4cad/models/clip4cad_gfa_v4_9.py`
2. `clip4cad/losses/gfa_v4_9_losses.py`
3. `notebooks/train_gfa_v4_9.ipynb`

See plan file: `C:\Users\User\.claude\plans\atomic-herding-adleman.md`

---

## Data Pipeline (Optimized)

### Pre-extracted Features
All features in FP16 for memory efficiency:
```
data/mmcad/
├── brep_autobrep_faces.npy       (N, F, 48) - Face FSQ features
├── brep_autobrep_edges.npy       (N, E, 12) - Edge FSQ features
├── pc_local_features.npy         (N, 49, 1024) - ShapeLLM local
├── pc_global_features.npy        (N, 1024) - ShapeLLM global
├── text_features.npy             (N, L, 3072) - Phi-4-mini
├── face_mask.npy                 (N, F) - Valid faces
├── edge_mask.npy                 (N, E) - Valid edges
└── text_mask.npy                 (N, L) - Valid tokens
```

### Dataloading Pattern (v2.4)
```python
from clip4cad.data import GFAMappedDataset, gfa_collate_fn

dataset = GFAMappedDataset(
    data_dir='data/mmcad',
    split='train',
    use_autobrep=True
)

dataloader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True,
    num_workers=4,
    collate_fn=gfa_collate_fn,
    pin_memory=True
)
```

**Key optimizations:**
- Memory-mapped .npy (mmap_mode='r')
- Pre-split files per epoch
- FP16 storage
- Efficient collation

---

## Common Issues & Solutions

### Issue: OOM (Out of Memory)
**Solutions:**
- Reduce batch size (512 → 256)
- Enable gradient checkpointing
- Use FP16 (already default)
- Check for memory leaks (detach gradients)

### Issue: NaN loss
**Solutions:**
- Clamp logits before sigmoid: `logits.clamp(-5, 5)`
- Safe softmax: `torch.nan_to_num(softmax(...), nan=0.0)`
- Add epsilon to divisions: `x / (y + 1e-8)`
- Check for inf in similarity matrix

### Issue: Low retrieval but training loss decreases
**Diagnosis:**
- Check if cosine is high for ALL pairs (model collapse)
- Monitor MARGIN not just average cosine
- Check if codebook is active (if using v4.x)

**Solutions:**
- Remove codebook (use v4.9)
- Increase contrastive loss weight
- Check data pipeline (are matched pairs actually matched?)

### Issue: Self-grounding not working
**Diagnosis:**
- Check cosine(z_self, z_guided)
- Should be > 0.8 by end of Stage 1

**Solutions:**
- Add direct contrastive loss to self-path (L_self)
- Don't rely only on distillation (L_distill)
- Increase λ_self

---

## Hardware Requirements

### Minimum
- GPU: 16GB VRAM (reduce batch to 256)
- RAM: 128GB
- Storage: 500GB SSD

### Recommended (Used)
- GPU: 24GB VRAM (RTX 4090)
- RAM: 256GB
- Storage: 1TB SSD

### Scaling
- Larger batch → better training (up to 512)
- More RAM → faster dataloading
- SSD → essential for 137K pre-extracted features

---

## Next Steps for New System

### Priority 1: Implement v4.9
1. Read `docs/EXPERIMENTAL_RESULTS_MASTER.md`
2. Read plan: `C:\Users\User\.claude\plans\atomic-herding-adleman.md`
3. Create v4.9 model file
4. Create v4.9 loss file
5. Create v4.9 training notebook
6. Train Stage 0 (8 epochs)
7. Monitor MARGIN (should reach 0.3+)
8. Train Stage 1 (20 epochs)
9. Target: R@1 > 50%

### Priority 2: If v4.9 Fails
- Fall back to v2.4 (proven 54.8%)
- Focus on data quality
- Check AutoBrep feature quality
- Investigate dataset issues

### Priority 3: If v4.9 Succeeds
- Add hard negative mining (Stage 2)
- Fine-grained discrimination
- Interpretability (codebook on frozen embeddings)
- Generative modeling (RL-based CAD generation)

---

## Contact & References

### Git Status
Branch: `claude/clip4cad-gfa-architecture-JlqEc`
Last commit: "feat: add comprehensive documentation for project archive"

### Key Papers
1. CLIP (Radford et al.) - Direct contrastive works
2. ULIP - 3D-language alignment
3. Modality Gap - Gap doesn't matter for retrieval (`modalitygap.pdf`)
4. AutoBrep - B-Rep feature extraction

### Dependencies
See `environment.yml` for complete conda environment

---

## Archive Status

✅ **Complete:**
- All experimental results documented
- All architectures documented
- Training logs preserved
- Key findings summarized
- Implementation plans written
- Code committed

🔄 **Ready for:**
- Porting to new system
- Continuing v4.9 implementation
- Reference for future work

---

**Last Updated:** 2026-03-26
**Author:** Claude Code (with human supervision)
**Status:** Production-ready archive
