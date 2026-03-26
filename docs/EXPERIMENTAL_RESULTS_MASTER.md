# CLIP4CAD-GFA: Complete Experimental Results & Architecture Evolution

**Document Version:** 2.0
**Last Updated:** 2026-03-26
**Purpose:** Comprehensive record of all experiments, architectures, and results for porting/reference

---

## Table of Contents

1. [Hardware Configuration](#hardware-configuration)
2. [Dataset Information](#dataset-information)
3. [Architecture Evolution Timeline](#architecture-evolution-timeline)
4. [Detailed Experiment Results](#detailed-experiment-results)
5. [Key Findings & Lessons Learned](#key-findings--lessons-learned)
6. [Best Configurations](#best-configurations)
7. [Failed Approaches & Why](#failed-approaches--why)
8. [Next Steps](#next-steps)

---

## Hardware Configuration

### Training Environment
- **GPU:** NVIDIA RTX 4090 (24GB VRAM)
- **RAM:** 256GB DDR4
- **OS:** Windows 10 Pro (10.0.19045)
- **CUDA:** Compatible with PyTorch 2.x
- **Storage:** SSD for dataset (137K+ samples)

### Memory Optimizations Used
- FP16 mixed precision training (GradScaler)
- Memory-mapped datasets (numpy mmap_mode='r')
- Pre-split batch loading
- Gradient checkpointing (for larger models)
- Batch size: 256-512 depending on model

---

## Dataset Information

### MM-CAD Dataset
- **Total Samples:** 137,000+ CAD models
- **Modalities:** B-Rep, Point Cloud, Text descriptions
- **Train/Val Split:** 90%/10%
- **Features:**
  - **B-Rep:** AutoBrep FSQ features (faces: 48-dim, edges: 12-dim)
  - **Point Cloud:** ShapeLLM features (local: 49×1024, global: 1024)
  - **Text:** Phi-4-mini features (3072-dim)

### Pre-extracted Features
All features stored in FP16 for memory efficiency:
- `brep_autobrep_faces.npy` - Face features
- `brep_autobrep_edges.npy` - Edge features
- `pc_local_features.npy` - Local PC features
- `pc_global_features.npy` - Global PC token
- `text_features.npy` - Text embeddings

---

## Architecture Evolution Timeline

### v2.4 - Baseline (Successful)
**Date:** Early experiments
**Architecture:** Text-guided grounding + self-grounding
**Status:** ✅ WORKING

**Key Components:**
- Modality-specific grounding projections
- Joint self-grounding training (not just distillation)
- Hierarchical aggregation (global → detail)
- BRep encoder with face/edge message passing

**Results:**
- Text→BRep R@1: 54.8%
- Text→PC R@1: 59.7%
- Self-grounding: WORKING (cosine > 0.8)

**Why it worked:**
- Separate projections for B-Rep vs PC
- Self-grounding trained with contrastive loss (not just KL)
- No conflicting losses (removed consistency loss)

---

### v4.0 - Hierarchical Codebook Introduction
**Date:** Mid experiments
**Architecture:** Added hierarchical codebook for gap-closing
**Status:** ⚠️ PARTIAL SUCCESS

**Key Changes from v2.4:**
- Added 3-level hierarchical codebook (category → type → spatial)
- Modality gap closing loss (L_ATP + L_CU)
- Self-query generation for text-free inference

**Architecture:**
- Category codes: 16
- Type codes per category: 8
- Spatial codes: 16
- Total codes: 16×8 + 16 = 144 codes

**Results:**
- Initial results showed promise
- Gap reduction achieved
- BUT: Started seeing retrieval degradation

---

### v4.2 - Curriculum Learning
**Date:** Mid experiments
**Architecture:** v4.0 + curriculum learning
**Status:** ⚠️ MIXED

**Key Changes:**
- Conditional self-query generation
- Curriculum learning schedule
- Gradual loss weight transitions

**Results:**
- Training more stable
- Gap closing worked
- Retrieval still below v2.4 baseline

---

### v4.8 - Expanded Codebook
**Date:** Recent
**Architecture:** Larger codebook + topology encoding
**Status:** ⚠️ TRAINING BUT SUBOPTIMAL

**Key Changes:**
- Increased codebook size:
  - Category: 20
  - Type per category: 10
  - Spatial: 32
  - Total: ~232 codes
- Added BFS-level topology encoding
- Edge-to-face message passing

**Configuration:**
```python
d = 256
d_proj = 128
num_msg_layers = 3
num_brep_tf_layers = 4
codebook: 20 category × 10 type + 32 spatial
```

**Results:**
- Model trains
- Gap decreases
- Retrieval remains low (not reaching v2.4 levels)

---

### v4.8.2 - Maximum Capacity (FAILED)
**Date:** Most recent experiment
**Architecture:** Largest codebook + smooth curriculum
**Status:** ❌ **COMPLETE FAILURE**

**Configuration:**
```python
d = 320
d_proj = 160
n_category = 20
n_type_per_cat = 10
n_spatial = 32
Total codes: 1040
Parameters: ~15M
```

**Training Log - Stage 1 (15 epochs):**

| Epoch | Loss | Gap(B) | Cos(B) | Div | CodeW | KLBlend |
|-------|------|--------|--------|-----|-------|---------|
| 1 | 5.5875 | 8.6105 | 0.7891 | 0.8111 | 0.10 | 0.03 |
| 5 | 4.9734 | 2.6198 | 0.9756 | 0.6627 | 0.30 | 0.17 |
| 10 | 4.8947 | 1.9827 | 0.9822 | 0.6287 | 0.30 | 0.33 |
| 15 | **4.7167** | **1.6533** | **0.9853** | 0.5603 | 0.30 | 0.40 |

**Stage 1 Retrieval (after 15 epochs):**
```
Text → BRep: R@1=0.0%, R@5=0.0%, R@10=0.1%
Text → PC:   R@1=0.0%, R@5=0.0%
BRep → PC:   R@1=0.0%, R@5=0.1%
```

**Training Log - Stage 2 (10 epochs):**

| Epoch | Loss | Gap(B) | Cos(B) | Align | ATPW | CUW |
|-------|------|--------|--------|-------|------|-----|
| 1 | 17.6723 | 3.4714 | 0.9335 | 45.9995 | 0.25 | 0.15 |
| 2 | 73.0861 | 4.9260 | 0.7878 | 136.6871 | 0.50 | 0.30 |
| 7 | 60.7485 | 2.4300 | 0.8765 | 111.7665 | 0.50 | 0.30 |
| 10 | **53.9645** | **2.2258** | 0.8611 | 98.2568 | 0.50 | 0.30 |

**Stage 2 Final Retrieval:**
```
Text → BRep: R@1=0.0%, R@5=0.0%, R@10=0.0%
Text → PC:   R@1=0.0%, R@5=0.0%
BRep → PC:   R@1=0.0%, R@5=0.0%
```

**Critical Observation:**
- **Cosine similarity: 0.9853** (very high!)
- **Retrieval: 0.0%** (complete failure!)
- **Problem:** ALL pairs have ~0.985 cosine, not just true pairs
- **Root cause:** Codebook information bottleneck

**Why it failed:**
```
137,000 unique CAD models
        ↓
1040 discrete codes, sparse top-k → ~15 active per sample
        ↓
z = weighted_sum(active_codes)
        ↓
Most models → nearly identical z

A gear and a shaft both activate "cylindrical" + "planar" codes.
Codebook captures WHAT TYPE but not WHICH INSTANCE.
Retrieval needs instance discrimination. Codebook destroys it.
```

---

## Detailed Experiment Results

### Experiment 1: GFA v2.4 Ablations

| Configuration | Text→BRep R@1 | Text→PC R@1 | Key Insight |
|---------------|---------------|-------------|-------------|
| Full model (λ_c=0.5) | 36.7% | - | Strong grounding HURTS |
| Global-only (no grounding) | 51.4% | 69.6% | Contrastive is foundation |
| Asymmetric (λ_c=0.05) | **54.8%** | 59.7% | Mild grounding helps |
| No consistency | 41.8% | - | Consistency over-constrains |
| HUS v2 (no cross-modal) | 2.2% | 12.8% | Cross-modal essential |

**Key Findings:**
1. Contrastive loss is the foundation (InfoNCE)
2. Heavy grounding constraints hurt performance
3. Asymmetric loss weighting works best
4. Self-grounding needs direct contrastive training

---

### Experiment 2: Codebook Size Scaling

| Version | Codes | d | Params | Text→BRep R@1 | Notes |
|---------|-------|---|--------|---------------|-------|
| v2.4 | None | 256 | ~8M | 54.8% | Baseline |
| v4.0 | 144 | 256 | ~10M | ~45% | Initial drop |
| v4.8 | 232 | 256 | ~12M | ~35% | Continued decline |
| v4.8.2 | 1040 | 320 | ~15M | **0.0%** | Complete failure |

**Trend:** Larger codebook → WORSE retrieval
**Conclusion:** Codebook is the problem, not the solution

---

### Experiment 3: Training Curriculum

**v4.8.2 Curriculum Schedule:**

**Stage 0: PC Anchoring (12 epochs)**
- BRep encoder learns to match PC
- PC encoder frozen
- Loss: InfoNCE + MSE
- Target: BRep-PC cosine > 0.85

**Stage 1: Codebook + Text (15 epochs)**
- Add text modality
- Codebook activated
- Smooth weight transitions:
  - Codebook weight: 0→0.3 over 3 epochs
  - Cosine → KL blend: gradual
- Result: High cosine but 0% retrieval

**Stage 2: Gap Closing (10 epochs)**
- Add L_ATP + L_CU
- Hard negative mining
- Weight ramp: 0→0.5 over 2 epochs
- Result: Gap closes but retrieval still 0%

**Finding:** Curriculum doesn't fix fundamental codebook bottleneck

---

## Key Findings & Lessons Learned

### 🔴 Critical Failures

1. **Codebook Bottleneck (v4.8.2)**
   - Discrete codes destroy instance-level information
   - High avg cosine + 0% retrieval = model collapse
   - 137K models → 1040 codes → information loss
   - **DO NOT USE CODEBOOK FOR RETRIEVAL**

2. **Gap-Closing for Retrieval (v4.x)**
   - Paper says: "Gap has limited impact on retrieval"
   - Confirmed: Gap-closing improves clustering, not retrieval
   - L_ATP + L_CU add complexity without benefit
   - **Don't optimize gap for retrieval tasks**

3. **Strong Grounding Constraints (v2 ablations)**
   - Consistency loss (L_consist) hurts: -13% R@1
   - Local contrastive creates false negatives
   - Over-constraining B-Rep ↔ PC correspondence
   - **Light grounding or no grounding is better**

4. **Self-Grounding via Distillation Only (v2 early)**
   - KL divergence alone insufficient
   - Self-grounding cosine stuck at 0.08
   - **Need direct contrastive loss for self-path**

### ✅ What Works

1. **Direct Contrastive Alignment (v2.4)**
   - Simple InfoNCE loss
   - No codebook
   - No gap-closing
   - **This is CLIP for CAD and it works**

2. **Modality-Specific Projections (v2.4)**
   - Separate W_g_brep and W_g_pc
   - B-Rep = discrete units, PC = spatial patches
   - Different semantics need different projections
   - **+4% R@1 improvement**

3. **Joint Self-Grounding Training (v2.4)**
   - Self-path gets InfoNCE loss too
   - Not just distillation (KL)
   - Enables text-free inference
   - **Cosine: 0.08 → 0.90+**

4. **FP16 + Memory-Mapped Data**
   - Batch size 512 fits in 24GB
   - Pre-split loading prevents CPU bottleneck
   - Faster training, same accuracy
   - **Essential for 137K dataset**

5. **BFS-Level Topology Encoding**
   - Adds hierarchical structure to B-Rep
   - Edge-to-face message passing
   - Helps encoder understand CAD structure
   - **Modest improvement in encoder quality**

---

## Best Configurations

### Recommended: GFA v2.4 (PROVEN)

```python
@dataclass
class GFAv24Config:
    # Dimensions
    d_face = 48          # AutoBrep
    d_edge = 12          # AutoBrep
    d_pc = 1024          # ShapeLLM
    d_text = 3072        # Phi-4-mini
    d_unified = 256      # Internal
    d_proj = 128         # Output embedding

    # Architecture
    num_slots = 12       # Feature slots
    num_heads = 8
    num_text_parser_layers = 2
    num_self_ground_layers = 2
    dropout = 0.1

    # Training
    batch_size = 512
    lr_stage1 = 3e-5
    lr_stage2 = 1e-5
    epochs_stage1 = 15
    epochs_stage2 = 20

    # Loss weights
    lambda_guided = 1.0
    lambda_self = 0.3        # Stage 1
    lambda_self_s2 = 0.5     # Stage 2
    lambda_distill = 0.1
    lambda_detail = 0.0      # Stage 1
    lambda_detail_s2 = 0.3   # Stage 2
```

**Expected Results:**
- Text→BRep R@1: 54-60%
- Text→PC R@1: 60-68%
- Self-grounding: Working (cosine > 0.9)
- Parameters: ~8-10M

---

### Alternative: GFA v4.9 (PROPOSED - NOT TESTED YET)

**Philosophy:** Remove codebook completely, pure contrastive alignment

```python
@dataclass
class GFAv49Config:
    # Dimensions
    d_face = 48
    d_edge = 12
    d_pc = 1024
    d_text = 3072
    d = 256              # Internal
    d_proj = 128         # Output

    # Architecture
    num_pool_queries = 8  # AttentionPooling (NO codebook!)
    num_msg_layers = 3
    num_brep_tf_layers = 4
    num_text_tf_layers = 2
    num_heads = 8
    dropout = 0.1

    # Training
    batch_size = 512
    init_tau = 0.07

    # Stages
    stage0_epochs = 8      # BRep-PC anchoring
    stage0_lr = 3e-4

    stage1_epochs = 20     # Full 3-way
    stage1_lr = 1e-4

    stage2_epochs = 5      # Hard negatives
    stage2_lr = 2e-5
```

**Key Differences from v4.8.2:**
- **NO codebook** (AttentionPooling with continuous queries)
- **NO gap-closing** (L_ATP, L_CU removed)
- **Pure InfoNCE** loss
- Simpler architecture, fewer parameters (~10M vs 15M)

**Motivation:**
- v4.8.2 proved codebook kills retrieval
- CLIP/ULIP/v2.4 all use direct contrastive
- AttentionPooling preserves instance information

**Status:** Designed but not implemented yet

---

## Failed Approaches & Why

### ❌ Hierarchical Codebook (v4.x)

**What was tried:**
- 3-level codebook: category → type → spatial
- Sparse top-k selection
- Learnable code embeddings
- Sizes from 144 to 1040 codes

**Why it failed:**
1. **Information bottleneck:** 137K models → 1040 codes = massive loss
2. **Sparse selection:** Only ~15 active codes per sample
3. **Type confusion:** "Gear" and "shaft" activate similar codes
4. **Instance collapse:** All samples → nearly identical weighted sums

**Evidence:**
- Cosine: 0.985 for ALL pairs (not just true pairs)
- Retrieval: 0.0% across all metrics
- Contrastive loss: 4.7 (barely improved from 5.5 initial)

**Conclusion:** Discrete codebooks incompatible with instance retrieval

---

### ❌ Gap-Closing for Retrieval (v4.x)

**What was tried:**
- L_ATP: Align to pretrained space
- L_CU: Cluster uniformity
- Various weight schedules

**Why it failed:**
1. Paper explicitly states: "Gap has limited impact on retrieval"
2. Gap-closing optimizes for clustering, not discrimination
3. Added complexity with no retrieval benefit
4. Competing objectives with InfoNCE

**Evidence:**
- Gap decreased: 8.6 → 1.6
- Retrieval stayed: 0.0%
- Training unstable (loss spikes in Stage 2)

**Conclusion:** Don't optimize gap for retrieval; use for clustering only

---

### ❌ Strong Grounding Constraints (v2.x)

**What was tried:**
- L_consist: Force F_brep[k] ≈ F_pc[k]
- L_local: Slot-level InfoNCE
- High lambda values (0.5)

**Why it failed:**
1. Over-constrains correspondence
2. B-Rep slots ≠ PC slots (different granularity)
3. Local contrastive creates false negatives
4. Fights global contrastive objective

**Evidence:**
- Full grounding (λ=0.5): 36.7% R@1
- Mild grounding (λ=0.05): 54.8% R@1
- Improvement: +18% absolute!

**Conclusion:** Light or no grounding is better

---

### ❌ Self-Grounding via Distillation Only (v2 early)

**What was tried:**
- Self-grounding with only L_distill = KL(G_self || G_guided)
- No direct contrastive loss on self-path

**Why it failed:**
1. KL alone is weak supervision
2. Self-grounding learns to mimic patterns, not semantics
3. Text-free inference broken

**Evidence:**
- Self-grounding cosine: 0.08 (random)
- Self → Text retrieval: broken

**Fix:**
- Add L_self = InfoNCE(z_self, z_text)
- Result: cosine 0.08 → 0.90+

**Conclusion:** Self-path needs direct contrastive training

---

## Next Steps

### Immediate: Implement v4.9

**Priority 1: Create v4.9 model**
- Copy EdgeMessageLayer from v4.8.x (topology encoding works)
- Implement AttentionPooling (replace codebook)
- Simple InfoNCE loss only
- Expected: R@1 > 50% (better than v4.8.2's 0%)

**Files to create:**
1. `clip4cad/models/clip4cad_gfa_v4_9.py`
2. `clip4cad/losses/gfa_v4_9_losses.py`
3. `notebooks/train_gfa_v4_9.ipynb`

**Training plan:**
- Stage 0: 8 epochs, BRep-PC anchoring
- Stage 1: 20 epochs, 3-way contrastive
- Monitor: MARGIN (pos_sim - neg_sim)

**Success criteria:**
- Stage 0: margin > 0.3
- Stage 1 ep 10: R@1 > 20%
- Stage 1 ep 20: R@1 > 40%

---

### Future Work

**If v4.9 works (R@1 > 50%):**

1. **Add interpretability AFTER retrieval works**
   - Train codebook on frozen embeddings
   - Use for visualization only, not training
   - Cluster analysis

2. **Fine-grained discrimination**
   - Hard negative mining (32 vs 64 teeth)
   - Detail-level InfoNCE
   - Hierarchical aggregation

3. **Generative modeling**
   - Use learned embeddings as conditioning
   - RL-based CAD generation
   - Retrieval-augmented generation

**If v4.9 doesn't reach v2.4 (R@1 < 54%):**
- Fall back to v2.4 architecture
- Focus on data quality improvements
- Investigate AutoBrep feature quality

---

## Appendix A: File Locations

### Model Files
- v2.4: `clip4cad/models/clip4cad_gfa_v2_4.py`
- v4.8.2: `clip4cad/models/clip4cad_gfa_v4_8_2.py`
- v4.9: `clip4cad/models/clip4cad_gfa_v4_9.py` (to be created)

### Loss Files
- v2.4: `clip4cad/losses/gfa_v2_4_losses.py`
- v4.8.2: `clip4cad/losses/gfa_v4_8_2_losses.py`
- v4.9: `clip4cad/losses/gfa_v4_9_losses.py` (to be created)

### Training Notebooks
- v2.4: `notebooks/train_gfa_v2_4.ipynb`
- v4.8.2: `notebooks/train_gfa_v4_8_2.ipynb`
- v4.9: `notebooks/train_gfa_v4_9.ipynb` (to be created)

### Documentation
- Architecture docs: `docs/CLIP4CAD_GFA_v*.md`
- This file: `docs/EXPERIMENTAL_RESULTS_MASTER.md`
- AutoBrep features: `docs/autobrep_features.md`

---

## Appendix B: Key Papers Referenced

1. **Modality Gap**
   - "Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning"
   - Key insight: Gap matters for clustering, not retrieval
   - Reference: `modalitygap.pdf`

2. **CLIP**
   - Direct contrastive alignment works
   - Simple InfoNCE loss
   - No intermediate representations

3. **ULIP**
   - 3-way contrastive for text/image/PC
   - Proven approach for multi-modal alignment

4. **AutoBrep**
   - FSQ-based B-Rep encoding
   - Pretrained face/edge encoders
   - Reference: `docs/autobrep_features.md`

---

## Appendix C: Quick Reference Commands

### Load v2.4 model
```python
from clip4cad.models import CLIP4CAD_GFA_v2_4
from clip4cad.losses import GFAv24Loss

config = GFAv24Config()
model = CLIP4CAD_GFA_v2_4(config).cuda()
criterion = GFAv24Loss(lambda_self=0.3)
```

### Load v4.8.2 model (failed)
```python
from clip4cad.models import CLIP4CAD_GFA_v482
from clip4cad.losses import GFAv482Loss

config = GFAv482Config()
model = CLIP4CAD_GFA_v482(config).cuda()
# DO NOT USE FOR RETRIEVAL - 0% R@1
```

### Training with FP16
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast(device_type='cuda'):
        outputs = model(batch)
        loss, losses = criterion(outputs)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

## Document History

- **v1.0 (2026-03-26):** Initial comprehensive documentation
- **v2.0 (2026-03-26):** Added v4.8.2 failure analysis and v4.9 proposal

**Status:** Complete record ready for porting to new system
