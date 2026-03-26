# CLIP4CAD-GFA v4.9: Direct Contrastive Alignment

**Status:** 🟡 Designed, not yet implemented
**Motivation:** v4.8.2 codebook caused complete retrieval failure (0% R@1)
**Philosophy:** Remove codebook, return to proven CLIP-style direct alignment

---

## Why v4.8.2 Failed

### The Codebook Bottleneck Problem

```
Training logs reveal the issue:
  Stage 1 Epoch 15:  Cos(B) = 0.985  but  R@1 = 0.0%

This means: ALL pairs have ~0.985 cosine, not just true pairs.
The model cannot distinguish sample A from sample B.
```

**Root cause visualization:**
```
137,000 unique CAD models
        │
        ▼
1040 discrete codes, sparse top-k → ~15 active per sample
        │
        ▼
z = weighted_sum(active_codes)
        │
        ▼
Most models → nearly identical z

Example:
  A gear: activates "cylindrical" + "planar" + "circular" codes
  A shaft: activates "cylindrical" + "planar" codes
  → Nearly identical representations!

Codebook captures WHAT TYPE but not WHICH INSTANCE.
Retrieval needs instance discrimination. Codebook destroys it.
```

**Contrastive loss evidence:**
- Initial: ~5.55 (= ln(256), random chance)
- After 15 epochs: ~4.7 (barely improved)
- For reference, working models: ~1.0-2.0

---

## v4.9 Solution: No Codebook, Pure Contrastive

### What Works (Empirical Evidence)

| Method | Approach | R@1 | Source |
|--------|----------|-----|--------|
| CLIP | Encoder → Pool → Project → InfoNCE | ✅ Works | OpenAI |
| ULIP | 3-way CLIP (text, image, PC) | ✅ Works | Paper |
| Our v2.4 | Text-guided grounding → InfoNCE | ✅ 54.8% | Experiments |
| Our v4.8.2 | Codebook grounding → InfoNCE | ❌ 0.0% | Experiments |

**Pattern:** Every working method uses **direct contrastive alignment, no intermediate codebook.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   GFA v4.9 Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUTS                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ BRep (AutoBr)│  │ PC (ShapeLLM)│  │ Text (Phi-4) │      │
│  │ faces: (F,48)│  │ local:(N,1024)│  │ tokens:(L,3072)│    │
│  │ edges: (E,12)│  │ global:(1024)│  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │              │
│         ▼                 ▼                 ▼              │
│  ┌───────────────────────────────────────────────┐         │
│  │         MODALITY ENCODERS                     │         │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │         │
│  │  │BRep Enc  │  │  PC Enc  │  │ Text Enc │   │         │
│  │  │3×MsgPass │  │ 2-layer  │  │2-layer TF│   │         │
│  │  │4-layer TF│  │   MLP    │  │          │   │         │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘   │         │
│  │       │             │             │         │         │
│  │  X_brep (N,d)  X_pc (49,d)  X_text (L,d)   │         │
│  └───────┼─────────────┼─────────────┼──────────┘         │
│          │             │             │                    │
│          ▼             ▼             ▼                    │
│  ┌───────────────────────────────────────────────┐         │
│  │   ATTENTION POOLING (per modality)            │         │
│  │                                               │         │
│  │  K=8 learnable queries cross-attend to tokens │         │
│  │  NO discrete codes, NO sparse selection      │         │
│  │                                               │         │
│  │  Q_brep (8,d) → AttentionPool → z_brep_raw   │         │
│  │  Q_pc (8,d) → AttentionPool → z_pc_raw       │         │
│  │  Q_text (8,d) → AttentionPool → z_text_raw   │         │
│  └───────┼─────────────┼─────────────┼──────────┘         │
│          │             │             │                    │
│          ▼             ▼             ▼                    │
│  ┌───────────────────────────────────────────────┐         │
│  │   PROJECTION HEADS (per modality)             │         │
│  │                                               │         │
│  │  brep_proj: MLP(d → d_proj)                  │         │
│  │  pc_proj: MLP(d → d_proj)                    │         │
│  │  text_proj: MLP(d → d_proj)                  │         │
│  │                                               │         │
│  │  z_brep (128)  z_pc (128)  z_text (128)      │         │
│  └───────────────────────────────────────────────┘         │
│                                                             │
│  LOSS: InfoNCE(z_text, z_brep, z_pc)                      │
│                                                             │
│  That's it. No codebook. No gap-closing. No curriculum.   │
│  Just strong contrastive alignment.                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. AttentionPooling (Replaces Codebook)

```python
class AttentionPooling(nn.Module):
    """
    Learnable queries cross-attend to tokens, then mean-pool.

    KEY DIFFERENCES from codebook:
    - Queries are CONTINUOUS, not discrete codes
    - Output preserves instance-level information
    - No sparse selection = no information loss
    """

    def __init__(self, d: int, num_queries: int = 8):
        self.queries = nn.Parameter(torch.randn(num_queries, d))
        self.cross_attn = nn.MultiheadAttention(d, num_heads=8)
        self.ffn = ... # Feed-forward

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, N, d)
        Q = self.queries.expand(B, -1, -1)  # (B, K, d)
        attn_out, _ = self.cross_attn(Q, X, X)
        Q = self.norm(Q + attn_out + self.ffn(...))
        z = Q.mean(dim=1)  # (B, d) - mean pool queries
        return z
```

**Why this works:**
- 8 continuous queries capture multiple aspects of each input
- No discrete selection → no information bottleneck
- Different instances → different query responses → different z

**Contrast with codebook:**
- Codebook: 137K instances → 15 active codes → collapse
- AttentionPool: 137K instances → 8 continuous vectors → preserved

---

### 2. TopologyBRepEncoder (Kept from v4.8.x)

```python
class TopologyBRepEncoder(nn.Module):
    """
    Topology-aware B-Rep encoder.
    Uses: face_features, edge_features, edge_to_faces, bfs_level
    """

    def __init__(self, config):
        self.face_proj = nn.Linear(48, d)
        self.edge_proj = nn.Linear(12, d)
        self.level_emb = nn.Embedding(32, d)  # BFS level

        # Message passing (3 layers)
        self.msg_layers = nn.ModuleList([
            EdgeMessageLayer(d) for _ in range(3)
        ])

        # Transformer (4 layers)
        self.transformer = nn.TransformerEncoder(...)

    def forward(self, face_feats, edge_feats, face_mask,
                edge_mask, edge_to_faces, bfs_level):
        F = self.face_proj(face_feats) + self.level_emb(bfs_level)
        E = self.edge_proj(edge_feats)

        # Message passing through topology
        for layer in self.msg_layers:
            F, E = layer(F, E, edge_to_faces, ...)

        # Concatenate and transform
        X = torch.cat([F, E], dim=1)
        X = self.transformer(X, src_key_padding_mask=~mask)
        return X, mask
```

**Why keep this:**
- BFS-level encoding adds hierarchical structure
- Edge-to-face message passing captures topology
- Works well in v4.8.x (encoder itself is good)
- Only the codebook was the problem

---

### 3. Simple Encoders (Text, PC)

```python
class TextEncoder(nn.Module):
    def __init__(self, config):
        self.proj = nn.Linear(3072, d)
        self.encoder = nn.TransformerEncoder(
            layers=2, d=d, heads=8
        )

    def forward(self, X, mask):
        X = self.proj(X)
        X = self.encoder(X, src_key_padding_mask=~mask)
        return X, mask


class PCEncoder(nn.Module):
    def __init__(self, config):
        self.proj = nn.Sequential(
            nn.Linear(1024, d),
            nn.GELU(),
            nn.Linear(d, d)
        )

    def forward(self, pc_local, pc_global):
        X = torch.cat([pc_local, pc_global.unsqueeze(1)], dim=1)
        return self.proj(X)
```

---

### 4. Projection Heads (Per-Modality)

```python
self.brep_proj = nn.Sequential(
    nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_proj)
)
self.text_proj = nn.Sequential(
    nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_proj)
)
self.pc_proj = nn.Sequential(
    nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_proj)
)
```

**Why separate heads:**
- Each encoder organizes its space differently
- Separate projections give flexibility
- Standard in CLIP/ULIP

---

## Loss Function

### Simple InfoNCE (That's All You Need)

```python
class GFAv49Loss(nn.Module):
    def __init__(self, label_smoothing: float = 0.05):
        self.label_smoothing = label_smoothing

    def forward(self, outputs, stage: int = 1):
        if stage == 0:
            return self._stage0(outputs)  # BRep ↔ PC
        else:
            return self._stage1(outputs)  # Text ↔ BRep ↔ PC

    def _stage1(self, outputs):
        z_text = F.normalize(outputs['z_text'], dim=-1)
        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)

        # 3-way symmetric InfoNCE
        loss_tb = self._infonce(z_text, z_brep, tau)
        loss_tp = self._infonce(z_text, z_pc, tau)
        loss_bp = self._infonce(z_brep, z_pc, tau)

        total = (loss_tb + loss_tp + loss_bp) / 3
        return total

    def _infonce(self, z_a, z_b, tau):
        logits = z_a @ z_b.T / tau
        labels = torch.arange(B, device=z_a.device)
        loss_ab = F.cross_entropy(logits, labels,
                                   label_smoothing=self.label_smoothing)
        loss_ba = F.cross_entropy(logits.T, labels,
                                   label_smoothing=self.label_smoothing)
        return (loss_ab + loss_ba) / 2
```

**That's the entire loss. No:**
- L_codebook (removed - was the bottleneck)
- L_diversity (removed - codebook gone)
- L_ATP (removed - gap-closing not needed)
- L_CU (removed - gap-closing not needed)
- L_consistency (removed - hurts performance)

**Just InfoNCE. Simple. Proven.**

---

## Training Procedure

### Stage 0: Anchor (8 epochs)
```
BRep encoder → AttentionPool → z_brep
     ↕ InfoNCE + MSE
PC encoder (frozen) → AttentionPool → z_pc

Goal: BRep-PC cosine > 0.7
LR: 3e-4, warmup 1 epoch
Batch: 512
```

### Stage 1: Full Alignment (20 epochs)
```
Text → z_text ─┐
BRep → z_brep ─┼─ 3-way InfoNCE
PC   → z_pc   ─┘

Goal: R@1 > 20% by epoch 10, > 50% by epoch 20
LR: 1e-4, warmup 2 epochs
Batch: 512
```

### Stage 2 (Optional): Fine-Tuning (5 epochs)
```
Same as Stage 1 + hard negative mining
LR: 2e-5
```

---

## Monitoring: The ONE Metric That Matters

### MARGIN

```python
@torch.no_grad()
def compute_contrastive_quality(z_a, z_b):
    sims = z_a @ z_b.T
    B = sims.shape[0]

    # Positive pairs: diagonal
    pos_mask = torch.eye(B, dtype=torch.bool)
    mean_pos = sims[pos_mask].mean().item()

    # Negative pairs: off-diagonal
    neg_mask = ~pos_mask
    mean_neg = sims[neg_mask].mean().item()

    return {
        'pos_sim': mean_pos,
        'neg_sim': mean_neg,
        'margin': mean_pos - neg_sim,  # THIS IS THE KEY
    }
```

**Healthy training:**
```
Epoch 1:  margin = 0.02  (barely above random)
Epoch 5:  margin = 0.15  (learning!)
Epoch 10: margin = 0.30  (good separation)
Epoch 20: margin = 0.50+ (strong discrimination)
```

**🚨 ALARM:** margin stays near 0 for >3 epochs → model collapsed

**v4.8.2 had:**
- Cosine: 0.985 (misleading - high for ALL pairs)
- Margin: ~0 (true diagnostic - no discrimination)

**v4.9 should have:**
- Positive cosine: 0.9+
- Negative cosine: 0.2-0.4
- Margin: 0.5+ (strong)

---

## Expected Results

### Timeline

| Epoch | Loss | Margin | Text→BRep R@1 | Notes |
|-------|------|--------|---------------|-------|
| Stage 0, ep 8 | ~3.0 | 0.3+ (BRep↔PC) | - | Anchoring done |
| Stage 1, ep 5 | ~4.0 | 0.1+ | ~5% | Starting to learn |
| Stage 1, ep 10 | ~3.0 | 0.2+ | ~20% | Good progress |
| Stage 1, ep 20 | ~2.0 | 0.4+ | ~40-60% | Target |

### Targets

| Metric | Target | Baseline (v2.4) | v4.8.2 (failed) |
|--------|--------|-----------------|-----------------|
| Text→BRep R@1 | 50-60% | 54.8% | 0.0% |
| Text→PC R@1 | 55-65% | 59.7% | 0.0% |
| BRep→PC R@1 | 30-40% | 29.4% | 0.0% |
| Margin | 0.5+ | - | ~0 |
| Parameters | ~10-12M | ~8M | ~15M |

---

## Why This Will Work

1. **No information bottleneck.**
   - AttentionPooling: 8 continuous queries preserve instance info
   - No discrete selection forcing similar models together
   - Different instances → different query responses → different z

2. **Single clear objective.**
   - InfoNCE: "make true pairs more similar than false pairs"
   - No competing losses pulling different directions
   - Proven in CLIP, ULIP, v2.4

3. **Proven approach.**
   - CLIP for images: works
   - ULIP for 3D: works
   - Our v2.4 for CAD: works (54.8%)
   - Only difference: better BRep encoder (topology-aware)

4. **Staged training handles cold start.**
   - Stage 0: BRep learns stable anchor (PC)
   - Stage 1: Add text gradually
   - Pre-trained text/PC encoders provide stable signal

5. **Easy to debug.**
   - If margin doesn't grow: problem in data or BRep encoder
   - Not buried in complex loss landscape with 7 terms
   - Clear diagnostic from day 1

---

## Configuration

```python
@dataclass
class GFAv49Config:
    # Input dimensions
    d_face: int = 48       # AutoBrep face FSQ
    d_edge: int = 12       # AutoBrep edge FSQ
    d_pc: int = 1024       # ShapeLLM
    d_text: int = 3072     # Phi-4-mini

    # Model dimensions
    d: int = 256           # Internal dimension
    d_proj: int = 128      # Output embedding

    # BRep encoder
    num_msg_layers: int = 3
    num_brep_tf_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_bfs_levels: int = 32

    # Text encoder
    num_text_tf_layers: int = 2

    # Attention pooling
    num_pool_queries: int = 8   # K learnable queries

    # Contrastive
    init_tau: float = 0.07
    label_smoothing: float = 0.05
```

---

## Implementation Checklist

- [ ] Create `clip4cad/models/clip4cad_gfa_v4_9.py`
  - [ ] GFAv49Config dataclass
  - [ ] AttentionPooling class
  - [ ] EdgeMessageLayer (copy from v4.8.x)
  - [ ] TopologyBRepEncoder
  - [ ] TextEncoder
  - [ ] PCEncoder
  - [ ] CLIP4CAD_GFA_v49 main class

- [ ] Create `clip4cad/losses/gfa_v4_9_losses.py`
  - [ ] GFAv49Loss class
  - [ ] compute_retrieval_metrics
  - [ ] compute_contrastive_quality

- [ ] Update `clip4cad/models/__init__.py`
  - [ ] Add v4.9 exports

- [ ] Update `clip4cad/losses/__init__.py`
  - [ ] Add v4.9 exports

- [ ] Create `notebooks/train_gfa_v4_9.ipynb`
  - [ ] Use v2.4 training harness pattern
  - [ ] GFAMappedDataset with use_autobrep=True
  - [ ] 3-stage training
  - [ ] Margin monitoring
  - [ ] Retrieval evaluation every 5 epochs

---

## FAQ

**Q: What about the modality gap?**
A: The paper says it doesn't matter for retrieval. If you later need clustering, add L_ATP + L_CU as a Stage 3. But get retrieval working first.

**Q: What about self-grounding?**
A: There's no separate self-path needed. z_brep is computed from geometry alone and aligned to z_text via contrastive loss. That IS self-grounding. Can add later if needed.

**Q: What about the hierarchical codebook for interpretability?**
A: Add it AFTER retrieval works. Train a codebook on top of frozen embeddings for analysis. Don't let it interfere with training.

**Q: What if this still gives 0% R@1?**
A: Then the problem is in the data pipeline (check matched pairs are actually matched) or BRep features (check AutoBrep quality). Monitor margin—it will tell you immediately if learning is happening.

**Q: Isn't this too simple?**
A: CLIP is "simple" and works for billions of image-text pairs. Complexity should be added only when simple approaches fail, and only to address a specific diagnosed failure mode. v4.8.2 was complex and failed. v2.4 was simpler and worked.

---

## References

- v4.8.2 failure analysis: `docs/EXPERIMENTAL_RESULTS_MASTER.md`
- v2.4 success: `docs/CLIP4CAD_GFA_v2_4_Architecture.md`
- Training instructions: `tempinstr.txt`
- CLIP paper: Radford et al.
- ULIP paper: 3D-language alignment
- Modality gap paper: `modalitygap.pdf`
