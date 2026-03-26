# CLIP4CAD-GFA v2: Grounded Feature Alignment for CAD Multimodal Learning

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Motivation & Lessons Learned](#2-motivation--lessons-learned)
3. [Architecture Overview](#3-architecture-overview)
4. [Component Details](#4-component-details)
5. [Loss Functions](#5-loss-functions)
6. [Training Procedure](#6-training-procedure)
7. [Implementation Guide](#7-implementation-guide)
8. [Expected Results](#8-expected-results)
9. [Ablation Plan](#9-ablation-plan)

---

## 1. Executive Summary

### What is GFA v2?

CLIP4CAD-GFA v2 is a multimodal representation learning architecture that aligns three modalities:
- **B-Rep** (Boundary Representation): CAD-native format with faces and edges
- **Point Cloud**: 3D spatial representation from ShapeLLM
- **Text**: Natural language descriptions with geometric detail

### Key Innovations

| Innovation | Description | Why It Matters |
|------------|-------------|----------------|
| **Modality-Specific Grounding** | Separate projections for B-Rep and PC | B-Rep tokens ≠ PC tokens |
| **Joint Self-Grounding Training** | Self-path learns via direct contrastive loss | Enables text-free inference |
| **Hierarchical Aggregation** | Global → Detail conditioning | Fine-grained discrimination |
| **Unified Embedding Space** | Single embedding for retrieval + generation | Future RL-based generation |

### Design Goals

1. **Unified Latent Space**: Same embedding works for retrieval AND future generative models
2. **Text-Free Inference**: Geometry encodes standalone without text input
3. **Fine-Grained Discrimination**: Distinguish 32 vs 64 teeth on a gear
4. **Interpretable Grounding**: Visualize what text attends to in geometry
5. **Leverage Hierarchical Data**: Use face/edge, local/global, title/description structure

---

## 2. Motivation & Lessons Learned

### GFA v1 Ablation Results

| Configuration | Text→BRep R@1 | Text→PC R@1 | Key Insight |
|---------------|---------------|-------------|-------------|
| Full model (λ_c=0.5) | 36.7% | - | Strong grounding HURTS |
| Global-only (no grounding) | 51.4% | 69.6% | Contrastive is foundation |
| Asymmetric (λ_c=0.05) | **54.8%** | 59.7% | Mild grounding helps |
| No consistency | 41.8% | - | Consistency over-constrains |
| HUS v2 (no cross-modal) | 2.2% | 12.8% | Cross-modal interaction essential |

### Root Causes of GFA v1 Failures

```
PROBLEM 1: Shared Grounding Projection
──────────────────────────────────────
GFA v1: X_g_brep = X_brep @ W_g_geo   ← Same W for both!
        X_g_pc   = X_pc   @ W_g_geo

Reality:
  • B-Rep tokens = Discrete semantic units (faces, edges)
  • PC tokens = Spatial patches with multimodal context
  These need DIFFERENT projections!

PROBLEM 2: Rigid Consistency Loss
─────────────────────────────────
GFA v1: L_consist = 1 - cosine(F_brep[k], F_pc[k])

This forces exact B-Rep ↔ PC correspondence, but:
  • "Fillet" in B-Rep = 1 face
  • "Fillet" in PC = parts of 3-4 patches
  Over-constraining!

PROBLEM 3: Local Contrastive Creates False Negatives
────────────────────────────────────────────────────
GFA v1: L_local = InfoNCE(F_brep[:, k], F_pc[:, k])

Problem: Many samples share features (holes, fillets)
Forcing slot-level discrimination fights global objective!

PROBLEM 4: Untrained Self-Grounding
───────────────────────────────────
GFA v1: Self-grounding only had distillation loss
Result: Self-grounding cosine similarity stuck at 0.08
        Text-free inference completely broken!
```

### HUS v2 Failure Analysis

```
PROBLEM: No Cross-Modal Interaction During Encoding
───────────────────────────────────────────────────
HUS v2:
  brep_out = encode_brep(Z_brep)    ← Independent!
  pc_out = encode_pc(Z_pc)          ← Independent!
  text_out = encode_text(H_text)    ← Independent!

Only alignment: Contrastive loss at the END
With only 111K samples, this isn't enough supervision!

Result: 2.2% R@1 (25x worse than GFA v1)

Gate stuck at 0.5000 for ALL epochs = learned nothing
```

### Key Insight

**Text must guide geometry encoding during training, but inference must work without text.**

This requires:
1. Text-guided path (training)
2. Self-grounding path (inference)
3. **Joint training of both paths** (not post-hoc distillation!)

---

## 3. Architecture Overview

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CLIP4CAD-GFA v2 ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  INPUTS                                                                         │
│  ──────                                                                         │
│  B-Rep: [Faces (F, d_f)] [Edges (E, d_e)]                                      │
│  PC:    [Local (L, d_pc)] [Global (1, d_pc)]                                   │
│  Text:  [Tokens (T, d_text)]                                                   │
│                                                                                 │
│                              ┌──────────────────┐                               │
│                              │  INPUT PROJECTION │                              │
│                              │  (Modality-Specific)│                            │
│                              └────────┬─────────┘                               │
│                                       │                                         │
│                    ┌──────────────────┼──────────────────┐                     │
│                    ▼                  ▼                  ▼                      │
│             ┌──────────┐       ┌──────────┐       ┌──────────┐                 │
│             │  X_brep  │       │   X_pc   │       │  X_text  │                 │
│             │ (N_b, d) │       │ (N_p, d) │       │  (T, d)  │                 │
│             └────┬─────┘       └────┬─────┘       └────┬─────┘                 │
│                  │                  │                  │                        │
│                  │                  │                  ▼                        │
│                  │                  │         ┌───────────────┐                │
│                  │                  │         │ TEXT PARSING  │                │
│                  │                  │         │ K Feature Slots│                │
│                  │                  │         │ + Confidence   │                │
│                  │                  │         └───────┬───────┘                │
│                  │                  │                 │                         │
│                  │                  │          T_feat (B, K, d)                 │
│                  │                  │          confidence (B, K)                │
│                  │                  │                 │                         │
│         ┌────────┴──────────────────┴─────────────────┤                        │
│         │                                             │                        │
│         ▼                                             ▼                        │
│  ┌─────────────────────────────┐        ┌─────────────────────────────┐       │
│  │    TEXT-GUIDED PATH         │        │      SELF-GROUNDING PATH    │       │
│  │    (Training)               │        │      (Inference)            │       │
│  ├─────────────────────────────┤        ├─────────────────────────────┤       │
│  │                             │        │                             │       │
│  │  Modality-Specific          │        │  Geometry-Adaptive          │       │
│  │  Grounding                  │        │  Queries                    │       │
│  │       │                     │        │       │                     │       │
│  │       ▼                     │        │       ▼                     │       │
│  │  G_text = softmax(          │        │  G_self = softmax(          │       │
│  │    T_feat @ X_geo.T)        │        │    Q_adapted @ X_geo.T)     │       │
│  │       │                     │        │       │                     │       │
│  │       ▼                     │        │       ▼                     │       │
│  │  Hierarchical               │        │  Hierarchical               │       │
│  │  Aggregation                │        │  Aggregation                │       │
│  │  (Global → Detail)          │        │  (Global → Detail)          │       │
│  │       │                     │        │       │                     │       │
│  │       ▼                     │        │       ▼                     │       │
│  │  z_guided                   │        │  z_self                     │       │
│  │                             │        │                             │       │
│  └──────────────┬──────────────┘        └──────────────┬──────────────┘       │
│                 │                                      │                       │
│                 │              LOSSES                  │                       │
│                 │              ──────                  │                       │
│                 │                                      │                       │
│                 ├───────► L_guided: InfoNCE(z_guided, z_text)                 │
│                 │                                      │                       │
│                 │         L_self: InfoNCE(z_self, z_text)  ◄────┤             │
│                 │                                      │                       │
│                 └───────► L_distill: KL(G_self || G_guided) ◄───┘             │
│                                                                                │
│                              ┌──────────────────┐                              │
│                              │ PROJECTION HEAD  │                              │
│                              │ (Shared)         │                              │
│                              └────────┬─────────┘                              │
│                                       │                                        │
│                                       ▼                                        │
│                              ┌──────────────────┐                              │
│                              │ UNIFIED EMBEDDING │                             │
│                              │ z ∈ R^128        │                              │
│                              └──────────────────┘                              │
│                                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
TRAINING:
  Text → Parse → T_feat, confidence
  Geometry + T_feat → Text-Guided Grounding → G_guided → z_guided
  Geometry → Self-Grounding → G_self → z_self
  
  Losses:
    1. L_guided = InfoNCE(z_brep_guided, z_pc_guided, z_text)  [PRIMARY]
    2. L_self = InfoNCE(z_brep_self, z_pc_self, z_text)        [KEY FIX!]
    3. L_distill = KL(G_self || G_guided)                      [AUXILIARY]

INFERENCE (No Text!):
  Geometry → Self-Grounding → G_self → z_self → Retrieval/Generation
```

---

## 4. Component Details

### 4.1 Input Projections

```python
# Dimensions from pre-trained encoders
d_face = 16    # AutoBrep face features (2×2×4 latent)
d_edge = 8     # AutoBrep edge features
d_pc = 1152    # ShapeLLM features
d_text = 3072  # Phi-4-mini features
d_unified = 256  # Internal dimension
```

#### B-Rep Projection (Face + Edge with Type Embeddings)

```python
class BRepProjection(nn.Module):
    def __init__(self, d_face, d_edge, d):
        super().__init__()
        self.proj_face = nn.Sequential(
            nn.Linear(d_face, d),
            nn.LayerNorm(d),
            nn.Dropout(0.1)
        )
        self.proj_edge = nn.Sequential(
            nn.Linear(d_edge, d),
            nn.LayerNorm(d),
            nn.Dropout(0.1)
        )
        # Learnable type embeddings to distinguish faces from edges
        self.face_type_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_type_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
    
    def forward(self, Z_face, Z_edge):
        """
        Args:
            Z_face: (B, F, d_face) - Face features from AutoBrep
            Z_edge: (B, E, d_edge) - Edge features from AutoBrep
        Returns:
            X_brep: (B, F+E, d) - Unified B-Rep tokens
        """
        X_face = self.proj_face(Z_face) + self.face_type_embed
        X_edge = self.proj_edge(Z_edge) + self.edge_type_embed
        X_brep = torch.cat([X_face, X_edge], dim=1)
        return X_brep
```

**Why type embeddings?** Faces and edges have different semantics. The model needs to know which tokens are faces vs edges for proper grounding.

#### Point Cloud Projection (Local + Global)

```python
class PCProjection(nn.Module):
    def __init__(self, d_pc, d):
        super().__init__()
        self.proj_local = nn.Sequential(
            nn.Linear(d_pc, d),
            nn.LayerNorm(d),
            nn.Dropout(0.1)
        )
        self.proj_global = nn.Sequential(
            nn.Linear(d_pc, d),
            nn.LayerNorm(d)
        )
    
    def forward(self, Z_local, Z_global):
        """
        Args:
            Z_local: (B, L, d_pc) - Local patch features (32 patches)
            Z_global: (B, 1, d_pc) - Global token from ShapeLLM
        Returns:
            X_pc: (B, L+1, d) - All PC tokens
        """
        X_local = self.proj_local(Z_local)
        X_global = self.proj_global(Z_global)
        X_pc = torch.cat([X_local, X_global], dim=1)
        return X_pc
```

**Note:** ShapeLLM provides 32 local patches + 1 global token. The global token captures overall shape identity.

---

### 4.2 Text Feature Parsing

The text parser extracts K feature slots from the text description, each representing a potential geometric feature mentioned.

```python
class TextFeatureParser(nn.Module):
    """
    Parse text into K feature slots using cross-attention.
    Each slot represents a geometric feature (e.g., "serrated teeth", "through-hole").
    """
    def __init__(self, d, K=12, num_layers=2, num_heads=8):
        super().__init__()
        self.K = K
        
        # Learnable feature queries
        self.feature_queries = nn.Parameter(torch.randn(K, d) * 0.02)
        
        # Cross-attention: queries attend to text tokens
        self.parser = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d,
                nhead=num_heads,
                dim_feedforward=d * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Predict which slots are active (confidence)
        self.confidence_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1)
        )
    
    def forward(self, X_text, text_mask=None):
        """
        Args:
            X_text: (B, T, d) - Projected text tokens
            text_mask: (B, T) - Valid token mask
        Returns:
            T_feat: (B, K, d) - K feature slot embeddings
            confidence: (B, K) - Importance weight per slot
        """
        B = X_text.shape[0]
        
        # Expand queries for batch
        queries = self.feature_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attend to text
        T_feat = self.parser(
            queries, X_text,
            memory_key_padding_mask=~text_mask if text_mask is not None else None
        )
        
        # Predict confidence (which slots are active)
        conf_logits = self.confidence_head(T_feat).squeeze(-1)
        confidence = torch.sigmoid(conf_logits.clamp(-5, 5))
        
        return T_feat, confidence
```

**Interpretation:**
- Slot 1 might attend to "cylindrical bore"
- Slot 2 might attend to "serrated teeth"
- Slot 3 might attend to "draft angle"
- Confidence indicates whether the text actually mentions this type of feature

---

### 4.3 Modality-Specific Grounding (KEY FIX #1)

```python
class ModalitySpecificGrounding(nn.Module):
    """
    Separate grounding projections for B-Rep and Point Cloud.
    
    KEY INSIGHT: B-Rep tokens are discrete semantic units (faces, edges),
    while PC tokens are spatial patches. They need different projections
    to map to the same grounding space.
    """
    def __init__(self, d, d_ground=128):
        super().__init__()
        
        # Text projection (shared - text is the anchor)
        self.proj_text = nn.Linear(d, d_ground)
        
        # B-Rep projection (deeper - needs to learn CAD semantics)
        self.proj_brep = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d_ground)
        )
        
        # PC projection (deeper - needs to learn spatial semantics)
        self.proj_pc = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d_ground)
        )
        
        # Learnable temperature per modality
        # Allows different "sharpness" of attention
        self.log_tau_brep = nn.Parameter(torch.log(torch.tensor(0.1)))
        self.log_tau_pc = nn.Parameter(torch.log(torch.tensor(0.1)))
    
    def compute_grounding(self, T_feat, X_geo, modality, geo_mask=None):
        """
        Compute grounding matrix: where does each text slot attend in geometry?
        
        Args:
            T_feat: (B, K, d) - Text feature slots
            X_geo: (B, N, d) - Geometry tokens
            modality: 'brep' or 'pc'
            geo_mask: (B, N) - Valid geometry tokens
        
        Returns:
            G: (B, K, N) - Grounding matrix (soft attention)
        """
        # Project to grounding space
        T_g = self.proj_text(T_feat)  # (B, K, d_g)
        
        if modality == 'brep':
            X_g = self.proj_brep(X_geo)  # (B, N, d_g)
            tau = self.log_tau_brep.exp().clamp(0.01, 1.0)
        else:
            X_g = self.proj_pc(X_geo)
            tau = self.log_tau_pc.exp().clamp(0.01, 1.0)
        
        # Scaled dot-product attention
        scores = torch.bmm(T_g, X_g.transpose(-2, -1))  # (B, K, N)
        scores = scores / (T_g.shape[-1] ** 0.5 * tau)
        
        # Mask invalid positions
        if geo_mask is not None:
            scores = scores.masked_fill(~geo_mask.unsqueeze(1), float('-inf'))
        
        # Softmax to get attention weights
        G = F.softmax(scores, dim=-1)
        G = torch.nan_to_num(G, nan=0.0)  # Handle all-masked rows
        
        return G
```

**Why separate projections?**

| Modality | Token Meaning | Example |
|----------|---------------|---------|
| B-Rep Face | Discrete geometric surface | "Cylindrical face of bore" |
| B-Rep Edge | Boundary between faces | "Fillet edge" |
| PC Patch | Spatial region of points | "Points in this 3D region" |

These have fundamentally different semantics. A shared projection assumes they're comparable, which they're not.

---

### 4.4 Hierarchical Aggregation

```python
class HierarchicalAggregator(nn.Module):
    """
    Two-level feature extraction:
    1. Global: Overall shape identity ("This is a gear")
    2. Detail: Fine-grained features ("It has 32 teeth")
    
    Global CONDITIONING detail: "I know it's a gear, so look for teeth"
    """
    def __init__(self, d, num_detail_queries=8):
        super().__init__()
        
        # Global extraction: single query for overall shape
        self.global_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.global_attn = nn.MultiheadAttention(d, 8, batch_first=True, dropout=0.1)
        self.global_norm = nn.LayerNorm(d)
        self.global_ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )
        
        # Detail extraction: multiple queries for fine features
        self.detail_queries = nn.Parameter(torch.randn(1, num_detail_queries, d) * 0.02)
        self.detail_attn = nn.MultiheadAttention(d, 8, batch_first=True, dropout=0.1)
        self.detail_norm = nn.LayerNorm(d)
        self.detail_ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d)
        )
        
        # Global-to-detail conditioning
        self.global_to_detail = nn.Linear(d, d)
        
        # Learned fusion weights
        self.fusion_gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, X_geo, G, confidence, geo_mask=None):
        """
        Args:
            X_geo: (B, N, d) - Geometry tokens
            G: (B, K, N) - Grounding matrix
            confidence: (B, K) - Slot confidence
            geo_mask: (B, N) - Valid token mask
        
        Returns:
            z_global: (B, d) - Global shape embedding
            z_detail: (B, d) - Detail features embedding
            z_unified: (B, d) - Fused embedding
        """
        B = X_geo.shape[0]
        
        # Compute importance weighting from grounding
        # Which geometry tokens are important based on text attention?
        importance = (G * confidence.unsqueeze(-1)).sum(dim=1)  # (B, N)
        importance = importance / (importance.sum(dim=-1, keepdim=True) + 1e-8)
        
        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 1: Global Extraction
        # ─────────────────────────────────────────────────────────────────────
        
        global_q = self.global_query.expand(B, -1, -1)
        z_global, attn_global = self.global_attn(
            global_q, X_geo, X_geo,
            key_padding_mask=~geo_mask if geo_mask is not None else None
        )
        z_global = self.global_norm(z_global + global_q)
        z_global = z_global + self.global_ffn(z_global)
        z_global = z_global.squeeze(1)  # (B, d)
        
        # ─────────────────────────────────────────────────────────────────────
        # LEVEL 2: Detail Extraction (Conditioned on Global)
        # ─────────────────────────────────────────────────────────────────────
        
        detail_q = self.detail_queries.expand(B, -1, -1)
        
        # Global context guides detail queries
        global_cond = self.global_to_detail(z_global).unsqueeze(1)
        detail_q = detail_q + global_cond
        
        z_detail, attn_detail = self.detail_attn(
            detail_q, X_geo, X_geo,
            key_padding_mask=~geo_mask if geo_mask is not None else None
        )
        z_detail = self.detail_norm(z_detail + detail_q)
        z_detail = z_detail + self.detail_ffn(z_detail)
        z_detail = z_detail.mean(dim=1)  # (B, d)
        
        # ─────────────────────────────────────────────────────────────────────
        # FUSION
        # ─────────────────────────────────────────────────────────────────────
        
        concat = torch.cat([z_global, z_detail], dim=-1)
        gate = self.fusion_gate(concat)  # (B, 2)
        z_unified = gate[:, 0:1] * z_global + gate[:, 1:2] * z_detail
        
        return z_global, z_detail, z_unified, attn_global, attn_detail
```

**Why hierarchical?**

For fine-grained discrimination (32 vs 64 teeth):
1. Global level identifies: "This is a gear"
2. Global conditions detail: "Look for teeth specifically"
3. Detail level extracts: "Count/characterize the teeth"

Without global conditioning, detail queries might attend to irrelevant features.

---

### 4.5 Joint Self-Grounding (KEY FIX #2)

```python
class JointSelfGrounding(nn.Module):
    """
    Self-grounding for text-free inference.
    
    KEY INSIGHT: Train jointly with contrastive loss, not just distillation!
    
    The queries adapt to each geometry input, learning to identify
    important regions WITHOUT text guidance.
    """
    def __init__(self, d, num_slots, num_layers=2, num_heads=8):
        super().__init__()
        self.num_slots = num_slots
        
        # Learnable base queries
        self.base_queries = nn.Parameter(torch.randn(num_slots, d) * 0.02)
        
        # Geometry-adaptive: queries attend to geometry to understand it
        self.query_adapter = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d,
                nhead=num_heads,
                dim_feedforward=d * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Modality-specific grounding (mirrors text-guided)
        self.proj_brep = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 128)
        )
        self.proj_pc = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 128)
        )
        self.proj_query = nn.Linear(d, 128)
        
        # Confidence prediction (which slots matter)
        self.confidence_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1)
        )
        
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.1)))
    
    def forward(self, X_geo, modality, geo_mask=None):
        """
        Compute self-grounding WITHOUT text input.
        
        Args:
            X_geo: (B, N, d) - Geometry tokens
            modality: 'brep' or 'pc'
            geo_mask: (B, N) - Valid token mask
        
        Returns:
            G_self: (B, K, N) - Self-grounding matrix
            confidence: (B, K) - Learned slot confidence
        """
        B = X_geo.shape[0]
        
        # Expand base queries
        Q = self.base_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Adapt queries to THIS specific geometry
        # (Cross-attention: queries learn about the geometry)
        Q = self.query_adapter(
            Q, X_geo,
            memory_key_padding_mask=~geo_mask if geo_mask is not None else None
        )
        
        # Predict confidence per slot
        conf_logits = self.confidence_head(Q).squeeze(-1)
        confidence = torch.sigmoid(conf_logits.clamp(-5, 5))
        
        # Compute grounding matrix (same as text-guided)
        Q_g = self.proj_query(Q)
        
        if modality == 'brep':
            X_g = self.proj_brep(X_geo)
        else:
            X_g = self.proj_pc(X_geo)
        
        tau = self.log_tau.exp().clamp(0.01, 1.0)
        scores = torch.bmm(Q_g, X_g.transpose(-2, -1)) / (128 ** 0.5 * tau)
        
        if geo_mask is not None:
            scores = scores.masked_fill(~geo_mask.unsqueeze(1), float('-inf'))
        
        G_self = F.softmax(scores, dim=-1)
        G_self = torch.nan_to_num(G_self, nan=0.0)
        
        return G_self, confidence
```

**Why geometry-adaptive queries?**

Fixed queries (GFA v1) can't adapt to different geometries:
- For a gear: queries should focus on teeth
- For a housing: queries should focus on mounting holes

Adaptive queries cross-attend to geometry first, learning what's important for THIS specific shape.

---

### 4.6 Complete Architecture

```python
class CLIP4CAD_GFA_v2(nn.Module):
    """
    CLIP4CAD-GFA v2: Grounded Feature Alignment for CAD Multimodal Learning
    
    Key innovations:
    1. Modality-specific grounding projections
    2. Joint self-grounding training (not post-hoc!)
    3. Hierarchical global→detail extraction
    4. Unified embedding for retrieval + generation
    """
    def __init__(self, config):
        super().__init__()
        
        d = config.d_unified  # 256
        K = config.num_slots  # 12
        
        # ─────────────────────────────────────────────────────────────────────
        # Input Projections
        # ─────────────────────────────────────────────────────────────────────
        
        self.brep_proj = BRepProjection(config.d_face, config.d_edge, d)
        self.pc_proj = PCProjection(config.d_pc, d)
        self.text_proj = nn.Sequential(
            nn.Linear(config.d_text, d),
            nn.LayerNorm(d),
            nn.Dropout(0.1)
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Text Parsing
        # ─────────────────────────────────────────────────────────────────────
        
        self.text_parser = TextFeatureParser(d, K)
        
        # ─────────────────────────────────────────────────────────────────────
        # Grounding
        # ─────────────────────────────────────────────────────────────────────
        
        self.grounding = ModalitySpecificGrounding(d, d_ground=128)
        self.self_grounding = JointSelfGrounding(d, K)
        
        # ─────────────────────────────────────────────────────────────────────
        # Hierarchical Aggregation
        # ─────────────────────────────────────────────────────────────────────
        
        self.hierarchical_agg = HierarchicalAggregator(d, num_detail_queries=8)
        
        # ─────────────────────────────────────────────────────────────────────
        # Projection Head (Shared)
        # ─────────────────────────────────────────────────────────────────────
        
        self.proj_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.d_proj)  # 128
        )
        
        # Learnable temperature
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.07)))
    
    def encode_text(self, H_text, text_mask=None):
        """Encode text into feature slots and global embedding."""
        X_text = self.text_proj(H_text)
        T_feat, confidence = self.text_parser(X_text, text_mask)
        
        # Global text embedding (confidence-weighted)
        z_text = (T_feat * confidence.unsqueeze(-1)).sum(dim=1)
        z_text = z_text / (confidence.sum(dim=1, keepdim=True) + 1e-8)
        
        return {
            'T_feat': T_feat,
            'confidence': confidence,
            'z_text': z_text,
            'X_text': X_text
        }
    
    def encode_geometry_guided(self, X_geo, T_feat, confidence, modality, geo_mask=None):
        """Text-guided geometry encoding (training path)."""
        
        # Compute grounding matrix
        G = self.grounding.compute_grounding(T_feat, X_geo, modality, geo_mask)
        
        # Extract grounded features
        F_grounded = torch.bmm(G, X_geo)  # (B, K, d)
        
        # Hierarchical aggregation
        z_global, z_detail, z_unified, attn_g, attn_d = self.hierarchical_agg(
            X_geo, G, confidence, geo_mask
        )
        
        return {
            'z_unified': z_unified,
            'z_global': z_global,
            'z_detail': z_detail,
            'G': G,
            'F_grounded': F_grounded,
            'attn_global': attn_g,
            'attn_detail': attn_d
        }
    
    def encode_geometry_self(self, X_geo, modality, geo_mask=None):
        """Self-grounding geometry encoding (inference path)."""
        
        # Self-grounding
        G_self, confidence = self.self_grounding(X_geo, modality, geo_mask)
        
        # Same aggregation pipeline
        F_grounded = torch.bmm(G_self, X_geo)
        z_global, z_detail, z_unified, attn_g, attn_d = self.hierarchical_agg(
            X_geo, G_self, confidence, geo_mask
        )
        
        return {
            'z_unified': z_unified,
            'z_global': z_global,
            'z_detail': z_detail,
            'G': G_self,
            'confidence': confidence
        }
    
    def forward(self, Z_face, Z_edge, Z_pc_local, Z_pc_global, H_text,
                face_mask=None, edge_mask=None, text_mask=None):
        """
        Full forward pass with both encoding paths.
        
        Args:
            Z_face: (B, F, d_face) - Face features from AutoBrep
            Z_edge: (B, E, d_edge) - Edge features from AutoBrep
            Z_pc_local: (B, L, d_pc) - Local PC features from ShapeLLM
            Z_pc_global: (B, 1, d_pc) - Global PC token
            H_text: (B, T, d_text) - Text features from Phi-4
            face_mask: (B, F) - Valid faces
            edge_mask: (B, E) - Valid edges
            text_mask: (B, T) - Valid text tokens
        
        Returns:
            Dictionary with all embeddings and intermediate outputs
        """
        # ─────────────────────────────────────────────────────────────────────
        # Project inputs
        # ─────────────────────────────────────────────────────────────────────
        
        X_brep = self.brep_proj(Z_face, Z_edge)
        brep_mask = torch.cat([face_mask, edge_mask], dim=1) if face_mask is not None else None
        
        X_pc = self.pc_proj(Z_pc_local, Z_pc_global)
        
        # ─────────────────────────────────────────────────────────────────────
        # Encode text
        # ─────────────────────────────────────────────────────────────────────
        
        text_out = self.encode_text(H_text, text_mask)
        z_text = self.proj_head(text_out['z_text'])
        
        # ─────────────────────────────────────────────────────────────────────
        # TEXT-GUIDED ENCODING (Training path)
        # ─────────────────────────────────────────────────────────────────────
        
        brep_guided = self.encode_geometry_guided(
            X_brep, text_out['T_feat'], text_out['confidence'], 'brep', brep_mask
        )
        pc_guided = self.encode_geometry_guided(
            X_pc, text_out['T_feat'], text_out['confidence'], 'pc', None
        )
        
        z_brep_guided = self.proj_head(brep_guided['z_unified'])
        z_pc_guided = self.proj_head(pc_guided['z_unified'])
        
        # Detail embeddings (for hard negative mining)
        z_brep_detail = self.proj_head(brep_guided['z_detail'])
        z_pc_detail = self.proj_head(pc_guided['z_detail'])
        
        # ─────────────────────────────────────────────────────────────────────
        # SELF ENCODING (Inference path)
        # ─────────────────────────────────────────────────────────────────────
        
        brep_self = self.encode_geometry_self(X_brep, 'brep', brep_mask)
        pc_self = self.encode_geometry_self(X_pc, 'pc', None)
        
        z_brep_self = self.proj_head(brep_self['z_unified'])
        z_pc_self = self.proj_head(pc_self['z_unified'])
        
        return {
            # Primary embeddings (text-guided)
            'z_brep': z_brep_guided,
            'z_pc': z_pc_guided,
            'z_text': z_text,
            
            # Self-encoded (for inference)
            'z_brep_self': z_brep_self,
            'z_pc_self': z_pc_self,
            
            # Detail level (for hard negatives)
            'z_brep_detail': z_brep_detail,
            'z_pc_detail': z_pc_detail,
            
            # Grounding matrices (for distillation + visualization)
            'G_brep_guided': brep_guided['G'],
            'G_pc_guided': pc_guided['G'],
            'G_brep_self': brep_self['G'],
            'G_pc_self': pc_self['G'],
            
            # Confidence
            'confidence': text_out['confidence'],
            'confidence_brep_self': brep_self['confidence'],
            'confidence_pc_self': pc_self['confidence'],
            
            # Temperature
            'tau': self.log_tau.exp().clamp(0.01, 1.0),
        }
```

---

## 5. Loss Functions

### 5.1 Loss Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              GFA v2 LOSS STRUCTURE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  L_total = 1.0 × L_guided                  [PRIMARY]                           │
│          + λ_self × L_self                 [KEY FIX: Self learns retrieval]    │
│          + λ_distill × L_distill           [Auxiliary: Grounding pattern]      │
│          + λ_detail × L_detail             [Optional: Hard negatives]          │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  L_guided: InfoNCE(z_brep_guided, z_pc_guided, z_text)                  │   │
│  │  ─────────────────────────────────────────────────────                  │   │
│  │  Standard 3-way contrastive: align text-guided embeddings with text    │   │
│  │  This is what worked in GFA v1 (the foundation)                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  L_self: InfoNCE(z_brep_self, z_pc_self, z_text)                       │   │
│  │  ───────────────────────────────────────────────                       │   │
│  │  KEY FIX: Self-encoded embeddings ALSO align with text!                │   │
│  │  This is NOT just distillation - it's direct retrieval training!       │   │
│  │                                                                         │   │
│  │  GFA v1 problem: Self-grounding only had distillation                  │   │
│  │  GFA v2 fix: Self-grounding has DIRECT contrastive loss                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  L_distill: KL(G_self || G_guided.detach())                            │   │
│  │  ──────────────────────────────────────────                            │   │
│  │  Auxiliary: Self-grounding should ATTEND to similar regions as         │   │
│  │  text-grounding. But this is not the main training signal!             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  L_detail: InfoNCE with hard negatives                                 │   │
│  │  ─────────────────────────────────────                                 │   │
│  │  Optional: For fine-grained discrimination (32 vs 64 teeth)            │   │
│  │  Add in Stage 2 after basic alignment is established                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Loss Implementation

```python
class GFAv2Loss(nn.Module):
    """
    GFA v2 Loss Function
    
    KEY INSIGHT: Self-grounding must learn DIRECTLY via contrastive loss,
    not just by mimicking text-grounding patterns (distillation alone fails).
    """
    def __init__(self, lambda_self=0.5, lambda_distill=0.1, lambda_detail=0.0):
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_distill = lambda_distill
        self.lambda_detail = lambda_detail
    
    def forward(self, outputs, hard_negatives=None):
        """
        Compute all losses.
        
        Args:
            outputs: Model forward outputs
            hard_negatives: Optional indices for hard negative mining
        
        Returns:
            Dictionary with all loss components
        """
        losses = {}
        tau = outputs['tau']
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. GUIDED CONTRASTIVE (Primary - what works)
        # ─────────────────────────────────────────────────────────────────────
        
        z_brep = F.normalize(outputs['z_brep'], dim=-1)
        z_pc = F.normalize(outputs['z_pc'], dim=-1)
        z_text = F.normalize(outputs['z_text'], dim=-1)
        
        losses['guided'] = self.infonce_3way(z_brep, z_pc, z_text, tau)
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. SELF CONTRASTIVE (KEY FIX!)
        # ─────────────────────────────────────────────────────────────────────
        
        z_brep_self = F.normalize(outputs['z_brep_self'], dim=-1)
        z_pc_self = F.normalize(outputs['z_pc_self'], dim=-1)
        
        # Self-encoded should ALSO align with text!
        # This is DIRECT retrieval training, not just distillation
        losses['self'] = self.infonce_3way(z_brep_self, z_pc_self, z_text, tau)
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. GROUNDING DISTILLATION (Auxiliary)
        # ─────────────────────────────────────────────────────────────────────
        
        G_brep_guided = outputs['G_brep_guided'].detach()  # Stop gradient!
        G_pc_guided = outputs['G_pc_guided'].detach()
        G_brep_self = outputs['G_brep_self']
        G_pc_self = outputs['G_pc_self']
        
        # KL divergence: self should attend to similar regions
        loss_distill_brep = F.kl_div(
            torch.log(G_brep_self + 1e-8),
            G_brep_guided,
            reduction='batchmean'
        )
        loss_distill_pc = F.kl_div(
            torch.log(G_pc_self + 1e-8),
            G_pc_guided,
            reduction='batchmean'
        )
        losses['distill'] = (loss_distill_brep + loss_distill_pc) / 2
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. DETAIL CONTRASTIVE (Optional - for hard negatives)
        # ─────────────────────────────────────────────────────────────────────
        
        if self.lambda_detail > 0:
            z_brep_d = F.normalize(outputs['z_brep_detail'], dim=-1)
            z_pc_d = F.normalize(outputs['z_pc_detail'], dim=-1)
            
            if hard_negatives is not None:
                losses['detail'] = self.hard_negative_infonce(
                    z_brep_d, z_text, hard_negatives, tau * 0.7
                )
            else:
                losses['detail'] = self.infonce_3way(z_brep_d, z_pc_d, z_text, tau)
        else:
            losses['detail'] = torch.tensor(0.0, device=tau.device)
        
        # ─────────────────────────────────────────────────────────────────────
        # TOTAL
        # ─────────────────────────────────────────────────────────────────────
        
        losses['total'] = (
            1.0 * losses['guided'] +
            self.lambda_self * losses['self'] +
            self.lambda_distill * losses['distill'] +
            self.lambda_detail * losses['detail']
        )
        
        return losses
    
    def infonce_3way(self, z_a, z_b, z_c, tau):
        """Standard 3-way InfoNCE loss."""
        B = z_a.shape[0]
        labels = torch.arange(B, device=z_a.device)
        
        loss = 0
        for zi, zj in [(z_a, z_c), (z_b, z_c), (z_a, z_b)]:
            logits = zi @ zj.T / tau
            loss += (F.cross_entropy(logits, labels) + 
                     F.cross_entropy(logits.T, labels)) / 2
        return loss / 3
    
    def hard_negative_infonce(self, z_geo, z_text, hard_negs, tau):
        """InfoNCE with hard negative emphasis."""
        B = z_geo.shape[0]
        labels = torch.arange(B, device=z_geo.device)
        
        logits = z_geo @ z_text.T / tau
        
        # Emphasize hard negatives (lower temperature = harder)
        if hard_negs is not None:
            for i, negs in enumerate(hard_negs):
                if negs is not None and len(negs) > 0:
                    logits[i, negs] = logits[i, negs] / 0.5
        
        return F.cross_entropy(logits, labels)
```

---

## 6. Training Procedure

### 6.1 Two-Stage Training

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING SCHEDULE                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  STAGE 1: Establish Alignment (Epochs 1-15)                                    │
│  ──────────────────────────────────────────                                    │
│  Goal: Learn basic text-geometry alignment                                     │
│                                                                                 │
│  Loss weights:                                                                 │
│    λ_self = 0.3      (start mild)                                             │
│    λ_distill = 0.1                                                             │
│    λ_detail = 0.0    (disabled)                                               │
│                                                                                 │
│  Learning rate: 3e-5                                                           │
│  Hard negatives: DISABLED                                                      │
│                                                                                 │
│  Validation metrics to track:                                                  │
│    - Text→BRep R@1 (target: ≥ 50%)                                            │
│    - Self-grounding cosine similarity (target: ≥ 0.8)                         │
│                                                                                 │
│  ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                 │
│  STAGE 2: Fine-Grained Discrimination (Epochs 16-35)                          │
│  ────────────────────────────────────────────────────                          │
│  Goal: Distinguish similar shapes (32 vs 64 teeth)                            │
│                                                                                 │
│  Loss weights:                                                                 │
│    λ_self = 0.5      (increase)                                               │
│    λ_distill = 0.2   (increase)                                               │
│    λ_detail = 0.3    (ENABLE)                                                 │
│                                                                                 │
│  Learning rate: 1e-5 (reduced)                                                 │
│  Hard negatives: ENABLED                                                       │
│                                                                                 │
│  Validation metrics to track:                                                  │
│    - Text→BRep R@1 (target: ≥ 55%)                                            │
│    - Fine-grained retrieval accuracy                                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Hard Negative Mining

```python
class HardNegativeMiner:
    """
    Mine hard negatives for fine-grained discrimination.
    
    Hard negatives: Samples that are similar but should be distinguished
    (e.g., gears with different tooth counts)
    """
    def __init__(self, k=10):
        self.k = k
        self.index = None
    
    def build_index(self, embeddings):
        """Build FAISS index from embeddings."""
        import faiss
        
        embeddings_np = embeddings.cpu().numpy().astype('float32')
        faiss.normalize_L2(embeddings_np)
        
        d = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings_np)
    
    def mine(self, query_embeddings, exclude_self=True):
        """
        Find hard negatives for each query.
        
        Returns:
            hard_neg_indices: List of arrays, one per query
        """
        import faiss
        
        query_np = query_embeddings.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_np)
        
        # Search for nearest neighbors
        _, indices = self.index.search(query_np, self.k + 1)
        
        hard_negatives = []
        for i, neighbors in enumerate(indices):
            if exclude_self:
                # Remove self from neighbors
                negs = [n for n in neighbors if n != i][:self.k]
            else:
                negs = neighbors[:self.k].tolist()
            hard_negatives.append(negs)
        
        return hard_negatives
```

### 6.3 Training Configuration

```python
@dataclass
class GFAv2Config:
    # Model dimensions
    d_face: int = 16
    d_edge: int = 8
    d_pc: int = 1152
    d_text: int = 3072
    d_unified: int = 256
    d_proj: int = 128
    d_ground: int = 128
    
    # Architecture
    num_slots: int = 12
    num_detail_queries: int = 8
    num_heads: int = 8
    num_parser_layers: int = 2
    num_self_ground_layers: int = 2
    dropout: float = 0.1
    
    # Training Stage 1
    stage1_epochs: int = 15
    stage1_lr: float = 3e-5
    stage1_lambda_self: float = 0.3
    stage1_lambda_distill: float = 0.1
    stage1_lambda_detail: float = 0.0
    
    # Training Stage 2
    stage2_epochs: int = 20
    stage2_lr: float = 1e-5
    stage2_lambda_self: float = 0.5
    stage2_lambda_distill: float = 0.2
    stage2_lambda_detail: float = 0.3
    
    # General
    batch_size: int = 512
    warmup_epochs: int = 2
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Hard negative mining
    hard_negative_k: int = 10
    mine_every_n_epochs: int = 5
```

---

## 7. Implementation Guide

### 7.1 File Structure

```
clip4cad/
├── models/
│   ├── gfa_v2.py              # Main model
│   ├── components/
│   │   ├── projections.py     # BRepProjection, PCProjection
│   │   ├── text_parser.py     # TextFeatureParser
│   │   ├── grounding.py       # ModalitySpecificGrounding
│   │   ├── self_grounding.py  # JointSelfGrounding
│   │   └── aggregation.py     # HierarchicalAggregator
│   └── losses.py              # GFAv2Loss
├── training/
│   ├── trainer.py             # Training loop
│   ├── hard_negatives.py      # HardNegativeMiner
│   └── config.py              # GFAv2Config
├── evaluation/
│   ├── retrieval.py           # Retrieval metrics
│   └── visualization.py       # Grounding visualization
└── data/
    ├── dataset.py             # MM-CAD dataset
    └── preprocessing.py       # Feature loading
```

### 7.2 Key Implementation Details

#### Numerical Stability (FP16)

```python
# Clamp logits BEFORE sigmoid
logits = self.confidence_head(features)
logits = logits.clamp(-5, 5)  # Prevent extreme values
confidence = torch.sigmoid(logits)

# Safe softmax
scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
G = F.softmax(scores, dim=-1)
G = torch.nan_to_num(G, nan=0.0)  # Handle all-masked rows

# Safe division
z = weighted_sum / (weights.sum() + 1e-8)

# Safe KL divergence
loss = F.kl_div(torch.log(G_self + 1e-8), G_guided, reduction='batchmean')
```

#### Gradient Checkpointing (Memory Efficiency)

```python
from torch.utils.checkpoint import checkpoint

class GFAv2WithCheckpointing(CLIP4CAD_GFA_v2):
    def encode_geometry_guided(self, X_geo, T_feat, confidence, modality, geo_mask=None):
        # Checkpoint the heavy computation
        return checkpoint(
            super().encode_geometry_guided,
            X_geo, T_feat, confidence, modality, geo_mask,
            use_reentrant=False
        )
```

### 7.3 Data Loading

```python
class MMCAD_Dataset(Dataset):
    """
    MM-CAD Dataset for GFA v2 training.
    
    Features are pre-extracted and stored in FP16.
    """
    def __init__(self, split='train', data_dir='data/mmcad'):
        self.split = split
        self.data_dir = Path(data_dir)
        
        # Load indices
        with open(self.data_dir / f'{split}_indices.json') as f:
            self.indices = json.load(f)
        
        # Memory-mapped feature files
        self.brep_faces = np.load(self.data_dir / 'brep_faces.npy', mmap_mode='r')
        self.brep_edges = np.load(self.data_dir / 'brep_edges.npy', mmap_mode='r')
        self.pc_local = np.load(self.data_dir / 'pc_local.npy', mmap_mode='r')
        self.pc_global = np.load(self.data_dir / 'pc_global.npy', mmap_mode='r')
        self.text_features = np.load(self.data_dir / 'text_features.npy', mmap_mode='r')
        
        # Load masks
        self.face_masks = np.load(self.data_dir / 'face_masks.npy', mmap_mode='r')
        self.edge_masks = np.load(self.data_dir / 'edge_masks.npy', mmap_mode='r')
        self.text_masks = np.load(self.data_dir / 'text_masks.npy', mmap_mode='r')
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        
        return {
            'Z_face': torch.from_numpy(self.brep_faces[i].copy()).float(),
            'Z_edge': torch.from_numpy(self.brep_edges[i].copy()).float(),
            'Z_pc_local': torch.from_numpy(self.pc_local[i].copy()).float(),
            'Z_pc_global': torch.from_numpy(self.pc_global[i].copy()).float(),
            'H_text': torch.from_numpy(self.text_features[i].copy()).float(),
            'face_mask': torch.from_numpy(self.face_masks[i].copy()).bool(),
            'edge_mask': torch.from_numpy(self.edge_masks[i].copy()).bool(),
            'text_mask': torch.from_numpy(self.text_masks[i].copy()).bool(),
        }
```

---

## 8. Expected Results

### 8.1 Performance Targets

| Metric | GFA v1 Best | GFA v2 Target | Notes |
|--------|-------------|---------------|-------|
| Text→BRep R@1 | 54.8% | **55-60%** | Primary benchmark |
| Text→PC R@1 | 59.7% | **62-68%** | Should improve with PC-specific grounding |
| PC→BRep R@1 | 29.4% | **32-38%** | Cross-modal consistency |
| Self cosine | 0.08 | **0.90+** | KEY: Self-grounding works |
| Text-free inference | Broken | **Works** | Enables standalone use |

### 8.2 Training Metrics to Monitor

```python
# Every epoch, log:
metrics = {
    # Loss components
    'loss/total': total_loss,
    'loss/guided': guided_loss,
    'loss/self': self_loss,
    'loss/distill': distill_loss,
    'loss/detail': detail_loss,
    
    # Self-grounding quality
    'self/cosine_brep': cosine_similarity(z_brep_guided, z_brep_self),
    'self/cosine_pc': cosine_similarity(z_pc_guided, z_pc_self),
    'self/grounding_kl': kl_divergence(G_self, G_guided),
    
    # Confidence statistics
    'conf/text_mean': confidence.mean(),
    'conf/text_std': confidence.std(),
    'conf/self_brep_mean': confidence_brep_self.mean(),
    'conf/self_pc_mean': confidence_pc_self.mean(),
    
    # Temperature
    'tau': tau.item(),
    
    # Retrieval (every 5 epochs)
    'retrieval/text_brep_r1': text_brep_r1,
    'retrieval/text_brep_r5': text_brep_r5,
    'retrieval/text_pc_r1': text_pc_r1,
    'retrieval/self_brep_r1': self_brep_r1,  # KEY: Self-grounding retrieval
}
```

### 8.3 Success Criteria

**Stage 1 (Epoch 15):**
- [ ] Text→BRep R@1 ≥ 50%
- [ ] Self cosine similarity ≥ 0.8
- [ ] Validation loss decreasing

**Stage 2 (Epoch 35):**
- [ ] Text→BRep R@1 ≥ 55%
- [ ] Self cosine similarity ≥ 0.9
- [ ] Self-grounding retrieval within 5% of guided

---

## 9. Ablation Plan

### 9.1 Core Ablations

| Ablation | Change | Purpose |
|----------|--------|---------|
| A1: No self contrastive | λ_self=0 | Verify self contrastive is essential |
| A2: No distillation | λ_distill=0 | Verify distillation helps |
| A3: Shared grounding | Single W_g | Verify modality-specific helps |
| A4: No hierarchy | Direct pooling | Verify hierarchy helps |
| A5: No hard negatives | λ_detail=0 in Stage 2 | Verify hard negatives help |

### 9.2 Expected Ablation Results

| Ablation | Expected Text→BRep R@1 | Expected Self Cosine |
|----------|------------------------|----------------------|
| Full GFA v2 | 55-60% | 0.90+ |
| A1: No self contrastive | 50-55% | 0.2-0.4 (broken!) |
| A2: No distillation | 53-57% | 0.80-0.85 |
| A3: Shared grounding | 50-53% | 0.85-0.90 |
| A4: No hierarchy | 52-55% | 0.85-0.90 |
| A5: No hard negatives | 53-57% | 0.90+ |

### 9.3 Ablation Priority

1. **A1 FIRST**: Must verify self contrastive is the key fix
2. **A3 SECOND**: Verify modality-specific grounding helps
3. **A2, A4, A5**: Lower priority, run if time permits

---

## Appendix A: Comparison with GFA v1

| Component | GFA v1 | GFA v2 | Change |
|-----------|--------|--------|--------|
| Grounding projection | Shared W_g_geo | Separate W_g_brep, W_g_pc | **Key fix** |
| Consistency loss | λ=0.5 (hurt) | **REMOVED** | Learned from ablations |
| Local contrastive | λ=0.5 (hurt) | **REMOVED** | Created false negatives |
| Self-grounding training | Distillation only | **Joint contrastive** | **Key fix** |
| Self-grounding queries | Fixed | Geometry-adaptive | Better adaptation |
| Aggregation | Direct weighted pool | Hierarchical global→detail | Fine-grained |
| Hard negatives | None | Detail-level mining | Discrimination |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Grounding** | Mapping text features to geometry regions via attention |
| **Self-grounding** | Grounding without text (for inference) |
| **Confidence** | Importance weight per feature slot (which slots are active) |
| **Hard negative** | Similar but different sample (32 vs 64 teeth) |
| **InfoNCE** | Contrastive loss that pushes positives together, negatives apart |

## Appendix C: Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| NaN loss | Numerical instability | Clamp logits, add epsilon |
| Self cosine stuck at 0.5 | Gate not learning | Simplify gate, check gradient flow |
| Text→BRep dropping | Overfitting | Early stopping, reduce lr |
| Grounding all uniform | Temperature too high | Reduce tau, check initialization |
