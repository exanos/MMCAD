# CLIP4CAD-GFA: Cross-Modal Representation Learning for CAD Models via Grounded Feature Alignment

## Technical Architecture Specification

---

## 1. Introduction and Problem Formulation

Computer-Aided Design (CAD) models serve as the digital foundation of modern manufacturing, encoding precise geometric and topological information through boundary representations (B-Rep). The challenge of learning unified representations that enable cross-modal retrieval between CAD geometry and natural language descriptions remains largely unexplored. Unlike general 3D object understanding, CAD-specific multimodal learning presents unique challenges: (1) descriptions explicitly reference geometric features ("central through-hole", "corner fillets") that correspond to specific B-Rep primitives, (2) multiple geometric representations (B-Rep, point cloud) encode the same underlying shape with different structural biases, and (3) industrial CAD models appear in arbitrary orientations, requiring rotation-robust representations.

We introduce **CLIP4CAD-GFA** (Grounded Feature Alignment), a cross-modal representation learning framework that addresses these challenges through three key innovations:

1. **Explicit Grounding Learning**: Rather than aligning only global embeddings, we learn a grounding matrix that maps text feature mentions to their corresponding geometric primitives.

2. **Cross-Modal Grounding Consistency**: We enforce that the same text mention should ground to geometrically corresponding regions across different shape representations (B-Rep and point cloud).

3. **Adaptive Feature Slot Utilization**: We handle variable-complexity descriptions through confidence-weighted feature slots that naturally adapt to the number of distinct features mentioned.

---

## 2. Preliminaries and Notation

### 2.1 Input Representations

We consider three modalities for each CAD model:

**Boundary Representation (B-Rep).** A B-Rep model consists of parametric surfaces $\mathcal{S} = \{S_i\}_{i=1}^{F}$, curves $\mathcal{C} = \{C_j\}_{j=1}^{E}$, and their topological connectivity $\mathcal{T}_{SC} \in \{0,1\}^{F \times E}$. Following Willis et al. [AutoBrep, 2025], we represent each face as a point grid $\mathbf{G}_i \in \mathbb{R}^{32 \times 32 \times 3}$ sampled uniformly in the parametric $(u,v)$ domain, and each edge as a point sequence $\mathbf{E}_j \in \mathbb{R}^{32 \times 3}$. These are encoded using AutoBrep's FSQ VAE encoder to yield face tokens $\mathbf{z}_i^{\text{face}} \in \mathbb{R}^{d_f}$ and edge tokens $\mathbf{z}_j^{\text{edge}} \in \mathbb{R}^{d_e}$, where $d_f = 48$ and $d_e = 12$.

The AutoBrep encoder architecture consists of:
- **Surface encoder**: 4 stages with channels [128, 256, 512, 512], each containing 2 ResBlocks with single GroupNorm, followed by channel projections between stages. Final conv_out maps 512 → 4 latent channels.
- **Edge encoder**: Similar 4-stage 1D architecture with channels [128, 256, 512, 512], processing edge point sequences.
- **Pretrained weights**: Automatically downloaded from HuggingFace (`SamGiantEagle/AutoBrep`) when not present locally.

**Point Cloud.** We use pre-computed features from a finetuned ShapeLLM/ReCon++ encoder [Qi et al., 2024] trained on 32K CAD models with multi-modal supervision (images, text, point clouds). The ReCon++ encoder processes 10,000 points through a 24-layer transformer with 16 attention heads and embed_dim=1024. We use the pooled local patch features concatenated with the global token, yielding $\mathbf{Z}^{\text{pc}} \in \mathbb{R}^{48 \times 1024}$, representing 32 spatial regions aggregated from the original 512 patch tokens plus a global shape summary.

**Text Description.** Each model is paired with a natural language description $T$ that explicitly references geometric features (e.g., "A cylindrical mounting boss with a central through-hole and four corner mounting slots"). We encode using a frozen large language model (Phi-4-mini, 3.8B parameters) to obtain contextualized token embeddings $\mathbf{H}^{\text{text}} \in \mathbb{R}^{L \times d_{\text{llm}}}$ where $d_{\text{llm}} = 3072$.

### 2.2 Design Rationale

A key observation motivating our approach is that CAD descriptions differ fundamentally from general object descriptions. When describing a mechanical part, engineers enumerate specific geometric features: "four corner holes", "central slot", "filleted edges". Each such mention corresponds to identifiable B-Rep primitives (the hole faces, the slot faces, the fillet faces). Standard CLIP-style alignment, which only matches global embeddings, discards this rich correspondence signal. Our architecture explicitly learns and leverages these correspondences.

---

## 3. Architecture Overview

The complete architecture comprises four stages, illustrated in Figure 1:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIP4CAD-GFA Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                    │
│  │   B-Rep      │   │  Point Cloud │   │    Text      │                    │
│  │  (Pre-comp)  │   │  (Pre-comp)  │   │  (Pre-comp)  │                    │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                    │
│         │                  │                  │                             │
│         ▼                  ▼                  ▼                             │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │           Unified Projection Layer (§3.1)            │                  │
│  │     X_brep ∈ R^{N_b×d}    X_pc ∈ R^{48×d}     X_text ∈ R^{L×d}         │
│  └──────────────────────────────────────────────────────┘                  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │         Adaptive Text Feature Parsing (§3.2)         │                  │
│  │    K learnable queries → K feature slots + confidence │                  │
│  │              T_feat ∈ R^{K×d}, c ∈ (0,1)^K           │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │      Bi-directional Grounding Module (§3.3)          │                  │
│  │         G_brep ∈ R^{K×N_b}    G_pc ∈ R^{K×48}        │                  │
│  │    Grounded features: F_brep, F_pc ∈ R^{K×d}         │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │      Cross-Modal Alignment Module (§3.4)             │                  │
│  │    Learned alignment: F_brep → F̃_brep, F_pc → F̃_pc  │                  │
│  │         Grounding consistency in aligned space        │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │      Global Aggregation & Projection (§3.5)          │                  │
│  │    Semantic importance weighting → global embeddings  │                  │
│  │         z_brep, z_pc, z_text ∈ R^{d_proj}            │                  │
│  └──────────────────────────────────────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

All encoder features are **pre-computed and cached** to enable efficient training. Only the projection, grounding, alignment, and aggregation modules (approximately 3.1M parameters) are trained.

---

## 3.1 Unified Projection Layer

The pre-computed encoder features have heterogeneous dimensions. We project all modalities to a common dimension $d = 256$ using learned linear projections with layer normalization:

$$\mathbf{X}^{\text{brep}} = \text{LayerNorm}(\mathbf{Z}^{\text{brep}} \mathbf{W}_{\text{brep}} + \mathbf{b}_{\text{brep}}) \in \mathbb{R}^{N_b \times d}$$

$$\mathbf{X}^{\text{pc}} = \text{LayerNorm}(\mathbf{Z}^{\text{pc}} \mathbf{W}_{\text{pc}} + \mathbf{b}_{\text{pc}}) \in \mathbb{R}^{48 \times d}$$

$$\mathbf{X}^{\text{text}} = \text{LayerNorm}(\mathbf{H}^{\text{text}} \mathbf{W}_{\text{text}} + \mathbf{b}_{\text{text}}) \in \mathbb{R}^{L \times d}$$

where $\mathbf{W}_{\text{brep}} \in \mathbb{R}^{d_{\text{brep}} \times d}$, $\mathbf{W}_{\text{pc}} \in \mathbb{R}^{1024 \times d}$, and $\mathbf{W}_{\text{text}} \in \mathbb{R}^{d_{\text{llm}} \times d}$ are learned projection matrices.

For B-Rep, we concatenate face and edge tokens with learned type embeddings:

$$\mathbf{Z}^{\text{brep}} = [\mathbf{z}_1^{\text{face}} + \mathbf{e}_{\text{face}}; \ldots; \mathbf{z}_F^{\text{face}} + \mathbf{e}_{\text{face}}; \mathbf{z}_1^{\text{edge}} + \mathbf{e}_{\text{edge}}; \ldots; \mathbf{z}_E^{\text{edge}} + \mathbf{e}_{\text{edge}}]$$

where $\mathbf{e}_{\text{face}}, \mathbf{e}_{\text{edge}} \in \mathbb{R}^{d_{\text{brep}}}$ are learned embeddings distinguishing faces from edges.

**Padding and Masking.** B-Rep models have variable numbers of primitives. We pad to maximum sizes $F_{\max} = 192$, $E_{\max} = 512$ (total $N_b = 704$ tokens) and maintain boolean masks $\mathbf{m}^{\text{brep}} \in \{0,1\}^{N_b}$ indicating valid tokens. Point cloud tokens have fixed count (48: 32 local patches + 16 global tokens) and require no masking. Text description tokens are padded to $L_{\max} = 512$ with mask $\mathbf{m}^{\text{text}}$.

---

## 3.2 Adaptive Text Feature Parsing

CAD descriptions naturally decompose into mentions of distinct geometric features. Rather than treating the description as a single embedding, we introduce $K$ learnable **feature slot queries** that attend to the text and extract distinct feature mentions. Critically, we predict a **confidence score** for each slot indicating whether it found meaningful content, enabling adaptive utilization based on description complexity.

### 3.2.1 Feature Slot Queries

We define $K = 12$ learnable query vectors $\mathbf{Q}^{\text{feat}} \in \mathbb{R}^{K \times d}$ with associated positional encodings $\mathbf{P}^{\text{feat}} \in \mathbb{R}^{K \times d}$, both initialized from $\mathcal{N}(0, 0.02^2)$.

The queries attend to the projected text tokens through a cross-attention block with $L_{\text{parse}} = 2$ layers:

$$\mathbf{T}^{\text{feat}} = \text{CrossAttnBlock}(\mathbf{Q}^{\text{feat}} + \mathbf{P}^{\text{feat}}, \mathbf{X}^{\text{text}}, \mathbf{X}^{\text{text}}; \mathbf{m}^{\text{text}}) \in \mathbb{R}^{K \times d}$$

Each cross-attention layer follows the pre-norm transformer architecture [Xiong et al., 2020]:

$$\hat{\mathbf{Q}}^{(l)} = \text{LayerNorm}(\mathbf{Q}^{(l-1)})$$
$$\mathbf{Q}^{(l)} = \mathbf{Q}^{(l-1)} + \text{MHA}(\hat{\mathbf{Q}}^{(l)}, \text{LayerNorm}(\mathbf{X}^{\text{text}}), \mathbf{X}^{\text{text}}; \mathbf{m}^{\text{text}})$$
$$\mathbf{Q}^{(l)} = \mathbf{Q}^{(l)} + \text{FFN}(\text{LayerNorm}(\mathbf{Q}^{(l)}))$$

where MHA denotes multi-head attention with $H = 8$ heads, and FFN is a two-layer feed-forward network with GELU activation and expansion factor 4.

### 3.2.2 Confidence Prediction

Not all descriptions mention $K$ distinct features. A simple description like "rectangular plate" activates perhaps one or two meaningful feature slots, while "mounting bracket with four corner holes, central boss, and filleted edges" activates more. To handle this variability, we predict a confidence score $c_k \in (0, 1)$ for each feature slot:

$$c_k = \sigma\left(\mathbf{w}_c^\top \text{GELU}(\mathbf{W}_c \mathbf{t}_k^{\text{feat}} + \mathbf{b}_c) + b_c'\right)$$

where $\mathbf{W}_c \in \mathbb{R}^{(d/4) \times d}$, $\mathbf{w}_c \in \mathbb{R}^{d/4}$, and $\sigma$ is the sigmoid function.

The confidence scores serve three purposes:
1. **Loss weighting**: Low-confidence slots contribute less to grounding losses
2. **Aggregation weighting**: Global embeddings weight feature slots by confidence  
3. **Interpretability**: Confidence indicates how many distinct features the model detected

We collect the confidence scores as $\mathbf{c} = [c_1, \ldots, c_K]^\top \in (0, 1)^K$.

---

## 3.3 Bi-directional Grounding Module

The core novelty of our architecture is the **grounding matrix** $\mathbf{G} \in \mathbb{R}^{K \times N}$ that explicitly maps each text feature slot to its corresponding geometric tokens. Unlike attention mechanisms used for compression [Chen et al., HCC-CAD, 2024], our grounding is specifically designed for cross-modal correspondence and is supervised by semantic consistency objectives.

### 3.3.1 Grounding Space Projection

To compute grounding, we project text feature slots and geometric tokens into a shared **grounding space** of dimension $d_g = 128$:

$$\mathbf{T}^g = \mathbf{T}^{\text{feat}} \mathbf{W}_g^{\text{text}} \in \mathbb{R}^{K \times d_g}$$
$$\mathbf{X}^g_{\text{brep}} = \mathbf{X}^{\text{brep}} \mathbf{W}_g^{\text{geo}} \in \mathbb{R}^{N_b \times d_g}$$
$$\mathbf{X}^g_{\text{pc}} = \mathbf{X}^{\text{pc}} \mathbf{W}_g^{\text{geo}} \in \mathbb{R}^{48 \times d_g}$$

Note that B-Rep and point cloud share the same geometric projection $\mathbf{W}_g^{\text{geo}}$, encouraging a unified grounding space for geometry.

### 3.3.2 Grounding Matrix Computation

The grounding matrix is computed as temperature-scaled softmax attention over geometric tokens:

$$\mathbf{G}^{\text{brep}}_{k,n} = \frac{\exp\left(\mathbf{t}_k^g \cdot \mathbf{x}_n^{g,\text{brep}} / (\sqrt{d_g} \cdot \tau_g)\right) \cdot m_n^{\text{brep}}}{\sum_{n'} \exp\left(\mathbf{t}_k^g \cdot \mathbf{x}_{n'}^{g,\text{brep}} / (\sqrt{d_g} \cdot \tau_g)\right) \cdot m_{n'}^{\text{brep}}}$$

where $\tau_g$ is a learnable temperature parameter initialized to 0.1 and clamped to $[0.01, 1.0]$.

Similarly for point cloud:

$$\mathbf{G}^{\text{pc}}_{k,n} = \frac{\exp\left(\mathbf{t}_k^g \cdot \mathbf{x}_n^{g,\text{pc}} / (\sqrt{d_g} \cdot \tau_g)\right)}{\sum_{n'=1}^{48} \exp\left(\mathbf{t}_k^g \cdot \mathbf{x}_{n'}^{g,\text{pc}} / (\sqrt{d_g} \cdot \tau_g)\right)}$$

The grounding matrix $\mathbf{G}_{k,:}$ forms a probability distribution over geometric tokens, indicating which regions correspond to feature mention $k$.

### 3.3.3 Grounded Feature Extraction

Using the grounding matrices, we extract grounded geometric features for each feature slot:

$$\mathbf{F}^{\text{brep}}_k = \sum_n \mathbf{G}^{\text{brep}}_{k,n} \cdot \mathbf{x}_n^{\text{brep}} \in \mathbb{R}^d$$
$$\mathbf{F}^{\text{pc}}_k = \sum_n \mathbf{G}^{\text{pc}}_{k,n} \cdot \mathbf{x}_n^{\text{pc}} \in \mathbb{R}^d$$

These grounded features capture the geometric information corresponding to each text feature mention. For a slot attending to "central through-hole", $\mathbf{F}^{\text{brep}}_k$ aggregates the hole face representations while $\mathbf{F}^{\text{pc}}_k$ aggregates the corresponding point cloud patches.

We also create fused multimodal features by combining grounded geometry with text:

$$\tilde{\mathbf{F}}^{\text{brep}}_k = \text{LayerNorm}(\mathbf{F}^{\text{brep}}_k + \mathbf{t}_k^{\text{feat}})$$
$$\tilde{\mathbf{F}}^{\text{pc}}_k = \text{LayerNorm}(\mathbf{F}^{\text{pc}}_k + \mathbf{t}_k^{\text{feat}})$$

---

## 3.4 Cross-Modal Alignment Module

A critical challenge is that B-Rep and point cloud encoders produce fundamentally different representations. AutoBrep encodes parametric surfaces with UV-grid structure, while ShapeLLM/ReCon++ encodes local point patches with multi-modal context from its pre-training on images and text. The same geometric region (e.g., a fillet face) may have quite different encoder representations in each modality.

To address this, we introduce **learned modality-specific alignment layers** that project grounded features into a shared comparison space before computing consistency losses.

### 3.4.1 Alignment Projection

Each geometric modality has a dedicated alignment network:

$$\hat{\mathbf{F}}^{\text{brep}}_k = \text{AlignNet}_{\text{brep}}(\mathbf{F}^{\text{brep}}_k) \in \mathbb{R}^{d_a}$$
$$\hat{\mathbf{F}}^{\text{pc}}_k = \text{AlignNet}_{\text{pc}}(\mathbf{F}^{\text{pc}}_k) \in \mathbb{R}^{d_a}$$

where $d_a = 128$ and each AlignNet is a two-layer MLP:

$$\text{AlignNet}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

with $\mathbf{W}_1 \in \mathbb{R}^{d \times d}$, $\mathbf{W}_2 \in \mathbb{R}^{d \times d_a}$.

### 3.4.2 Rationale

The alignment layers serve a crucial role: they allow the model to learn the mapping between encoder representation spaces rather than assuming direct comparability. This is analogous to the projection heads used in contrastive learning [Chen et al., SimCLR, 2020], where representations are projected before computing the contrastive loss, allowing the encoder to retain information that may not be directly comparable but is useful for the representation.

---

## 3.5 Semantic Importance and Global Aggregation

For retrieval tasks, we require single vector embeddings per modality. We compute these through **semantic importance weighting**, where geometric tokens are weighted by how strongly they are grounded in text.

### 3.5.1 Semantic Importance Scores

For each geometric token, we compute its semantic importance as the total grounding attention it receives, weighted by feature slot confidence:

$$\text{imp}_n^{\text{brep}} = \sum_{k=1}^{K} c_k \cdot \mathbf{G}^{\text{brep}}_{k,n}$$

This differs from attention-coverage-based importance [Chen et al., HCC-CAD, 2024] in a fundamental way: we weight by **semantic grounding** (what the text mentions) rather than **statistical coverage** (what the global queries attended to). Tokens corresponding to mentioned features receive high importance; background regions receive low importance.

We normalize to obtain a probability distribution:

$$\bar{\text{imp}}_n^{\text{brep}} = \frac{\text{imp}_n^{\text{brep}} \cdot m_n^{\text{brep}}}{\sum_{n'} \text{imp}_{n'}^{\text{brep}} \cdot m_{n'}^{\text{brep}}}$$

### 3.5.2 Global Embedding Computation

The global geometric embedding is the importance-weighted average of projected tokens:

$$\mathbf{z}^{\text{brep}}_{\text{geo}} = \sum_n \bar{\text{imp}}_n^{\text{brep}} \cdot \mathbf{x}_n^{\text{brep}} \in \mathbb{R}^d$$

For text, we use confidence-weighted averaging of feature slots:

$$\mathbf{z}^{\text{text}} = \frac{\sum_k c_k \cdot \mathbf{t}_k^{\text{feat}}}{\sum_k c_k} \in \mathbb{R}^d$$

### 3.5.3 Contrastive Projection Head

Following standard practice [Radford et al., CLIP, 2021], we project global embeddings through a learned head before computing contrastive losses:

$$\mathbf{z}^{\text{brep}}_{\text{proj}} = \text{ProjHead}(\mathbf{z}^{\text{brep}}_{\text{geo}}) \in \mathbb{R}^{d_{\text{proj}}}$$

where $d_{\text{proj}} = 128$ and ProjHead is a two-layer MLP with GELU activation.

---

## 4. Training Objectives

Our training combines four complementary objectives that together supervise both fine-grained grounding and global alignment.

### 4.1 Grounding Consistency Loss (Novel)

The central novel objective enforces that the same text feature mention should ground to geometrically corresponding regions across modalities. We compute this in the aligned space:

$$\mathcal{L}_{\text{consist}} = \frac{1}{|\mathcal{K}_{\text{active}}|} \sum_{k \in \mathcal{K}_{\text{active}}} \left(1 - \frac{\hat{\mathbf{F}}^{\text{brep}}_k \cdot \hat{\mathbf{F}}^{\text{pc}}_k}{\|\hat{\mathbf{F}}^{\text{brep}}_k\| \|\hat{\mathbf{F}}^{\text{pc}}_k\|}\right)$$

where $\mathcal{K}_{\text{active}} = \{k : c_k > \tau_c\}$ is the set of active feature slots (confidence above threshold $\tau_c = 0.3$).

**Interpretation.** This loss forces the model to learn grounding that is geometrically consistent. If feature slot 3 attends to "corner holes" in the text, then $\mathbf{G}^{\text{brep}}_{3,:}$ should attend to the hole faces in B-Rep while $\mathbf{G}^{\text{pc}}_{3,:}$ should attend to the corresponding point cloud patches. The alignment layers then map these different representations to a common space where they can be directly compared.

### 4.2 Grounding Diversity Loss (Novel)

Without regularization, all feature slot queries might collapse to attending to the same geometric regions (typically the most prominent features). We encourage diverse grounding through an overlap penalty:

$$\mathcal{L}_{\text{diverse}} = \frac{1}{|\mathcal{K}_{\text{active}}|(|\mathcal{K}_{\text{active}}|-1)} \sum_{k \neq k' \in \mathcal{K}_{\text{active}}} \left(\mathbf{G}^{\text{brep}}_{k,:} \cdot \mathbf{G}^{\text{brep}}_{k',:} + \mathbf{G}^{\text{pc}}_{k,:} \cdot \mathbf{G}^{\text{pc}}_{k',:}\right)$$

This penalizes pairwise overlap between grounding distributions of different slots, encouraging each active slot to attend to distinct geometric regions.

### 4.3 Local Contrastive Loss

We align grounded features across modalities within each feature slot using InfoNCE [van den Oord et al., 2018]. For a batch of $B$ samples:

$$\mathcal{L}_{\text{local}} = \frac{1}{|\mathcal{K}_{\text{active}}|} \sum_{k \in \mathcal{K}_{\text{active}}} \mathcal{L}_{\text{NCE}}(\{\tilde{\mathbf{F}}^{\text{brep}}_{i,k}\}_{i=1}^B, \{\tilde{\mathbf{F}}^{\text{pc}}_{i,k}\}_{i=1}^B)$$

where:

$$\mathcal{L}_{\text{NCE}}(\mathbf{A}, \mathbf{B}) = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\bar{\mathbf{a}}_i \cdot \bar{\mathbf{b}}_i / \tau)}{\sum_{j=1}^{B} \exp(\bar{\mathbf{a}}_i \cdot \bar{\mathbf{b}}_j / \tau)}$$

with $\bar{\mathbf{a}}_i = \mathbf{a}_i / \|\mathbf{a}_i\|$ denoting L2 normalization and $\tau$ the learnable temperature.

This loss ensures that corresponding feature slots align across modalities for the same sample while being discriminable from other samples.

### 4.4 Global Contrastive Loss

For global retrieval, we align the projected embeddings across all modality pairs:

$$\mathcal{L}_{\text{global}} = \frac{1}{3}\left(\mathcal{L}_{\text{NCE}}(\mathbf{Z}^{\text{brep}}_{\text{proj}}, \mathbf{Z}^{\text{text}}_{\text{proj}}) + \mathcal{L}_{\text{NCE}}(\mathbf{Z}^{\text{pc}}_{\text{proj}}, \mathbf{Z}^{\text{text}}_{\text{proj}}) + \mathcal{L}_{\text{NCE}}(\mathbf{Z}^{\text{brep}}_{\text{proj}}, \mathbf{Z}^{\text{pc}}_{\text{proj}})\right)$$

Each pairwise loss is computed symmetrically (both directions).

### 4.5 Confidence Regularization

To prevent the model from collapsing all confidences to zero (which would minimize the grounding losses trivially), we add a mild regularization encouraging confident predictions:

$$\mathcal{L}_{\text{conf}} = -\frac{1}{K} \sum_{k=1}^{K} c_k \log c_k + (1-c_k) \log(1-c_k)$$

This entropy regularization encourages confident (near 0 or 1) predictions rather than uncertain (near 0.5) ones.

### 4.6 Total Loss

The total training objective is:

$$\mathcal{L} = \lambda_g \mathcal{L}_{\text{global}} + \lambda_l \mathcal{L}_{\text{local}} + \lambda_c \mathcal{L}_{\text{consist}} + \lambda_d \mathcal{L}_{\text{diverse}} + \lambda_r \mathcal{L}_{\text{conf}}$$

with default weights $\lambda_g = 1.0$, $\lambda_l = 0.5$, $\lambda_c = 0.5$, $\lambda_d = 0.2$, $\lambda_r = 0.1$.

---

## 5. Orientation-Aware Representations

Unlike general 3D understanding tasks, CAD model descriptions frequently contain explicit directional references such as "vertical mounting plates", "horizontal through-holes", "top-facing surface", and "side mounting bosses". These orientation-specific semantic associations require representations that preserve directional information rather than achieving rotation invariance.

### 5.1 Design Rationale

Standard rotation augmentation would train the model to produce identical embeddings for differently-oriented versions of the same model. However, this conflicts with our goal of learning accurate text-geometry correspondences when the text explicitly references orientation.

Consider the description: "A bracket with vertical mounting holes on the left side and horizontal slots on the bottom." Rotation-invariant features would be unable to distinguish "vertical" from "horizontal" orientations, degrading grounding quality.

### 5.2 Canonical Orientation

We process all models in their canonical orientation as defined in the source datasets. The ShapeLLM features encode models without rotation augmentation, preserving the directional semantics that the text encoder can learn to ground.

### 5.3 Storage Analysis

Without rotation augmentation, storage requirements are significantly reduced:

| Modality | Per-sample size | Total (169K samples) |
|----------|----------------|---------------------|
| B-Rep | ~25 KB | ~4.2 GB |
| Point Cloud (ShapeLLM) | ~135 KB | ~23 GB |
| Text | ~2 MB | ~488 GB |
| **Total** | | **~365 GB** |

This represents a ~3× reduction from the rotation-augmented approach (~1 TB), enabling efficient training on standard hardware.

### 5.4 Mixed Precision Training and FP16 Stability

To enable efficient training with large batch sizes and datasets, we implement mixed precision (FP16) training with several critical stability enhancements:

#### Confidence Clamping

The confidence predictor outputs logits that are clamped before sigmoid activation to prevent numerical instability (implementation at `clip4cad_gfa.py:580-583`):

```python
logits = self.confidence_predictor(T_feat).squeeze(-1)  # (B, K)
logits = logits.clamp(min=-5, max=5)  # Tight range for FP16
confidence = torch.sigmoid(logits)   # sigmoid(-5)≈0.007, sigmoid(5)≈0.993
```

**Rationale:** In FP16, `sigmoid(10)` can round to exactly 1.0, causing `log(1-conf) = -inf` in loss functions. Clamping to [-5, 5] ensures confidence ∈ [0.007, 0.993], preventing NaN.

#### Curriculum-Based Slot Activation Fallback

During early training (epoch < 30), if all slots have confidence below threshold, we fall back to activating the top-3 slots to prevent degenerate solutions where no features are extracted. After epoch 30, we trust the model but ensure at least 1 slot is always active.

#### Confidence Floor Loss

To prevent confidence collapse (where all slots converge to low confidence), we add a soft constraint that encourages mean confidence ≥ 0.3:

$$\mathcal{L}_{\text{floor}} = \max(0, 0.3 - \text{mean}(\mathbf{c}))$$

This prevents the model from learning a degenerate uniform distribution.

---

## 6. Hard Negative Mining

Following insights from OpenSHAPE [Liu et al., 2023], we employ offline hard negative mining to improve contrastive learning efficiency.

### 6.1 Mining Procedure

After an initial training phase (Stage 1, see §7), we extract embeddings for all training samples and construct a k-NN graph in embedding space. For each sample $i$:

1. Find $k = 20$ nearest neighbors by geometric embedding similarity
2. Filter out neighbors $j$ where text embedding similarity exceeds threshold: $\cos(\mathbf{z}_i^{\text{text}}, \mathbf{z}_j^{\text{text}}) > 0.8$
3. Remaining neighbors are **hard negatives**: geometrically similar but semantically different

### 6.2 Hard Negative Batch Construction

During Stage 2 training, we construct batches to include hard negatives:

1. Sample $s = 16$ seed indices uniformly
2. For each seed, add up to 3 of its hard negatives
3. Fill remaining batch slots with random samples

This ensures geometrically confusing pairs appear together in batches, forcing the model to learn fine-grained discriminative features.

### 6.3 Hard Negative Persistence

Mined hard negatives are cached to disk for reproducibility:

- **File**: `outputs/hard_negatives/hard_negatives.json`
- **Format**:
```json
{
  "sample_id_1": [neg_id_1, neg_id_2, ..., neg_id_k],
  "sample_id_2": [...]
}
```

When resuming training in stage 2, hard negatives are automatically reloaded from this file, ensuring consistent negative sampling across training runs.

---

## 7. Training Procedure

We employ a two-stage training strategy to establish grounding before global discrimination.

**Reduced Training Schedule:** Through empirical testing, we found the model converges effectively with half the originally documented epochs (15 stage 1 + 20 stage 2 = 35 total). The learning rate is reduced to 3e-5 (from 1e-4) to prevent confidence collapse in early training when using mixed precision (FP16).

### 7.1 Stage 1: Grounding Establishment (Epochs 1-15)

Focus on learning meaningful grounding with reduced global alignment pressure:

- **Active losses**: $\mathcal{L}_{\text{consist}}$, $\mathcal{L}_{\text{diverse}}$, $\mathcal{L}_{\text{local}}$, $\mathcal{L}_{\text{conf}}$
- **Global loss weight**: $\lambda_g = 0.2$ (reduced)
- **Learning rate**: $3 \times 10^{-5}$
- **Batch construction**: Random sampling
- **Epochs**: 15 (reduced from originally documented 30)

This stage teaches the model which text mentions correspond to which geometric regions without strong pressure to discriminate between samples.

#### Stage 1 Checkpoint Saving

At the transition from stage 1 to stage 2, a separate checkpoint is saved for ablation studies:

- **File**: `checkpoint_stage1_final.pt`
- **Purpose**: Enables evaluation of stage 1 grounding alone vs. full two-stage training
- **Contents**: Model weights, optimizer state, scheduler state, best val loss, epoch, stage

This checkpoint can be used to measure the marginal contribution of hard negative mining and cross-modal alignment (stage 2) over grounding alone (stage 1).

### 7.2 Stage 2: Global Alignment with Hard Negatives (Epochs 16-35)

Add hard negative mining and full global alignment:

- **Active losses**: All losses at full weight
- **Global loss weight**: $\lambda_g = 1.0$
- **Learning rate**: $3 \times 10^{-5}$ (maintained from stage 1)
- **Batch construction**: Hard negative sampling
- **Epochs**: 20 (epochs 16-35, reduced from originally documented 40)

Hard negatives are mined using Stage 1 embeddings at the transition point.

### 7.3 Optimization Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Base learning rate | $3 \times 10^{-5}$ |
| Weight decay | 0.05 |
| Batch size | 512 (with in-memory loading) or 64 (default) |
| LR schedule | Cosine annealing to $1 \times 10^{-6}$ |
| Warmup | 3 epochs (linear) |
| Gradient clipping | 1.0 (max norm) |
| Precision | Mixed (FP16) with stability enhancements |
| Contrastive temperature init | 0.07 (learnable) |
| Grounding temperature init | 0.1 (learnable) |

---

## 8. Data Loading and Pre-computation Strategies

### 8.1 Pre-split Text File Loading

For large-scale training with 200GB+ text embeddings, loading the full file for each split (train/val/test) is inefficient. The dataset supports pre-split files with automatic discovery:

**Priority order** (implementation at `gfa_dataset.py:745-750`):
1. `C:/Users/User/Desktop/text_splits/{split}_text_embeddings.h5` (SSD, fastest)
2. `{text_path.parent}/{split}_text_embeddings.h5`
3. `{text_path.parent}/text_splits/{split}_text_embeddings.h5`
4. `D:/Defect_Det/MMCAD/data/aligned/text_splits/{split}_text_embeddings.h5` (HDD fallback)

**Pre-processing script:**
```bash
python scripts/preprocess_text_splits.py \
    --text-file "path/to/text_embeddings.h5" \
    --data-root "data/" \
    --output-dir "path/to/ssd/text_splits" \
    --fp16  # Saves 50% space
```

This creates split-specific files (e.g., `train_text_embeddings.h5` with only 111k samples instead of full 169k) that load 5-10× faster.

### 8.2 In-Memory Loading for Large-Scale Training

The `GFAMappedDataset` supports loading the entire training set to RAM:

```python
train_dataset = GFAMappedDataset(
    data_root="data/",
    split="train",
    load_to_memory=True,  # Loads ~200GB to RAM
    ...
)
```

**Trade-off:**
- Initial load: ~5 minutes (one-time cost)
- Training iteration speed: 3-4× faster (no disk I/O)
- RAM requirement: ~200GB for 111k training samples

**Validation strategy:** Keep validation on disk (`load_to_memory=False`) to save RAM.

### 8.3 UID Mapping for Aligned Training

Since different modalities (B-Rep, point cloud, text) may have different indexing in their HDF5 files, we use a canonical UID mapping:

**Structure** (`uid_mapping.json`):
```json
{
  "canonical_uid": {
    "brep_idx": 1234,
    "pc_idx": 5678,
    "text_idx": 9012
  }
}
```

This allows efficient aligned loading without duplicating data across modality-specific files.

---

## 9. Interactive Training with Jupyter Notebooks

For rapid experimentation and hyperparameter tuning, a Jupyter notebook training workflow is provided at `notebooks/train_gfa.ipynb`.

### 13.1 Key Benefits

1. **Persistent data loading:** Load 200GB dataset once, keep in RAM across training runs
2. **Fast iteration:** Restart training from any epoch without reloading data
3. **Easy debugging:** Inspect model state, visualize attention weights, check gradients
4. **Hyperparameter experimentation:** Try different configs without 5-minute reload penalty

### 13.2 Workflow Structure

**Cell 1-2:** Imports and configuration

**Cell 3:** Load datasets to RAM (~5 min, run once)
```python
train_dataset = GFAMappedDataset(..., load_to_memory=True)  # 111k samples, 203GB
val_dataset = GFAMappedDataset(..., load_to_memory=False)   # Stay on disk
```

**Cell 4-5:** Initialize model and trainer

**Cell 5.5:** Resume from checkpoint (optional)
```python
checkpoint_path = Path("notebooks/outputs/gfa_111k/checkpoint_best.pt")
trainer.resume(str(checkpoint_path))  # Loads model + optimizer + scheduler
```

**Cell 6:** Train
```python
trainer.train()  # Automatically saves checkpoints every 5 epochs
```

**Cell 7-9:** Validation, memory monitoring, utilities

### 13.3 Critical Configuration for Notebook Training

```python
config.training.num_workers = 0  # Data in RAM, no multiprocessing needed
```

**Why:** DataLoader workers would duplicate the 200GB dataset in RAM (main + worker1 + worker2 = 600GB). With data in RAM, single-process iteration is actually faster.

### 13.4 Module Reloading for Live Code Updates

When fixing bugs or tuning hyperparameters, reload modules without restarting kernel:

```python
import importlib
from clip4cad.models import clip4cad_gfa
importlib.reload(clip4cad_gfa)  # Picks up code changes without data reload
```

This enables sub-second iteration on model architecture changes.

---

## 10. Inference

At inference time, we compute embeddings for single-modality inputs to enable cross-modal retrieval.

### 12.1 Text-to-Geometry Retrieval

Given a text query, we compute the text embedding:

1. Encode text with frozen LLM to get $\mathbf{H}^{\text{text}}$
2. Project: $\mathbf{X}^{\text{text}} = \text{Proj}_{\text{text}}(\mathbf{H}^{\text{text}})$
3. Parse features: $\mathbf{T}^{\text{feat}} = \text{TextParser}(\mathbf{X}^{\text{text}})$
4. Predict confidences: $\mathbf{c} = \text{ConfidencePredictor}(\mathbf{T}^{\text{feat}})$
5. Aggregate: $\mathbf{z}^{\text{text}} = \sum_k c_k \mathbf{t}_k^{\text{feat}} / \sum_k c_k$
6. Project: $\mathbf{z}^{\text{text}}_{\text{proj}} = \text{ProjHead}(\mathbf{z}^{\text{text}})$

The database of geometric embeddings is pre-computed. Retrieval is performed via nearest neighbor search in the projected embedding space.

### 8.2 Geometry-to-Text/Geometry Retrieval

Given B-Rep or point cloud input, we need to compute geometric embeddings **without** text grounding. We employ a **self-grounding** mechanism where the geometric tokens serve as their own "pseudo-text":

1. Encode geometry to get $\mathbf{X}^{\text{geo}}$
2. Apply learned global queries (separate from text feature queries) to extract summary: $\mathbf{Q}^{\text{self}} \in \mathbb{R}^{K \times d}$
3. Compute self-grounding: $\mathbf{G}^{\text{self}} = \text{softmax}(\mathbf{Q}^{\text{self}} \mathbf{X}^{\text{geo}\top} / \tau)$
4. Aggregate with uniform confidence: $\mathbf{z}^{\text{geo}} = \text{mean}_n(\bar{\text{imp}}_n \cdot \mathbf{x}_n)$
5. Project: $\mathbf{z}^{\text{geo}}_{\text{proj}} = \text{ProjHead}(\mathbf{z}^{\text{geo}})$

The self-grounding queries are initialized from the trained text feature queries and optionally fine-tuned.

---

## 11. Theoretical Analysis

### 13.1 Relationship to Prior Work

**Comparison with CLIP [Radford et al., 2021].** Standard CLIP aligns global image and text embeddings through contrastive learning. Our approach extends this by (1) decomposing text into feature mentions, (2) learning explicit grounding to geometric primitives, and (3) enforcing grounding consistency across geometric representations.

**Comparison with HCC-CAD [Chen et al., 2024].** HCC-CAD compresses point cloud tokens using attention-coverage-based importance, selecting tokens that global queries under-attended to. Our approach differs fundamentally: we use semantic grounding (what text mentions) rather than statistical coverage (what attention missed) to determine importance. This is a semantic criterion vs. a statistical criterion.

**Comparison with HoLA [Liu et al., 2025].** HoLA encodes B-Rep topology by training an intersection module that predicts curve geometry from surface pairs. This is a generative approach that encodes topology through reconstruction. Our approach encodes semantics through text grounding—a discriminative approach that encodes meaning through cross-modal correspondence.

### 13.2 Why Grounding Consistency Works

The grounding consistency loss $\mathcal{L}_{\text{consist}}$ provides supervision that neither modality alone could provide. Consider learning to ground "corner fillet":

- From B-Rep alone: The model might attend to any small face
- From point cloud alone: The model might attend to any curved region
- With consistency: The model must find the region that (1) is small in B-Rep, (2) is curved in point cloud, and (3) corresponds across modalities—this uniquely identifies fillets

The alignment layers $\text{AlignNet}_{\text{brep}}$ and $\text{AlignNet}_{\text{pc}}$ learn the mapping between encoder representation spaces, allowing comparison despite their different structural biases.

---

## 12. Implementation Details

### 12.1 Pre-computation Pipeline

All encoder features are pre-computed using scripts in the `scripts/` directory:

**B-Rep Features** (`scripts/precompute_brep_features_step.py`):
- Reads STEP files directly using pythonOCC (OpenCASCADE bindings)
- Extracts face UV grids (32×32×3) and edge point curves (32×3) from B-Rep topology
- Encodes using AutoBrep-style FSQ VAE encoder with pretrained weights (auto-downloaded from HuggingFace)
- Architecture: 4-stage encoder with channels [128, 256, 512, 512], ResBlocks with single GroupNorm
- Outputs: 48-dimensional face features, 12-dimensional edge features
- Features: Multiprocessing for parallel geometry extraction (max 60 workers on Windows), checkpointing every N batches, resume support
- Storage: HDF5 with LZF compression

```bash
# Example usage
python scripts/precompute_brep_features_step.py \
    --step-dir ../data/extracted_step_files \
    --csv ../data/169k.csv \
    --output-dir ../data/embeddings \
    --batch-size 32 --num-workers 60
```

**Point Cloud Features** (Pre-computed ShapeLLM/ReCon++ embeddings):
- Pre-computed using finetuned ShapeLLM/ReCon++ encoder
- Model trained on 32K CAD models with multi-modal (image, text) supervision
- Architecture: 24-layer transformer, 16 heads, embed_dim=1024, 512 groups pooled to 32
- Features stored in HDF5 files with structure:
  - `local_features`: [N, 32, 1024] - Pooled local geometric patches
  - `global_token`: [N, 16, 1024] - 16 global tokens from ReCon++ aggregation
  - `filenames`: [N] - Model identifiers for UID mapping
- Combined usage: 48 tokens (local + global) at 1024 dimensions
- Storage: ~135KB per model (48×1024×4 bytes)

**Text Features** (`scripts/precompute_text_embeddings_csv.py`):
- Reads titles and descriptions from CSV file (columns: uid, title, description)
- Encodes using Phi-4-mini (3.8B parameters) with frozen weights
- Stores full sequence of hidden states at final layer (dimension 3072)
- Features: Checkpointing, resume support, batched inference

### 12.2 Model Architecture Summary

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| Unified projections (B-Rep, PC, Text) | ~1.1M | Yes |
| Text feature parser (2-layer cross-attn) | ~530K | Yes |
| Confidence predictor | ~17K | Yes |
| Grounding projections | ~65K | Yes |
| Alignment networks (B-Rep, PC) | ~200K | Yes |
| Projection heads (global, local) | ~130K | Yes |
| Feature queries & self-grounding | ~6K | Yes |
| **Total** | **~3.1M** | Yes |

The model is extremely lightweight (~3M trainable parameters), enabling rapid iteration and training on consumer hardware.

### 12.3 Computational Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | Single NVIDIA RTX 4090 (24GB) |
| Training time | ~4 hours (70 epochs) |
| Pre-computation time | ~8 hours (one-time, B-Rep only) |
| Storage | ~365 GB (orientation-aware, no rotation augmentation) |
| Peak GPU memory | ~6 GB |

---

## 13. Evaluation Protocol

### 13.1 Cross-Modal Retrieval

We evaluate on three retrieval tasks:
- **Text → B-Rep**: Given text description, retrieve matching B-Rep model
- **Text → Point Cloud**: Given text description, retrieve matching point cloud
- **B-Rep ↔ Point Cloud**: Cross-geometric-modality retrieval

Metrics: Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank (MRR)

### 13.2 Grounding Quality

We manually annotate 200 test samples with ground-truth correspondences between text mentions and B-Rep faces. We evaluate:
- **Grounding Precision**: Fraction of top-attended faces that are correct
- **Grounding Recall**: Fraction of correct faces in top-K attention

### 13.3 Ablation Studies

We ablate each architectural component:
1. Without grounding consistency loss
2. Without grounding diversity loss
3. Without confidence weighting (fixed K=12)
4. Without alignment layers (direct comparison)
5. Without hard negative mining
6. With rotation augmentation (vs. orientation-aware design)

### 13.4 Orientation Sensitivity

Since our approach preserves directional information rather than enforcing rotation invariance:
- **Orientation Grounding Accuracy**: Whether directional terms ("vertical", "horizontal") correctly ground to appropriately-oriented geometric regions
- **Directional Description Retrieval**: Retrieval precision for descriptions with explicit orientation references

---

## 14. Limitations and Future Work

**Limitation 1: Dependency on Description Quality.** Our approach relies on descriptions that explicitly mention geometric features. Vague descriptions ("a mechanical part") provide weak grounding signal. Future work could incorporate construction sequence information as an additional grounding source.

**Limitation 2: Fixed Maximum Feature Slots.** While confidence weighting handles variable complexity, the maximum K=12 slots may be insufficient for highly complex parts with many distinct features. Adaptive slot allocation could address this.

**Limitation 3: No Topology Encoding.** Unlike HoLA, we do not explicitly encode B-Rep topology. The grounding mechanism implicitly captures some topological information (adjacent faces often share text mentions), but explicit topology encoding could improve performance on topology-sensitive tasks.

**Future Direction: Generation Guidance.** The grounded feature representations could serve as conditioning for generative models. Given a text description, the grounding matrix indicates which geometric regions should be generated with what properties—a form of semantic layout for CAD generation.

---

## References

- Willis, K. D. D., et al. (2025). AutoBrep: Autoregressive B-Rep Generation with Unified Topology and Geometry. *SIGGRAPH*.
- Yu, X., et al. (2022). Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling. *CVPR*.
- Xue, L., et al. (2024). ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding. *CVPR*.
- Chen, X., et al. (2024). HCC-CAD: Hierarchical Compensated Compression for Efficient 3D Vision-Language Models. *NeurIPS*.
- Liu, M., et al. (2025). HoLA: B-Rep Generation using a Holistic Latent Representation. *SIGGRAPH*.
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML*.
- van den Oord, A., et al. (2018). Representation Learning with Contrastive Predictive Coding. *arXiv*.
- Liu, M., et al. (2023). OpenSHAPE: Scaling Up 3D Shape Representation Towards Open-World Understanding. *NeurIPS*.
- Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML*.
- Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture. *ICML*.
- Qi, H., et al. (2024). ShapeLLM: Universal 3D Object Understanding for Embodied Interaction. *ECCV*.