
# CLIP4CAD-H: Technical Architecture Specification

## 1. Problem Statement and Motivation

We aim to learn a unified embedding space for CAD models where representations from different modalities—B-Rep, point cloud, and natural language—can be directly compared. Given a CAD model, we have access to:

- **B-Rep representation**: Parametric surfaces $\mathcal{S} = \{S_i\}_{i=1}^{F}$, curves $\mathcal{C} = \{C_j\}_{j=1}^{E}$, and topology $\mathcal{T}_{SC} \in \{0,1\}^{F \times E}$
- **Point cloud**: $\mathcal{P} \in \mathbb{R}^{N \times 3}$ sampled from the model surface
- **Hierarchical text**: A concise title $T_{\text{title}}$ and detailed description $T_{\text{desc}}$

The key challenges we address:

1. **Information asymmetry**: Text descriptions are sequential and constructive ("a cylinder with two holes"), while geometric representations present complete 3D structure simultaneously. Standard CLIP-style alignment may learn superficial correlations rather than geometric understanding.

2. **Granularity mismatch**: Titles describe global shape ("hexagonal bracket"), while descriptions enumerate local features ("four corner fillets, central through-hole"). A single embedding level cannot capture both.

3. **Rotation sensitivity**: CAD models in datasets appear in arbitrary orientations. Semantically identical models oriented differently should produce similar embeddings.

## 2. Architecture Overview

Our architecture processes each modality through three stages:

```
Input → Pretrained Encoder → Unified Projection → Hierarchical Compression → Output Embeddings
```

The hierarchical compression module is shared across geometric modalities and produces:
- **Global features** $\mathbf{F}_g \in \mathbb{R}^{N_g \times d}$: Coarse structural information
- **Detail features** $\mathbf{F}_d \in \mathbb{R}^{N_d \times d}$: Fine-grained local information

Text is processed through a parallel hierarchical encoder producing:
- **Title embedding** $\mathbf{t}_g \in \mathbb{R}^{d}$: Global semantic embedding
- **Feature embeddings** $\mathbf{T}_d \in \mathbb{R}^{N_d \times d}$: Local feature mentions

Training aligns global geometric features with title embeddings, and detail features with description feature embeddings.

## 3. Modality-Specific Encoders

### 3.1 B-Rep Encoder

We adopt the encoding scheme from AutoBrep [Willis et al., 2025], which represents B-Rep primitives as point grids sampled uniformly in the parametric domain.

**Face representation.** Each face $S_i$ is sampled on a $32 \times 32$ grid in the $(u, v)$ parameter domain:

$$\mathbf{G}_i[u, v] = S_i\left(u_{\min} + \frac{u}{31}(u_{\max} - u_{\min}), \; v_{\min} + \frac{v}{31}(v_{\max} - v_{\min})\right)$$

for $u, v \in \{0, 1, \ldots, 31\}$, yielding $\mathbf{G}_i \in \mathbb{R}^{32 \times 32 \times 3}$.

**Edge representation.** Each edge $C_j$ is sampled at 32 uniform points along its parameter domain, yielding $\mathbf{E}_j \in \mathbb{R}^{32 \times 3}$.

**Normalization.** Following AutoBrep, we apply:
1. **UV-origin normalization**: Rotate/flip the grid so the lexicographically smallest corner point is at index $(0, 0)$
2. **Bounding box normalization**: Scale each grid independently to $[-1, 1]^3$

**Encoder architecture.** We use the Deep Compression Autoencoder architecture from AutoBrep:

*Face encoder* $\mathcal{E}_{\text{face}}: \mathbb{R}^{32 \times 32 \times 3} \rightarrow \mathbb{R}^{d_f}$:
```
Conv2d(3→32, k=4, s=2, p=1) → GroupNorm → GELU    # 32×32 → 16×16
Conv2d(32→64, k=4, s=2, p=1) → GroupNorm → GELU   # 16×16 → 8×8
Conv2d(64→128, k=4, s=2, p=1) → GroupNorm → GELU  # 8×8 → 4×4
Conv2d(128→8, k=4, s=2, p=1)                       # 4×4 → 2×2
Flatten                                            # 2×2×8 → 32
```

Output dimension: $d_f = 32$.

*Edge encoder* $\mathcal{E}_{\text{edge}}: \mathbb{R}^{32 \times 3} \rightarrow \mathbb{R}^{d_e}$:
```
Conv1d(3→32, k=4, s=2, p=1) → GroupNorm → GELU    # 32 → 16
Conv1d(32→64, k=4, s=2, p=1) → GroupNorm → GELU   # 16 → 8
Conv1d(64→128, k=4, s=2, p=1) → GroupNorm → GELU  # 8 → 4
Conv1d(128→8, k=4, s=2, p=1)                       # 4 → 2
Flatten                                            # 2×8 → 16
```

Output dimension: $d_e = 16$.

**Note on pretrained weights.** AutoBrep releases pretrained encoder weights trained on ABC-1M with reconstruction + FSQ quantization objectives. We use the encoder portion and extract continuous features before the FSQ layer. If pretrained weights are unavailable, we initialize randomly and rely on our training objectives.

**Output.** For a B-Rep model with $F$ faces and $E$ edges:

$$\mathbf{Z}^{\text{face}} = [\mathcal{E}_{\text{face}}(\mathbf{G}_1); \ldots; \mathcal{E}_{\text{face}}(\mathbf{G}_F)] \in \mathbb{R}^{F \times 32}$$
$$\mathbf{Z}^{\text{edge}} = [\mathcal{E}_{\text{edge}}(\mathbf{E}_1); \ldots; \mathcal{E}_{\text{edge}}(\mathbf{E}_E)] \in \mathbb{R}^{E \times 16}$$

### 3.2 Point Cloud Encoder

We use Point-BERT [Yu et al., 2022] with ULIP-2 [Xue et al., 2024] pretrained weights.

**Tokenization.** The point cloud $\mathcal{P} \in \mathbb{R}^{N \times 3}$ (we use $N = 2048$) is processed as:

1. **Farthest Point Sampling**: Select $M = 512$ seed points
2. **KNN Grouping**: For each seed, group $k = 32$ nearest neighbors
3. **Local encoding**: Each group is encoded by a mini-PointNet (shared MLP + max pooling)

This produces $M = 512$ group tokens plus one prepended CLS token.

**Transformer encoding.** The tokens pass through a 12-layer transformer with:
- Hidden dimension: 384
- Attention heads: 6
- FFN dimension: 1536

**Output.** $\mathbf{Z}^{\text{pc}} \in \mathbb{R}^{513 \times 384}$ (including CLS token).

**Note.** Point-BERT operates on raw XYZ coordinates. The ULIP-2 weights were trained with image-language supervision, providing semantically meaningful features without CAD-specific fine-tuning.

### 3.3 Text Encoder

We use a pretrained LLM (Phi-4-mini 3.8B or Qwen2.5-3B) with frozen weights.

**Title encoding.** The title $T_{\text{title}}$ is tokenized and encoded:

$$\mathbf{H}^{\text{title}} = \text{LLM}(T_{\text{title}}) \in \mathbb{R}^{L_t \times d_{\text{LLM}}}$$

where $d_{\text{LLM}} = 3072$ for Phi-4-mini. Since these LLMs use causal attention, we take the **last non-padding token's hidden state** as the sequence representation:

$$\mathbf{h}^{\text{title}} = \mathbf{H}^{\text{title}}[\text{last\_token\_idx}] \in \mathbb{R}^{d_{\text{LLM}}}$$

**Description encoding.** The description $T_{\text{desc}}$ is similarly encoded:

$$\mathbf{H}^{\text{desc}} = \text{LLM}(T_{\text{desc}}) \in \mathbb{R}^{L_d \times d_{\text{LLM}}}$$

We project to the unified dimension:

$$\tilde{\mathbf{H}}^{\text{desc}} = \mathbf{H}^{\text{desc}} \mathbf{W}_{\text{proj}} \in \mathbb{R}^{L_d \times d}$$

where $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{d_{\text{LLM}} \times d}$.

## 4. Unified Input Projection

All modality encoders produce different output dimensions. We project to a common dimension $d = 256$.

**B-Rep projection:**

$$\tilde{\mathbf{Z}}^{\text{face}} = \text{LayerNorm}(\mathbf{Z}^{\text{face}} \mathbf{W}_f + \mathbf{b}_f) \in \mathbb{R}^{F \times d}$$
$$\tilde{\mathbf{Z}}^{\text{edge}} = \text{LayerNorm}(\mathbf{Z}^{\text{edge}} \mathbf{W}_e + \mathbf{b}_e) \in \mathbb{R}^{E \times d}$$

where $\mathbf{W}_f \in \mathbb{R}^{32 \times d}$, $\mathbf{W}_e \in \mathbb{R}^{16 \times d}$.

We concatenate and add modality embeddings:

$$\mathbf{X}^{\text{brep}} = [\tilde{\mathbf{Z}}^{\text{face}}; \tilde{\mathbf{Z}}^{\text{edge}}] + \mathbf{e}_{\text{brep}} \in \mathbb{R}^{(F+E) \times d}$$

**Point cloud projection:**

$$\mathbf{X}^{\text{pc}} = \text{LayerNorm}(\mathbf{Z}^{\text{pc}} \mathbf{W}_p + \mathbf{b}_p) + \mathbf{e}_{\text{pc}} \in \mathbb{R}^{513 \times d}$$

where $\mathbf{W}_p \in \mathbb{R}^{384 \times d}$.

**Padding and masking.** B-Rep models have variable numbers of faces and edges. We pad to maximum sizes $F_{\max} = 64$, $E_{\max} = 128$ and maintain boolean masks $\mathbf{m}^{\text{face}} \in \{0,1\}^{F_{\max}}$, $\mathbf{m}^{\text{edge}} \in \{0,1\}^{E_{\max}}$ indicating valid (non-padding) tokens.

## 5. Hierarchical Compression Module

This is the core contribution. The module compresses variable-length token sequences into fixed-size representations with explicit global-local structure.

### 5.1 Global Structure Compression (GSC)

We define $N_g = 8$ learnable query vectors $\mathbf{Q}_g \in \mathbb{R}^{N_g \times d}$ that will learn to capture global shape structure.

**Initialization.** Both queries and their positional encodings are initialized from $\mathcal{N}(0, 0.02^2)$:

$$\mathbf{Q}_g^{(0)} \sim \mathcal{N}(0, 0.02^2), \quad \mathbf{P}_g \sim \mathcal{N}(0, 0.02^2)$$

**Cross-attention layers.** We apply $L = 2$ cross-attention layers. For layer $l$:

$$\hat{\mathbf{Q}}_g^{(l-1)} = \text{LayerNorm}(\mathbf{Q}_g^{(l-1)})$$
$$\hat{\mathbf{X}} = \text{LayerNorm}(\mathbf{X})$$

Multi-head attention with $H = 8$ heads:

$$\text{head}_h = \text{Attention}(\hat{\mathbf{Q}}_g^{(l-1)} \mathbf{W}_Q^h, \; \hat{\mathbf{X}} \mathbf{W}_K^h, \; \mathbf{X} \mathbf{W}_V^h; \; \mathbf{m})$$

where $\mathbf{W}_Q^h, \mathbf{W}_K^h, \mathbf{W}_V^h \in \mathbb{R}^{d \times (d/H)}$ and $\mathbf{m}$ is the padding mask.

The attention operation is:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}; \mathbf{m}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d/H}} + \mathbf{M}\right)\mathbf{V}$$

where $\mathbf{M}_{ij} = -\infty$ if token $j$ is padding (i.e., $m_j = 0$), else $0$.

Concatenate heads and project:

$$\text{MHA}(\mathbf{Q}, \mathbf{X}; \mathbf{m}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}_O$$

Update with residual connection:

$$\mathbf{Q}_g^{(l)} = \mathbf{Q}_g^{(l-1)} + \text{MHA}(\hat{\mathbf{Q}}_g^{(l-1)}, \hat{\mathbf{X}}; \mathbf{m})$$

Apply feed-forward network:

$$\mathbf{Q}_g^{(l)} = \mathbf{Q}_g^{(l)} + \text{FFN}(\text{LayerNorm}(\mathbf{Q}_g^{(l)}))$$

where $\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$ with $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$, $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$.

**Output.** After $L$ layers: $\mathbf{F}_g = \mathbf{Q}_g^{(L)} \in \mathbb{R}^{N_g \times d}$.

**Attention weight extraction.** During the forward pass, we store the attention weights from each layer and head:

$$\boldsymbol{\alpha}^{(l,h)} \in \mathbb{R}^{N_g \times M}$$

where $M$ is the input sequence length. These are used for adaptive detail mining.

### 5.2 Adaptive Detail Mining (ADM)

The global queries may not adequately attend to fine-grained features. ADM identifies important but under-attended regions.

**Coverage computation.** For each input token $i$, we compute how strongly it was attended to by the global queries. We use the **maximum** attention weight across all layers, heads, and queries:

$$A_i^{\text{cov}} = \max_{l \in [L], h \in [H], j \in [N_g]} \alpha_{j,i}^{(l,h)}$$

This measures whether *any* global query strongly attended to token $i$. Using max (rather than sum) avoids the normalization issues with summing softmax outputs.

**Importance prediction.** We predict the intrinsic importance of each token using a learned MLP:

$$I_i = \sigma\left(\mathbf{w}_2^\top \text{GELU}(\mathbf{W}_1 \mathbf{x}_i + \mathbf{b}_1) + b_2\right)$$

where $\mathbf{W}_1 \in \mathbb{R}^{(d/4) \times d}$, $\mathbf{w}_2 \in \mathbb{R}^{d/4}$, and $\sigma$ is the sigmoid function. Output: $I_i \in (0, 1)$.

**Complementary score.** We identify tokens that are important but under-covered:

$$S_i^c = I_i \cdot (1 - A_i^{\text{cov}})$$

Tokens with high importance $I_i$ and low coverage $A_i^{\text{cov}}$ receive high complementary scores.

**Masking.** For padding tokens, we set $S_i^c = -\infty$ before selection.

**Top-K selection.** We select the $K = 64$ tokens with highest complementary scores:

$$\mathcal{I}_{\text{sel}} = \text{argtopk}(\mathbf{S}^c, K)$$
$$\mathbf{X}_{\text{sel}} = \{\mathbf{x}_i : i \in \mathcal{I}_{\text{sel}}\} \in \mathbb{R}^{K \times d}$$

**Detail compression.** We define $N_d = 8$ learnable detail queries $\mathbf{Q}_d \in \mathbb{R}^{N_d \times d}$ with positional encodings $\mathbf{P}_d$. These attend to the selected tokens through the same cross-attention mechanism (2 layers, 8 heads):

$$\mathbf{F}_d = \text{CrossAttnLayers}(\mathbf{Q}_d + \mathbf{P}_d, \mathbf{X}_{\text{sel}}, \mathbf{X}_{\text{sel}})$$

**Output.** Detail features: $\mathbf{F}_d \in \mathbb{R}^{N_d \times d}$.

### 5.3 Combined Output

**Unified representation:**

$$\mathbf{Z} = [\mathbf{F}_g; \mathbf{F}_d] \in \mathbb{R}^{(N_g + N_d) \times d}$$

For our default settings, $\mathbf{Z} \in \mathbb{R}^{16 \times 256}$.

**Global embedding for contrastive learning:**

$$\mathbf{z}_g = \frac{1}{N_g} \sum_{i=1}^{N_g} \mathbf{f}_g^{(i)} \in \mathbb{R}^d$$

We then project through a learned head:

$$\mathbf{z}_{\text{proj}} = \mathbf{W}_{\text{proj}}^{(2)} \text{GELU}(\mathbf{W}_{\text{proj}}^{(1)} \mathbf{z}_g + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)} \in \mathbb{R}^{d_{\text{proj}}}$$

where $d_{\text{proj}} = 128$.

## 6. Hierarchical Text Encoder

The text encoder mirrors the hierarchical structure of the compression module.

### 6.1 Title Branch (Global)

Project the LLM's last-token representation:

$$\mathbf{t}_g = \mathbf{W}_t^{(2)} \text{GELU}(\mathbf{W}_t^{(1)} \mathbf{h}^{\text{title}} + \mathbf{b}_t^{(1)}) + \mathbf{b}_t^{(2)} \in \mathbb{R}^d$$

Then project to contrastive space:

$$\mathbf{t}_{g,\text{proj}} = \text{ProjHead}(\mathbf{t}_g) \in \mathbb{R}^{d_{\text{proj}}}$$

### 6.2 Description Branch (Local)

We introduce $N_d = 8$ learnable feature queries $\mathbf{Q}_{\text{feat}} \in \mathbb{R}^{N_d \times d}$ that attend to the projected description tokens $\tilde{\mathbf{H}}^{\text{desc}} \in \mathbb{R}^{L_d \times d}$.

**Cross-attention:**

$$\mathbf{T}_d = \text{CrossAttn}(\mathbf{Q}_{\text{feat}} + \mathbf{P}_{\text{feat}}, \tilde{\mathbf{H}}^{\text{desc}}, \tilde{\mathbf{H}}^{\text{desc}}; \mathbf{m}^{\text{desc}})$$

where $\mathbf{m}^{\text{desc}}$ is the padding mask for description tokens.

Apply FFN:

$$\mathbf{T}_d = \mathbf{T}_d + \text{FFN}(\text{LayerNorm}(\mathbf{T}_d))$$

**Confidence prediction.** Not all descriptions mention multiple features. We predict a confidence score for each feature query indicating whether it found a relevant mention:

$$c_i = \sigma(\mathbf{w}_c^\top \text{GELU}(\mathbf{W}_c \mathbf{t}_d^{(i)} + \mathbf{b}_c) + b_c') \in (0, 1)$$

**Output:**
- Feature embeddings: $\mathbf{T}_d \in \mathbb{R}^{N_d \times d}$
- Confidence scores: $\mathbf{c} \in (0, 1)^{N_d}$

Project to contrastive space:

$$\mathbf{T}_{d,\text{proj}} = \text{ProjHead}(\mathbf{T}_d) \in \mathbb{R}^{N_d \times d_{\text{proj}}}$$

## 7. Training Objectives

### 7.1 Global Contrastive Loss

We align global embeddings across modalities using InfoNCE [van den Oord et al., 2018].

For a batch of $B$ samples with L2-normalized embeddings $\{\mathbf{z}_i^{(a)}\}_{i=1}^B$ and $\{\mathbf{z}_i^{(b)}\}_{i=1}^B$ from modalities $a$ and $b$:

$$\mathcal{L}_{\text{global}}^{a \to b} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\mathbf{z}_i^{(a)} \cdot \mathbf{z}_i^{(b)} / \tau)}{\sum_{j=1}^{B} \exp(\mathbf{z}_i^{(a)} \cdot \mathbf{z}_j^{(b)} / \tau)}$$

The symmetric loss:

$$\mathcal{L}_{\text{global}}^{(a,b)} = \frac{1}{2}\left(\mathcal{L}_{\text{global}}^{a \to b} + \mathcal{L}_{\text{global}}^{b \to a}\right)$$

We compute this for all available modality pairs. If modalities $\{a_1, \ldots, a_k\}$ are present:

$$\mathcal{L}_{\text{global}} = \frac{2}{k(k-1)} \sum_{i < j} \mathcal{L}_{\text{global}}^{(a_i, a_j)}$$

**Temperature.** We use a learnable temperature initialized to $\tau = 0.07$:

$$\tau = \exp(\log \tau_{\text{param}})$$

clamped to $[0.01, 1.0]$ during training.

### 7.2 Local Contrastive Loss

We align geometric detail features $\mathbf{F}_{d,\text{proj}} \in \mathbb{R}^{N_d \times d_{\text{proj}}}$ with text feature embeddings $\mathbf{T}_{d,\text{proj}} \in \mathbb{R}^{N_d \times d_{\text{proj}}}$ using bipartite matching.

**Confidence thresholding.** We only consider text features with confidence above threshold $\tau_c = 0.5$:

$$\mathcal{I}_{\text{active}} = \{j : c_j > \tau_c\}$$

If $|\mathcal{I}_{\text{active}}| = 0$, we skip the local loss for this sample.

**Cost matrix.** For sample $n$, compute the cost between all geometric detail features and active text features:

$$C_{ij}^{(n)} = 1 - \frac{\mathbf{f}_{d,i}^{(n)}}{\|\mathbf{f}_{d,i}^{(n)}\|} \cdot \frac{\mathbf{t}_{d,j}^{(n)}}{\|\mathbf{t}_{d,j}^{(n)}\|}, \quad i \in [N_d], j \in \mathcal{I}_{\text{active}}^{(n)}$$

**Hungarian matching.** Find the optimal assignment $\pi^*$ minimizing total cost:

$$\pi^* = \arg\min_{\pi} \sum_{i=1}^{\min(N_d, |\mathcal{I}_{\text{active}}|)} C_{i, \pi(i)}$$

This is solved using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

**Loss.** The confidence-weighted matching loss:

$$\mathcal{L}_{\text{local}}^{(n)} = \frac{1}{|\mathcal{I}_{\text{active}}^{(n)}|} \sum_{(i,j) \in \text{Match}^{(n)}} c_j^{(n)} \cdot C_{ij}^{(n)}$$

Averaged over the batch:

$$\mathcal{L}_{\text{local}} = \frac{1}{|\{n : |\mathcal{I}_{\text{active}}^{(n)}| > 0\}|} \sum_{n} \mathcal{L}_{\text{local}}^{(n)}$$

### 7.3 Reconstruction Loss (Auxiliary)

To encourage the unified representation to encode actual geometry, we include an auxiliary reconstruction objective. We emphasize that this is a **regularization term**, not a high-fidelity reconstruction target—the compression ratio is extreme (~4K values encoding ~200K output values).

**Decoder architecture.** From the unified representation $\mathbf{Z} \in \mathbb{R}^{16 \times 256}$:

1. Flatten: $\mathbf{z}_{\text{flat}} = \text{Flatten}(\mathbf{Z}) \in \mathbb{R}^{4096}$
2. Predict face latents: $\hat{\mathbf{Z}}^{\text{face}} = \text{MLP}(\mathbf{z}_{\text{flat}}) \in \mathbb{R}^{F_{\max} \times 32}$
3. Decode each face: $\hat{\mathbf{G}}_i = \mathcal{D}_{\text{face}}(\hat{\mathbf{z}}_i^{\text{face}}) \in \mathbb{R}^{32 \times 32 \times 3}$

The face decoder $\mathcal{D}_{\text{face}}$ mirrors the encoder with transposed convolutions.

**Loss.** L1 reconstruction loss on valid faces only:

$$\mathcal{L}_{\text{recon}} = \frac{1}{\sum_i m_i^{\text{face}}} \sum_{i=1}^{F_{\max}} m_i^{\text{face}} \|\mathbf{G}_i - \hat{\mathbf{G}}_i\|_1$$

Edge reconstruction follows analogously.

**Topology loss (optional).** We can additionally predict face-edge adjacency:

$$\hat{a}_{ij} = \sigma(\text{MLP}([\hat{\mathbf{z}}_i^{\text{face}}; \hat{\mathbf{z}}_j^{\text{face}}]))$$

$$\mathcal{L}_{\text{topo}} = \text{BCE}(\hat{\mathbf{A}}, \mathbf{A}^{\text{gt}})$$

where $\mathbf{A}^{\text{gt}} \in \{0,1\}^{F \times E}$ is the ground-truth adjacency.

### 7.4 Total Loss

$$\mathcal{L} = \lambda_g \mathcal{L}_{\text{global}} + \lambda_l \mathcal{L}_{\text{local}} + \lambda_r \mathcal{L}_{\text{recon}} + \lambda_t \mathcal{L}_{\text{topo}}$$

Default weights: $\lambda_g = 1.0$, $\lambda_l = 0.5$, $\lambda_r = 0.3$, $\lambda_t = 0.1$.

## 8. Handling Rotation Variance

The pretrained encoders (AutoBrep, Point-BERT) are **not** rotation invariant—rotating the input produces different encoder outputs. We address this through data augmentation rather than architectural changes.

**Augmentation strategy.** During training, we apply random rotations to all geometric modalities of the same sample:
- Sample rotation matrix $\mathbf{R}$ from discrete set: 90° increments around each axis (24 possible rotations)
- Apply to point cloud: $\mathcal{P}' = \mathcal{P} \mathbf{R}^\top$
- Apply to B-Rep point grids: $\mathbf{G}_i' = \mathbf{G}_i \mathbf{R}^\top$ (applied to all points in the grid)

**Expected outcome.** With sufficient augmentation, the model should learn representations that are similar for rotated versions of the same model. This is **learned approximate invariance**, not guaranteed architectural invariance.

**Validation plan.** We will evaluate rotation robustness by:
1. Computing embeddings for test models in canonical orientation
2. Computing embeddings for the same models rotated by random angles
3. Measuring cosine similarity between original and rotated embeddings

We expect high similarity (>0.9) for augmented rotations and moderate similarity for out-of-distribution rotations.

## 9. Implementation Specifications

### 9.1 Dimensions Summary

| Component | Dimension | Description |
|-----------|-----------|-------------|
| $d$ | 256 | Unified feature dimension |
| $d_{\text{proj}}$ | 128 | Contrastive projection dimension |
| $d_f$ | 32 | AutoBrep face latent |
| $d_e$ | 16 | AutoBrep edge latent |
| $d_{\text{pc}}$ | 384 | Point-BERT output |
| $d_{\text{LLM}}$ | 3072 | Phi-4-mini hidden dimension |
| $N_g$ | 8 | Global queries |
| $N_d$ | 8 | Detail queries |
| $K$ | 64 | Selected tokens for ADM |
| $H$ | 8 | Attention heads |
| $L$ | 2 | Cross-attention layers per level |
| $F_{\max}$ | 64 | Maximum faces |
| $E_{\max}$ | 128 | Maximum edges |
| $N$ | 2048 | Point cloud size |

### 9.2 Trainable vs Frozen Parameters

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| AutoBrep encoders | ~2M | No (frozen) |
| Point-BERT | ~22M | No (frozen) |
| LLM (Phi-4-mini) | ~3.8B | No (frozen) |
| Input projections | ~300K | Yes |
| Hierarchical compression | ~2M | Yes |
| Text projections | ~1M | Yes |
| Projection heads | ~100K | Yes |
| Reconstruction decoder | ~3M | Yes |
| **Total trainable** | **~6.4M** | |

### 9.3 Training Configuration

```yaml
# Optimizer
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 0.05
betas: [0.9, 0.999]

# Schedule
epochs: 100
scheduler: CosineAnnealingLR
min_lr: 1e-6
warmup_epochs: 5

# Batch
batch_size: 32
gradient_accumulation: 2
effective_batch_size: 64

# Precision
mixed_precision: fp16
gradient_checkpointing: true
max_grad_norm: 1.0

# Loss weights
lambda_global: 1.0
lambda_local: 0.5
lambda_recon: 0.3
lambda_topo: 0.1
```

### 9.4 Training Stages

**Stage 1 (Epochs 1-40): Establish geometric grounding**
- Enable: $\mathcal{L}_{\text{global}}$, $\mathcal{L}_{\text{recon}}$
- Disable: $\mathcal{L}_{\text{local}}$
- Learning rate: 1e-4

**Stage 2 (Epochs 41-100): Add local alignment**
- Enable: All losses
- Learning rate: 5e-5 (reduced)
- $\lambda_r$ reduced to 0.2

## 10. Open Questions and Design Choices to Validate

### 10.1 Architecture Choices

1. **Number of queries.** We set $N_g = N_d = 8$ based on HCC-CAD. Ablation needed for $\{4, 8, 16, 32\}$.

2. **Detail selection count.** We set $K = 64$. If too small, important details are missed; if too large, noise is included.

3. **Coverage computation.** We use max-attention. Alternative: mean-attention or learned aggregation.

4. **Confidence threshold.** We set $\tau_c = 0.5$. This significantly affects local loss computation.

### 10.2 Training Choices

1. **Loss weights.** Current weights are heuristic. Hyperparameter search needed.

2. **Stage transition.** We switch at epoch 40. Earlier or later transition may be better.

3. **Rotation augmentation.** We use discrete 90° rotations. Continuous rotations may improve generalization but slow training.

### 10.3 Encoder Choices

1. **Pretrained weights availability.** AutoBrep weights may require retraining if not released.

2. **LLM selection.** Phi-4-mini vs Qwen2.5-3B vs smaller models. Trade-off between quality and memory.

3. **Point cloud normals.** Current design uses XYZ only. Adding normals requires modifying Point-BERT or using different encoder.

### 10.4 Evaluation Plan

1. **Cross-modal retrieval**: Text→B-Rep, Text→PointCloud, B-Rep→PointCloud (and vice versa)
2. **Zero-shot classification**: Using text prompts for category names
3. **Rotation robustness**: Similarity between original and rotated embeddings
4. **Ablation studies**: Each architectural component and loss term

---

## Appendix: Notation Reference

| Symbol | Description |
|--------|-------------|
| $\mathcal{S}, \mathcal{C}, \mathcal{T}$ | B-Rep surfaces, curves, topology |
| $\mathcal{P}$ | Point cloud |
| $T_{\text{title}}, T_{\text{desc}}$ | Title and description text |
| $\mathbf{G}_i$ | Face point grid |
| $\mathbf{E}_j$ | Edge point sequence |
| $\mathbf{X}$ | Input token sequence |
| $\mathbf{Q}_g, \mathbf{Q}_d$ | Global and detail queries |
| $\mathbf{F}_g, \mathbf{F}_d$ | Global and detail features |
| $\mathbf{Z}$ | Unified representation |
| $\mathbf{z}_g$ | Global embedding |
| $\mathbf{t}_g, \mathbf{T}_d$ | Text global and local embeddings |
| $\mathbf{c}$ | Text feature confidence scores |
| $\boldsymbol{\alpha}$ | Attention weights |
| $A^{\text{cov}}$ | Attention coverage |
| $I$ | Importance scores |
| $S^c$ | Complementary scores |
| $\tau$ | Temperature |
| $\lambda_*$ | Loss weights |

---

This document represents a technical proposal and implementation plan. All architectural choices and hyperparameters are subject to empirical validation. Claims about model properties (e.g., rotation robustness) are hypotheses to be tested, not established facts.