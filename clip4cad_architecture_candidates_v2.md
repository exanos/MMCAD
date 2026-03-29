# CLIP4CAD Architecture Candidates: Mathematical Foundations

**Version:** 2.0 (Complete rewrite incorporating mHC-lite, Gated DeltaNet, and full research synthesis)
**Context:** SGP'26 resubmission, April 13 deadline, 2x A100 GPUs, 137K MM-CAD dataset
**Reports synthesized:** B-Rep Tokenization, Efficient Architecture, JEPA for 3D CAD, Latent Reasoning, Multimodal 3D Foundation Models, Multimodal Latent Spaces, SE3/Sheaf/BRep
**Experiments synthesized:** 20+ architectures from Jan 18 - Mar 2026, best result 54.78% Text->BRep R@1

---

## 0. Preamble: What the experimental history tells us

Your 20+ experiments converge on five hard facts that constrain every architecture decision:

**Fact 1: InfoNCE is the only loss that reliably works for retrieval.** Every addition (codebook commitment, gap-closing L_ATP/L_CU, heavy grounding, local contrastive) degraded performance. The one exception: asymmetric mild grounding (+48% over symmetric). GFA v1 had 5 competing losses at 37.07%. Global-only had 1 loss at 51.37%. Asymmetric had effectively 2 losses at 54.78%.

**Fact 2: The B-Rep encoder is the bottleneck.** AutoBrep FSQ features are 48-dim per face, 12-dim per edge, aggressively compressed latent codes from a VQ-VAE with codebook size 1000 and levels [8,5,5,5]. Your EdgeMessageLayer does scatter-gather message passing with gating. The PC encoder (DGCNN) outputs 1024-dim. The text encoder (Phi-4-mini frozen) is 3072-dim. The B-Rep pathway is representationally starved. Evidence: Text->PC R@1 reached 69.63% (global-only) while Text->BRep peaked at 54.78%. The PC path has 15 percentage points of headroom the B-Rep path lacks.

**Fact 3: Discrete bottlenecks destroy instance discrimination.** The v4.x codebook family proves this with monotonic decline: v4.0 (144 codes) ~45%, v4.2 (144 + curriculum) ~40-45%, v4.8 (232 codes) ~35%, v4.8.2 (1040 codes) 0.0%. The root cause is mathematically precise: 137K unique instances mapped through sparse top-k selection (~15 active codes) into a weighted sum that collapses instance-specific variation. A gear with 32 teeth and a gear with 64 teeth both activate "cylindrical" + "teeth" + "planar" codes and become indistinguishable.

**Fact 4: Asymmetric modality treatment is essential.** B-Rep needs lambda_c=0.08, PC needs lambda_c=0.02 (4:1 ratio). This reflects the fundamental gap between a randomly-initialized encoder (B-Rep) and one with pretrained multimodal features (ShapeLLM for PC). OMIB (Wu et al., ICML 2025) formalizes exactly this principle.

**Fact 5: Architecture matters more than data scale.** 111k vs 166k gave comparable results (~54-55%). The ceiling is architectural, not data-limited.

**The implication:** To beat 54.78%, you need (a) a better B-Rep encoder, (b) a principled continuous compression mechanism, and (c) exploitation of the three-level text hierarchy that is currently completely unused.

---

## 1. Universal Infrastructure: Applied to ALL Candidate Architectures

These two components are not candidates to be ablated against each other. They are universal upgrades applied to every architecture variant, because their cost is negligible and their benefits are orthogonal to the specific architectural choices.

### 1.1 Manifold Hyperconnections Lite (mHC-lite)

**Source:** mHC (arXiv:2512.24880, DeepSeek-AI), mHC-lite (arXiv:2601.05732, Yang & Gao), KromHC (arXiv:2601.21579)

**What it does:** Expands the single residual stream into n parallel streams that cross-mix through doubly stochastic matrices at every layer boundary. Standard residual connections (y = x + F(x)) limit topological complexity to additive bypass. mHC-lite enables richer signal routing without increasing FLOPs.

**Why it matters for CLIP4CAD specifically:** Your B-Rep transformer is 6 layers, your text transformer is 4 layers. These are shallow networks. mHC-lite with n=3 streams gives a shallow network the effective expressivity of a significantly deeper one. The gradient stabilization allows more aggressive learning rates, which directly matters when training for 35 epochs where convergence speed counts. The original mHC paper quantifies the problem: unconstrained hyper-connections produce composite gain magnitudes up to ~3000x across deep networks. mHC constrains mixing matrices to the Birkhoff polytope via 20 Sinkhorn-Knopp iterations per layer, but mHC-lite eliminates this entirely.

**Mathematical formulation:**

mHC-lite uses the Birkhoff-von Neumann theorem: any doubly stochastic matrix can be expressed as a convex combination of permutation matrices. For n=3 streams, there are 3! = 6 permutation matrices.

At initialization, generate all n! permutation matrices as fixed (non-learnable) tensors. Introduce n! learnable scalar coefficients. Apply softmax to ensure valid convex combination:

```
M = sum_{i=1}^{n!} softmax(alpha)_i * P_i
```

where P_i are the n! permutation matrices and alpha are learnable scalars. This is doubly stochastic **by construction**, not by approximation. Variance of column sums is exactly 1.0. No iterations, no approximation gap, no custom CUDA kernels.

**Implementation:**

```python
import itertools

class mHCLiteConnection(nn.Module):
    """Manifold Hyper-Connection Lite (Birkhoff-von Neumann reparameterization).
    
    Replaces standard residual connections. n=3 recommended for shallow networks.
    Parameter cost: n! scalars per connection point = 6 floats for n=3.
    """
    def __init__(self, d_model, n_streams=3):
        super().__init__()
        self.n_streams = n_streams
        self.d_model = d_model
        
        # Generate all n! permutation matrices as fixed buffers
        perms = list(itertools.permutations(range(n_streams)))
        perm_tensors = []
        for p in perms:
            P = torch.zeros(n_streams, n_streams)
            for i, j in enumerate(p):
                P[i, j] = 1.0
            perm_tensors.append(P)
        self.register_buffer('perm_matrices', torch.stack(perm_tensors))  # (n!, n, n)
        
        # n! learnable coefficients (initialized to uniform -> identity-like mixing)
        self.coeffs = nn.Parameter(torch.zeros(len(perms)))
    
    def get_mixing_matrix(self):
        """Returns doubly stochastic mixing matrix."""
        alpha = F.softmax(self.coeffs, dim=0)  # (n!,)
        M = torch.einsum('p,pij->ij', alpha, self.perm_matrices)  # (n, n)
        return M
    
    def forward(self, streams):
        """
        Args:
            streams: (batch, seq_len, n_streams, d_model)
        Returns:
            mixed: (batch, seq_len, n_streams, d_model)
        """
        M = self.get_mixing_matrix()  # (n, n)
        return torch.einsum('ij,bsnj->bsni', M, streams)
```

**Integration into transformer layers:**

```python
class mHCTransformerBlock(nn.Module):
    """Transformer block with mHC-lite residual streams."""
    def __init__(self, d_model, n_heads, n_streams=3, dropout=0.1):
        super().__init__()
        self.n_streams = n_streams
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mhc_pre_attn = mHCLiteConnection(d_model, n_streams)
        self.mhc_pre_ffn = mHCLiteConnection(d_model, n_streams)
    
    def forward(self, streams, mask=None):
        # streams: (B, S, n_streams, d)
        # Mix streams before attention
        streams = self.mhc_pre_attn(streams)
        # Use first stream for attention (others provide gradient diversity)
        x = streams[:, :, 0, :]  # (B, S, d)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), 
                          key_padding_mask=mask)[0]
        # Broadcast attention output back to all streams
        streams = streams.clone()
        streams[:, :, 0, :] = x
        # Mix before FFN
        streams = self.mhc_pre_ffn(streams)
        x = streams[:, :, 0, :]
        x = x + self.ffn(self.norm2(x))
        streams[:, :, 0, :] = x
        return streams
```

**Parameter cost for entire CLIP4CAD model:**
- 6-layer B-Rep encoder: 12 mHC connections * 6 coefficients = 72 floats
- 4-layer text encoder: 8 mHC connections * 6 coefficients = 48 floats
- Total: 120 floats = 480 bytes

This is not a hyperparameter. It's free expressivity.

**Compatibility:** Validated at scale in DeepSeek-V3 (with MLA + MoE). mHC wraps layer outputs; it is orthogonal to the attention mechanism, FFN structure, and loss function. Compatible with every candidate architecture below.

---

### 1.2 Gated DeltaNet Hybrid Attention for B-Rep Encoder

**Source:** Gated DeltaNet (arXiv:2412.06464, ICLR 2025, NVIDIA); Qwen3-Next/Qwen3.5 hybrid design; Systematic analysis (arXiv:2507.06457)

**What it does:** Replaces the majority of standard self-attention layers with Gated DeltaNet layers that maintain a fixed-size recurrent state (O(1) memory) instead of a growing KV cache (O(n) memory). A minority of layers retain full attention for long-range retrieval.

**Why it matters for CLIP4CAD specifically:** Your face tokens are ordered by BFS traversal. BFS produces strong locality: adjacent tokens in the sequence are topologically adjacent on the B-Rep model. This is exactly the regime where linear/recurrent attention excels. Local geometric patterns (face type sequences, edge convexity patterns, surface continuity) are sequential and local. But B-Rep also has long-range dependencies: two faces on opposite sides of a model might share a through-hole feature, or a fillet on one end might constrain geometry on the other. These require full attention.

**How Gated DeltaNet works:**

Gated DeltaNet combines two mechanisms:

From Mamba2: an adaptive decay gate alpha_t that enables rapid bulk memory erasure. The recurrent state S_t decays by factor alpha_t at each step.

From Delta Networks: a delta rule that performs precise surgical key-value association updates. Instead of simply adding new information, it first removes the old association for the current key before inserting the new one.

The update rule:

```
S_t = alpha_t * S_{t-1} + beta_t * v_t * (k_t - alpha_t * S_{t-1}^T * k_t)^T
```

This maintains a fixed-size recurrent state (d_model x d_head matrix) regardless of sequence length. For BFS-ordered face tokens, the model can maintain a running representation of "the shape so far" and precisely update it as each new face is encountered, rather than the blurry accumulation standard SSMs suffer from. The gated decay lets it forget irrelevant earlier faces when BFS traversal jumps to a different topological branch.

**Optimal hybrid ratio:**

The systematic analysis in arXiv:2507.06457 compared GLA, RetNet, Mamba, HGRN-2, DeltaNet, and Gated DeltaNet in hybrid configurations. Key findings:
- Most linear attention variants peak at the 3:1 linear-to-full ratio
- Gated DeltaNet consistently outperforms other linear variants in hybrid settings
- The 25% full attention layers provide "retrieval anchors" for precise token lookup
- Diminishing returns below 25% full attention

Qwen3-Next-80B-A3B validates this at scale: performance matches Qwen3-32B at less than 10% of GPU hours and 10x throughput at >32K context.

**For a 6-layer B-Rep encoder, use 2:1 ratio (not 3:1):**

```
Layer 1: Gated DeltaNet    (local BFS pattern processing)
Layer 2: Gated DeltaNet    (local sequential features)
Layer 3: Full Attention     (global: catch long-range topology like through-holes)
Layer 4: Gated DeltaNet    (local refinement with topology context)
Layer 5: Gated DeltaNet    (local refinement)
Layer 6: Full Attention     (global: final holistic representation)
```

The ratio is 2:1 rather than 3:1 because B-Rep sequences are short (max 192 faces) and need proportionally more global steps than NLP models processing thousands of tokens. Two full-attention layers at positions 3 and 6 create two "synchronization points" where the model can integrate information across the entire shape.

**Implementation:**

The official PyTorch implementation is at github.com/NVlabs/GatedDeltaNet. For integration:

```python
from gated_deltanet import GatedDeltaNetLayer  # NVIDIA implementation

class HybridBRepTransformer(nn.Module):
    """6-layer hybrid: 4 Gated DeltaNet + 2 Full Attention, with mHC-lite."""
    def __init__(self, d_model=384, n_heads=8, n_streams=3, dropout=0.1):
        super().__init__()
        self.n_streams = n_streams
        self.layers = nn.ModuleList()
        self.mhc_connections = nn.ModuleList()
        
        for i in range(6):
            if i in [2, 5]:  # Full attention at layers 3 and 6 (0-indexed: 2, 5)
                self.layers.append(FullAttentionBlock(d_model, n_heads, dropout))
            else:
                self.layers.append(GatedDeltaNetBlock(d_model, n_heads, dropout))
            self.mhc_connections.append(mHCLiteConnection(d_model, n_streams))
    
    def forward(self, x, mask=None):
        B, S, d = x.shape
        # Initialize n_streams copies
        streams = x.unsqueeze(2).expand(-1, -1, self.n_streams, -1).clone()
        
        for layer, mhc in zip(self.layers, self.mhc_connections):
            streams = mhc(streams)
            # Process primary stream through layer
            x = streams[:, :, 0, :]
            x = layer(x, mask=mask)
            streams[:, :, 0, :] = x
        
        return streams[:, :, 0, :]  # Return primary stream
```

**Memory and compute impact:**
- 4 of 6 layers: O(1) memory instead of O(n), O(n) compute instead of O(n^2)
- On 192-token sequences the absolute savings are modest (~1.5x), but the fixed-size recurrent state acts as implicit regularization against overfitting on 137K samples
- The official NVIDIA kernels provide optimized Triton implementations

**For the text encoder (4 layers):** Use 3:1 ratio: 3 Gated DeltaNet + 1 Full Attention (layer 4). Text sequences from Phi-4-mini features are already contextualized; the transformer just needs to compress them.

**For the point cloud encoder:** DGCNN already works well (Text->PC at 69.63%). Don't change it. The bottleneck is B-Rep, not PC.

---

## 2. Candidate Architecture A: VIB-Contrastive

**Core idea:** Add a Variational Information Bottleneck compression term to the existing InfoNCE loss. The minimal theoretically-grounded change.

**Theoretical justification:**

Achille and Soatto ("Emergence of Invariance and Disentanglement in Deep Representations," JMLR 2018, arXiv:1706.01350) prove:

**Proposition 3.1 (Invariance-Minimality Equivalence):** A sufficient statistic z of x for y is maximally insensitive to nuisance factors n if and only if it is minimal. Formally: I(z; n) <= I(z; x) - I(x; y), with equality when z is a minimal sufficient statistic.

For B-Rep, nuisance variables include: non-canonical rotations (UV-Net loses 63.51pp under rotation per FoV-Net benchmarks), tessellation density, UV-grid sampling resolution, FSQ quantization artifacts from AutoBrep's [8,5,5,5] codebook. The theorem guarantees that VIB strips these automatically without requiring explicit augmentation for each nuisance type.

Additionally, Almudever et al. (ICML 2025, arXiv:2506.04870) show VIB directly regularizes multimodal alignment when added to InfoNCE, with an "Information Homeostasis" effect: when using a learnable temperature tau alongside beta, the optimizer decreases tau to counteract compression as beta increases, reaching a coupled equilibrium. This is directly relevant since your model uses a learnable log_tau.

**Why it should work on your specific failure modes:**

VIB addresses the v4.8.2 collapse mode through a *continuous* mechanism rather than the *discrete* codebook mechanism that failed. The codebook destroyed instance discrimination by mapping continuous features to a small set of discrete codes. VIB compresses through a soft KL penalty that forces the encoder to encode only cross-modally relevant information, while preserving the continuous representation that retrieval requires. It's the correct information bottleneck, applied correctly.

### Mathematical formulation

Each encoder outputs a distribution, not a point:

```
q_brep(z | x_brep) = N(mu_brep(x_brep), diag(sigma^2_brep(x_brep)))
q_pc(z | x_pc)     = N(mu_pc(x_pc), diag(sigma^2_pc(x_pc)))
q_text(z | x_text)  = N(mu_text(x_text), diag(sigma^2_text(x_text)))
```

During training, sample via reparameterization: `z = mu + sigma * epsilon, epsilon ~ N(0,I)`. During inference, use `z = mu` (deterministic).

The loss:

```
L = L_InfoNCE(z_brep, z_text, tau) 
  + L_InfoNCE(z_brep, z_pc, tau) 
  + L_InfoNCE(z_text, z_pc, tau)
  + beta_brep * KL(q_brep(z|x) || N(0,I))
  + beta_pc   * KL(q_pc(z|x)   || N(0,I))
  + beta_text * KL(q_text(z|x)  || N(0,I))
```

KL for diagonal Gaussian (closed form, no sampling needed):

```
KL(q || p) = 0.5 * sum_j (sigma_j^2 + mu_j^2 - 1 - log(sigma_j^2))
```

**Adaptive beta via OMIB (Wu et al., ICML 2025, arXiv:2505.19996):**

Your asymmetric lambda finding (0.08 BRep, 0.02 PC) discovered empirically what OMIB formalizes. OMIB's r factor:

```
r = 1 - tanh(ln( E[KL(p(y|z_pc) || p(y))] / E[KL(p(y|z_brep) || p(y))] ))
```

When the text LLM dominates fusion (its prediction closely matches joint prediction), the denominator KL is small, driving r upward. Then:

```
beta_brep = beta_max * (1 - r)    [weaker encoder, less compression]
beta_pc   = beta_max * r           [stronger encoder, more compression]  
beta_text = beta_min               [frozen, minimal compression]
```

OMIB proves theoretical bounds: beta_max = I(X; Z) / I(Z; Y), beta_min computed via MINE estimator. This prevents over-compression (posterior collapse) and under-compression (nuisance leakage) by construction.

**Hyperparameter sweep:** `beta_max in {1e-4, 1e-3, 1e-2, 1e-1}`. Based on Alemi et al. (ICLR 2017): optimal classification at beta = 1e-3, matched deterministic accuracy at beta = 1e-2 with added adversarial robustness. Expect optimal for retrieval around 1e-3.

**Implementation:**

```python
class VIBProjectionHead(nn.Module):
    """Projection head that outputs mu and log_var for VIB."""
    def __init__(self, d_in, d_proj, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_in, d_in), nn.GELU(), nn.Dropout(dropout)
        )
        self.mu_head = nn.Linear(d_in, d_proj)
        self.logvar_head = nn.Linear(d_in, d_proj)
        # Initialize logvar to small negative -> small initial variance
        nn.init.constant_(self.logvar_head.bias, -5.0)
    
    def forward(self, x, sample=True):
        h = self.shared(x)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        
        if sample and self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        
        return z, mu, log_var

def kl_divergence(mu, log_var):
    """KL(N(mu, sigma^2) || N(0, I)), closed form."""
    return 0.5 * torch.mean(torch.sum(mu.pow(2) + log_var.exp() - 1 - log_var, dim=-1))
```

**Implementation cost:** ~80 lines. Each ProjectionHead becomes VIBProjectionHead. KL terms added to loss.

**Risk:** Low. VIB is the most extensively validated component here. Worst case: beta too high causes posterior collapse (all z -> N(0,I), retrieval = random). Detected by monitoring KL magnitude: if KL drops to near-zero, beta is too high. If KL stays very large, beta is too low.

**Expected gain:** 2-5% R@1 from automatic invariance to geometric nuisances. Provides theoretical foundation for other candidates.

---

## 3. Candidate Architecture B: Sheaf B-Rep Encoder

**Core idea:** Replace the EdgeMessageLayer with sheaf neural network message passing on the B-Rep face-adjacency graph, followed by Gated DeltaNet hybrid transformer layers.

**Theoretical justification:**

B-Rep graphs are inherently heterophilic: a planar face (zero curvature) is topologically adjacent to a cylindrical fillet (high curvature). Standard GNN message passing minimizes Dirichlet energy, driving adjacent node representations toward a global constant. Bodnar et al. (NeurIPS 2022, arXiv:2202.04579) showed that the sheaf Laplacian resolves this: learned restriction maps model anti-correlation between adjacent nodes, and the infinite-time diffusion limit converges to diverse harmonic substructures rather than a single constant.

B-Rep is naturally a CW complex: vertices = 0-cells, edges = 1-cells, faces = 2-cells. This maps directly onto the combinatorial complexes used in topological deep learning. ETNN (Battiloro et al., ICLR 2025, arXiv:2405.15429) proved that E(n)-equivariant topological neural networks on combinatorial complexes are more expressive than standard geometric GNNs while using less than half the memory.

**Critical novelty:** No SE(3)/E(n) equivariant architecture has been applied to B-Rep data, and no sheaf method has been tested on CAD graphs. This structural gap (confirmed by extensive search across 60+ papers in the SE3/Sheaf report) represents a genuine research contribution.

Additionally, BRepGAT (Lee et al., JCDE 2023) is the only B-Rep paper that mentions oversmoothing at all. No systematic study of oversmoothing in B-Rep GNNs exists. Sheaf message passing with learned restriction maps directly addresses this gap.

### Mathematical formulation

**Stalk assignment:** Each face node i gets a stalk vector space of dimension d_s. Each edge e connecting faces i,j has its own edge stalk.

**Restriction maps:** For each directed adjacency (face i, edge e), learn a linear map F_{i<e} : R^d_s -> R^d_s.

Three parameterization options (Bodnar et al. tested all three):

1. **Diagonal** (d_s parameters per map): F_{i<e} = diag(f_theta(x_i || x_j)). Cheapest. O(d_s) per edge.
2. **Orthogonal** (BuNN-style, Bamberger et al., ICLR 2025 Spotlight): O(n d^2) parameters total (not per-edge). Proven first uniform universal approximation result for GNNs (Theorem 5.3 of BuNN). Norm-preserving via Householder reflections.
3. **General linear** (d_s^2 parameters per map): Most expressive. O(d_s^2) per edge.

Start with diagonal for speed, ablate up to orthogonal if performance warrants.

**Sheaf Laplacian:** The block sheaf Laplacian Delta_F has blocks:

```
Delta_F[i,i] = sum_{e: i in e} F_{i<e}^T F_{i<e}          [diagonal blocks]
Delta_F[i,j] = -F_{i<e}^T F_{j<e}   for e = (i,j)         [off-diagonal blocks]
```

**Sheaf diffusion update (per layer):**

```
X^(t+1) = X^(t) - sigma(Delta_F^(t) * W1^t * X^t * W2^t)
```

where W1, W2 are learnable weight matrices and sigma is GELU activation.

**Concrete implementation:**

```python
class SheafBRepLayer(nn.Module):
    """Sheaf neural network layer for B-Rep face-adjacency graphs.
    
    Learns anisotropic diffusion via diagonal restriction maps.
    Handles heterophilic adjacency (planar face next to cylindrical fillet).
    """
    def __init__(self, d_model, d_stalk=64, dropout=0.1):
        super().__init__()
        self.d_stalk = d_stalk
        self.proj_in = nn.Linear(d_model, d_stalk)
        self.proj_out = nn.Linear(d_stalk, d_model)
        
        # MLP to predict diagonal restriction map from concatenated face features
        self.restriction_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_stalk)
        )
        
        self.W1 = nn.Linear(d_stalk, d_stalk)
        self.W2 = nn.Linear(d_stalk, d_stalk)
        self.norm = nn.LayerNorm(d_stalk)
        self.gate = nn.Sequential(nn.Linear(d_stalk * 2, d_stalk), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, F_feat, edge_to_faces, face_mask, edge_mask):
        """
        Args:
            F_feat: (B, N_f, d_model) face features
            edge_to_faces: (B, N_e, 2) edge-to-face adjacency indices
            face_mask: (B, N_f) valid face mask
            edge_mask: (B, N_e) valid edge mask
        Returns:
            (B, N_f, d_model) updated face features
        """
        B, N_f, d = F_feat.shape
        N_e = edge_to_faces.shape[1]
        
        # Project to stalk dimension
        X = self.proj_in(F_feat)  # (B, N_f, d_stalk)
        
        # Get face indices for each edge
        f1_idx = edge_to_faces[:, :, 0].clamp(0, N_f - 1).long()
        f2_idx = edge_to_faces[:, :, 1].clamp(0, N_f - 1).long()
        valid = edge_mask.bool() & (edge_to_faces[:, :, 0] >= 0) & (edge_to_faces[:, :, 1] >= 0)
        
        # Gather face features for restriction map computation
        f1_feat = torch.gather(F_feat, 1, f1_idx.unsqueeze(-1).expand(-1, -1, d))
        f2_feat = torch.gather(F_feat, 1, f2_idx.unsqueeze(-1).expand(-1, -1, d))
        
        # Diagonal restriction maps: R_{i<e} for each edge
        R = self.restriction_mlp(torch.cat([f1_feat, f2_feat], dim=-1))  # (B, N_e, d_stalk)
        R = R * valid.unsqueeze(-1).float()
        
        # Sheaf Laplacian diffusion step
        # Compute disagreement: R * (X[f1] - X[f2])  (diagonal case: element-wise)
        X_f1 = torch.gather(X, 1, f1_idx.unsqueeze(-1).expand(-1, -1, self.d_stalk))
        X_f2 = torch.gather(X, 1, f2_idx.unsqueeze(-1).expand(-1, -1, self.d_stalk))
        diff = X_f1 - X_f2
        weighted_diff = R * diff  # (B, N_e, d_stalk)
        
        # Aggregate: each face accumulates disagreements from its incident edges
        face_update = torch.zeros_like(X)
        update_count = torch.zeros(B, N_f, 1, device=X.device)
        
        edge_msg = R * weighted_diff * valid.unsqueeze(-1).float()
        count_one = valid.unsqueeze(-1).float()
        
        face_update.scatter_add_(1, f1_idx.unsqueeze(-1).expand(-1, -1, self.d_stalk), edge_msg)
        face_update.scatter_add_(1, f2_idx.unsqueeze(-1).expand(-1, -1, self.d_stalk), -edge_msg)
        update_count.scatter_add_(1, f1_idx.unsqueeze(-1), count_one)
        update_count.scatter_add_(1, f2_idx.unsqueeze(-1), count_one)
        
        # Normalize by degree
        face_update = face_update / (update_count + 1e-8)
        
        # Transform and gate
        transformed = self.W2(F.gelu(self.W1(face_update)))
        transformed = self.dropout(transformed)
        gate_val = self.gate(torch.cat([X, transformed], dim=-1))
        X_new = self.norm(X - gate_val * transformed)  # diffusion: subtract Laplacian term
        
        return self.proj_out(X_new) * face_mask.unsqueeze(-1).float()
```

**Full B-Rep encoder architecture (Sheaf + Hybrid DeltaNet + mHC-lite):**

```
AutoBrep FSQ face features (48-dim) + edge features (12-dim)
    |
    v
Linear projection: face (48 -> 384), edge (12 -> 384)
    + BFS level embedding: nn.Embedding(32, 384)
    |
    v
[SheafBRepLayer x 2]  -- topology-aware message passing, learns restriction maps
    |                     handles heterophily between planar/curved faces
    v
[Gated DeltaNet + mHC-lite]  -- Layer 1: local BFS sequential patterns
[Gated DeltaNet + mHC-lite]  -- Layer 2: local refinement
[Full Attention  + mHC-lite]  -- Layer 3: global topology (through-holes, symmetry)
[Gated DeltaNet + mHC-lite]  -- Layer 4: local with global context
[Gated DeltaNet + mHC-lite]  -- Layer 5: local refinement
[Full Attention  + mHC-lite]  -- Layer 6: final holistic representation
    |
    v
AttentionPooling (16 continuous queries, 2-layer cross-attention)
    |
    v
Projection head -> z_brep (D_PROJ = 256)
```

The sheaf layers run first to enrich face features with topology-aware information, then the hybrid transformer stack processes the enriched sequence. This is cleaner than interleaving sheaf and attention because the sheaf layer operates on the graph structure while the transformer operates on the BFS-ordered sequence.

**Memory analysis (from SE3/Sheaf report Section 8.1):**
- Sheaf with d_stalk=64 diagonal maps: ~64 params per edge, ~750 edges per graph -> ~48K params per sheaf layer -> trivially fits in memory
- Total sheaf layers: 2 * ~48K = ~96K additional parameters
- Full model: well within single A100

**Training time estimate (from SE3/Sheaf report Section 8.2):**
- NSD-style sheaf on 137K graphs of ~200 faces: ~30-90 min/epoch
- With hybrid DeltaNet (4 layers O(1), 2 layers O(n^2)): faster than pure attention
- Projected: 50-100 epochs in 2-4 days on 2xA100

**Implementation cost:** ~200 lines for SheafBRepLayer + integration. Plus ~150 lines for Gated DeltaNet hybrid wrapper (or use NVIDIA's official implementation).

**Risk:** Medium. Sheaf NNs have never been tested on B-Rep graphs. Diagonal restriction maps might not be expressive enough for all topological relationships. Ablation: diagonal -> orthogonal (BuNN-style) -> general linear.

**Expected gain:** 5-10% R@1 from better B-Rep representations. The B-Rep encoder is the primary bottleneck; improving it directly improves all Text->BRep and PC->BRep metrics.

---

## 4. Candidate Architecture C: Hierarchical Nested Contrastive

**Core idea:** Exploit the three caption levels (title, detailed description, application keywords) to create a hierarchically structured latent space via nested dropout, inspired by LoST (CVPR 2026, arXiv:2603.17995) and Matryoshka Representation Learning (NeurIPS 2022).

**Theoretical justification:**

LoST proves that nested dropout with semantic alignment (RIDA) creates a natural information hierarchy where early tokens encode coarse identity and later tokens add geometric detail. With 1 token, their DiT decoder generates semantically plausible complete shapes. With more tokens, it progressively snaps to instance-specific geometry. LoST achieves 128 tokens vs OctGPT's ~50K or Llama-Mesh's ~3758, a 90-99.9% reduction.

The provenance of nested dropout: Rippel, Gelbart, Adams (ICML 2014) proved that in the linear case, nested dropout recovers PCA. Variational Nested Dropout (Cui et al., IEEE TPAMI 2023, arXiv:2101.11353) made rates learnable via Gumbel-Softmax with a "tail index."

Your three caption levels are a natural semantic hierarchy:
- **Title:** "A helical gear" (coarse identity, ~5-10 words)
- **Application keywords:** "power transmission, automotive, drivetrain" (functional context)
- **Detailed description:** "A helical gear with 24 teeth, module 2.5, 20-degree pressure angle, case-hardened steel, 80mm pitch diameter" (full geometric specification)

Forcing all three into the same flat embedding wastes capacity. The first 128 dimensions should suffice for coarse identity; the next 128 should add functional context; the final 128 should capture geometric specifics.

**Connection to information theory (from Multimodal Latent Spaces report Section 6):**

No existing paper formally connects VIB's soft compression with LoST's hard hierarchy. The bridge would require showing that nested dropout induces an effective beta schedule: the first sub-vector experiences maximal compression (only coarse identity survives), later sub-vectors experience minimal compression (fine detail passes through). This connection is implicit in the mechanics but has not been rigorously established, making it a theoretical contribution for the paper.

### Mathematical formulation

**Latent space structure:** 384-dimensional embedding with three nested subspaces:

```
z_core = z[:128]      # coarse geometric identity (title-level)
z_mid  = z[:256]      # + functional/ontological context (keyword-level)
z_full = z[:384]      # + precise geometric specification (description-level)
```

**Text projection heads:** Three separate projections from frozen Phi-4-mini features:

```
h_title       = proj_title(text_features)       -> R^128
h_keywords    = proj_keywords(text_features)    -> R^256  
h_description = proj_description(text_features) -> R^384
```

**Nested dropout during training:** Sample truncation level each batch:

```
L ~ Categorical({128, 256, 384}, p={0.3, 0.3, 0.4})
```

Zero out all dimensions beyond L in z_brep and z_pc:

```
z_truncated = z * mask(L)   where mask(L)[i] = 1 if i < L, else 0
```

This forces the encoder to front-load semantically critical information into early dimensions.

**Hierarchical contrastive loss:**

```
L = w_core * L_InfoNCE(z_brep[:128], h_title, tau)
  + w_mid  * L_InfoNCE(z_brep[:256], h_keywords, tau)
  + w_full * L_InfoNCE(z_brep[:384], h_description, tau)
  + analogous terms for z_pc
  + L_InfoNCE(z_brep, z_pc, tau)  [full-dim cross-modal]
```

with `w_core = 1.0, w_mid = 1.0, w_full = 1.5` (description carries most information).

**RIDA-inspired relational alignment (optional, from LoST):**

Instead of direct InfoNCE between z_core and h_title, align their pairwise distance matrices:

```
D_brep[i,j] = ||z_brep_i[:128] - z_brep_j[:128]||_2
D_text[i,j] = ||h_title_i - h_title_j||_2
L_RIDA_core = ||D_brep - D_text||_F^2 / B^2
```

RIDA (Relational Inter-Distance Alignment) comes from Park et al. (CVPR 2019, arXiv:1904.05068) and was adapted for 3D by LoST. It's softer than InfoNCE (doesn't require exact pairing) and preserves local neighborhood structure. The Aristotelian calibration result (Groger et al., arXiv:2602.14486) confirms local relational structure, not global spectral alignment, is the correct target.

**Why nested and not independent subspaces:**

Independent subspaces (disjoint partitions like z[0:128], z[128:256], z[256:384]) lose the inclusion constraint. The nested structure guarantees z_core is always the most compressed and invariant, because it must suffice alone when later dimensions are dropped. Achille-Soatto's invariance-minimality equivalence applies at each level: z[:128] under maximum compression is automatically maximally invariant, z[:256] under moderate compression preserves functional but not geometric nuisances, z[:384] under minimal compression retains full detail.

**Practical benefit for retrieval:** Hierarchical retrieval is now possible. Coarse search in 128-dim (fast, semantic), then refinement in 256-dim, then final ranking in 384-dim. This is architecturally impossible with a flat embedding.

**Implementation cost:** ~200 lines. Three text projection heads, nested dropout mask, hierarchical loss computation.

**Risk:** Medium. Depends on Phi-4-mini differentiating title vs description vs keywords in its embedding space. Pre-check: compute cosine similarity between title and description embeddings for the same object. If they're already >0.95, the hierarchy won't emerge from the text side. Mitigation: train separate lightweight heads for each text level rather than relying on the frozen embedding to differentiate.

**Expected gain:** 5-10% R@1 on Text->BRep, with hierarchical retrieval capability as a bonus.

---

## 5. Candidate Architecture D: SIGReg + Predictive Auxiliary

**Core idea:** Add (1) SIGReg regularization from LeJEPA/LeWorldModel for principled anti-collapse, and (2) a lightweight cross-modal predictor that predicts z_brep from z_text.

**Theoretical justification:**

SIGReg (Balestriero et al., arXiv:2511.08544):
- **Theorem 1:** Among distributions with scalar covariance constraint, isotropic Gaussian N(0, I_d) uniquely minimizes worst-case downstream prediction risk across arbitrary tasks.
- **Cramer-Wold theorem (Lemma 3):** A high-dimensional distribution matches isotropic Gaussian iff every 1D projection matches standard normal. This enables practical enforcement via 1D tests.
- Enforced via Epps-Pulley test statistic on M=512 random projections. O(N) per projection, fully differentiable, bounded gradients.

CLIPred (NeurIPS 2025):
- Joint I-JEPA + CLIP with shared encoder
- Optimal lambda = 0.01 (predictive weighted 100x lower than JEPA; for us, contrastive should dominate similarly)
- "Particularly important when training data is limited" (your 137K vs their ImageNet scale)

### Mathematical formulation

**SIGReg loss:**

Generate M = 512 random unit-norm directions u^(m) in R^d (fixed at initialization, not learned):

```
For each direction u^(m):
    h^(m) = Z @ u^(m)                     # project batch embeddings to 1D
    h_norm = (h^(m) - mean(h^(m))) / std(h^(m))  # normalize
    T^(m) = EppsPulley(h_norm)             # test 1D Gaussianity

SIGReg(Z) = (1/M) * sum_{m=1}^{M} T^(m)
```

The Epps-Pulley statistic for a sample {h_1, ..., h_N}:

```
T = (2/N) * sum_{j<k} exp(-0.5*(h_j - h_k)^2) 
  - (2*sqrt(2)/N) * sum_j exp(-0.25 * h_j^2) 
  + 1
```

Properties critical for deep learning:
- O(N^2) within each projection, but only M projections needed, total O(N^2 * M)
- For batch=128, M=512, d=384: ~33M operations, negligible vs attention (~28M per layer)
- Bounded gradients: no numerical explosions
- Statistically consistent against all alternatives (proven by Epps & Pulley, Biometrika 1983)
- Single hyperparameter: lambda_sig

Reference implementation: `lejepa.univariate.EppsPulley(num_points=17)` + `lejepa.multivariate.SlicingUnivariateTest(num_slices=1024)`

**Predictive auxiliary:**

A 2-layer MLP predictor:

```
z_brep_hat = predictor(z_text) = W2 * GELU(W1 * z_text + b1) + b2
```

Loss with stop-gradient on target:

```
L_pred = (1/B) * sum_i ||z_brep_hat_i - sg(z_brep_i)||_2^2
```

The stop-gradient on z_brep is critical: without it, the B-Rep encoder could collapse to produce easily-predictable representations. With stop-gradient, only the predictor and text pathway receive gradients from this loss.

**Total loss:**

```
L = L_InfoNCE(z_brep, z_text, tau) 
  + L_InfoNCE(z_brep, z_pc, tau) 
  + L_InfoNCE(z_text, z_pc, tau)
  + lambda_sig * SIGReg(cat[z_brep, z_pc, z_text])
  + lambda_pred * L_pred
```

Following CLIPred: lambda_pred = 0.01
Following LeWorldModel: lambda_sig = 0.1

**Implementation:**

```python
class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (LeJEPA/LeWorldModel)."""
    def __init__(self, d_model, n_projections=512):
        super().__init__()
        # Fixed random projections (not learned)
        directions = torch.randn(d_model, n_projections)
        directions = F.normalize(directions, dim=0)
        self.register_buffer('directions', directions)
    
    def epps_pulley(self, h):
        """Epps-Pulley test statistic for 1D sample h."""
        h = (h - h.mean()) / (h.std() + 1e-8)
        N = h.shape[0]
        # Pairwise term
        diff = h.unsqueeze(0) - h.unsqueeze(1)  # (N, N)
        pairwise = torch.exp(-0.5 * diff.pow(2)).sum() * (2.0 / (N * N))
        # Marginal term
        marginal = torch.exp(-0.25 * h.pow(2)).sum() * (2.0 * 1.41421 / N)
        return pairwise - marginal + 1.0
    
    def forward(self, Z):
        """Z: (B, d) batch of embeddings."""
        projections = Z @ self.directions  # (B, M)
        stats = torch.stack([self.epps_pulley(projections[:, m]) 
                            for m in range(projections.shape[1])])
        return stats.mean()

class CrossModalPredictor(nn.Module):
    """Predicts z_brep from z_text. Lightweight 2-layer MLP."""
    def __init__(self, d_proj, hidden_mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_proj, d_proj * hidden_mult),
            nn.GELU(),
            nn.Linear(d_proj * hidden_mult, d_proj)
        )
    def forward(self, z_text):
        return self.net(z_text)
```

**Implementation cost:** ~120 lines total.

**Risk:** Low. Both components well-validated independently.

**Expected gain:** 3-6% R@1 from SIGReg anti-collapse + predictive alignment.

---

## 6. Candidate Architecture E: Full Stack (A + B + C + D infrastructure)

**Core idea:** Combine all components: Sheaf B-Rep encoder (B), VIB compression (A), hierarchical nested contrastive (C), SIGReg + predictive auxiliary (D), all on mHC-lite + Gated DeltaNet hybrid infrastructure.

### Combined mathematical formulation

**B-Rep encoder:** SheafBRepLayer x 2 -> HybridDeltaNet (4 GDN + 2 Full Attn) with mHC-lite at every layer.

**Distributional output:** VIB projection heads outputting (mu, log_var) per modality.

**Hierarchical structure:** 384-dim with nested dropout at {128, 256, 384}.

**Per-level VIB with decreasing beta (bridging soft IB and hard hierarchy):**

```
beta_128 = beta_max           # maximum compression at coarsest level
beta_256 = beta_max * 0.1     # moderate compression at mid level
beta_384 = beta_max * 0.01    # minimal compression at finest level
```

This creates a continuous information-theoretic analogue of LoST's discrete hierarchy. The first 128 dimensions are under heavy KL pressure: only coarse semantic identity survives (invariance-minimality equivalence at maximum compression). The last 128 dimensions are under minimal pressure: fine geometric detail passes through.

**Total loss:**

```
L = sum_{l in {128,256,384}} [
      w_l * L_InfoNCE(z_brep[:l], h_text_l, tau)
    + w_l * L_InfoNCE(z_pc[:l], h_text_l, tau)
    + beta_l_brep * KL(q_brep(z[:l]|x) || N(0,I))
    + beta_l_pc   * KL(q_pc(z[:l]|x)   || N(0,I))
  ]
  + L_InfoNCE(z_brep, z_pc, tau)          [full-dim cross-modal]
  + lambda_sig * SIGReg(cat[mu_brep, mu_pc])  [on means, not samples]
  + lambda_pred * ||predictor(z_text) - sg(z_brep)||^2
```

**Staged training curriculum (essential, based on COCONUT finding that direct latent training fails):**

```
Stage 0 (epochs 1-8):   Sheaf BRep + PC only, flat InfoNCE, no VIB, no hierarchy
                         Purpose: anchor B-Rep encoder to PC space
                         
Stage 1 (epochs 9-20):  Add text, add VIB at beta_max=1e-4, flat 384-dim
                         Enable SIGReg at lambda=0.1
                         Purpose: establish cross-modal alignment

Stage 2 (epochs 21-35): Enable nested dropout, hierarchical loss, increase beta_max to 1e-3
                         Add predictive auxiliary at lambda=0.01
                         Purpose: refine hierarchical structure
```

This mirrors the multi-stage curriculum that was essential for all your successful experiments (15+20 epochs worked; single-stage HUS at 30 epochs failed).

**Implementation cost:** ~500 lines total. But this is the sum of independently testable components.

**Risk:** High complexity. Three simultaneous changes plus infrastructure changes. Mitigated by staged training and the fact that each component is independently validated in its own candidate.

**Expected gain:** 10-20% R@1 if everything works together. The ablation narrative for the paper: "We identify four independent bottlenecks and show each contributes independently."

---

## 7. Testing Order and Timeline

Given the April 13 deadline (15 days):

**Days 1-2: Infrastructure + Baseline**
- Implement mHC-lite (180 lines, universal)
- Implement Gated DeltaNet hybrid for B-Rep encoder (use NVIDIA official code, ~150 lines wrapper)
- Run v4.9 baseline with just infrastructure upgrades
- This establishes the new baseline with zero architectural risk

**Days 2-4: Low-risk candidates in parallel**
- Implement A (VIB, ~80 lines) on infrastructure baseline
- Implement D (SIGReg + predictor, ~120 lines) on infrastructure baseline
- Run both overnight, evaluate in morning
- Compare: infrastructure-only vs +VIB vs +SIGReg+pred

**Days 4-7: B-Rep encoder upgrade**
- Implement B (Sheaf encoder, ~200 lines)
- Run B on infrastructure baseline
- Run B + best of {A, D}
- This is the riskiest single component; needs 3 days for implementation + debugging

**Days 7-10: Hierarchical text**
- Implement C (nested contrastive, ~200 lines)
- Run C standalone on infrastructure baseline
- Run C + B + best of {A, D}

**Days 10-13: Full stack + ablations**
- Run E (full combination) if components individually work
- Run ablation matrix for paper: remove each component one at a time
- Generate final numbers

**Days 13-15: Paper writing buffer**

The key principle: every component is independently testable and provides an independent ablation point. You never need "everything to work at once" to have publishable results. Even if only 2 of 4 candidates improve over baseline, the ablation study tells a clear story.

---

## 8. Connection to Multimodal CoT Vision

Your insight about multimodal reasoning tokens for engineering design is architecturally sound and connects to CLIP4CAD in a specific way.

**The research path:**

1. **Paper 1 (current, SGP'26):** CLIP4CAD establishes a hierarchically-structured multimodal latent space for B-Rep. If the nested structure works, z[:128] reliably encodes "what kind of thing," z[128:256] encodes "what it's for," z[256:384] encodes "its precise geometry."

2. **Paper 2 (future):** Train a JEPA-style predictor in this space. Given z_t (current B-Rep state) and action a_t (CAD operation: extrude, fillet, boolean), predict z_{t+1} using AdaLN conditioning (following LeWorldModel). The predictor's latent states during rollout are the multimodal CoT tokens: they encode geometric imagination without passing through text.

3. **Paper 3 (future):** Integrate these tokens into an LLM reasoning loop. Following Mirage (arXiv:2506.17218), the LLM generates latent geometric tokens interleaved with text tokens during design reasoning. Following COCONUT (arXiv:2412.06769, ICLR 2025), these tokens maintain superposition over multiple design alternatives (the BFS property) before committing.

**Why CAD is uniquely suited for this:**

Mirage showed linguistic interference degrades spatial planning from 0.72 to 0.47 when forced through text CoT. CAD design is fundamentally spatial planning. Latent visual/geometric reasoning should bypass this interference.

CAD operations are deterministic. Unlike general visual reasoning, you can always verify the imagined result against the actual CAD kernel output. This makes CAD a uniquely suitable domain for verifiable latent reasoning, analogous to how code and math are verifiable domains for text-based reasoning.

The AIDL approach (Jones et al., Pacific Graphics/CGF 2025, arXiv:2502.09819) already separates LLM strategy from solver precision for 2D CAD. The multimodal CoT extends this to 3D: the LLM reasons about design intent via latent geometric tokens, and a deterministic CAD kernel verifies/executes the results.

**What the current paper must accomplish for this to work:** The latent space must (a) preserve instance discrimination (retrieval), (b) encode semantic hierarchy (nested structure), and (c) be smooth enough for a future predictor to operate in (VIB/SIGReg regularization). All three are addressed by the candidate architectures above.

---

## 9. Key Numbers for Paper Framing

| System | Text->BRep R@1 | Text->PC R@1 | Source |
|--------|---------------|-------------|--------|
| Random | ~0.07% | ~0.07% | Theoretical (1/val_size) |
| Symmetric GFA baseline | 37.07% | 36.25% | Your experiment |
| Global-Only InfoNCE | 51.37% | 69.63% | Your ablation |
| Asymmetric GFA (current best) | 54.78% | 59.66% | Your ablation |
| DuoDuo CLIP (ICLR 2025) | N/A (no B-Rep) | RR@1=15.90 | Literature (Text2Shape) |
| OpenShape (NeurIPS 2023) | N/A | RR@1=10.53 | Literature (Text2Shape) |
| Uni3D-giant (ICLR 2024) | N/A | RR@1=10.78 | Literature (Text2Shape) |

No directly comparable B-Rep text retrieval benchmark exists. You are establishing this benchmark. The paper contribution is: (1) MM-CAD dataset with 3-level engineered captions for 192K B-Rep models, (2) CLIP4CAD architecture with sheaf encoding + VIB + hierarchical alignment, (3) first Text-to-BRep retrieval results, (4) ablation study demonstrating independent contributions.

---

## 10. What NOT To Do

Hard constraints from experimental history:

1. **No discrete codebooks for retrieval.** v4.0 through v4.8.2 prove this with monotonic decline to 0%.
2. **No gap-closing losses (L_ATP, L_CU).** v4.8.2: gap closed successfully (8.61 -> 1.65), retrieval still 0%. Gap and retrieval are independent.
3. **No local contrastive loss.** Global-only at 51.37% vs baseline with local contrastive at 37.07%. Local contrastive creates false negatives in CAD data where similar parts legitimately share geometric features.
4. **No unified encoders.** HUS: gates stuck at 0.5, R@1 < 3%. Modality-specific encoding is essential.
5. **No symmetric loss weights.** Asymmetric (+48%) vs symmetric (baseline). The modalities are fundamentally asymmetric: random-init B-Rep vs pretrained PC.
6. **Don't monitor average cosine.** v4.8.2: cosine 0.985, retrieval 0.0%. Monitor margin (pos_sim - neg_sim).
7. **Don't over-complicate the loss.** GFA v1 had 5 competing losses at 37.07%. Asymmetric GFA had 2 effective losses at 54.78%. Every loss you add must have independent theoretical justification AND empirical validation on your data.
8. **Don't trust "more training fixes bad architecture."** v4.8.2 trained for 37 full epochs across 3 stages. Still 0%. HUS trained for 30 epochs. Still <3%.

---

## 11. References Cited in This Document

### Architecture sources
- mHC: arXiv:2512.24880 (DeepSeek-AI)
- mHC-lite: arXiv:2601.05732 (Yang & Gao)
- KromHC: arXiv:2601.21579
- Gated DeltaNet: arXiv:2412.06464 (ICLR 2025, NVIDIA)
- Qwen3-Next hybrid: Qwen3.5 technical blog
- Hybrid analysis: arXiv:2507.06457

### Information theory
- VIB: Alemi et al., ICLR 2017, arXiv:1612.00410
- Achille-Soatto invariance-minimality: JMLR 2018, arXiv:1706.01350
- OMIB: Wu et al., ICML 2025, arXiv:2505.19996
- Almudever et al. VIB alignment: ICML 2025, arXiv:2506.04870
- Modality gap: Liang et al., NeurIPS 2022, arXiv:2203.02053

### Sheaf neural networks
- NSD: Bodnar et al., NeurIPS 2022, arXiv:2202.04579
- BuNN: Bamberger et al., ICLR 2025 Spotlight, arXiv:2405.15540
- PolyNSD: Borgi et al., arXiv:2512.00242
- ETNN: Battiloro et al., ICLR 2025, arXiv:2405.15429
- Copresheaf TNN: NeurIPS 2025, arXiv:2505.21251

### JEPA and predictive objectives
- LeWorldModel: arXiv:2603.19312
- LeJEPA/SIGReg: arXiv:2511.08544
- CLIPred: NeurIPS 2025
- V-JEPA 2: arXiv:2506.09985
- VL-JEPA: arXiv:2512.10942, ICLR 2026

### Hierarchical latent spaces
- LoST: CVPR 2026, arXiv:2603.17995
- Matryoshka Representation Learning: NeurIPS 2022
- Nested Dropout: Rippel et al., ICML 2014
- Variational Nested Dropout: Cui et al., IEEE TPAMI 2023, arXiv:2101.11353
- RIDA/RKD: Park et al., CVPR 2019, arXiv:1904.05068
- Aristotelian calibration: Groger et al., arXiv:2602.14486

### Latent reasoning (future direction)
- COCONUT: Hao et al., ICLR 2025, arXiv:2412.06769
- Mirage: Yang et al., arXiv:2506.17218
- AIDL: Jones et al., Pacific Graphics/CGF 2025, arXiv:2502.09819
- CAD-Coder: NeurIPS 2025, arXiv:2505.19713

### B-Rep learning
- UV-Net: Jayaraman et al., CVPR 2021, arXiv:2006.10211
- BRepNet: Lambourne et al., CVPR 2021
- AAGNet: Wu et al., RCIM 2024
- BRepFormer: Dai et al., ACM ICMR 2025, arXiv:2504.07378
- FoV-Net: Ballegeer & Benoit, arXiv:2602.24084
- AutoBrep: Xu et al., SIGGRAPH Asia 2025, arXiv:2512.03018
