"""
Hierarchical Text Encoder

Two-branch encoding:
- Title -> single global embedding (for global alignment)
- Description -> N feature embeddings via learnable queries (for local alignment)
- Confidence scores indicate which queries found relevant content
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any


class HierarchicalTextEncoder(nn.Module):
    """
    Two-branch text encoder for hierarchical alignment.

    Uses a frozen LLM backbone with trainable projection layers.
    """

    def __init__(
        self,
        llm_name: str = "microsoft/phi-2",
        d_unified: int = 256,
        n_local_features: int = 8,
        freeze_llm: bool = True,
        use_fp16: bool = True,
    ):
        """
        Args:
            llm_name: HuggingFace model name for LLM
            d_unified: Unified output dimension
            n_local_features: Number of local feature queries
            freeze_llm: Whether to freeze LLM weights
            use_fp16: Whether to load LLM in FP16
        """
        super().__init__()

        self.d_unified = d_unified
        self.n_local_features = n_local_features
        self.freeze_llm = freeze_llm
        self.llm_name = llm_name

        # Will be initialized lazily or by explicit call
        self.llm = None
        self.tokenizer = None
        self.d_llm = None

        self._use_fp16 = use_fp16

        # Placeholder for dimensions (set after LLM load)
        self._projections_initialized = False

    def _init_projections(self, d_llm: int):
        """Initialize projection layers after knowing LLM dimension."""
        if self._projections_initialized:
            return

        self.d_llm = d_llm

        # Title branch (global)
        self.title_proj = nn.Sequential(
            nn.Linear(d_llm, self.d_unified),
            nn.LayerNorm(self.d_unified),
            nn.GELU(),
            nn.Linear(self.d_unified, self.d_unified),
        )

        # Description branch (local) - projection
        self.desc_proj = nn.Sequential(
            nn.Linear(d_llm, self.d_unified),
            nn.LayerNorm(self.d_unified),
        )

        # Feature queries for local alignment
        self.feature_queries = nn.Parameter(
            torch.randn(self.n_local_features, self.d_unified) * 0.02
        )
        self.feature_pos = nn.Parameter(
            torch.randn(self.n_local_features, self.d_unified) * 0.02
        )

        # Cross-attention for feature extraction
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=self.d_unified,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        self.feature_norm = nn.LayerNorm(self.d_unified)
        self.feature_ffn = nn.Sequential(
            nn.Linear(self.d_unified, self.d_unified * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_unified * 2, self.d_unified),
        )
        self.feature_ffn_norm = nn.LayerNorm(self.d_unified)

        # Confidence predictor (which queries found relevant content)
        self.confidence_mlp = nn.Sequential(
            nn.Linear(self.d_unified, self.d_unified // 4),
            nn.GELU(),
            nn.Linear(self.d_unified // 4, 1),
            nn.Sigmoid(),
        )

        self._projections_initialized = True

    def load_llm(self, device: Optional[torch.device] = None):
        """
        Explicitly load the LLM.

        Call this before training to initialize the model.
        """
        if self.llm is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        print(f"Loading LLM: {self.llm_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        dtype = torch.float16 if self._use_fp16 else torch.float32
        self.llm = AutoModel.from_pretrained(
            self.llm_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        # Get hidden dimension
        if hasattr(self.llm.config, "hidden_size"):
            d_llm = self.llm.config.hidden_size
        elif hasattr(self.llm.config, "n_embd"):
            d_llm = self.llm.config.n_embd
        else:
            d_llm = 2048  # Default fallback

        # Freeze LLM
        if self.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()

        # Initialize projections
        self._init_projections(d_llm)

        # Move to device if specified
        if device is not None:
            self.llm = self.llm.to(device)

        print(f"LLM loaded with hidden_dim={d_llm}")

    def get_tokenizer(self):
        """Get the tokenizer (load if needed)."""
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _get_last_token_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the hidden state of the last non-padding token.
        For causal LLMs, this is the sequence representation.

        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, L]

        Returns:
            [B, D]
        """
        # Find index of last non-padding token
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, seq_lengths]
        return last_hidden

    def forward(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        desc_input_ids: torch.Tensor,
        desc_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode title and description hierarchically.

        Args:
            title_input_ids: [B, L_title]
            title_attention_mask: [B, L_title]
            desc_input_ids: [B, L_desc]
            desc_attention_mask: [B, L_desc]

        Returns:
            t_global: [B, d_unified] global title embedding
            t_local: [B, n_local, d_unified] local feature embeddings
            t_local_confidence: [B, n_local] confidence scores
        """
        if self.llm is None:
            raise RuntimeError("LLM not loaded. Call load_llm() first.")

        B = title_input_ids.size(0)
        device = title_input_ids.device

        # Encode title
        with torch.no_grad() if self.freeze_llm else torch.enable_grad():
            title_out = self.llm(
                input_ids=title_input_ids,
                attention_mask=title_attention_mask,
            )
            title_hidden = title_out.last_hidden_state  # [B, L, d_llm]

        # Get last token for causal LLM
        title_last = self._get_last_token_hidden(title_hidden, title_attention_mask)
        t_global = self.title_proj(title_last.float())  # [B, d_unified]

        # Encode description
        with torch.no_grad() if self.freeze_llm else torch.enable_grad():
            desc_out = self.llm(
                input_ids=desc_input_ids,
                attention_mask=desc_attention_mask,
            )
            desc_hidden = desc_out.last_hidden_state  # [B, L_desc, d_llm]

        # Project description tokens
        desc_proj = self.desc_proj(desc_hidden.float())  # [B, L_desc, d_unified]

        # Initialize feature queries
        queries = self.feature_queries.unsqueeze(0).expand(B, -1, -1) + self.feature_pos

        # Cross-attention to description tokens
        key_padding_mask = ~desc_attention_mask.bool()  # True = ignore

        attn_out, _ = self.feature_attn(
            query=queries,
            key=desc_proj,
            value=desc_proj,
            key_padding_mask=key_padding_mask,
        )

        # Residual + norm
        t_local = self.feature_norm(queries + attn_out)

        # FFN
        t_local = self.feature_ffn_norm(t_local + self.feature_ffn(t_local))

        # Confidence scores
        t_local_confidence = self.confidence_mlp(t_local).squeeze(-1)  # [B, n_local]

        return {
            "t_global": t_global,
            "t_local": t_local,
            "t_local_confidence": t_local_confidence,
        }

    def encode_title_only(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode title only (for inference)."""
        if self.llm is None:
            raise RuntimeError("LLM not loaded. Call load_llm() first.")

        with torch.no_grad() if self.freeze_llm else torch.enable_grad():
            title_out = self.llm(
                input_ids=title_input_ids,
                attention_mask=title_attention_mask,
            )
            title_hidden = title_out.last_hidden_state

        title_last = self._get_last_token_hidden(title_hidden, title_attention_mask)
        t_global = self.title_proj(title_last.float())

        return t_global
