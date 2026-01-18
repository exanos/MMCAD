"""
Hierarchical Text Encoder

Two-branch encoding:
- Title -> single global embedding (for global alignment)
- Description -> N feature embeddings via learnable queries (for local alignment)

Supports:
1. Live LLM inference (for inference/evaluation)
2. Pre-computed embeddings (for training - much faster)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path


class TextProjectionHead(nn.Module):
    """
    Projection head for text features.

    Used to project raw LLM hidden states or pre-computed embeddings
    to the unified representation space.
    """

    def __init__(
        self,
        d_input: int,
        d_unified: int,
        n_local_features: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_input = d_input
        self.d_unified = d_unified
        self.n_local_features = n_local_features

        # Title branch (global) - project last token hidden state
        self.title_proj = nn.Sequential(
            nn.Linear(d_input, d_unified),
            nn.LayerNorm(d_unified),
            nn.GELU(),
            nn.Linear(d_unified, d_unified),
        )

        # Description branch (local) - project token sequence
        self.desc_proj = nn.Sequential(
            nn.Linear(d_input, d_unified),
            nn.LayerNorm(d_unified),
        )

        # Feature queries for local alignment (cross-attention)
        self.feature_queries = nn.Parameter(
            torch.randn(n_local_features, d_unified) * 0.02
        )
        self.feature_pos = nn.Parameter(
            torch.randn(n_local_features, d_unified) * 0.02
        )

        # Cross-attention for feature extraction
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=d_unified,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        self.feature_norm = nn.LayerNorm(d_unified)
        self.feature_ffn = nn.Sequential(
            nn.Linear(d_unified, d_unified * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_unified * 2, d_unified),
        )
        self.feature_ffn_norm = nn.LayerNorm(d_unified)

        # Confidence predictor (which queries found relevant content)
        self.confidence_mlp = nn.Sequential(
            nn.Linear(d_unified, d_unified // 4),
            nn.GELU(),
            nn.Linear(d_unified // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        title_hidden: torch.Tensor,
        desc_hidden: torch.Tensor,
        desc_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Project LLM hidden states to unified space.

        Args:
            title_hidden: [B, d_input] - last token hidden from title
            desc_hidden: [B, L_desc, d_input] - all token hiddens from description
            desc_mask: [B, L_desc] - attention mask (1=valid, 0=padding)

        Returns:
            t_global: [B, d_unified]
            t_local: [B, n_local, d_unified]
            t_local_confidence: [B, n_local]
        """
        B = title_hidden.size(0)

        # Project title to global embedding
        t_global = self.title_proj(title_hidden)  # [B, d_unified]

        # Project description tokens
        desc_proj = self.desc_proj(desc_hidden)  # [B, L_desc, d_unified]

        # Initialize feature queries
        queries = self.feature_queries.unsqueeze(0).expand(B, -1, -1) + self.feature_pos

        # Cross-attention to description tokens
        key_padding_mask = None
        if desc_mask is not None:
            key_padding_mask = ~desc_mask.bool()  # True = ignore

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


class HierarchicalTextEncoder(nn.Module):
    """
    Two-branch text encoder for hierarchical alignment.

    Supports two modes:
    1. Live mode: Uses LLM to encode text on-the-fly
    2. Cached mode: Uses pre-computed embeddings (faster for training)
    """

    # Supported models and their hidden dimensions
    SUPPORTED_MODELS = {
        # Phi models
        "microsoft/phi-4-mini": 3072,
        "microsoft/Phi-4-mini-instruct": 3072,
        "microsoft/phi-2": 2560,
        # Qwen models
        "Qwen/Qwen2.5-3B-Instruct": 2048,
        "Qwen/Qwen2.5-1.5B-Instruct": 1536,
        "Qwen/Qwen3-4B": 2560,
    }

    def __init__(
        self,
        llm_name: str = "microsoft/Phi-4-mini-instruct",
        d_unified: int = 256,
        n_local_features: int = 8,
        freeze_llm: bool = True,
        use_fp16: bool = True,
        use_cached_embeddings: bool = False,
    ):
        """
        Args:
            llm_name: HuggingFace model name for LLM
            d_unified: Unified output dimension
            n_local_features: Number of local feature queries
            freeze_llm: Whether to freeze LLM weights
            use_fp16: Whether to load LLM in FP16
            use_cached_embeddings: If True, expects pre-computed embeddings
        """
        super().__init__()

        self.d_unified = d_unified
        self.n_local_features = n_local_features
        self.freeze_llm = freeze_llm
        self.llm_name = llm_name
        self.use_cached_embeddings = use_cached_embeddings
        self._use_fp16 = use_fp16

        # Get LLM hidden dimension
        if llm_name in self.SUPPORTED_MODELS:
            self.d_llm = self.SUPPORTED_MODELS[llm_name]
        else:
            # Try to infer from name or use default
            self.d_llm = 2048
            print(f"Warning: Unknown model {llm_name}, assuming d_llm={self.d_llm}")

        # Initialize projection head (always needed)
        self.projection = TextProjectionHead(
            d_input=self.d_llm,
            d_unified=d_unified,
            n_local_features=n_local_features,
        )

        # LLM will be initialized lazily (only if not using cached embeddings)
        self.llm = None
        self.tokenizer = None

    def load_llm(self, device: Optional[torch.device] = None):
        """
        Load the LLM for live inference.

        Not needed if using cached embeddings.
        """
        if self.use_cached_embeddings:
            print("Using cached embeddings, skipping LLM load")
            return

        if self.llm is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        print(f"Loading LLM: {self.llm_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_name,
            trust_remote_code=True,
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with appropriate dtype
        dtype = torch.float16 if self._use_fp16 else torch.float32

        try:
            self.llm = AutoModel.from_pretrained(
                self.llm_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            )
        except Exception as e:
            print(f"Flash attention not available, using eager: {e}")
            self.llm = AutoModel.from_pretrained(
                self.llm_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        # Verify hidden dimension
        if hasattr(self.llm.config, "hidden_size"):
            actual_d_llm = self.llm.config.hidden_size
        elif hasattr(self.llm.config, "n_embd"):
            actual_d_llm = self.llm.config.n_embd
        else:
            actual_d_llm = self.d_llm

        if actual_d_llm != self.d_llm:
            print(f"Warning: Expected d_llm={self.d_llm}, got {actual_d_llm}")
            # Reinitialize projection if dimension changed
            self.d_llm = actual_d_llm
            self.projection = TextProjectionHead(
                d_input=self.d_llm,
                d_unified=self.d_unified,
                n_local_features=self.n_local_features,
            )

        # Freeze LLM
        if self.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()

        # Move to device
        if device is not None:
            self.llm = self.llm.to(device)

        print(f"LLM loaded: {self.llm_name} (hidden_dim={self.d_llm})")

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
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self.tokenizer

    def _get_last_token_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the hidden state of the last non-padding token.

        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, L]

        Returns:
            [B, D]
        """
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, seq_lengths.long()]
        return last_hidden

    @torch.no_grad()
    def extract_llm_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract raw LLM hidden states (for pre-computation).

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]

        Returns:
            last_hidden: [B, d_llm] - last token hidden state
            all_hidden: [B, L, d_llm] - all token hidden states
        """
        if self.llm is None:
            raise RuntimeError("LLM not loaded. Call load_llm() first.")

        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # [B, L, d_llm]

        # Get last token hidden state
        last_hidden = self._get_last_token_hidden(hidden_states, attention_mask)

        return last_hidden, hidden_states

    def forward_live(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        desc_input_ids: torch.Tensor,
        desc_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with live LLM inference.
        """
        if self.llm is None:
            raise RuntimeError("LLM not loaded. Call load_llm() first.")

        # Extract title features
        with torch.no_grad():
            title_last, _ = self.extract_llm_features(
                title_input_ids, title_attention_mask
            )

            # Extract description features
            _, desc_hidden = self.extract_llm_features(
                desc_input_ids, desc_attention_mask
            )

        # Project through trainable head
        return self.projection(
            title_hidden=title_last.float(),
            desc_hidden=desc_hidden.float(),
            desc_mask=desc_attention_mask,
        )

    def forward_cached(
        self,
        title_embedding: torch.Tensor,
        desc_embedding: torch.Tensor,
        desc_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with pre-computed embeddings.

        Args:
            title_embedding: [B, d_llm] - pre-computed title last token hidden
            desc_embedding: [B, L_desc, d_llm] - pre-computed description hiddens
            desc_mask: [B, L_desc] - attention mask for description
        """
        return self.projection(
            title_hidden=title_embedding,
            desc_hidden=desc_embedding,
            desc_mask=desc_mask,
        )

    def forward(
        self,
        title_input_ids: Optional[torch.Tensor] = None,
        title_attention_mask: Optional[torch.Tensor] = None,
        desc_input_ids: Optional[torch.Tensor] = None,
        desc_attention_mask: Optional[torch.Tensor] = None,
        # Cached embeddings (alternative to input_ids)
        title_embedding: Optional[torch.Tensor] = None,
        desc_embedding: Optional[torch.Tensor] = None,
        desc_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode title and description hierarchically.

        Supports both live inference and cached embeddings.
        """
        # Use cached embeddings if provided
        if title_embedding is not None and desc_embedding is not None:
            return self.forward_cached(title_embedding, desc_embedding, desc_mask)

        # Otherwise use live inference
        if title_input_ids is None or desc_input_ids is None:
            raise ValueError(
                "Either (title_input_ids, desc_input_ids) or "
                "(title_embedding, desc_embedding) must be provided"
            )

        return self.forward_live(
            title_input_ids,
            title_attention_mask,
            desc_input_ids,
            desc_attention_mask,
        )

    def encode_title_only(
        self,
        title_input_ids: Optional[torch.Tensor] = None,
        title_attention_mask: Optional[torch.Tensor] = None,
        title_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode title only (for inference)."""
        if title_embedding is not None:
            return self.projection.title_proj(title_embedding)

        if self.llm is None:
            raise RuntimeError("LLM not loaded. Call load_llm() first.")

        with torch.no_grad():
            title_last, _ = self.extract_llm_features(
                title_input_ids, title_attention_mask
            )

        return self.projection.title_proj(title_last.float())
