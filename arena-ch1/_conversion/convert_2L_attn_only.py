"""
2-layer attention-only transformer model definition (plain PyTorch).

Architecture matching callummcdougall/attn_only_2L_half:
  - n_layers=2, n_heads=12, d_model=768, d_head=64
  - d_vocab=50278, n_ctx=2048
  - attn_only=True (no MLP blocks)
  - normalization_type=None (no LayerNorm anywhere)
  - positional_embedding_type="standard" (learned absolute)
  - tokenizer: EleutherAI/gpt-neox-20b (GPTNeoXTokenizerFast)

Forward pass:
  1. residual = W_E[tokens] + W_pos[positions]
  2. For each layer i in [0, 1]:
       residual = residual + MultiHeadAttention_i(residual)
  3. logits = residual @ W_U + b_U
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Model definition
# ============================================================

class Attention(nn.Module):
    """Multi-head causal self-attention (no layernorm, matching TL conventions)."""

    def __init__(self, n_heads: int, d_model: int, d_head: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.scale = math.sqrt(d_head)

        # Per-head projection weights, stored in TL layout: [n_heads, d_model, d_head]
        self.W_Q = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_K = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_V = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_O = nn.Parameter(torch.empty(n_heads, d_head, d_model))

        self.b_Q = nn.Parameter(torch.zeros(n_heads, d_head))
        self.b_K = nn.Parameter(torch.zeros(n_heads, d_head))
        self.b_V = nn.Parameter(torch.zeros(n_heads, d_head))
        self.b_O = nn.Parameter(torch.zeros(d_model))

        # Causal mask (not a parameter, just a buffer)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(n_ctx, n_ctx, dtype=torch.bool)),
            persistent=False,
        )
        self.register_buffer(
            "IGNORE",
            torch.tensor(float("-inf")),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, pos, d_model]
        Returns:
            out: [batch, pos, d_model]
        """
        batch, seq_len, _ = x.shape

        # Compute Q, K, V via einsum (matches TL's simple_attn_linear)
        q = torch.einsum("bpd,hdi->bphi", x, self.W_Q) + self.b_Q
        k = torch.einsum("bpd,hdi->bphi", x, self.W_K) + self.b_K
        v = torch.einsum("bpd,hdi->bphi", x, self.W_V) + self.b_V

        # Rearrange for attention: [batch, n_heads, pos, d_head]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        # Attention scores: [batch, n_heads, pos, pos]
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale

        # Causal mask
        causal_mask = self.mask[:seq_len, :seq_len]
        attn_scores = torch.where(
            causal_mask[None, None, :, :],
            attn_scores,
            self.IGNORE,
        )

        # Softmax with NaN protection (matching TL)
        pattern = F.softmax(attn_scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)

        # Apply attention to values
        v = v.permute(0, 2, 1, 3)  # [batch, n_heads, pos, d_head]
        z = pattern @ v  # [batch, n_heads, pos, d_head]
        z = z.permute(0, 2, 1, 3)  # [batch, pos, n_heads, d_head]

        # Output projection
        z_flat = z.reshape(batch, seq_len, self.n_heads * self.d_head)
        w_o = self.W_O.permute(2, 0, 1).reshape(self.d_model, self.n_heads * self.d_head)
        out = F.linear(z_flat, w_o, self.b_O)

        return out


class TransformerBlock(nn.Module):
    """Single attention-only transformer block (no LN, no MLP)."""

    def __init__(self, n_heads: int, d_model: int, d_head: int, n_ctx: int):
        super().__init__()
        self.attn = Attention(n_heads, d_model, d_head, n_ctx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


class AttnOnly2L(nn.Module):
    """
    2-layer attention-only transformer matching the callummcdougall/attn_only_2L_half
    architecture.

    No LayerNorm, no MLP, standard learned positional embeddings.

    The forward() accepts either positional args (tokens tensor) or HuggingFace-style
    keyword arguments (input_ids=, attention_mask=) so it can be used with both
    nnsight.NNsight and nnsight.LanguageModel.
    """

    def __init__(
        self,
        n_layers: int = 2,
        n_heads: int = 12,
        d_model: int = 768,
        d_head: int = 64,
        d_vocab: int = 50278,
        n_ctx: int = 2048,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.d_vocab = d_vocab
        self.n_ctx = n_ctx

        # Embeddings
        self.embed = nn.Embedding(d_vocab, d_model)  # token embedding
        self.pos_embed = nn.Embedding(n_ctx, d_model)  # positional embedding

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads, d_model, d_head, n_ctx)
            for _ in range(n_layers)
        ])

        # Unembed (no final layernorm since normalization_type=None)
        self.unembed = nn.Linear(d_model, d_vocab)

    def forward(
        self,
        tokens: torch.Tensor = None,
        *,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, pos] integer token IDs (positional argument)
            input_ids: [batch, pos] integer token IDs (keyword argument, HF-style)
            attention_mask: ignored (accepted for HF compatibility)
        Returns:
            logits: [batch, pos, d_vocab]
        """
        # Accept either positional or keyword input_ids
        if tokens is None and input_ids is not None:
            tokens = input_ids
        elif tokens is None:
            raise ValueError("Must provide either tokens or input_ids")

        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)

        # Embed
        residual = self.embed(tokens) + self.pos_embed(positions)

        # Transformer blocks
        for block in self.blocks:
            residual = block(residual)

        # Unembed
        logits = self.unembed(residual)
        return logits


# ============================================================
# Convenience: load model from saved state_dict
# ============================================================

def load_model(
    state_dict_path: str | None = None,
    device: str = "cpu",
) -> AttnOnly2L:
    """Load the converted model from a saved state dict.

    If state_dict_path is None or the file doesn't exist, downloads from
    HuggingFace Hub (woog/arena-attn-only-2L).
    """
    if state_dict_path is None or not os.path.exists(state_dict_path):
        from huggingface_hub import hf_hub_download
        state_dict_path = hf_hub_download(
            repo_id="woog/arena-attn-only-2L",
            filename="attn_only_2L_converted.pt",
        )
    model = AttnOnly2L()
    sd = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model
