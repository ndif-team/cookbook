"""
Grokking modular arithmetic model definition (plain PyTorch).

Architecture:
  - n_layers=1, n_heads=4, d_model=128, d_head=32
  - d_vocab=114 (p=113, plus one "=" token), n_ctx=3
  - d_mlp=512, act_fn='relu'
  - normalization_type=None (no LayerNorm)
  - positional_embedding_type='standard' (learned absolute)
  - No tokenizer (integer token IDs directly)

Forward pass:
  1. residual = W_E[tokens] + W_pos[positions]
  2. residual = residual + MultiHeadAttention(residual)
  3. mlp_in = residual @ W_in + b_in
  4. mlp_act = relu(mlp_in)
  5. mlp_out = mlp_act @ W_out + b_out
  6. residual = residual + mlp_out
  7. logits = residual @ W_U + b_U

The model exposes intermediate activations via Identity hook points:
  - blocks.0.attn.hook_pattern: attention pattern [batch, n_heads, pos_q, pos_k]
  - blocks.0.mlp.hook_pre: pre-ReLU activations [batch, pos, d_mlp]
  - blocks.0.mlp.hook_post: post-ReLU activations [batch, pos, d_mlp]

These can be accessed via NNsight tracing.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Model definition
# ============================================================

class HookPoint(nn.Module):
    """Identity module used as a hook point for NNsight tracing."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Attention(nn.Module):
    """Multi-head causal self-attention (no layernorm, matching TL conventions)."""

    def __init__(self, n_heads: int, d_model: int, d_head: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.scale = math.sqrt(d_head)

        # Per-head projection weights in TL layout: [n_heads, d_model, d_head]
        self.W_Q = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_K = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_V = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_O = nn.Parameter(torch.empty(n_heads, d_head, d_model))

        self.b_Q = nn.Parameter(torch.zeros(n_heads, d_head))
        self.b_K = nn.Parameter(torch.zeros(n_heads, d_head))
        self.b_V = nn.Parameter(torch.zeros(n_heads, d_head))
        self.b_O = nn.Parameter(torch.zeros(d_model))

        # Hook point for attention pattern
        self.hook_pattern = HookPoint()

        # Causal mask
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
            attn_out: [batch, pos, d_model]
        """
        batch, seq_len, _ = x.shape

        # Q, K, V: [batch, pos, n_heads, d_head]
        q = torch.einsum("bpd,hdi->bphi", x, self.W_Q) + self.b_Q
        k = torch.einsum("bpd,hdi->bphi", x, self.W_K) + self.b_K
        v = torch.einsum("bpd,hdi->bphi", x, self.W_V) + self.b_V

        # Rearrange for attention: [batch, n_heads, pos, d_head]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        # Attention scores: [batch, n_heads, pos_q, pos_k]
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale

        # Causal mask
        causal_mask = self.mask[:seq_len, :seq_len]
        attn_scores = torch.where(
            causal_mask[None, None, :, :],
            attn_scores,
            self.IGNORE,
        )

        # Attention pattern (with hook point for NNsight)
        pattern = F.softmax(attn_scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = self.hook_pattern(pattern)

        # Apply attention to values
        v = v.permute(0, 2, 1, 3)  # [batch, n_heads, pos, d_head]
        z = pattern @ v  # [batch, n_heads, pos, d_head]
        z = z.permute(0, 2, 1, 3)  # [batch, pos, n_heads, d_head]

        # Output projection
        z_flat = z.reshape(batch, seq_len, self.n_heads * self.d_head)
        w_o = self.W_O.permute(2, 0, 1).reshape(self.d_model, self.n_heads * self.d_head)
        out = F.linear(z_flat, w_o, self.b_O)

        return out


class MLP(nn.Module):
    """Single-layer MLP with ReLU activation and hook points for pre/post activations."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        # Stored in TL layout: W_in [d_model, d_mlp], W_out [d_mlp, d_model]
        self.W_in = nn.Parameter(torch.empty(d_model, d_mlp))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.empty(d_mlp, d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))

        # Hook points for pre/post activations
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, pos, d_model]
        Returns:
            out: [batch, pos, d_model]
        """
        pre = x @ self.W_in + self.b_in  # [batch, pos, d_mlp]
        pre = self.hook_pre(pre)
        post = F.relu(pre)
        post = self.hook_post(post)
        out = post @ self.W_out + self.b_out  # [batch, pos, d_model]
        return out


class TransformerBlock(nn.Module):
    """Single transformer block with attention + MLP (no LayerNorm)."""

    def __init__(self, n_heads: int, d_model: int, d_head: int, d_mlp: int, n_ctx: int):
        super().__init__()
        self.attn = Attention(n_heads, d_model, d_head, n_ctx)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class GrokkingTransformer(nn.Module):
    """
    1-layer transformer for modular arithmetic (grokking), matching the
    architecture from the ARENA grokking exercises.

    No LayerNorm, standard learned positional embeddings, ReLU MLP.

    Exposes intermediate activations via HookPoint modules:
      - blocks[i].attn.hook_pattern: attention pattern [batch, n_heads, pos_q, pos_k]
      - blocks[i].mlp.hook_pre: pre-ReLU activations [batch, pos, d_mlp]
      - blocks[i].mlp.hook_post: post-ReLU activations [batch, pos, d_mlp]
    """

    def __init__(
        self,
        n_layers: int = 1,
        n_heads: int = 4,
        d_model: int = 128,
        d_head: int = 32,
        d_mlp: int = 512,
        d_vocab: int = 114,
        n_ctx: int = 3,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.d_mlp = d_mlp
        self.d_vocab = d_vocab
        self.n_ctx = n_ctx

        # Embeddings
        self.embed = nn.Embedding(d_vocab, d_model)
        self.pos_embed = nn.Embedding(n_ctx, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads, d_model, d_head, d_mlp, n_ctx)
            for _ in range(n_layers)
        ])

        # Unembed
        self.unembed = nn.Linear(d_model, d_vocab)

    def forward(
        self,
        tokens: torch.Tensor = None,
        *,
        input_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, pos] integer token IDs
        Returns:
            logits: [batch, pos, d_vocab]
        """
        if tokens is None and input_ids is not None:
            tokens = input_ids
        elif tokens is None:
            raise ValueError("Must provide either tokens or input_ids")

        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)

        residual = self.embed(tokens) + self.pos_embed(positions)

        for block in self.blocks:
            residual = block(residual)

        logits = self.unembed(residual)
        return logits


# ============================================================
# Convenience: load model from saved state_dict
# ============================================================

def load_model(
    state_dict_path: str | None = None,
    device: str = "cpu",
) -> GrokkingTransformer:
    """Load the converted model from a saved state dict.

    If state_dict_path is None or the file doesn't exist, downloads from
    HuggingFace Hub (woog/arena-grokking-checkpoints).
    """
    if state_dict_path is None or not os.path.exists(state_dict_path):
        from huggingface_hub import hf_hub_download
        state_dict_path = hf_hub_download(
            repo_id="woog/arena-grokking-checkpoints",
            filename="grokking_model_converted.pt",
        )
    model = GrokkingTransformer()
    sd = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model.to(device)
