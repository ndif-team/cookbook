"""
OthelloGPT model definition (plain PyTorch).

Architecture matching NeelNanda/Othello-GPT-Transformer-Lens (synthetic model):
  - n_layers=8, n_heads=8, d_model=512, d_head=64, d_mlp=2048
  - d_vocab=61, n_ctx=59
  - act_fn="gelu" (torch.nn.functional.gelu)
  - normalization_type="LNPre" (center + normalize, NO learnable params)
  - No tokenizer (raw integer token IDs 0..60)

Forward pass (LNPre = pre-norm with no learnable scale/bias):
  1. residual = W_E[tokens] + W_pos[positions]
  2. For each layer i in [0..7]:
       a. ln1_out = LayerNormPre(residual)
       b. attn_out = MultiHeadAttention(ln1_out)
       c. residual = residual + attn_out
       d. ln2_out = LayerNormPre(residual)
       e. mlp_out = MLP(ln2_out)   # W_in, b_in, GELU, W_out, b_out
       f. residual = residual + mlp_out
  3. ln_final_out = LayerNormPre(residual)
  4. logits = ln_final_out @ W_U + b_U

LayerNormPre (no learnable params):
  x = x - x.mean(-1, keepdim=True)
  scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
  return x / scale
"""

import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Model definition
# ============================================================

class LayerNormPre(nn.Module):
    """Pre-norm layer normalization without learnable parameters.
    Centers and normalizes, matching TL's LayerNormPre."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(-1, keepdim=True)
        scale = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return x / scale


class Attention(nn.Module):
    """Multi-head causal self-attention matching TL conventions."""

    def __init__(self, n_heads: int, d_model: int, d_head: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.scale = math.sqrt(d_head)

        # Per-head projection weights in TL layout
        self.W_Q = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_K = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_V = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_O = nn.Parameter(torch.empty(n_heads, d_head, d_model))

        self.b_Q = nn.Parameter(torch.zeros(n_heads, d_head))
        self.b_K = nn.Parameter(torch.zeros(n_heads, d_head))
        self.b_V = nn.Parameter(torch.zeros(n_heads, d_head))
        self.b_O = nn.Parameter(torch.zeros(d_model))

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

        # Q, K, V projections
        q = torch.einsum("bpd,hdi->bphi", x, self.W_Q) + self.b_Q
        k = torch.einsum("bpd,hdi->bphi", x, self.W_K) + self.b_K
        v = torch.einsum("bpd,hdi->bphi", x, self.W_V) + self.b_V

        # Rearrange for attention: [batch, n_heads, pos, d_head]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        # Attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale

        # Causal mask
        causal_mask = self.mask[:seq_len, :seq_len]
        attn_scores = torch.where(
            causal_mask[None, None, :, :],
            attn_scores,
            self.IGNORE,
        )

        # Softmax
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


class MLP(nn.Module):
    """Feed-forward MLP with GELU activation."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(d_model, d_mlp))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.empty(d_mlp, d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match TL's batch_addmm: pre_act = x @ W_in + b_in
        pre_act = torch.addmm(self.b_in, x.reshape(-1, x.shape[-1]), self.W_in).reshape(
            *x.shape[:-1], -1
        )
        post_act = F.gelu(pre_act)
        out = torch.addmm(self.b_out, post_act.reshape(-1, post_act.shape[-1]), self.W_out).reshape(
            *x.shape[:-1], -1
        )
        return out


class TransformerBlock(nn.Module):
    """Single pre-norm transformer block with attention + MLP."""

    def __init__(self, n_heads: int, d_model: int, d_head: int, d_mlp: int, n_ctx: int, eps: float = 1e-5):
        super().__init__()
        self.ln1 = LayerNormPre(eps=eps)
        self.attn = Attention(n_heads, d_model, d_head, n_ctx)
        self.ln2 = LayerNormPre(eps=eps)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        # Pre-norm MLP
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        return x


class OthelloGPT(nn.Module):
    """
    8-layer transformer trained to predict legal Othello moves.

    Architecture matches NeelNanda/Othello-GPT-Transformer-Lens (synthetic model).
    Pre-norm (LayerNormPre) with no learnable LN parameters.

    Forward pass accepts either positional tokens arg or keyword input_ids.
    Returns logits of shape [batch, seq_len, 61].
    """

    def __init__(
        self,
        n_layers: int = 8,
        n_heads: int = 8,
        d_model: int = 512,
        d_head: int = 64,
        d_mlp: int = 2048,
        d_vocab: int = 61,
        n_ctx: int = 59,
        eps: float = 1e-5,
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
            TransformerBlock(n_heads, d_model, d_head, d_mlp, n_ctx, eps=eps)
            for _ in range(n_layers)
        ])

        # Final layer norm (pre-norm before unembed)
        self.ln_final = LayerNormPre(eps=eps)

        # Unembed
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
            tokens: [batch, pos] integer token IDs (0..60)
            input_ids: alternative keyword argument for tokens
        Returns:
            logits: [batch, pos, d_vocab]
        """
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

        # Final norm + unembed
        residual = self.ln_final(residual)
        logits = self.unembed(residual)
        return logits


# ============================================================
# Convenience: load model from saved state_dict
# ============================================================

def convert_from_hf(save_path: str | None = None) -> dict:
    """Download the TransformerLens checkpoint from HuggingFace and convert to our format.

    Downloads from NeelNanda/Othello-GPT-Transformer-Lens and remaps weight keys
    to match OthelloGPT's nn.Module structure.
    """
    from huggingface_hub import hf_hub_download

    tl_path = hf_hub_download(
        repo_id="NeelNanda/Othello-GPT-Transformer-Lens",
        filename="synthetic_model.pth",
    )
    tl_sd = torch.load(tl_path, map_location="cpu", weights_only=True)

    new_sd = {}
    for k, v in tl_sd.items():
        if k == "embed.W_E":
            new_sd["embed.weight"] = v
        elif k == "pos_embed.W_pos":
            new_sd["pos_embed.weight"] = v
        elif k == "unembed.W_U":
            new_sd["unembed.weight"] = v.T  # nn.Linear stores [out, in]
        elif k == "unembed.b_U":
            new_sd["unembed.bias"] = v
        elif k.endswith(".mask") or k.endswith(".IGNORE"):
            continue  # buffers created by model init
        else:
            new_sd[k] = v

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(new_sd, save_path)

    return new_sd


def load_model(
    state_dict_path: str | None = None,
    device: str = "cpu",
) -> OthelloGPT:
    """Load the converted model from a saved state dict.

    If state_dict_path is None or the file doesn't exist, automatically
    downloads from NeelNanda/Othello-GPT-Transformer-Lens and converts.
    """
    if state_dict_path is None:
        state_dict_path = str(Path(__file__).resolve().parent / "othello_gpt_converted.pt")
    if os.path.exists(state_dict_path):
        sd = torch.load(state_dict_path, map_location=device, weights_only=True)
    else:
        print(f"Converted model not found at {state_dict_path}, downloading and converting...")
        sd = convert_from_hf(save_path=state_dict_path)
    model = OthelloGPT()
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model
