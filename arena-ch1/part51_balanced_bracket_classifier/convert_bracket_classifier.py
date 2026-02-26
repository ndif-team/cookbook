"""
Bracket classifier model definition (plain PyTorch).

Architecture:
- 3-layer bidirectional transformer with 2 attention heads per layer
- d_model=56, d_head=28, d_mlp=56, n_ctx=42
- d_vocab=5 (start=0, pad=1, end=2, open=3, close=4)
- d_vocab_out=2 (binary classification: unbalanced vs balanced)
- ReLU activation in MLP
- No causal mask (bidirectional attention)
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Plain PyTorch model ──────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.w = nn.Parameter(t.ones(d_model))
        self.b = nn.Parameter(t.zeros(d_model))
        self.eps = eps

    def forward(self, x: t.Tensor) -> t.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, correction=0)
        return self.w * (x - mean) / (std + self.eps) + self.b


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        # Per-head Q, K, V, O weight matrices
        self.W_Q = nn.Parameter(t.empty(n_heads, d_model, d_head))
        self.W_K = nn.Parameter(t.empty(n_heads, d_model, d_head))
        self.W_V = nn.Parameter(t.empty(n_heads, d_model, d_head))
        self.W_O = nn.Parameter(t.empty(n_heads, d_head, d_model))
        self.b_Q = nn.Parameter(t.zeros(n_heads, d_head))
        self.b_K = nn.Parameter(t.zeros(n_heads, d_head))
        self.b_V = nn.Parameter(t.zeros(n_heads, d_head))
        self.b_O = nn.Parameter(t.zeros(d_model))

    def forward(self, x: t.Tensor, padding_mask: t.Tensor | None = None) -> t.Tensor:
        """
        x: (batch, seq, d_model)
        padding_mask: (batch, seq) bool, True where tokens are padding
        Returns: (batch, seq, d_model), attn_patterns: (batch, n_heads, seq_q, seq_k)
        """
        # x: [batch, seq, d_model]
        # Q/K/V: [batch, seq, n_heads, d_head]
        # W_Q shape: [n_heads, d_model, d_head]
        q = t.einsum("bsd,ndh->bsnh", x, self.W_Q) + self.b_Q  # [b, s, n, dh]
        k = t.einsum("bsd,ndh->bsnh", x, self.W_K) + self.b_K
        v = t.einsum("bsd,ndh->bsnh", x, self.W_V) + self.b_V

        # Attention scores: [batch, n_heads, seq_q, seq_k]
        attn_scores = t.einsum("bqnh,bknh->bnqk", q, k) / math.sqrt(self.d_head)

        # Apply padding mask if provided
        if padding_mask is not None:
            # padding_mask: [batch, seq] -> [batch, 1, 1, seq_k]
            attn_scores = attn_scores.masked_fill(
                padding_mask[:, None, None, :], -1e5
            )

        attn_probs = F.softmax(attn_scores, dim=-1)  # [b, n, sq, sk]

        # Weighted sum of values
        z = t.einsum("bnqk,bknh->bqnh", attn_probs, v)  # [b, sq, n, dh]

        # Per-head results before summing; W_O shape: [n_heads, d_head, d_model]
        result = t.einsum("bsnh,nhd->bsnd", z, self.W_O)  # [b, s, n, d_model] -- this is correct since W_O is [n, d_head, d_model]

        # Sum over heads and add bias
        out = result.sum(dim=2) + self.b_O  # [b, s, d_model]

        return out, attn_probs, result


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Parameter(t.empty(d_model, d_mlp))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.empty(d_mlp, d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))

    def forward(self, x: t.Tensor) -> t.Tensor:
        pre = x @ self.W_in + self.b_in
        post = F.relu(pre)
        out = post @ self.W_out + self.b_out
        return out, pre, post


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_mlp: int):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, d_head)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: t.Tensor, padding_mask: t.Tensor | None = None):
        # Attention sublayer
        ln1_out = self.ln1(x)
        attn_out, attn_probs, attn_result = self.attn(ln1_out, padding_mask)
        resid_mid = x + attn_out

        # MLP sublayer
        ln2_out = self.ln2(resid_mid)
        mlp_out, mlp_pre, mlp_post = self.mlp(ln2_out)
        resid_post = resid_mid + mlp_out

        return resid_post, attn_probs, attn_result, mlp_out, mlp_pre, mlp_post, resid_mid, ln1_out, ln2_out


class BracketClassifier(nn.Module):
    """
    A 3-layer bidirectional transformer for bracket classification.

    Config:
        n_ctx=42, d_model=56, d_head=28, n_heads=2, d_mlp=56, n_layers=3
        d_vocab=5, d_vocab_out=2, act_fn=relu, attention_dir=bidirectional

    Attributes accessible for interpretability:
        - embed.W_E: [d_vocab, d_model]
        - pos_embed.W_pos: [n_ctx, d_model]
        - blocks[i].ln1, blocks[i].ln2: LayerNorm
        - blocks[i].attn: Attention (W_Q, W_K, W_V, W_O, b_Q, b_K, b_V, b_O)
        - blocks[i].mlp: MLP (W_in, b_in, W_out, b_out)
        - ln_final: LayerNorm
        - unembed.W_U: [d_model, d_vocab_out]
        - unembed.b_U: [d_vocab_out]
    """
    def __init__(
        self,
        n_ctx: int = 42,
        d_model: int = 56,
        d_head: int = 28,
        n_heads: int = 2,
        d_mlp: int = 56,
        n_layers: int = 3,
        d_vocab: int = 5,
        d_vocab_out: int = 2,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.d_vocab = d_vocab
        self.d_vocab_out = d_vocab_out

        # Embeddings
        self.embed = nn.Module()
        self.embed.W_E = nn.Parameter(t.empty(d_vocab, d_model))
        self.pos_embed = nn.Module()
        self.pos_embed.W_pos = nn.Parameter(t.empty(n_ctx, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_head, d_mlp)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_final = LayerNorm(d_model)

        # Unembedding
        self.unembed = nn.Module()
        self.unembed.W_U = nn.Parameter(t.empty(d_model, d_vocab_out))
        self.unembed.b_U = nn.Parameter(t.zeros(d_vocab_out))

    def forward(self, tokens: t.Tensor = None, padding_mask: t.Tensor | None = None, *, input_ids: t.Tensor = None, **kwargs):
        """
        tokens: (batch, seq) integer token IDs
        padding_mask: (batch, seq) bool tensor, True where tokens are padding
        input_ids: alternative keyword argument for tokens (HF/NNsight compatibility)
        Returns: logits (batch, seq, d_vocab_out)
        """
        if tokens is None and input_ids is not None:
            tokens = input_ids
        elif tokens is None:
            raise ValueError("Must provide either tokens or input_ids")
        # Embedding
        tok_embed = self.embed.W_E[tokens]  # [batch, seq, d_model]
        pos_embed = self.pos_embed.W_pos[:tokens.shape[1]]  # [seq, d_model]
        x = tok_embed + pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x, padding_mask)[0]

        # Final layer norm + unembed
        x = self.ln_final(x)
        logits = x @ self.unembed.W_U + self.unembed.b_U

        return logits
