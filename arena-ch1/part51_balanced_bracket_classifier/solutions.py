# %%

"""
Solutions for ARENA 1.4.2 - Balanced Bracket Classifier
Adapted for NNsight + plain PyTorch (no TransformerLens)
"""

import json
import math
import sys
from pathlib import Path

import circuitsvis as cv
import einops
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Bool, Float, Int
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenameConfig, AttnProbFunction
from sklearn.linear_model import LinearRegression
from torch import Tensor
from tqdm import tqdm

# Add parent directory for plotly_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plotly_utils
from plotly_utils import bar, hist, to_numpy

device = t.device("cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu")
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"


class FakeConfig:
    """nnterp calls model.config with __contains__ checks."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)
    def __contains__(self, item): return hasattr(self, item)
    def __getitem__(self, item): return getattr(self, item)


class SourceSoftmaxAttnProbFunction(AttnProbFunction):
    """Universal attention prob accessor using NNsight's source tracing."""
    def get_attention_prob_source(self, attention_module, return_module_source=False):
        if return_module_source: return attention_module.source
        return attention_module.source.F_softmax_0


def wrap_bracket_model(model):
    """Wrap a BracketClassifier with StandardizedTransformer."""
    model.config = FakeConfig(num_attention_heads=2, hidden_size=56, vocab_size=5)
    rename_config = RenameConfig(
        layers_name="blocks", attn_name="attn", mlp_name="mlp",
        ln_final_name="ln_final", lm_head_name="unembed",
        attn_prob_source=SourceSoftmaxAttnProbFunction(),
    )
    wrapped = StandardizedTransformer(
        model, rename_config=rename_config,
        device_map=None, check_renaming=False,
    )
    wrapped.attention_probabilities.enabled = True
    return wrapped


# %%
# ==================== Model Definition ====================

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
        padding_mask: (batch, seq) bool, True for padding positions
        Returns: out (batch, seq, d_model)
        """
        q = t.einsum("bsd,ndh->bsnh", x, self.W_Q) + self.b_Q
        k = t.einsum("bsd,ndh->bsnh", x, self.W_K) + self.b_K
        v = t.einsum("bsd,ndh->bsnh", x, self.W_V) + self.b_V

        attn_scores = t.einsum("bqnh,bknh->bnqk", q, k) / math.sqrt(self.d_head)

        if padding_mask is not None:
            attn_scores = attn_scores.masked_fill(padding_mask[:, None, None, :], -1e5)

        attn_probs = F.softmax(attn_scores, dim=-1)

        z = t.einsum("bnqk,bknh->bqnh", attn_probs, v)
        result = t.einsum("bsnh,nhd->bsnd", z, self.W_O)
        out = result.sum(dim=2) + self.b_O

        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Parameter(t.empty(d_model, d_mlp))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.empty(d_mlp, d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.activation = nn.ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        pre = x @ self.W_in + self.b_in
        post = self.activation(pre)
        return post @ self.W_out + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_mlp: int):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, d_head)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: t.Tensor, padding_mask: t.Tensor | None = None) -> t.Tensor:
        ln1_out = self.ln1(x)
        attn_out = self.attn(ln1_out, padding_mask)
        resid_mid = x + attn_out

        ln2_out = self.ln2(resid_mid)
        mlp_out = self.mlp(ln2_out)
        return resid_mid + mlp_out


class BracketClassifier(nn.Module):
    """
    3-layer bidirectional transformer for bracket classification.

    Config: n_ctx=42, d_model=56, d_head=28, n_heads=2, d_mlp=56, n_layers=3,
            d_vocab=5, d_vocab_out=2, act_fn=relu, attention_dir=bidirectional
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

        self.embed = nn.Module()
        self.embed.W_E = nn.Parameter(t.empty(d_vocab, d_model))
        self.pos_embed = nn.Module()
        self.pos_embed.W_pos = nn.Parameter(t.empty(n_ctx, d_model))

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_head, d_mlp) for _ in range(n_layers)
        ])
        self.ln_final = LayerNorm(d_model)

        self.unembed = nn.Module()
        self.unembed.W_U = nn.Parameter(t.empty(d_model, d_vocab_out))
        self.unembed.b_U = nn.Parameter(t.zeros(d_vocab_out))

    def forward(self, tokens: t.Tensor | None = None, padding_mask: t.Tensor | None = None, input_ids: t.Tensor | None = None, **kwargs):
        """
        tokens: (batch, seq) integer token IDs
        padding_mask: (batch, seq) bool tensor, True where padding
        input_ids: alias for tokens (used by NNsight's StandardizedTransformer)
        Returns: logits (batch, seq, d_vocab_out)
        """
        if input_ids is not None:
            tokens = input_ids
        tok_embed = self.embed.W_E[tokens]
        pos_embed = self.pos_embed.W_pos[:tokens.shape[1]]
        x = tok_embed + pos_embed

        for block in self.blocks:
            x = block(x, padding_mask)

        x = self.ln_final(x)
        logits = x @ self.unembed.W_U + self.unembed.b_U
        return logits


# %%
# ==================== Load model and data ====================

from brackets_datasets import BracketsDataset, SimpleTokenizer

if MAIN:
    section_dir = Path(__file__).resolve().parent

    # Load model
    raw_model = BracketClassifier().to(device)
    state_dict = t.load(section_dir / "bracket_classifier_converted.pt", map_location=device, weights_only=True)
    raw_model.load_state_dict(state_dict)
    raw_model.eval()

    # Wrap with nnterp for activation access
    model = wrap_bracket_model(raw_model)

    # Tokenizer
    tokenizer = SimpleTokenizer("()")

    print(tokenizer.tokenize("()"))
    print(tokenizer.tokenize(["()", "()()"]))
    print(tokenizer.i_to_t)
    print(tokenizer.t_to_i)
    print(tokenizer.decode(t.tensor([[0, 3, 4, 2, 1, 1]])))

# %%
# ==================== Helper: run model with padding mask ====================

def run_model(model: StandardizedTransformer, toks: t.Tensor, pad_token: int = 1) -> t.Tensor:
    """Run the bracket classifier model, computing padding mask from tokens."""
    padding_mask = (toks == pad_token).to(toks.device)
    return model._model(toks, padding_mask=padding_mask)


def _padding_mask(toks: t.Tensor) -> t.Tensor:
    """Compute padding mask for bracket tokens (PAD_TOKEN=1)."""
    return (toks == 1).to(toks.device)


# %%
# ==================== Dataset ====================

if MAIN:
    N_SAMPLES = 5000
    with open(section_dir / "brackets_data.json") as f:
        data_tuples = json.load(f)
        print(f"loaded {len(data_tuples)} examples, using {N_SAMPLES}")
        data_tuples = data_tuples[:N_SAMPLES]

    data = BracketsDataset(data_tuples).to(device)
    data_mini = BracketsDataset(data_tuples[:100]).to(device)

# %%

if MAIN:
    hist(
        [len(x) for x, _ in data_tuples],
        nbins=data.seq_length,
        title="Sequence lengths of brackets in dataset",
        labels={"x": "Seq len"},
    )

# %%

if MAIN:
    examples = [
        "()()",
        "(())",
        "))((",
        "()",
        "((()()()()))",
        "(()()()(()(())()",
        "()(()(((())())()))",
    ]
    labels = [True, True, False, True, True, False, True]
    toks = tokenizer.tokenize(examples).to(device)

    logits = run_model(model, toks)[:, 0]
    prob_balanced = logits.softmax(-1)[:, 1]

    print(
        "Model confidence:\n"
        + "\n".join(
            [
                f"{ex:18} : {prob:<8.4%} : label={int(label)}"
                for ex, prob, label in zip(examples, prob_balanced, labels)
            ]
        )
    )

# %%

def run_model_on_data(
    model: StandardizedTransformer, data: BracketsDataset, batch_size: int = 200
) -> Float[Tensor, "batch 2"]:
    """Return logits for each example (at position 0, the classification position)."""
    all_logits = []
    for i in tqdm(range(0, len(data.strs), batch_size)):
        toks = data.toks[i : i + batch_size]
        logits = run_model(model, toks)[:, 0]
        all_logits.append(logits)
    all_logits = t.cat(all_logits)
    assert all_logits.shape == (len(data), 2)
    return all_logits


if MAIN:
    test_set = data
    n_correct = (run_model_on_data(model, test_set).argmax(-1).bool() == test_set.isbal).sum()
    print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")

# %%
# ==================== Algorithmic Solutions ====================

def is_balanced_forloop(parens: str) -> bool:
    """
    Return True if the parens are balanced.
    Parens is just the ( and ) characters, no begin or end tokens.
    """
    cumsum = 0
    for paren in parens:
        cumsum += 1 if paren == "(" else -1
        if cumsum < 0:
            return False
    return cumsum == 0


if MAIN:
    for parens, expected in zip(examples, labels):
        actual = is_balanced_forloop(parens)
        assert expected == actual, f"{parens}: expected {expected} got {actual}"
    print("All tests for `is_balanced_forloop` passed!")

# %%

def is_balanced_vectorized(tokens: Float[Tensor, "seq_len"]) -> bool:
    """
    Return True if the parens are balanced.
    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    """
    table = t.tensor([0, 0, 0, 1, -1])
    change = table[tokens]
    altitude = t.cumsum(change, -1)
    no_total_elevation_failure = altitude[-1] == 0
    no_negative_failure = altitude.min() >= 0
    return (no_total_elevation_failure & no_negative_failure).item()


if MAIN:
    for tokens, expected in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_vectorized(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("All tests for `is_balanced_vectorized` passed!")

# %%
# ==================== Moving Backwards ====================

def get_post_final_ln_dir(model: StandardizedTransformer) -> Float[Tensor, "d_model"]:
    """
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced).
    Unbalanced = class 0, balanced = class 1.
    """
    W_U = model._model.unembed.W_U
    return W_U[:, 0] - W_U[:, 1]


if MAIN:
    post_final_ln_dir = get_post_final_ln_dir(model)
    print(f"post_final_ln_dir shape: {post_final_ln_dir.shape}")

# %%
# ==================== LayerNorm fitting ====================

def get_ln_fit(
    model: StandardizedTransformer,
    data: BracketsDataset,
    layernorm: str,
    seq_pos: int | None = None,
) -> tuple[LinearRegression, float]:
    """
    Fits a linear regression from pre-layernorm to post-layernorm activations.

    layernorm: one of "ln_final", "blocks.{i}.ln1", "blocks.{i}.ln2"
    seq_pos: if None, aggregate over all positions; otherwise, use only that position.

    Returns: (fitted LinearRegression, r^2 score)
    """
    padding_mask = _padding_mask(data.toks)

    if layernorm == "ln_final":
        n_layers = model._model.n_layers
        with model.trace(data.toks, padding_mask=padding_mask):
            inputs_t = model.blocks[n_layers - 1].output.save()
            outputs_t = model.ln_final.output.save()
    else:
        parts = layernorm.split(".")
        layer = int(parts[1])
        ln_type = parts[2]

        if ln_type == "ln1":
            if layer == 0:
                inputs_t = (
                    model._model.embed.W_E[data.toks]
                    + model._model.pos_embed.W_pos[:data.toks.shape[1]].unsqueeze(0).expand(data.toks.shape[0], -1, -1)
                )
                with model.trace(data.toks, padding_mask=padding_mask):
                    outputs_t = model.blocks[0].ln1.output.save()
            else:
                with model.trace(data.toks, padding_mask=padding_mask):
                    inputs_t = model.blocks[layer - 1].output.save()
                    outputs_t = model.blocks[layer].ln1.output.save()
        else:  # ln2
            if layer == 0:
                embed_input = (
                    model._model.embed.W_E[data.toks]
                    + model._model.pos_embed.W_pos[:data.toks.shape[1]].unsqueeze(0).expand(data.toks.shape[0], -1, -1)
                )
                with model.trace(data.toks, padding_mask=padding_mask):
                    attn_out = model.blocks[0].attn.output.save()
                    outputs_t = model.blocks[0].ln2.output.save()
                inputs_t = embed_input + attn_out
            else:
                with model.trace(data.toks, padding_mask=padding_mask):
                    block_input = model.blocks[layer - 1].output.save()
                    attn_out = model.blocks[layer].attn.output.save()
                    outputs_t = model.blocks[layer].ln2.output.save()
                inputs_t = block_input + attn_out

    inputs = to_numpy(inputs_t)
    outputs = to_numpy(outputs_t)

    if seq_pos is None:
        inputs = einops.rearrange(inputs, "batch seq d_model -> (batch seq) d_model")
        outputs = einops.rearrange(outputs, "batch seq d_model -> (batch seq) d_model")
    else:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]

    final_ln_fit = LinearRegression().fit(inputs, outputs)
    r2 = final_ln_fit.score(inputs, outputs)

    return (final_ln_fit, r2)


if MAIN:
    _, r2 = get_ln_fit(model, data, layernorm="ln_final", seq_pos=0)
    print(f"r^2 for LN_final, at sequence position 0: {r2:.4f}")
    _, r2 = get_ln_fit(model, data, layernorm="blocks.1.ln1", seq_pos=None)
    print(f"r^2 for LN1, layer 1, over all sequence positions: {r2:.4f}")

# %%

def get_pre_final_ln_dir(
    model: StandardizedTransformer, data: BracketsDataset
) -> Float[Tensor, "d_model"]:
    """
    Returns the direction in residual stream (pre ln_final, at sequence position 0)
    which most points in the direction of making an unbalanced classification.
    """
    post_final_ln_dir = get_post_final_ln_dir(model)

    final_ln_fit = get_ln_fit(model, data, layernorm="ln_final", seq_pos=0)[0]
    final_ln_coefs = t.from_numpy(final_ln_fit.coef_).to(device)

    return final_ln_coefs.T @ post_final_ln_dir


if MAIN:
    pre_final_ln_dir = get_pre_final_ln_dir(model, data)
    print(f"pre_final_ln_dir shape: {pre_final_ln_dir.shape}")

# %%
# ==================== Output by components ====================

def get_out_by_components(
    model: StandardizedTransformer, data: BracketsDataset
) -> Float[Tensor, "component batch seq_pos emb"]:
    """
    Returns tensor of shape [10, dataset_size, seq_pos, emb] representing the output
    of each model component.

    Components in order:
        [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]

    Per-head attention results are computed manually from attention patterns and weights,
    since nnsight traces module-level I/O (not per-head intermediates).
    """
    n_layers = model._model.n_layers

    # Embeddings (no trace needed — just weight lookups)
    tok_embed = model._model.embed.W_E[data.toks]
    pos_embed = model._model.pos_embed.W_pos[:data.toks.shape[1]].unsqueeze(0).expand(data.toks.shape[0], -1, -1)

    # Save ln1 outputs (for v computation), attention patterns, and MLP outputs
    ln1_outputs = [None] * n_layers
    patterns = [None] * n_layers
    mlp_outputs = [None] * n_layers

    padding_mask = _padding_mask(data.toks)
    with model.trace(data.toks, padding_mask=padding_mask):
        for i in range(n_layers):
            ln1_outputs[i] = model.blocks[i].ln1.output.save()
            patterns[i] = model.attention_probabilities[i].save()
            mlp_outputs[i] = model.blocks[i].mlp.output.save()

    # Build output: embeddings first
    out = (tok_embed + pos_embed).unsqueeze(0)

    for layer in range(n_layers):
        # Compute per-head attention results from patterns and weights
        attn = model._model.blocks[layer].attn
        v = t.einsum("bsd,ndh->bsnh", ln1_outputs[layer], attn.W_V) + attn.b_V
        z = t.einsum("bnqk,bknh->bqnh", patterns[layer], v)
        result = t.einsum("bsnh,nhd->bsnd", z, attn.W_O)  # [batch, seq, n_heads, d_model]

        out = t.cat([
            out,
            einops.rearrange(result, "batch seq heads emb -> heads batch seq emb"),
            mlp_outputs[layer].unsqueeze(0),
        ])

    return out


if MAIN:
    print(f"out_by_components shape: {get_out_by_components(model, data_mini).shape}")

# %%

if MAIN:
    # Verify that the sum of components equals the input to the final layernorm
    # Need to add b_O biases (summed across layers)
    b_O_sum = sum(model._model.blocks[i].attn.b_O for i in range(model._model.n_layers))
    out_by_components = get_out_by_components(model, data)
    summed_terms = out_by_components.sum(dim=0) + b_O_sum

    padding_mask = _padding_mask(data.toks)
    with model.trace(data.toks, padding_mask=padding_mask):
        final_ln_input = model.blocks[model._model.n_layers - 1].output.save()

    t.testing.assert_close(summed_terms, final_ln_input)
    print("Tests passed: sum of components equals final LN input!")

# %%

if MAIN:
    out_by_components_seq0 = out_by_components[:, :, 0, :]  # [component=10, batch, d_model]
    pre_final_ln_dir = get_pre_final_ln_dir(model, data)  # [d_model]

    out_by_component_in_unbalanced_dir = einops.einsum(
        out_by_components_seq0,
        pre_final_ln_dir,
        "comp batch d_model, d_model -> comp batch",
    )
    out_by_component_in_unbalanced_dir -= (
        out_by_component_in_unbalanced_dir[:, data.isbal].mean(dim=1).unsqueeze(1)
    )

    plotly_utils.hists_per_comp(out_by_component_in_unbalanced_dir, data, xaxis_range=[-10, 20])

# %%
# ==================== Failure type classification ====================

def is_balanced_vectorized_return_both(
    toks: Int[Tensor, "batch seq"],
) -> tuple[Bool[Tensor, "batch"], Bool[Tensor, "batch"]]:
    table = t.tensor([0, 0, 0, 1, -1]).to(device)
    change = table[toks.to(device)].flip(-1)
    altitude = t.cumsum(change, -1)
    total_elevation_failure = altitude[:, -1] != 0
    negative_failure = altitude.max(-1).values > 0
    return total_elevation_failure, negative_failure


if MAIN:
    total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(data.toks)

    h20_in_unbalanced_dir = out_by_component_in_unbalanced_dir[7]
    h21_in_unbalanced_dir = out_by_component_in_unbalanced_dir[8]

# %%

if MAIN:
    failure_types_dict = {
        "both failures": negative_failure & total_elevation_failure,
        "just neg failure": negative_failure & ~total_elevation_failure,
        "just total elevation failure": ~negative_failure & total_elevation_failure,
        "balanced": ~negative_failure & ~total_elevation_failure,
    }

    plotly_utils.plot_failure_types_scatter(
        h20_in_unbalanced_dir, h21_in_unbalanced_dir, failure_types_dict, data
    )

# %%

if MAIN:
    plotly_utils.plot_contribution_vs_open_proportion(
        h20_in_unbalanced_dir,
        "Head 2.0 contribution vs proportion of open brackets '('",
        failure_types_dict,
        data,
    )

# %%

if MAIN:
    plotly_utils.plot_contribution_vs_open_proportion(
        h21_in_unbalanced_dir,
        "Head 2.1 contribution vs proportion of open brackets '('",
        failure_types_dict,
        data,
    )

# %%
# ==================== Attention pattern analysis ====================

def get_attn_probs(
    model: StandardizedTransformer, data: BracketsDataset, layer: int, head: int
) -> Tensor:
    """Returns attention probabilities for a specific head. Shape: (N_SAMPLES, seq_q, seq_k)."""
    padding_mask = _padding_mask(data.toks)
    with model.trace(data.toks, padding_mask=padding_mask):
        pattern = model.attention_probabilities[layer].save()
    return pattern[:, head, :, :]


if MAIN:
    attn_probs_20 = get_attn_probs(model, data, 2, 0)
    attn_probs_20_open_query0 = attn_probs_20[data.starts_open].mean(0)[0]

    bar(
        attn_probs_20_open_query0,
        title="Avg Attention Probabilities for query 0, first token '(', head 2.0",
        width=700,
        template="simple_white",
        labels={"x": "Sequence position", "y": "Attn prob"},
    )

# %%
# ==================== OV and pre-head 2.0 analysis ====================

def get_WOV(model: StandardizedTransformer, layer: int, head: int) -> Float[Tensor, "d_model d_model"]:
    """Returns the W_OV matrix for a particular layer and head."""
    W_V = model._model.blocks[layer].attn.W_V[head]  # [d_model, d_head]
    W_O = model._model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    return W_V @ W_O


def get_pre_20_dir(model: StandardizedTransformer, data: BracketsDataset) -> Float[Tensor, "d_model"]:
    """
    Returns the direction propagated back through the OV matrix of head 2.0
    and then through the layernorm before layer 2 attention heads.
    """
    W_OV = get_WOV(model, 2, 0)

    layer2_ln_fit, r2 = get_ln_fit(model, data, layernorm="blocks.2.ln1", seq_pos=1)
    layer2_ln_coefs = t.from_numpy(layer2_ln_fit.coef_).to(device)

    pre_final_ln_dir = get_pre_final_ln_dir(model, data)

    return layer2_ln_coefs.T @ W_OV @ pre_final_ln_dir


if MAIN:
    pre_20_dir = get_pre_20_dir(model, data)
    print(f"pre_20_dir shape: {pre_20_dir.shape}")

# %%

if MAIN:
    pre_layer2_outputs_seqpos1 = out_by_components[:-3, :, 1, :]
    out_by_component_in_pre_20_unbalanced_dir = einops.einsum(
        pre_layer2_outputs_seqpos1,
        get_pre_20_dir(model, data),
        "comp batch emb, emb -> comp batch",
    )
    out_by_component_in_pre_20_unbalanced_dir -= out_by_component_in_pre_20_unbalanced_dir[
        :, data.isbal
    ].mean(-1, True)

    plotly_utils.hists_per_comp(
        out_by_component_in_pre_20_unbalanced_dir, data, xaxis_range=(-5, 12)
    )

# %%

if MAIN:
    plotly_utils.mlp_attribution_scatter(
        out_by_component_in_pre_20_unbalanced_dir, data, failure_types_dict
    )

# %%
# ==================== Neuron-level analysis ====================

def get_out_by_neuron(
    model: StandardizedTransformer, data: BracketsDataset, layer: int, seq: int | None = None
) -> Float[Tensor, "batch *seq neuron d_model"]:
    """
    Computes the output vector written by each neuron in the specified MLP layer.

    If seq is not None, only look at that sequence position.
    """
    W_out = model._model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    padding_mask = _padding_mask(data.toks)
    with model.trace(data.toks, padding_mask=padding_mask):
        f_x_W_in = model.blocks[layer].mlp.activation.output.save()  # post-ReLU [batch, seq, d_mlp]

    if seq is not None:
        f_x_W_in = f_x_W_in[:, seq, :]  # [batch, d_mlp]

    out = einops.einsum(
        f_x_W_in,
        W_out,
        "... neuron, neuron d_model -> ... neuron d_model",
    )
    return out


def get_out_by_neuron_in_20_dir(
    model: StandardizedTransformer, data: BracketsDataset, layer: int
) -> Float[Tensor, "batch neurons"]:
    """
    Projects each neuron's output vector onto the pre-head-2.0 unbalanced direction.
    """
    out_by_neuron_seqpos1 = get_out_by_neuron(model, data, layer, seq=1)

    return einops.einsum(
        out_by_neuron_seqpos1,
        get_pre_20_dir(model, data),
        "batch neuron d_model, d_model -> batch neuron",
    )


# %%

def get_out_by_neuron_in_20_dir_less_memory(
    model: StandardizedTransformer, data: BracketsDataset, layer: int
) -> Float[Tensor, "batch neurons"]:
    """
    Same as get_out_by_neuron_in_20_dir but more memory efficient.
    """
    W_out = model._model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    padding_mask = _padding_mask(data.toks)
    with model.trace(data.toks, padding_mask=padding_mask):
        f_x_W_in = model.blocks[layer].mlp.activation.output.save()  # post-ReLU

    f_x_W_in = f_x_W_in[:, 1, :]  # [batch, d_mlp]

    pre_20_dir = get_pre_20_dir(model, data)  # [d_model]

    W_out_in_20_dir = W_out @ pre_20_dir  # [d_mlp]
    out_by_neuron_in_20_dir = f_x_W_in * W_out_in_20_dir  # [batch, d_mlp]

    return out_by_neuron_in_20_dir


# %%

if MAIN:
    for layer in range(2):
        neurons_in_unbalanced_dir = get_out_by_neuron_in_20_dir_less_memory(model, data, layer)[
            to_numpy(data.starts_open), :
        ]
        plotly_utils.plot_neurons(neurons_in_unbalanced_dir, model, data, failure_types_dict, layer)

# %%
# ==================== Q and K extraction ====================

def get_q_and_k_for_given_input(
    model: StandardizedTransformer,
    tokenizer: SimpleTokenizer,
    parens: str,
    layer: int,
) -> tuple[Float[Tensor, "seq n_heads d_head"], Float[Tensor, "seq n_heads d_head"]]:
    """
    Returns the queries and keys (at the given layer) for the given parens string.
    Computed from the ln1 output and attention weight matrices.
    """
    toks = tokenizer.tokenize(parens).to(device)
    padding_mask = _padding_mask(toks)

    with model.trace(toks, padding_mask=padding_mask):
        ln1_out = model.blocks[layer].ln1.output.save()

    attn = model._model.blocks[layer].attn
    q = t.einsum("bsd,ndh->bsnh", ln1_out, attn.W_Q) + attn.b_Q
    k = t.einsum("bsd,ndh->bsnh", ln1_out, attn.W_K) + attn.b_K

    return q[0], k[0]  # Remove batch dim


# %%
# ==================== Activation patching with NNsight ====================

if MAIN:
    layer = 0
    all_left_parens = "(" * 40
    all_right_parens = ")" * 40

    q0_all_left, k0_all_left = get_q_and_k_for_given_input(model, tokenizer, all_left_parens, layer)
    q0_all_right, k0_all_right = get_q_and_k_for_given_input(model, tokenizer, all_right_parens, layer)
    k0_avg = (k0_all_left + k0_all_right) / 2

    # Compute attention pattern with all-left queries and average keys
    attn = model._model.blocks[layer].attn
    attn_scores = t.einsum("sqnh,sknh->snqk", q0_all_left.unsqueeze(0), k0_avg.unsqueeze(0)) / math.sqrt(attn.d_head)
    avg_head_attn_pattern = F.softmax(attn_scores[0], dim=-1)  # [n_heads, seq_q, seq_k]

    labels = ["[start]", *[f"{i+1}" for i in range(40)], "[end]"]
    display(
        cv.attention.attention_heads(
            tokens=labels,
            attention=avg_head_attn_pattern,
            attention_head_names=["0.0", "0.1"],
            max_value=avg_head_attn_pattern.max().item(),
            mask_upper_tri=False,
        )
    )

# %%

if MAIN:
    # For displaying attention pattern at query position 1 with query patched to all-left
    data_len_40 = BracketsDataset.with_length(data_tuples, 40).to(device)
    balanced_toks_40 = data_len_40.toks[data_len_40.isbal]
    padding_mask_bal = (balanced_toks_40 == tokenizer.PAD_TOKEN).to(device)

    # Get keys from balanced data via trace (save ln1 output, compute k from weights)
    with model.trace(balanced_toks_40, padding_mask=padding_mask_bal):
        ln1_out_bal = model.blocks[0].ln1.output.save()

    attn = model._model.blocks[0].attn
    k_bal = t.einsum("bsd,ndh->bsnh", ln1_out_bal, attn.W_K) + attn.b_K

    # Use the all-left query (broadcast over batch)
    q_left = q0_all_left.unsqueeze(0).expand(k_bal.shape[0], -1, -1, -1)  # [batch, seq, n_heads, d_head]
    attn_scores_patched = t.einsum("bqnh,bknh->bnqk", q_left, k_bal) / math.sqrt(attn.d_head)
    # Apply padding mask
    attn_scores_patched = attn_scores_patched.masked_fill(padding_mask_bal[:, None, None, :], -1e5)
    attn_probs_patched = F.softmax(attn_scores_patched, dim=-1)

    # Plot average attention at position 1, head 0
    avg_attn_at_pos1 = attn_probs_patched[:, 0, 1].mean(0)  # [seq_k]

    bar(
        to_numpy(avg_attn_at_pos1),
        title="Average attn probabilities on data at posn 1, with query token = '('",
        labels={"index": "Sequence position of key", "value": "Average attn over dataset"},
        height=500,
        width=800,
        yaxis_range=[0, 0.1],
        template="simple_white",
    )

# %%
# ==================== OV circuit analysis ====================

def embedding(
    model: StandardizedTransformer, tokenizer: SimpleTokenizer, char: str
) -> Float[Tensor, "d_model"]:
    assert char in ("(", ")")
    idx = tokenizer.t_to_i[char]
    return model._model.embed.W_E[idx]


if MAIN:
    W_OV = model._model.blocks[0].attn.W_V[0] @ model._model.blocks[0].attn.W_O[0]

    layer0_ln_fit = get_ln_fit(model, data, layernorm="blocks.0.ln1", seq_pos=None)[0]
    layer0_ln_coefs = t.from_numpy(layer0_ln_fit.coef_).to(device)

    v_L = embedding(model, tokenizer, "(") @ layer0_ln_coefs.T @ W_OV
    v_R = embedding(model, tokenizer, ")") @ layer0_ln_coefs.T @ W_OV

    print(f"Cosine similarity: {t.cosine_similarity(v_L, v_R, dim=0).item():.4f}")

# %%
# ==================== Cosine similarity analysis ====================

def cos_sim_with_MLP_weights(
    model: StandardizedTransformer, v: Float[Tensor, "d_model"], layer: int
) -> Float[Tensor, "d_mlp"]:
    """
    Returns cosine similarity between v and each column of W_in.
    """
    v_unit = v / v.norm()
    W_in = model._model.blocks[layer].mlp.W_in
    W_in_unit = W_in / W_in.norm(dim=0)
    return einops.einsum(v_unit, W_in_unit, "d_model, d_model d_mlp -> d_mlp")


def avg_squared_cos_sim(v: Float[Tensor, "d_model"], n_samples: int = 1000) -> float:
    """Returns average squared cosine similarity between v and random vectors."""
    v2 = t.randn(n_samples, v.shape[0]).to(device)
    v2 /= v2.norm(dim=1, keepdim=True)
    v1 = v / v.norm()
    return (v1 * v2).pow(2).sum(1).mean().item()


if MAIN:
    print("Avg squared cosine similarity of v_R with ...\n")

    cos_sim_mlp0 = cos_sim_with_MLP_weights(model, v_R, 0)
    print(f"...MLP input directions in layer 0: {cos_sim_mlp0.pow(2).mean():.4f}")

    cos_sim_mlp1 = cos_sim_with_MLP_weights(model, v_R, 1)
    print(f"...MLP input directions in layer 1: {cos_sim_mlp1.pow(2).mean():.4f}")

    cos_sim_rand = avg_squared_cos_sim(v_R)
    print(f"...random vectors of len = d_model: {cos_sim_rand:.4f}")

# %%
# ==================== Adversarial examples ====================

if MAIN:
    adversarial_examples = ["()", "(())", "))"]

    def tallest_balanced_bracket(length: int) -> str:
        return "".join(["(" for _ in range(length)] + [")" for _ in range(length)])

    i_max = 30
    adversarial_examples.append(
        tallest_balanced_bracket(i_max // 2)
        + ")("
        + tallest_balanced_bracket((40 - i_max) // 2 - 1)
    )

    m = max(len(ex) for ex in adversarial_examples)
    toks = tokenizer.tokenize(adversarial_examples).to(device)
    probs = run_model(model, toks)[:, 0].softmax(-1)[:, 1]
    print(
        "\n".join(
            [
                f"{ex:{m}} -> {p:.4%} balanced confidence"
                for (ex, p) in zip(adversarial_examples, probs)
            ]
        )
    )
