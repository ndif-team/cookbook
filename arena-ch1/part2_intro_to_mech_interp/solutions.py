# %%

import sys
from pathlib import Path

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenameConfig, AttnProbFunction
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

# Add parent dir for plotly_utils
arena_ch1_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arena_ch1_dir))

from convert_2L_attn_only import load_model
from factored_matrix import FactoredMatrix
from plotly_utils import (
    hist,
    imshow,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
    to_numpy,
)

import part2_intro_to_mech_interp.tests as tests

t.set_grad_enabled(False)
device = t.device("cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu")

MAIN = __name__ == "__main__"


class FakeConfig:
    """nnterp calls model.config with __contains__ checks."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)
    def __contains__(self, item): return hasattr(self, item)
    def __getitem__(self, item): return getattr(self, item)


class SourceSoftmaxAttnProbFunction(AttnProbFunction):
    """Accesses F.softmax() output inside any custom Attention.forward()."""
    def get_attention_prob_source(self, attention_module, return_module_source=False):
        if return_module_source: return attention_module.source
        return attention_module.source.F_softmax_0

# %%

# ==================================================
# SECTION 1: NNsight Introduction & GPT-2 Exploration
# ==================================================

if MAIN:
    gpt2_small = StandardizedTransformer(
        "openai-community/gpt2", device_map="auto", dtype=t.float32,
        enable_attention_probs=True,
    )

# %%

if MAIN:
    model_description_text = """## Loading Models

    HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

    For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""

    tokens = gpt2_small.tokenizer(model_description_text, return_tensors="pt").input_ids.to(device)
    with gpt2_small.trace(tokens):
        logits = gpt2_small.logits.save()
    logits = logits.squeeze(0)  # [seq, vocab]

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits[:-1], tokens.squeeze()[1:])
    print("Model loss:", loss.item())

# %%

if MAIN:
    print(gpt2_small.tokenizer.tokenize("gpt2"))
    print(gpt2_small.tokenizer("gpt2", return_tensors="pt").input_ids)
    print(gpt2_small.tokenizer.decode([50256, 70, 457, 17]))

# %%

if MAIN:
    tokens = gpt2_small.tokenizer(model_description_text, return_tensors="pt").input_ids.to(device)
    with gpt2_small.trace(tokens):
        logits = gpt2_small.logits.save()
    logits = logits.squeeze(0)  # [seq, vocab]
    prediction = logits.argmax(dim=-1)[:-1]

    true_tokens = tokens.squeeze()[1:]
    is_correct = prediction == true_tokens

    print(f"Model accuracy: {is_correct.sum()}/{len(true_tokens)}")
    print(f"Correct tokens: {gpt2_small.tokenizer.batch_decode(prediction[is_correct].unsqueeze(-1))}")

# %%

# ==================================================
# SECTION 2: Caching Activations with NNsight
# ==================================================

if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
    gpt2_tokens = gpt2_small.tokenizer(gpt2_text, return_tensors="pt").input_ids.to(device)

    # Cache: run a trace and save everything we need
    # NNsight requires accessing modules in forward-execution order
    with gpt2_small.trace(gpt2_tokens):
        # Save attention internals for layer 0 (c_attn runs early in forward pass)
        gpt2_c_attn_out_0 = gpt2_small.layers[0].self_attn.c_attn.output.save()
        gpt2_logits = gpt2_small.logits.save()

    print(type(gpt2_logits), gpt2_logits.shape)

# %%

if MAIN:
    # Compute attention patterns from saved c_attn output
    d_model = 768
    n_heads = 12
    d_head = d_model // n_heads

    # c_attn output is [batch, seq, 3*d_model] = concatenated Q, K, V
    qkv = gpt2_c_attn_out_0.squeeze(0)  # [seq, 3*d_model]
    q, k, v = qkv.split(d_model, dim=-1)  # each [seq, d_model]

    # Reshape to [seq, n_heads, d_head]
    q = q.view(-1, n_heads, d_head)
    k = k.view(-1, n_heads, d_head)

    # Compute attention scores
    seq_len = q.shape[0]
    attn_scores = einops.einsum(q, k, "seqQ n h, seqK n h -> n seqQ seqK")
    mask = t.triu(t.ones((seq_len, seq_len), dtype=t.bool, device=device), diagonal=1)
    attn_scores.masked_fill_(mask, -1e9)
    layer0_pattern = (attn_scores / d_head**0.5).softmax(-1)

    print("Attention pattern shape:", layer0_pattern.shape)
    print("Tests passed!")

# %%

if MAIN:
    gpt2_str_tokens = [gpt2_small.tokenizer.decode(t_) for t_ in gpt2_tokens.squeeze()]

    print("Layer 0 Head Attention Patterns:")
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_str_tokens,
            attention=layer0_pattern,
        )
    )

# %%

# ==================================================
# SECTION 3: 2L Attention-Only Model
# ==================================================

if MAIN:
    # Load the converted 2L attention-only model (downloads from HF Hub if not cached locally)
    local_2l_path = arena_ch1_dir / "_conversion" / "attn_only_2L_converted.pt"
    attn_2l = load_model(
        state_dict_path=str(local_2l_path) if local_2l_path.exists() else None,
        device=str(device),
    )
    attn_2l = attn_2l.to(device)
    tokenizer_2l = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Wrap with nnterp StandardizedTransformer for consistent accessors
    attn_2l.config = FakeConfig(num_attention_heads=12, hidden_size=768, vocab_size=50278)
    rename_config = RenameConfig(
        layers_name="blocks", attn_name="attn", mlp_name=None,
        ln_final_name=None, lm_head_name="unembed",
        attn_prob_source=SourceSoftmaxAttnProbFunction(),
        ignore_mlp=True,
    )
    model = StandardizedTransformer(
        attn_2l, rename_config=rename_config, tokenizer=tokenizer_2l,
        device_map=None, check_renaming=False,
    )
    model.attention_probabilities.enabled = True

# %%


def get_attention_patterns_2l(
    model: StandardizedTransformer, tokens_or_text
) -> list[Tensor]:
    """
    Get attention patterns for all layers/heads using nnterp's attention_probabilities accessor.

    Args:
        model: The nnterp-wrapped StandardizedTransformer
        tokens_or_text: Either a string or a tensor of token ids [batch, seq]

    Returns a list of tensors, one per layer, each of shape [n_heads, seq, seq]
    (batch dimension squeezed if batch_size=1) or [batch, n_heads, seq, seq].
    """
    if isinstance(tokens_or_text, str):
        tokens_or_text = model.tokenizer(tokens_or_text, return_tensors="pt").input_ids.to(device)

    n_layers = model._model.n_layers
    patterns = [None] * n_layers
    with model.trace(tokens_or_text):
        for i in range(n_layers):
            patterns[i] = model.attention_probabilities[i].save()

    # Squeeze batch dim if batch_size == 1
    return [p.squeeze(0) if p.shape[0] == 1 else p for p in patterns]


if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

    patterns = get_attention_patterns_2l(model, text)
    str_tokens = [tokenizer_2l.decode(t_) for t_ in tokenizer_2l(text).input_ids]

    for layer in range(model._model.n_layers):
        display(
            cv.attention.attention_patterns(
                tokens=str_tokens, attention=patterns[layer]
            )
        )

# %%


def current_attn_detector(patterns: list[Tensor]) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads.
    """
    attn_heads = []
    for layer in range(len(patterns)):
        for head in range(patterns[layer].shape[0]):
            attention_pattern = patterns[layer][head]
            score = attention_pattern.diagonal().mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads


def prev_attn_detector(patterns: list[Tensor]) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads.
    """
    attn_heads = []
    for layer in range(len(patterns)):
        for head in range(patterns[layer].shape[0]):
            attention_pattern = patterns[layer][head]
            score = attention_pattern.diagonal(-1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads


def first_attn_detector(patterns: list[Tensor]) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads.
    """
    attn_heads = []
    for layer in range(len(patterns)):
        for head in range(patterns[layer].shape[0]):
            attention_pattern = patterns[layer][head]
            score = attention_pattern[:, 0].mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads


if MAIN:
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(patterns)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(patterns)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(patterns)))

# %%

# ==================================================
# SECTION 4: Induction Heads
# ==================================================


def generate_repeated_tokens(
    model, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens.

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)
    bos_id = model.tokenizer.bos_token_id
    if bos_id is None:
        bos_id = model.tokenizer.eos_token_id
    d_vocab = model._model.d_vocab
    prefix = (t.ones(batch_size, 1) * bos_id).long()
    rep_tokens_half = t.randint(0, d_vocab, (batch_size, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens


def run_and_cache_model_repeated_tokens(
    model: StandardizedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it,
    returning (tokens, logits, patterns) in a single trace.

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        patterns: list of attention pattern tensors per layer
    """
    rep_tokens = generate_repeated_tokens(model, seq_len, batch_size)
    n_layers = model._model.n_layers
    saved_patterns = [None] * n_layers

    with model.trace(rep_tokens):
        for i in range(n_layers):
            saved_patterns[i] = model.attention_probabilities[i].save()
        rep_logits = model.lm_head.output.save()

    # Squeeze batch dim for patterns if batch=1
    patterns = [p.squeeze(0) if p.shape[0] == 1 else p for p in saved_patterns]
    return rep_tokens, rep_logits, patterns


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs


if MAIN:
    seq_len = 50
    batch_size = 1
    (rep_tokens, rep_logits, rep_patterns) = run_and_cache_model_repeated_tokens(
        model, seq_len, batch_size
    )
    rep_str = [tokenizer_2l.decode(t_) for t_ in rep_tokens.squeeze()]
    log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

    tests.test_get_log_probs(get_log_probs)

    print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

    plot_loss_difference(log_probs, rep_str, seq_len)

# %%

if MAIN:
    for layer in range(model._model.n_layers):
        display(
            cv.attention.attention_patterns(
                tokens=rep_str, attention=rep_patterns[layer]
            )
        )

# %%


def induction_attn_detector(patterns: list[Tensor]) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads.

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    attn_heads = []
    for layer in range(len(patterns)):
        for head in range(patterns[layer].shape[0]):
            attention_pattern = patterns[layer][head]
            seq_len = (attention_pattern.shape[-1] - 1) // 2
            score = attention_pattern.diagonal(-seq_len + 1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads


if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_patterns)))

# %%

# ==================================================
# SECTION 5: Induction Head Scoring (with in-trace interventions)
# ==================================================

if MAIN:
    seq_len = 50
    batch_size = 10
    rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

    # Compute induction scores using nnterp's attention_probabilities accessor
    n_layers = model._model.n_layers
    n_heads = model._model.n_heads
    induction_score_store = t.zeros((n_layers, n_heads), device=device)

    # Get attention patterns via nnterp in a single trace
    patterns_10 = get_attention_patterns_2l(model, rep_tokens_10)

    for layer_idx in range(n_layers):
        pattern = patterns_10[layer_idx]  # [batch, n_heads, seq, seq]
        induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
        induction_score = einops.reduce(
            induction_stripe, "batch head_index position -> head_index", "mean"
        )
        induction_score_store[layer_idx, :] = induction_score

    imshow(
        induction_score_store,
        labels={"x": "Head", "y": "Layer"},
        title="Induction Score by Head",
        text_auto=".2f",
        width=900,
        height=350,
    )

# %%

# GPT-2 small induction head scoring
if MAIN:
    seq_len = 50
    batch_size = 10
    # Generate repeated tokens for GPT-2 (different vocab size)
    t.manual_seed(0)
    gpt2_bos = gpt2_small.tokenizer.bos_token_id
    gpt2_prefix = (t.ones(batch_size, 1) * gpt2_bos).long()
    gpt2_half = t.randint(0, 50257, (batch_size, seq_len), dtype=t.int64)
    rep_tokens_gpt2 = t.cat([gpt2_prefix, gpt2_half, gpt2_half], dim=-1).to(device)

    gpt2_n_layers = 12
    gpt2_n_heads = 12
    induction_score_store_gpt2 = t.zeros((gpt2_n_layers, gpt2_n_heads), device=device)

    # Use nnterp's attention_probabilities accessor for GPT-2
    gpt2_patterns = [None] * gpt2_n_layers
    with gpt2_small.trace(rep_tokens_gpt2):
        for i in range(gpt2_n_layers):
            gpt2_patterns[i] = gpt2_small.attention_probabilities[i].save()

    for layer_idx in range(gpt2_n_layers):
        pattern = gpt2_patterns[layer_idx]  # [batch, n_heads, seq, seq]
        induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
        induction_score = einops.reduce(
            induction_stripe, "batch head_index position -> head_index", "mean"
        )
        induction_score_store_gpt2[layer_idx, :] = induction_score

    imshow(
        induction_score_store_gpt2,
        labels={"x": "Head", "y": "Layer"},
        title="Induction Score by Head (GPT-2 Small)",
        text_auto=".1f",
        width=700,
        height=500,
    )

# %%

# ==================================================
# SECTION 6: Logit Attribution (2L model)
# ==================================================


def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"],
) -> Float[Tensor, "seq-1 n_components"]:
    """
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dims)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dims)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]

    direct_attributions = einops.einsum(
        W_U_correct_tokens, embed[:-1], "emb seq, seq emb -> seq"
    )
    l1_attributions = einops.einsum(
        W_U_correct_tokens, l1_results[:-1], "emb seq, seq nhead emb -> seq nhead"
    )
    l2_attributions = einops.einsum(
        W_U_correct_tokens, l2_results[:-1], "emb seq, seq nhead emb -> seq nhead"
    )
    return t.concat(
        [direct_attributions.unsqueeze(-1), l1_attributions, l2_attributions], dim=-1
    )


def get_attn_head_results(
    model: StandardizedTransformer, tokens: Tensor
) -> tuple[Tensor, list[Tensor]]:
    """
    Run the 2L model and return per-head output contributions for each layer.
    Uses nnterp's attention_probabilities to get patterns, then decomposes per-head
    by computing V from weights and projecting through W_O.

    Returns:
        embed: [seq, d_model] — token + positional embeddings
        results: list of [seq, n_heads, d_model] tensors (one per layer)
    """
    raw = model._model
    n_layers = raw.n_layers

    saved_patterns = [None] * n_layers
    layer_outs = [None] * n_layers
    with model.trace(tokens):
        embed_out = model.embed.output.save()          # [batch, seq, d_model]
        pos_embed_out = model.pos_embed.output.save()   # [seq, d_model]
        for i in range(n_layers):
            saved_patterns[i] = model.attention_probabilities[i].save()
            layer_outs[i] = model.layers_output[i].save()

    embed = (embed_out.squeeze(0) + pos_embed_out)  # [seq, d_model]

    # Input to each layer's attention
    inputs = [
        embed_out + pos_embed_out,  # [batch, seq, d_model] for layer 0
        layer_outs[0],              # [batch, seq, d_model] for layer 1
    ]

    results = []
    for layer_idx in range(n_layers):
        attn = raw.blocks[layer_idx].attn
        inp = inputs[layer_idx]
        pattern = saved_patterns[layer_idx]  # [batch, n_heads, seq, seq]

        # V = input @ W_V + b_V
        v = t.einsum("bsd,hdi->bshi", inp, attn.W_V) + attn.b_V  # [batch, seq, n_heads, d_head]
        v = v.permute(0, 2, 1, 3)  # [batch, n_heads, seq, d_head]

        # z = pattern @ V per head
        z = pattern @ v  # [batch, n_heads, seq, d_head]
        z = z.permute(0, 2, 1, 3)  # [batch, seq, n_heads, d_head]

        # Per-head result: z_h @ W_O_h
        result = t.einsum("bsnh,nhm->bsnm", z, attn.W_O)  # [batch, seq, n_heads, d_model]
        results.append(result.squeeze(0))  # [seq, n_heads, d_model]

    return embed, results


if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    tokens = tokenizer_2l(text, return_tensors="pt").input_ids.to(device)

    with model.trace(tokens):
        logits_out = model.lm_head.output.save()

    embed, head_results = get_attn_head_results(model, tokens)
    l1_results = head_results[0]
    l2_results = head_results[1]

    # W_U for the 2L model: unembed.weight is [d_vocab, d_model], we need [d_model, d_vocab]
    W_U = model._model.unembed.weight.T  # [d_model, d_vocab]

    with t.inference_mode():
        logit_attr = logit_attribution(embed, l1_results, l2_results, W_U, tokens[0])
        correct_token_logits = logits_out[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
        print("Tests passed!")

    tests.test_logit_attribution(logit_attribution, model, tokens)

# %%

if MAIN:
    str_tokens = [tokenizer_2l.decode(t_) for t_ in tokens.squeeze()]
    plot_logit_attribution(model, logit_attr, tokens, title="Logit attribution (demo prompt)")

# %%

if MAIN:
    seq_len = 50
    rep_tokens_1 = generate_repeated_tokens(model, seq_len, batch_size=1)

    embed_rep, head_results_rep = get_attn_head_results(model, rep_tokens_1)
    l1_results_rep = head_results_rep[0]
    l2_results_rep = head_results_rep[1]

    logit_attr_rep = logit_attribution(
        embed_rep, l1_results_rep, l2_results_rep, W_U, rep_tokens_1.squeeze()
    )
    plot_logit_attribution(
        model,
        logit_attr_rep,
        rep_tokens_1.squeeze(),
        title="Logit attribution (random induction prompt)",
    )

# %%

# ==================================================
# SECTION 7: Ablation Studies
# ==================================================


def get_ablation_scores(
    model: StandardizedTransformer,
    tokens: Int[Tensor, "batch seq"],
    ablation_type: str = "zero",
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss
    from ablating the output of each head.

    For zero ablation: zeroes the attention pattern for the target head inside an nnterp trace.
    For mean ablation: computes a correction to replace head output with batch-mean output.
    """
    raw = model._model
    n_layers, n_heads = raw.n_layers, raw.n_heads
    ablation_scores = t.zeros((n_layers, n_heads), device=device)
    seq_len = (tokens.shape[1] - 1) // 2

    # Baseline loss
    with model.trace(tokens):
        logits_clean = model.lm_head.output.save()
    loss_no_ablation = -get_log_probs(logits_clean, tokens)[:, -(seq_len - 1) :].mean()

    # Pre-compute for mean ablation: patterns, layer inputs → per-head z corrections
    if ablation_type == "mean":
        all_patterns = [None] * n_layers
        layer_outs = [None] * n_layers
        with model.trace(tokens):
            embed_out = model.embed.output.save()
            pos_embed_out = model.pos_embed.output.save()
            for i in range(n_layers):
                all_patterns[i] = model.attention_probabilities[i].save()
                layer_outs[i] = model.layers_output[i].save()
        layer_inputs = [embed_out + pos_embed_out, layer_outs[0]]

    for layer in tqdm(range(n_layers)):
        # Pre-compute V and z for all heads in this layer (mean ablation only)
        if ablation_type == "mean":
            attn = raw.blocks[layer].attn
            v = t.einsum("bsd,hdi->bshi", layer_inputs[layer], attn.W_V) + attn.b_V
            v = v.permute(0, 2, 1, 3)  # [batch, n_heads, seq, d_head]
            z_all = all_patterns[layer] @ v  # [batch, n_heads, seq, d_head]

        for head in range(n_heads):
            with model.trace(tokens):
                if ablation_type == "zero":
                    model.attention_probabilities[layer][:, head] = 0.0
                else:
                    z_h = z_all[:, head]  # [batch, seq, d_head]
                    mean_z_h = z_h.mean(dim=0, keepdim=True).expand_as(z_h)
                    correction = (mean_z_h - z_h) @ attn.W_O[head]  # [batch, seq, d_model]
                    model.attentions_output[layer] = model.attentions_output[layer] + correction
                logits = model.lm_head.output.save()
            loss = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


if MAIN:
    rep_tokens_1 = generate_repeated_tokens(model, seq_len=50, batch_size=1)
    ablation_scores = get_ablation_scores(model, rep_tokens_1)

    tests.test_get_ablation_scores(ablation_scores, model, rep_tokens_1)

# %%

if MAIN:
    imshow(
        ablation_scores,
        labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
        title="Loss Difference After Ablating Heads",
        text_auto=".2f",
        width=900,
        height=350,
    )

# %%

if MAIN:
    rep_tokens_batch = generate_repeated_tokens(model, seq_len=50, batch_size=10)
    mean_ablation_scores = get_ablation_scores(
        model, rep_tokens_batch, ablation_type="mean"
    )

    imshow(
        mean_ablation_scores,
        labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
        title="Loss Difference After Ablating Heads (Mean)",
        text_auto=".2f",
        width=900,
        height=350,
    )

# %%

# ==================================================
# SECTION 8: OV and QK Circuit Analysis
# ==================================================

if MAIN:
    head_index = 4
    layer = 1

    W_O = model._model.blocks[layer].attn.W_O[head_index]  # [d_head, d_model]
    W_V = model._model.blocks[layer].attn.W_V[head_index]  # [d_model, d_head]
    W_E = model._model.embed.weight  # [d_vocab, d_model]
    W_U = model._model.unembed.weight.T  # [d_model, d_vocab]

    # OV circuit: FactoredMatrix(W_V, W_O) — avoids materializing d_vocab x d_vocab
    OV_circuit = FactoredMatrix(W_V, W_O)
    # Full circuit: W_E @ OV @ W_U → FactoredMatrix, never materializes d_vocab x d_vocab
    full_OV_circuit = W_E @ OV_circuit @ W_U

    tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)

# %%

if MAIN:
    indices = t.randint(0, model._model.d_vocab, (200,))
    full_OV_circuit_sample = full_OV_circuit[indices][:, indices].AB

    imshow(
        to_numpy(full_OV_circuit_sample),
        labels={"x": "Logits on output token", "y": "Input token"},
        title="Full OV circuit for copying head",
        width=700,
        height=600,
    )

# %%


def top_1_acc(full_OV_circuit: FactoredMatrix, batch_size: int = 1000) -> float:
    """
    Return the fraction of the time that the maximum value is on the circuit diagonal.
    """
    total = 0

    for indices in t.split(t.arange(full_OV_circuit.shape[0], device=device), batch_size):
        AB_slice = full_OV_circuit[indices].AB
        total += (t.argmax(AB_slice, dim=1) == indices).float().sum().item()

    return total / full_OV_circuit.shape[0]


if MAIN:
    print(
        f"Fraction of time that the best logit is on diagonal: {top_1_acc(full_OV_circuit):.4f}"
    )

# %%

if MAIN:
    # Combined OV circuit for heads 4 and 10
    W_V_4 = model._model.blocks[1].attn.W_V[4]  # [d_model, d_head]
    W_O_4 = model._model.blocks[1].attn.W_O[4]  # [d_head, d_model]
    W_V_10 = model._model.blocks[1].attn.W_V[10]
    W_O_10 = model._model.blocks[1].attn.W_O[10]

    W_V_both = t.cat([W_V_4, W_V_10], dim=-1)  # [d_model, 2*d_head]
    W_O_both = t.cat([W_O_4, W_O_10], dim=0)  # [2*d_head, d_model]

    W_OV_eff = W_E @ FactoredMatrix(W_V_both, W_O_both) @ W_U

    print(
        f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(W_OV_eff):.4f}"
    )

# %%

if MAIN:
    layer = 0
    head_index = 7

    W_pos = model._model.pos_embed.weight  # [n_ctx, d_model]
    W_Q = model._model.blocks[layer].attn.W_Q[head_index]  # [d_model, d_head]
    W_K = model._model.blocks[layer].attn.W_K[head_index]  # [d_model, d_head]
    W_QK = W_Q @ W_K.T  # [d_model, d_model]
    pos_by_pos_scores = W_pos @ W_QK @ W_pos.T  # [n_ctx, n_ctx]

    # Mask, scale and softmax
    mask = t.tril(t.ones_like(pos_by_pos_scores)).bool()
    pos_by_pos_pattern = t.where(
        mask, pos_by_pos_scores / model._model.d_head**0.5, -1.0e6
    ).softmax(-1)

    print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")
    imshow(
        to_numpy(pos_by_pos_pattern[:200, :200]),
        labels={"x": "Key", "y": "Query"},
        title="Attention patterns for prev-token QK circuit, first 200 indices",
        width=700,
        height=600,
    )

# %%

# ==================================================
# SECTION 9: Residual Stream Decomposition
# ==================================================


def decompose_qk_input(
    model: StandardizedTransformer, tokens: Tensor
) -> Float[Tensor, "n_heads+2 posn d_model"]:
    """
    Retrieves all the input tensors to the first attention layer, and concatenates them along
    the 0th dim.

    The [i, :, :]th element is y_i. The sum of these along dim 0 should be the input to layer 1's attention.
    Components: [embed, pos_embed, head_0_result, head_1_result, ..., head_11_result]
    """
    raw = model._model

    with model.trace(tokens):
        embed_out = model.embed.output.save()          # [batch, seq, d_model]
        pos_embed_out = model.pos_embed.output.save()   # [seq, d_model]
        pattern0 = model.attention_probabilities[0].save()  # [batch, n_heads, seq, seq]

    embed = embed_out.squeeze(0)  # [seq, d_model]
    inp = embed + pos_embed_out   # [seq, d_model]

    # Per-head results from layer 0: V then z @ W_O per head
    attn = raw.blocks[0].attn
    v = t.einsum("sd,hdi->shi", inp, attn.W_V) + attn.b_V  # [seq, n_heads, d_head]
    z = pattern0.squeeze(0) @ v.permute(1, 0, 2)  # [n_heads, seq, d_head]
    head_results = t.einsum("nsh,nhm->nsm", z, attn.W_O)  # [n_heads, seq, d_model]

    y0 = embed.unsqueeze(0)        # [1, seq, d_model]
    y1 = pos_embed_out.unsqueeze(0)  # [1, seq, d_model]

    return t.concat([y0, y1, head_results], dim=0)


def decompose_q(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: StandardizedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of query vectors for each decomposed QK input.
    """
    W_Q = model._model.blocks[1].attn.W_Q[ind_head_index]
    return einops.einsum(
        decomposed_qk_input, W_Q, "n seq d_model, d_model d_head -> n seq d_head"
    )


def decompose_k(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: StandardizedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of key vectors for each decomposed QK input.
    """
    W_K = model._model.blocks[1].attn.W_K[ind_head_index]
    return einops.einsum(
        decomposed_qk_input, W_K, "n seq d_model, d_model d_head -> n seq d_head"
    )


if MAIN:
    seq_len = 50
    batch_size = 1
    rep_tokens_1 = generate_repeated_tokens(model, seq_len, batch_size)

    ind_head_index = 4

    decomposed_qk_input = decompose_qk_input(model, rep_tokens_1)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index, model)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index, model)

    component_labels = ["Embed", "PosEmbed"] + [
        f"0.{h}" for h in range(model._model.n_heads)
    ]
    for decomposed_input, name in [
        (decomposed_q, "query"),
        (decomposed_k, "key"),
    ]:
        imshow(
            to_numpy(decomposed_input.pow(2).sum([-1])),
            labels={"x": "Position", "y": "Component"},
            title=f"Norms of components of {name}",
            y=component_labels,
            width=800,
            height=400,
        )

# %%


def decompose_attn_scores(
    decomposed_q: Float[Tensor, "q_comp q_pos d_head"],
    decomposed_k: Float[Tensor, "k_comp k_pos d_head"],
    model: StandardizedTransformer,
) -> Float[Tensor, "q_comp k_comp q_pos k_pos"]:
    """
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]
    """
    return (
        einops.einsum(
            decomposed_q,
            decomposed_k,
            "q_comp q_pos d_head, k_comp k_pos d_head -> q_comp k_comp q_pos k_pos",
        )
        / (model._model.d_head**0.5)
    )


if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k, model)

    tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k, model)

    q_label = "Embed"
    k_label = "0.7"
    decomposed_scores_from_pair = decomposed_scores[
        component_labels.index(q_label), component_labels.index(k_label)
    ]

    imshow(
        to_numpy(t.tril(decomposed_scores_from_pair)),
        title=f"Attention score contributions from query = {q_label}, key = {k_label}<br>(by query & key sequence positions)",
        width=700,
    )

    decomposed_stds = einops.reduce(
        decomposed_scores,
        "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp",
        t.std,
    )
    imshow(
        to_numpy(decomposed_stds),
        labels={"x": "Key Component", "y": "Query Component"},
        title="Std dev of attn score contributions across sequence positions<br>(by query & key comp)",
        x=component_labels,
        y=component_labels,
        width=700,
    )

# %%

# ==================================================
# SECTION 10: K-Composition Circuit
# ==================================================


def find_K_comp_full_circuit(
    model: StandardizedTransformer, prev_token_head_index: int, ind_head_index: int
) -> FactoredMatrix:
    """
    Returns a FactoredMatrix representing the full K-composition circuit.
    The circuit is Q @ K.T = [d_vocab, d_vocab], kept in factored form.
    """
    raw = model._model
    W_E = raw.embed.weight  # [d_vocab, d_model]
    W_Q = raw.blocks[1].attn.W_Q[ind_head_index]  # [d_model, d_head]
    W_K = raw.blocks[1].attn.W_K[ind_head_index]  # [d_model, d_head]
    W_O = raw.blocks[0].attn.W_O[prev_token_head_index]  # [d_head, d_model]
    W_V = raw.blocks[0].attn.W_V[prev_token_head_index]  # [d_model, d_head]

    Q = W_E @ W_Q  # [d_vocab, d_head]
    K = W_E @ W_V @ W_O @ W_K  # [d_vocab, d_head]
    return FactoredMatrix(Q, K.T)  # [d_vocab, d_vocab] in factored form


if MAIN:
    prev_token_head_index = 7
    ind_head_index = 4
    K_comp_circuit = find_K_comp_full_circuit(
        model, prev_token_head_index, ind_head_index
    )

    print(
        f"Token frac where max-activating key = same token: {top_1_acc(K_comp_circuit.T):.4f}"
    )

    tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

# %%

# ==================================================
# SECTION 11: Composition Scores
# ==================================================


def get_comp_score(
    W_A: Float[Tensor, "in_A out_A"], W_B: Float[Tensor, "out_A out_B"]
) -> float:
    """
    Return the composition score between W_A and W_B.
    """
    W_A_norm = W_A.pow(2).sum().sqrt()
    W_B_norm = W_B.pow(2).sum().sqrt()
    W_AB_norm = (W_A @ W_B).pow(2).sum().sqrt()

    return (W_AB_norm / (W_A_norm * W_B_norm)).item()


if MAIN:
    tests.test_get_comp_score(get_comp_score)

    # Get all QK and OV matrices for the 2L model
    # W_QK[layer, head] = W_Q @ W_K^T: [d_model, d_model]
    # W_OV[layer, head] = W_V @ W_O: [d_model, d_model]
    raw = model._model
    n_heads = raw.n_heads

    W_QK = t.zeros(2, n_heads, raw.d_model, raw.d_model, device=device)
    W_OV = t.zeros(2, n_heads, raw.d_model, raw.d_model, device=device)

    for layer_idx in range(2):
        for head_idx in range(n_heads):
            W_Q = raw.blocks[layer_idx].attn.W_Q[head_idx]
            W_K = raw.blocks[layer_idx].attn.W_K[head_idx]
            W_V = raw.blocks[layer_idx].attn.W_V[head_idx]
            W_O = raw.blocks[layer_idx].attn.W_O[head_idx]
            W_QK[layer_idx, head_idx] = W_Q @ W_K.T
            W_OV[layer_idx, head_idx] = W_V @ W_O

    composition_scores = {
        "Q": t.zeros(n_heads, n_heads).to(device),
        "K": t.zeros(n_heads, n_heads).to(device),
        "V": t.zeros(n_heads, n_heads).to(device),
    }

    for i in tqdm(range(n_heads)):
        for j in range(n_heads):
            composition_scores["Q"][i, j] = get_comp_score(W_OV[0, i], W_QK[1, j])
            composition_scores["K"][i, j] = get_comp_score(W_OV[0, i], W_QK[1, j].T)
            composition_scores["V"][i, j] = get_comp_score(W_OV[0, i], W_OV[1, j])

    for comp_type in ["Q", "K", "V"]:
        plot_comp_scores(model, composition_scores[comp_type], f"{comp_type} Composition Scores")

# %%


def generate_single_random_comp_score(model: StandardizedTransformer) -> float:
    """
    Write a function which generates a single composition score for random matrices.
    """
    raw = model._model
    W_A_left = t.empty(raw.d_model, raw.d_head)
    W_B_left = t.empty(raw.d_model, raw.d_head)
    W_A_right = t.empty(raw.d_model, raw.d_head)
    W_B_right = t.empty(raw.d_model, raw.d_head)

    for W in [W_A_left, W_B_left, W_A_right, W_B_right]:
        nn.init.kaiming_uniform_(W, a=np.sqrt(5))

    W_A = W_A_left @ W_A_right.T
    W_B = W_B_left @ W_B_right.T

    return get_comp_score(W_A, W_B)


if MAIN:
    n_samples = 300
    comp_scores_baseline = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        comp_scores_baseline[i] = generate_single_random_comp_score(model)

    print("\nMean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())

    hist(
        comp_scores_baseline,
        nbins=50,
        width=800,
        labels={"x": "Composition score"},
        title="Random composition scores",
    )

# %%

if MAIN:
    baseline = comp_scores_baseline.mean()
    for comp_type, comp_scores in composition_scores.items():
        plot_comp_scores(
            model, comp_scores, f"{comp_type} Composition Scores", baseline=baseline
        )

# %%


def get_batched_comp_scores(
    W_As: FactoredMatrix, W_Bs: FactoredMatrix
) -> Tensor:
    """
    Computes compositional scores between pairs of matrices using FactoredMatrix.

    W_As: FactoredMatrix with shape [n_A, d_model, d_model]
    W_Bs: FactoredMatrix with shape [n_B, d_model, d_model]

    Returns: [n_A, n_B] composition scores
    """
    W_As = FactoredMatrix(
        W_As.A.reshape(-1, 1, *W_As.A.shape[-2:]),
        W_As.B.reshape(-1, 1, *W_As.B.shape[-2:]),
    )
    W_Bs = FactoredMatrix(
        W_Bs.A.reshape(1, -1, *W_Bs.A.shape[-2:]),
        W_Bs.B.reshape(1, -1, *W_Bs.B.shape[-2:]),
    )
    W_ABs = W_As @ W_Bs
    return W_ABs.norm() / (W_As.norm() * W_Bs.norm())


if MAIN:
    raw = model._model
    # Build W_OV and W_QK as FactoredMatrix objects
    W_OV_0 = FactoredMatrix(raw.blocks[0].attn.W_V, raw.blocks[0].attn.W_O)
    W_QK_1 = FactoredMatrix(raw.blocks[1].attn.W_Q, raw.blocks[1].attn.W_K.transpose(-1, -2))
    W_OV_1 = FactoredMatrix(raw.blocks[1].attn.W_V, raw.blocks[1].attn.W_O)

    composition_scores_batched = dict()
    composition_scores_batched["Q"] = get_batched_comp_scores(W_OV_0, W_QK_1)
    composition_scores_batched["K"] = get_batched_comp_scores(W_OV_0, W_QK_1.T)
    composition_scores_batched["V"] = get_batched_comp_scores(W_OV_0, W_OV_1)

    t.testing.assert_close(composition_scores_batched["Q"], composition_scores["Q"])
    t.testing.assert_close(composition_scores_batched["K"], composition_scores["K"])
    t.testing.assert_close(composition_scores_batched["V"], composition_scores["V"])
    print("Tests passed - your `get_batched_comp_scores` function is working!")

# %%

# ==================================================
# SECTION 12: Targeted Ablation
# ==================================================

if MAIN:
    seq_len = 50
    rep_tokens_1 = generate_repeated_tokens(model, seq_len, batch_size=1)

    def ablation_induction_score(
        model: StandardizedTransformer,
        tokens: Tensor,
        prev_head_index: int | None,
        ind_head_index: int,
    ) -> float:
        """
        Takes as input the index of the L0 head and the index of the L1 head, and then runs with
        the previous token head ablated and returns the induction score for the ind_head_index.
        Uses nnterp's attention_probabilities to both ablate and read patterns.
        """
        seq_len_half = (tokens.shape[1] - 1) // 2

        with model.trace(tokens):
            # Ablate layer 0 head by zeroing its attention pattern
            if prev_head_index is not None:
                model.attention_probabilities[0][:, prev_head_index] = 0.0
            # Get layer 1's attention pattern (affected by layer 0 ablation)
            pattern1 = model.attention_probabilities[1].save()

        # Extract induction score for specific head
        induction_score = (
            pattern1[0, ind_head_index].diag(-(seq_len_half - 1)).mean().item()
        )
        return induction_score

    baseline_induction_score = ablation_induction_score(model, rep_tokens_1, None, 4)
    print(f"Induction score for no ablations: {baseline_induction_score:.5f}\n")
    for i in range(model._model.n_heads):
        new_induction_score = ablation_induction_score(model, rep_tokens_1, i, 4)
        induction_score_change = new_induction_score - baseline_induction_score
        print(f"Ablation score change for head {i:02}: {induction_score_change:+.5f}")

# %%
