# %%

import re
import sys
from functools import partial
from itertools import product
from pathlib import Path
from typing import Callable, Literal

import circuitsvis as cv
import einops
import numpy as np
import plotly.express as px
import torch as t
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from rich import print as rprint
from rich.table import Column, Table
from torch import Tensor
from tqdm.auto import tqdm
from nnterp import StandardizedTransformer

t.set_grad_enabled(False)
device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part41_indirect_object_identification.tests as tests
from plotly_utils import bar, imshow, line, scatter

MAIN = __name__ == "__main__"

# %%

# === Model config constants for GPT-2 Small ===
N_LAYERS = 12
N_HEADS = 12
D_MODEL = 768
D_HEAD = 64  # D_MODEL // N_HEADS

# %%

if MAIN:
    model = StandardizedTransformer(
        "openai-community/gpt2",
        dtype=t.float32,
        enable_attention_probs=True,
    )
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

# %%

if MAIN:
    # Here is where we test on a single prompt
    # Result: high probability on Mary, as we expect
    example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
    example_answer = " Mary"

    input_ids = tokenizer(example_prompt, return_tensors="pt")["input_ids"].to(device)
    with model.trace(input_ids):
        logits = model.logits.save()

    # Get the prediction for the last token
    last_logits = logits[0, -1, :]
    probs = t.softmax(last_logits, dim=-1)
    answer_token = tokenizer.encode(example_answer)[0]
    print(f"Probability of '{example_answer}': {probs[answer_token].item():.4f}")

    # Show top 5 predictions
    top5 = t.topk(probs, 5)
    for i in range(5):
        token_str = tokenizer.decode(top5.indices[i].item())
        print(f"  Top {i+1}: '{token_str}' ({top5.values[i].item():.4f})")

# %%

if MAIN:
    prompt_format = [
        "When John and Mary went to the shops,{} gave the bag to",
        "When Tom and James went to the park,{} gave the ball to",
        "When Dan and Sid went to the shops,{} gave an apple to",
        "After Martin and Amy went to the park,{} gave a drink to",
    ]
    name_pairs = [
        (" Mary", " John"),
        (" Tom", " James"),
        (" Dan", " Sid"),
        (" Martin", " Amy"),
    ]

    # Define 8 prompts, in 4 groups of 2 (with adjacent prompts having answers swapped)
    prompts = [
        prompt.format(name)
        for (prompt, names) in zip(prompt_format, name_pairs)
        for name in names[::-1]
    ]
    # Define the answers for each prompt, in the form (correct, incorrect)
    answers = [names[::i] for names in name_pairs for i in (1, -1)]
    # Define the answer tokens (same shape as the answers)
    answer_tokens = t.concat(
        [
            t.tensor([tokenizer.encode(name)[0] for name in names]).unsqueeze(0)
            for names in answers
        ]
    )

    rprint(prompts)
    rprint(answers)
    rprint(answer_tokens)

    table = Table("Prompt", "Correct", "Incorrect", title="Prompts & Answers:")

    for prompt, answer in zip(prompts, answers):
        table.add_row(prompt, repr(answer[0]), repr(answer[1]))

    rprint(table)

# %%

if MAIN:
    tokens = tokenizer(prompts, padding=True, return_tensors="pt")["input_ids"].to(device)

    # Run the model and get logits
    with model.trace(tokens):
        original_logits = model.logits.save()

# %%

if MAIN:

    def logits_to_ave_logit_diff(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Int[Tensor, "batch 2"] = answer_tokens,
        per_prompt: bool = False,
    ) -> Float[Tensor, "*batch"]:
        """
        Returns logit difference between the correct and incorrect answer.

        If per_prompt=True, return the array of differences rather than the average.
        """
        # Only the final logits are relevant for the answer
        final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
        # Get the logits corresponding to the indirect object / subject tokens respectively
        answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(
            dim=-1, index=answer_tokens.to(final_logits.device)
        )
        # Find logit difference
        correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
        answer_logit_diff = correct_logits - incorrect_logits
        return answer_logit_diff if per_prompt else answer_logit_diff.mean()

    tests.test_logits_to_ave_logit_diff(logits_to_ave_logit_diff)

    original_per_prompt_diff = logits_to_ave_logit_diff(
        original_logits, answer_tokens, per_prompt=True
    )
    print("Per prompt logit difference:", original_per_prompt_diff)
    original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
    print("Average logit difference:", original_average_logit_diff)

    cols = [
        "Prompt",
        Column("Correct", style="rgb(0,200,0) bold"),
        Column("Incorrect", style="rgb(255,0,0) bold"),
        Column("Logit Difference", style="bold"),
    ]
    table = Table(*cols, title="Logit differences")

    for prompt, answer, logit_diff in zip(prompts, answers, original_per_prompt_diff):
        table.add_row(prompt, repr(answer[0]), repr(answer[1]), f"{logit_diff.item():.3f}")

    rprint(table)


# %%

# === Logit Diff Directions ===
# In the original, this used model.tokens_to_residual_directions which applies W_U.
# For NNsight/GPT-2, the unembedding matrix is model._model.lm_head.weight (shape [vocab, d_model]).
# tokens_to_residual_directions(token) = W_U.T[token] normalized... but TL with center_unembed=True
# centers the columns of W_U. We'll skip centering since we don't fold_ln or center weights.
# Instead we compute logit diff directly from raw logits.

if MAIN:
    W_U = model._model.lm_head.weight  # [vocab, d_model]

    # answer_residual_directions: for each batch element, get the unembedding directions
    # for the correct and incorrect answers
    answer_residual_directions = W_U[answer_tokens]  # [batch, 2, d_model]
    print("Answer residual directions shape:", answer_residual_directions.shape)

    correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(
        dim=1
    )
    logit_diff_directions = (
        correct_residual_directions - incorrect_residual_directions
    )  # [batch d_model]
    print("Logit difference directions shape:", logit_diff_directions.shape)


# %%

# === Residual Stream Decomposition ===
# We need to collect the residual stream at each layer. Since we don't have cache.decompose_resid(),
# we manually collect each component's contribution.

if MAIN:
    # Collect all layer outputs and the final logits in one trace
    # layers_output[i] gives the residual stream AFTER layer i (i.e., resid_post for layer i)
    layer_outputs_list = [None] * N_LAYERS

    with model.trace(tokens):
        embeddings = model.token_embeddings.save()
        for i in range(N_LAYERS):
            layer_outputs_list[i] = model.layers_output[i].save()
        traced_logits = model.logits.save()

    # Stack all layer outputs: shape [n_layers, batch, seq, d_model]
    all_layer_outputs = t.stack([layer_outputs_list[i] for i in range(N_LAYERS)])

    # The final residual stream is the output of the last layer
    final_residual_stream = all_layer_outputs[-1]  # [batch, seq, d_model]
    print(f"Final residual stream shape: {final_residual_stream.shape}")
    final_token_residual_stream = final_residual_stream[:, -1, :]  # [batch, d_model]

    # NOTE: Without fold_ln, we should apply the final layer norm before projecting.
    # The final LN is at model._model.transformer.ln_f
    ln_f = model._model.transformer.ln_f

    scaled_final_token_residual_stream = ln_f(final_token_residual_stream)

    average_logit_diff = einops.einsum(
        scaled_final_token_residual_stream,
        logit_diff_directions.to(scaled_final_token_residual_stream.device),
        "batch d_model, batch d_model ->",
    ) / len(prompts)

    print(f"Calculated average logit diff: {average_logit_diff:.10f}")
    print(f"Original logit difference:     {original_average_logit_diff:.10f}")

    # Note: these won't match exactly because TL uses center_unembed=True and fold_ln=True.
    # With raw HF GPT-2, we get the uncentered result. The logit diff from logits is the
    # ground truth; this residual stream projection is an approximation when LN isn't folded.
    print("(These may not match exactly due to LN not being folded into weights)")


# %%

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    logit_diff_directions: Float[Tensor, "batch d_model"] = None,
) -> Float[Tensor, "..."]:
    """
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream.

    NOTE: This is a simplified version that does NOT apply LayerNorm (since we can't
    fold LN into weights with raw HF models). The projection gives a useful relative
    signal for comparing components even without LN.
    """
    batch_size = residual_stack.size(-2)
    return (
        einops.einsum(
            residual_stack,
            logit_diff_directions,
            "... batch d_model, batch d_model -> ...",
        )
        / batch_size
    )


# %%

# === Accumulated Residual Stream (Logit Lens) ===
if MAIN:
    # Build accumulated residual stream: embeddings, then after each layer
    # (and optionally after attn / after MLP within each layer)
    # For simplicity, we just do after each full layer (resid_post)

    # Collect attn outputs and MLP outputs for decomposition
    attn_outputs_list = [None] * N_LAYERS
    mlp_outputs_list = [None] * N_LAYERS

    with model.trace(tokens):
        emb = model.token_embeddings.save()
        for i in range(N_LAYERS):
            attn_outputs_list[i] = model.attentions_output[i].save()
            mlp_outputs_list[i] = model.mlps_output[i].save()

    # Get outputs at the last sequence position
    emb_last = emb[:, -1, :]  # [batch, d_model]
    attn_last = t.stack([attn_outputs_list[i][:, -1, :] for i in range(N_LAYERS)])  # [n_layers, batch, d_model]
    mlp_last = t.stack([mlp_outputs_list[i][:, -1, :] for i in range(N_LAYERS)])  # [n_layers, batch, d_model]

    # Accumulated residual: start with embeddings, add attn then mlp for each layer
    # Components: embed, then for each layer: after_attn (embed + sum of prev + this attn), after_mlp
    accumulated = []
    labels = []
    running = emb_last.clone()
    accumulated.append(running.clone())
    labels.append("embed")

    for i in range(N_LAYERS):
        running = running + attn_last[i]
        accumulated.append(running.clone())
        labels.append(f"L{i}_attn")
        running = running + mlp_last[i]
        accumulated.append(running.clone())
        labels.append(f"L{i}_mlp")

    accumulated_residual = t.stack(accumulated)  # [2*n_layers+1, batch, d_model]

    logit_lens_logit_diffs = residual_stack_to_logit_diff(
        accumulated_residual, logit_diff_directions
    )

    line(
        logit_lens_logit_diffs,
        hovermode="x unified",
        title="Logit Difference From Accumulated Residual Stream",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=labels,
        width=800,
    )

# %%

# === Per-Layer Residual Decomposition ===
if MAIN:
    # Each component's individual contribution to logit diff direction
    per_layer_residual = t.stack(
        [emb_last] + [attn_last[i] for i in range(N_LAYERS)] + [mlp_last[i] for i in range(N_LAYERS)]
    )
    per_layer_labels = (
        ["embed"]
        + [f"L{i}_attn" for i in range(N_LAYERS)]
        + [f"L{i}_mlp" for i in range(N_LAYERS)]
    )
    per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, logit_diff_directions)

    line(
        per_layer_logit_diffs,
        hovermode="x unified",
        title="Logit Difference From Each Layer",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=per_layer_labels,
        width=800,
    )

# %%

# === Per-Head Logit Attribution ===
# We need individual head outputs. For GPT-2, the attention output for each head is:
#   z[head] @ W_O[head] where z = attn_output before the output projection
# But nnterp gives us the combined attention output (after c_proj).
# To get per-head outputs, we need to hook into the attention mechanism more deeply.
#
# Strategy: Access the internal attention value (z) before output projection,
# then manually compute each head's contribution as z[:, :, head, :] @ W_O[layer, head]

def split_qkv(c_attn_weight, d_model, n_heads):
    """Split GPT-2's fused c_attn weight into separate Q, K, V weight matrices.

    Args:
        c_attn_weight: shape [d_model, 3*d_model]
        d_model: model dimension
        n_heads: number of attention heads

    Returns:
        W_Q: [n_heads, d_model, d_head]
        W_K: [n_heads, d_model, d_head]
        W_V: [n_heads, d_model, d_head]
    """
    d_head = d_model // n_heads
    W_Q, W_K, W_V = c_attn_weight.split(d_model, dim=1)
    return (
        W_Q.view(d_model, n_heads, d_head).permute(1, 0, 2),
        W_K.view(d_model, n_heads, d_head).permute(1, 0, 2),
        W_V.view(d_model, n_heads, d_head).permute(1, 0, 2),
    )


def get_W_O(model, layer):
    """Get the output weight matrix for a given layer, split by head.
    Returns: [n_heads, d_head, d_model]
    """
    W_O_full = model._model.transformer.h[layer].attn.c_proj.weight  # [d_model, d_model]
    return W_O_full.view(N_HEADS, D_HEAD, D_MODEL)


def get_per_head_output(model, tokens, layer):
    """Get the output of each attention head at a given layer.
    Returns: [batch, seq, n_heads, d_model]
    """
    with model.trace(tokens):
        c_proj_input = model.layers[layer].self_attn.c_proj.input.save()

    z = c_proj_input.view(c_proj_input.shape[0], c_proj_input.shape[1], N_HEADS, D_HEAD)
    W_O = get_W_O(model, layer)

    per_head_out = einops.einsum(
        z, W_O, "batch seq n_heads d_head, n_heads d_head d_model -> batch seq n_heads d_model"
    )
    return per_head_out


if MAIN:
    # Compute per-head logit diffs for all layers
    # This collects the per-head output at the last token position for each layer
    per_head_logit_diffs = t.zeros(N_LAYERS, N_HEADS, device=device)

    for layer in tqdm(range(N_LAYERS), desc="Computing per-head logit diffs"):
        head_output = get_per_head_output(model, tokens, layer)  # [batch, seq, n_heads, d_model]
        head_output_last = head_output[:, -1, :, :]  # [batch, n_heads, d_model]
        # For each head, compute the logit diff direction projection
        for head in range(N_HEADS):
            per_head_logit_diffs[layer, head] = residual_stack_to_logit_diff(
                head_output_last[:, head, :].unsqueeze(0),  # [1, batch, d_model]
                logit_diff_directions,
            ).item()

    imshow(
        per_head_logit_diffs,
        labels={"x": "Head", "y": "Layer"},
        title="Logit Difference From Each Head",
        width=600,
    )


# %%

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    """
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].
    """
    i = t.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(i.cpu().numpy(), tensor.shape)).T.tolist()


if MAIN:
    k = 3

    for head_type in ["Positive", "Negative"]:
        # Get the heads with largest (or smallest) contribution to the logit difference
        top_heads = topk_of_Nd_tensor(
            per_head_logit_diffs * (1 if head_type == "Positive" else -1), k
        )

        # Get all their attention patterns
        attn_patterns = []
        for layer, head in top_heads:
            with model.trace(tokens):
                ap = model.attention_probabilities[layer].save()
            attn_patterns.append(ap[0, head])  # First batch element

        attn_patterns_for_important_heads = t.stack(attn_patterns)

        # Display results
        str_tokens = [tokenizer.decode(t_id) for t_id in tokens[0]]
        display(HTML(f"<h2>Top {k} {head_type} Logit Attribution Heads</h2>"))
        display(
            cv.attention.attention_heads(
                attention=attn_patterns_for_important_heads,
                tokens=str_tokens,
                attention_head_names=[f"{layer}.{head}" for layer, head in top_heads],
            )
        )


# %%

# === ACTIVATION PATCHING ===

if MAIN:
    clean_tokens = tokens
    # Swap each adjacent pair to get corrupted tokens
    indices = [i + 1 if i % 2 == 0 else i - 1 for i in range(len(tokens))]
    corrupted_tokens = clean_tokens[indices]

    print(
        "Clean string 0:    ",
        tokenizer.decode(clean_tokens[0]),
        "\nCorrupted string 0:",
        tokenizer.decode(corrupted_tokens[0]),
    )

    with model.trace(clean_tokens):
        clean_logits = model.logits.save()
    with model.trace(corrupted_tokens):
        corrupted_logits = model.logits.save()

    clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
    print(f"Clean logit diff: {clean_logit_diff:.4f}")

    corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
    print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")


# %%

if MAIN:

    def ioi_metric(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Int[Tensor, "batch 2"] = answer_tokens,
        corrupted_logit_diff: float = corrupted_logit_diff,
        clean_logit_diff: float = clean_logit_diff,
    ) -> Float[Tensor, ""]:
        """
        Linear function of logit diff, calibrated so that it equals 0 when performance is same as on
        corrupted input, and 1 when performance is same as on clean input.
        """
        patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
        return (patched_logit_diff - corrupted_logit_diff) / (
            clean_logit_diff - corrupted_logit_diff
        )

    t.testing.assert_close(ioi_metric(clean_logits).item(), 1.0)
    t.testing.assert_close(ioi_metric(corrupted_logits).item(), 0.0)
    t.testing.assert_close(ioi_metric((clean_logits + corrupted_logits) / 2).item(), 0.5)


# %%

# === Activation Patching: Residual Stream by Position and Layer ===

def get_act_patch_resid_pre(
    model: StandardizedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_tokens: Float[Tensor, "batch pos"],
    patching_metric: Callable,
) -> Float[Tensor, "layer pos"]:
    """
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from a clean run.

    Uses NNsight sessions to patch clean activations into corrupted runs.
    """
    seq_len = corrupted_tokens.size(1)
    results = t.zeros(N_LAYERS, seq_len, device=device, dtype=t.float32)

    for layer in tqdm(range(N_LAYERS), desc="Patching resid_pre by layer"):
        # Cache clean activations once per layer (doesn't depend on position)
        with model.trace(clean_tokens):
            if layer == 0:
                clean_act = model.token_embeddings.save()
            else:
                clean_act = model.layers_output[layer - 1].save()

        for position in range(seq_len):
            with model.trace(corrupted_tokens):
                if layer == 0:
                    model.token_embeddings[:, position, :] = clean_act[:, position, :]
                else:
                    model.layers_output[layer - 1][:, position, :] = clean_act[:, position, :]
                patched_logits = model.logits.save()

            results[layer, position] = patching_metric(patched_logits)

    return results


if MAIN:
    act_patch_resid_pre = get_act_patch_resid_pre(
        model, corrupted_tokens, clean_tokens, ioi_metric
    )

    str_tokens = [tokenizer.decode(t_id) for t_id in clean_tokens[0]]
    labels = [f"{tok} {i}" for i, tok in enumerate(str_tokens)]

    imshow(
        act_patch_resid_pre,
        labels={"x": "Position", "y": "Layer"},
        x=labels,
        title="resid_pre Activation Patching",
        width=700,
    )


# %%

# === Activation Patching: Block outputs (resid_pre, attn_out, mlp_out) ===

def get_act_patch_block_every(
    model: StandardizedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_tokens: Float[Tensor, "batch pos"],
    patching_metric: Callable,
) -> Float[Tensor, "3 layer pos"]:
    """
    Returns an array of results of patching each position at each layer in the residual
    stream, for residual stream, attention output, and MLP output.
    """
    seq_len = corrupted_tokens.size(1)
    results = t.zeros(3, N_LAYERS, seq_len, device=device, dtype=t.float32)

    # Component 0: resid_pre (same as before)
    for layer in tqdm(range(N_LAYERS), desc="Patching resid_pre"):
        for position in range(seq_len):
            with model.session():
                with model.trace(clean_tokens):
                    if layer == 0:
                        clean_act = model.token_embeddings.save()
                    else:
                        clean_act = model.layers_output[layer - 1].save()
                with model.trace(corrupted_tokens):
                    if layer == 0:
                        model.token_embeddings[:, position, :] = clean_act[:, position, :]
                    else:
                        model.layers_output[layer - 1][:, position, :] = clean_act[:, position, :]
                    patched_logits = model.logits.save()
            results[0, layer, position] = patching_metric(patched_logits)

    # Component 1: attn_out
    for layer in tqdm(range(N_LAYERS), desc="Patching attn_out"):
        for position in range(seq_len):
            with model.session():
                with model.trace(clean_tokens):
                    clean_attn = model.attentions_output[layer].save()
                with model.trace(corrupted_tokens):
                    model.attentions_output[layer][:, position, :] = clean_attn[:, position, :]
                    patched_logits = model.logits.save()
            results[1, layer, position] = patching_metric(patched_logits)

    # Component 2: mlp_out
    for layer in tqdm(range(N_LAYERS), desc="Patching mlp_out"):
        for position in range(seq_len):
            with model.session():
                with model.trace(clean_tokens):
                    clean_mlp = model.mlps_output[layer].save()
                with model.trace(corrupted_tokens):
                    model.mlps_output[layer][:, position, :] = clean_mlp[:, position, :]
                    patched_logits = model.logits.save()
            results[2, layer, position] = patching_metric(patched_logits)

    return results


if MAIN:
    act_patch_block_every = get_act_patch_block_every(
        model, corrupted_tokens, clean_tokens, ioi_metric
    )

    imshow(
        act_patch_block_every,
        x=labels,
        facet_col=0,
        facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
        title="Logit Difference From Patched Attn Head Output",
        labels={"x": "Sequence Position", "y": "Layer"},
        width=1200,
    )


# %%

# === Activation Patching: Per-Head (all positions) ===
# For per-head patching, we need to patch individual head outputs.
# We do this by patching at the c_proj input level (the concatenated z vectors).

def get_act_patch_attn_head_out_all_pos(
    model: StandardizedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_tokens: Float[Tensor, "batch pos"],
    patching_metric: Callable,
) -> Float[Tensor, "layer head"]:
    """
    Returns an array of results of patching at all positions for each head in each
    layer (patching the head's output before it's added to the residual stream).
    """
    results = t.zeros(N_LAYERS, N_HEADS, device=device, dtype=t.float32)

    for layer in tqdm(range(N_LAYERS), desc="Patching attn heads"):
        for head in range(N_HEADS):
            # Patch the head's slice of the c_proj input (the z vector for this head)
            h_start = head * D_HEAD
            h_end = (head + 1) * D_HEAD

            with model.session():
                with model.trace(clean_tokens):
                    clean_z = model.layers[layer].self_attn.c_proj.input.save()
                with model.trace(corrupted_tokens):
                    model.layers[layer].self_attn.c_proj.input[:, :, h_start:h_end] = clean_z[:, :, h_start:h_end]
                    patched_logits = model.logits.save()

            results[layer, head] = patching_metric(patched_logits)

    return results


if MAIN:
    act_patch_attn_head_out_all_pos = get_act_patch_attn_head_out_all_pos(
        model, corrupted_tokens, clean_tokens, ioi_metric
    )

    imshow(
        act_patch_attn_head_out_all_pos,
        labels={"y": "Layer", "x": "Head"},
        title="attn_head_out Activation Patching (All Pos)",
        width=600,
    )


# %%

# === Activation Patching: Per-Head Q/K/V/Pattern ===

def get_act_patch_attn_head_all_pos_every(
    model: StandardizedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_tokens: Float[Tensor, "batch pos"],
    patching_metric: Callable,
) -> Float[Tensor, "5 layer head"]:
    """
    Returns patching results for Output, Query, Key, Value, and Attention Pattern
    for each head in each layer.
    """
    results = t.zeros(5, N_LAYERS, N_HEADS, device=device, dtype=t.float32)

    # Component 0: Output (z) - same as get_act_patch_attn_head_out_all_pos
    for layer in tqdm(range(N_LAYERS), desc="Patching output (z)"):
        for head in range(N_HEADS):
            h_start = head * D_HEAD
            h_end = (head + 1) * D_HEAD
            with model.session():
                with model.trace(clean_tokens):
                    clean_z = model.layers[layer].self_attn.c_proj.input.save()
                with model.trace(corrupted_tokens):
                    model.layers[layer].self_attn.c_proj.input[:, :, h_start:h_end] = clean_z[:, :, h_start:h_end]
                    patched_logits = model.logits.save()
            results[0, layer, head] = patching_metric(patched_logits)

    # For Q, K, V patching, we need to patch the inputs to the attention computation
    # In GPT-2, c_attn produces [Q, K, V] concatenated. We can patch individual slices.
    # c_attn output shape: [batch, seq, 3*d_model]
    # Q = [:, :, 0:d_model], K = [:, :, d_model:2*d_model], V = [:, :, 2*d_model:3*d_model]
    # Per head: Q_h = [:, :, h*d_head:(h+1)*d_head] within the Q slice

    for comp_idx, comp_name, offset in [
        (1, "query", 0),
        (2, "key", D_MODEL),
        (3, "value", 2 * D_MODEL),
    ]:
        for layer in tqdm(range(N_LAYERS), desc=f"Patching {comp_name}"):
            for head in range(N_HEADS):
                h_start = offset + head * D_HEAD
                h_end = offset + (head + 1) * D_HEAD
                with model.session():
                    with model.trace(clean_tokens):
                        clean_qkv = model.layers[layer].self_attn.c_attn.output.save()
                    with model.trace(corrupted_tokens):
                        model.layers[layer].self_attn.c_attn.output[:, :, h_start:h_end] = clean_qkv[:, :, h_start:h_end]
                        patched_logits = model.logits.save()
                results[comp_idx, layer, head] = patching_metric(patched_logits)

    # Component 4: Attention pattern
    for layer in tqdm(range(N_LAYERS), desc="Patching attention pattern"):
        for head in range(N_HEADS):
            with model.session():
                with model.trace(clean_tokens):
                    clean_pattern = model.attention_probabilities[layer].save()
                with model.trace(corrupted_tokens):
                    model.attention_probabilities[layer][:, head] = clean_pattern[:, head]
                    patched_logits = model.logits.save()
            results[4, layer, head] = patching_metric(patched_logits)

    return results


if MAIN:
    act_patch_attn_head_all_pos_every = get_act_patch_attn_head_all_pos_every(
        model, corrupted_tokens, clean_tokens, ioi_metric
    )

    imshow(
        act_patch_attn_head_all_pos_every,
        facet_col=0,
        facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
        title="Activation Patching Per Head (All Pos)",
        labels={"x": "Head", "y": "Layer"},
        width=1200,
    )


# %%

# === IOI Dataset ===

from part41_indirect_object_identification.ioi_dataset import NAMES, IOIDataset

# %%

if MAIN:
    N = 25
    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=tokenizer,
        prepend_bos=False,
        seed=1,
        device=str(device),
    )

# %%

if MAIN:
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")


# %%

def format_prompt(sentence: str) -> str:
    """Format a prompt by underlining names (for rich print)"""
    return (
        re.sub(
            "(" + "|".join(NAMES) + ")", lambda x: f"[u bold dark_orange]{x.group(0)}[/]", sentence
        )
        + "\n"
    )


def make_table(cols, colnames, title="", n_rows=5, decimals=4):
    """Makes and displays a table, from cols rather than rows (using rich print)"""
    table = Table(*colnames, title=title)
    rows = list(zip(*cols))
    f = lambda x: x if isinstance(x, str) else f"{x:.{decimals}f}"
    for row in rows[:n_rows]:
        table.add_row(*list(map(f, row)))
    rprint(table)


if MAIN:
    make_table(
        colnames=["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
        cols=[
            map(format_prompt, ioi_dataset.sentences),
            [tokenizer.decode(tid) for tid in ioi_dataset.s_tokenIDs],
            [tokenizer.decode(tid) for tid in ioi_dataset.io_tokenIDs],
            map(format_prompt, abc_dataset.sentences),
        ],
        title="Sentences from IOI vs ABC distribution",
    )

# %%

if MAIN:

    def logits_to_ave_logit_diff_2(
        logits: Float[Tensor, "batch seq d_vocab"],
        ioi_dataset: IOIDataset = ioi_dataset,
        per_prompt=False,
    ) -> Float[Tensor, "*batch"]:
        """
        Returns logit difference between the correct and incorrect answer.
        """
        io_logits: Float[Tensor, "batch"] = logits[
            range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs
        ]
        s_logits: Float[Tensor, "batch"] = logits[
            range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs
        ]
        answer_logit_diff = io_logits - s_logits
        return answer_logit_diff if per_prompt else answer_logit_diff.mean()

    # Get logits for IOI and ABC datasets
    with model.trace(ioi_dataset.toks):
        ioi_logits_original = model.logits.save()
    with model.trace(abc_dataset.toks):
        abc_logits_original = model.logits.save()

    ioi_per_prompt_diff = logits_to_ave_logit_diff_2(ioi_logits_original, per_prompt=True)
    abc_per_prompt_diff = logits_to_ave_logit_diff_2(abc_logits_original, per_prompt=True)

    ioi_average_logit_diff = logits_to_ave_logit_diff_2(ioi_logits_original).item()
    abc_average_logit_diff = logits_to_ave_logit_diff_2(abc_logits_original).item()

    print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
    print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")

    make_table(
        colnames=["IOI prompt", "IOI logit diff", "ABC prompt", "ABC logit diff"],
        cols=[
            map(format_prompt, ioi_dataset.sentences),
            ioi_per_prompt_diff,
            map(format_prompt, abc_dataset.sentences),
            abc_per_prompt_diff,
        ],
        title="Sentences from IOI vs ABC distribution",
    )


# %%

if MAIN:

    def ioi_metric_2(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float = ioi_average_logit_diff,
        corrupted_logit_diff: float = abc_average_logit_diff,
        ioi_dataset: IOIDataset = ioi_dataset,
    ) -> float:
        """
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI
        dataset), and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        """
        patched_logit_diff = logits_to_ave_logit_diff_2(logits, ioi_dataset)
        return (patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

    print(f"IOI metric (IOI dataset): {ioi_metric_2(ioi_logits_original):.4f}")
    print(f"IOI metric (ABC dataset): {ioi_metric_2(abc_logits_original):.4f}")


# %%

# === PATH PATCHING ===

if MAIN:

    def get_path_patch_head_to_final_resid_post(
        model: StandardizedTransformer,
        patching_metric: Callable,
        new_dataset: IOIDataset = abc_dataset,
        orig_dataset: IOIDataset = ioi_dataset,
    ) -> Float[Tensor, "layer head"]:
        """
        Performs path patching with:
            sender head = (each head, looped through, one at a time)
            receiver node = final value of residual stream

        For each sender head:
        1. Run on orig input, cache all head z outputs
        2. Run on new input, cache all head z outputs
        3. Run on orig input, but replace the sender head's z with the new cache value,
           and freeze all other heads to orig values.
        4. The final residual stream captures the effect of just this head changing.
        """
        results = t.zeros(N_LAYERS, N_HEADS, device=device, dtype=t.float32)

        # Step 1: Cache all z values for both datasets
        orig_z_cache = {}
        new_z_cache = {}

        for layer in range(N_LAYERS):
            with model.trace(orig_dataset.toks):
                orig_z_cache[layer] = model.layers[layer].self_attn.c_proj.input.save()
            with model.trace(new_dataset.toks):
                new_z_cache[layer] = model.layers[layer].self_attn.c_proj.input.save()

        # Step 2-3: For each sender head, freeze all heads but patch sender from new
        for sender_layer, sender_head in tqdm(
            list(product(range(N_LAYERS), range(N_HEADS))),
            desc="Path patching to final resid"
        ):
            h_start = sender_head * D_HEAD
            h_end = (sender_head + 1) * D_HEAD

            with model.trace(orig_dataset.toks):
                # Freeze all heads to their orig values, except patch sender head from new
                for layer in range(N_LAYERS):
                    z_target = orig_z_cache[layer].clone()
                    if layer == sender_layer:
                        z_target[:, :, h_start:h_end] = new_z_cache[layer][:, :, h_start:h_end]
                    model.layers[layer].self_attn.c_proj.input[:] = z_target
                patched_logits = model.logits.save()

            results[sender_layer, sender_head] = patching_metric(patched_logits)

        return results

    path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(
        model, ioi_metric_2
    )

    imshow(
        100 * path_patch_head_to_final_resid_post,
        title="Direct effect on logit difference",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff. variation"},
        coloraxis=dict(colorbar_ticksuffix="%"),
        width=600,
    )


# %%

# === Path Patching: Head to Heads (S-Inhibition Head Values) ===

if MAIN:

    def get_path_patch_head_to_heads(
        receiver_heads: list[tuple[int, int]],
        receiver_input: str,
        model: StandardizedTransformer,
        patching_metric: Callable,
        new_dataset: IOIDataset = abc_dataset,
        orig_dataset: IOIDataset = ioi_dataset,
    ) -> Float[Tensor, "layer head"]:
        """
        Performs path patching with:
            sender head = (each head, looped through)
            receiver node = input to a later head (Q, K, or V)

        Uses a 3-step process:
        1. Cache z values for orig and new datasets
        2. Run on orig with sender patched from new, all others frozen -> cache receiver inputs
        3. Run on orig, patching in the receiver inputs from step 2
        """
        assert receiver_input in ("k", "q", "v")
        receiver_layers = set(layer for layer, head in receiver_heads)
        max_receiver_layer = max(receiver_layers)

        results = t.zeros(max_receiver_layer, N_HEADS, device=device, dtype=t.float32)

        # Cache z for both datasets
        orig_z_cache = {}
        new_z_cache = {}
        for layer in range(N_LAYERS):
            with model.trace(orig_dataset.toks):
                orig_z_cache[layer] = model.layers[layer].self_attn.c_proj.input.save()
            with model.trace(new_dataset.toks):
                new_z_cache[layer] = model.layers[layer].self_attn.c_proj.input.save()

        # Determine which component offset to use for patching
        if receiver_input == "q":
            comp_offset = 0
        elif receiver_input == "k":
            comp_offset = D_MODEL
        else:  # v
            comp_offset = 2 * D_MODEL

        for sender_layer, sender_head in tqdm(
            list(product(range(max_receiver_layer), range(N_HEADS))),
            desc=f"Path patching to {receiver_input} of receiver heads"
        ):
            h_start_sender = sender_head * D_HEAD
            h_end_sender = (sender_head + 1) * D_HEAD

            # Step 2: Run with sender patched, all others frozen, cache receiver QKV
            receiver_qkv_cache = {}
            with model.trace(orig_dataset.toks):
                for layer in range(N_LAYERS):
                    # Save receiver QKV BEFORE c_proj (c_attn runs before c_proj in forward pass)
                    if layer in receiver_layers:
                        receiver_qkv_cache[layer] = model.layers[layer].self_attn.c_attn.output.save()
                    z_target = orig_z_cache[layer].clone()
                    if layer == sender_layer:
                        z_target[:, :, h_start_sender:h_end_sender] = new_z_cache[layer][:, :, h_start_sender:h_end_sender]
                    model.layers[layer].self_attn.c_proj.input[:] = z_target

            # Step 3: Run on orig, patching in receiver inputs from step 2
            with model.trace(orig_dataset.toks):
                for r_layer in sorted(receiver_layers):
                    heads_in_layer = [h for l, h in receiver_heads if l == r_layer]
                    for h in heads_in_layer:
                        r_start = comp_offset + h * D_HEAD
                        r_end = comp_offset + (h + 1) * D_HEAD
                        model.layers[r_layer].self_attn.c_attn.output[:, :, r_start:r_end] = receiver_qkv_cache[r_layer][:, :, r_start:r_end]
                patched_logits = model.logits.save()

            results[sender_layer, sender_head] = patching_metric(patched_logits)

        return results

    s_inhibition_value_path_patching_results = get_path_patch_head_to_heads(
        receiver_heads=[(8, 6), (8, 10), (7, 9), (7, 3)],
        receiver_input="v",
        model=model,
        patching_metric=ioi_metric_2,
    )

    imshow(
        100 * s_inhibition_value_path_patching_results,
        title="Direct effect on S-Inhibition Heads' values",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff.<br>variation"},
        width=600,
        coloraxis=dict(colorbar_ticksuffix="%"),
    )


# %%

# === Scatter: Embedding vs Attention (Figure 3c from paper) ===

def scatter_embedding_vs_attn(
    attn_from_end_to_io: Float[Tensor, "batch"],
    attn_from_end_to_s: Float[Tensor, "batch"],
    projection_in_io_dir: Float[Tensor, "batch"],
    projection_in_s_dir: Float[Tensor, "batch"],
    layer: int,
    head: int,
    N: int,
):
    scatter(
        x=t.concat([attn_from_end_to_io, attn_from_end_to_s], dim=0),
        y=t.concat([projection_in_io_dir, projection_in_s_dir], dim=0),
        color=["IO"] * N + ["S"] * N,
        title=f"Projection of the output of {layer}.{head} along the name<br>embedding vs attention probability on name",
        title_x=0.5,
        labels={"x": "Attn prob on name", "y": "Dot w Name Embed", "color": "Name type"},
        color_discrete_sequence=["#72FF64", "#C9A5F7"],
        width=650,
    )


if MAIN:

    def calculate_and_show_scatter_embedding_vs_attn(
        layer: int,
        head: int,
        ioi_dataset: IOIDataset = ioi_dataset,
        model: StandardizedTransformer = model,
    ) -> None:
        """
        Creates and plots a figure equivalent to 3(c) in the paper.
        """
        N_data = len(ioi_dataset)
        W_O = get_W_O(model, layer)  # [n_heads, d_head, d_model]
        W_U = model._model.lm_head.weight  # [vocab, d_model]

        # Get attention z vectors (c_proj input) and attention patterns
        with model.trace(ioi_dataset.toks):
            attn_probs = model.attention_probabilities[layer].save()
            z_all = model.layers[layer].self_attn.c_proj.input.save()

        # Extract this head's z vectors: [batch, seq, d_head]
        h_start = head * D_HEAD
        h_end = (head + 1) * D_HEAD
        z_head = z_all[:, :, h_start:h_end]

        # Compute output: z @ W_O -> [batch, seq, d_model]
        output = einops.einsum(z_head, W_O[head], "batch seq d_head, d_head d_model -> batch seq d_model")

        # Get output at the end token position
        output_on_end_token = output[
            t.arange(N_data), ioi_dataset.word_idx["end"]
        ]  # [batch, d_model]

        # Get the unembedding directions
        io_unembedding = W_U[ioi_dataset.io_tokenIDs]  # [batch, d_model]
        s_unembedding = W_U[ioi_dataset.s_tokenIDs]  # [batch, d_model]

        # Compute projections
        projection_in_io_dir = (output_on_end_token * io_unembedding).sum(-1)
        projection_in_s_dir = (output_on_end_token * s_unembedding).sum(-1)

        # Get attention probs from END -> IO and END -> S1
        attn_from_end_to_io = attn_probs[
            t.arange(N_data), head, ioi_dataset.word_idx["end"], ioi_dataset.word_idx["IO"]
        ]
        attn_from_end_to_s = attn_probs[
            t.arange(N_data), head, ioi_dataset.word_idx["end"], ioi_dataset.word_idx["S1"]
        ]

        scatter_embedding_vs_attn(
            attn_from_end_to_io,
            attn_from_end_to_s,
            projection_in_io_dir,
            projection_in_s_dir,
            layer,
            head,
            N_data,
        )

    calculate_and_show_scatter_embedding_vs_attn(9, 9)   # name mover head 9.9
    calculate_and_show_scatter_embedding_vs_attn(11, 10)  # negative name mover head 11.10


# %%

# === Copying Scores ===

def get_copying_scores(
    model: StandardizedTransformer, k: int = 5, names: list = NAMES
) -> Float[Tensor, "2 layer-1 head"]:
    """
    Gets copying scores (both positive and negative) as described in page 6 of the IOI paper.
    """
    results = t.zeros((2, N_LAYERS, N_HEADS), device=device)

    W_E = model._model.transformer.wte.weight  # [vocab, d_model]
    W_U = model._model.lm_head.weight  # [vocab, d_model]
    ln_f = model._model.transformer.ln_f

    # Get MLP0 output for name embeddings
    name_tokens = t.tensor(
        [tokenizer.encode(" " + name)[0] for name in names], device=device
    ).unsqueeze(1)  # [batch, 1]
    name_embeddings = W_E[name_tokens.squeeze(1)].unsqueeze(1)  # [batch, 1, d_model]

    # Apply layer 0's MLP to get extended embeddings
    # We need to run through layer 0's LN2 and MLP
    ln2_0 = model._model.transformer.h[0].ln_2
    mlp_0 = model._model.transformer.h[0].mlp
    resid_after_mlp0 = name_embeddings + mlp_0(ln2_0(name_embeddings))

    for layer in range(1, N_LAYERS):
        W_V_full = model._model.transformer.h[layer].attn.c_attn.weight[:, 2*D_MODEL:3*D_MODEL]
        W_O_full = model._model.transformer.h[layer].attn.c_proj.weight

        for head in range(N_HEADS):
            # Get W_V and W_O for this head
            W_V_head = W_V_full[:, head*D_HEAD:(head+1)*D_HEAD]  # [d_model, d_head]
            W_O_head = W_O_full[head*D_HEAD:(head+1)*D_HEAD, :]  # [d_head, d_model]
            W_OV = W_V_head @ W_O_head  # [d_model, d_model]

            # Apply OV circuit (positive and negative)
            resid_after_OV_pos = resid_after_mlp0 @ W_OV
            resid_after_OV_neg = resid_after_mlp0 @ (-W_OV)

            # Get logits
            logits_pos = (ln_f(resid_after_OV_pos) @ W_U.T).squeeze(1)  # [batch, vocab]
            logits_neg = (ln_f(resid_after_OV_neg) @ W_U.T).squeeze(1)

            # Check how many are in top k
            topk_pos = t.topk(logits_pos, dim=-1, k=k).indices
            in_topk = (topk_pos == name_tokens).any(-1)
            bottomk_neg = t.topk(logits_neg, dim=-1, k=k).indices
            in_bottomk = (bottomk_neg == name_tokens).any(-1)

            results[:, layer - 1, head] = t.tensor(
                [in_topk.float().mean(), in_bottomk.float().mean()]
            )

    return results


if MAIN:
    copying_results = get_copying_scores(model)

    imshow(
        copying_results,
        facet_col=0,
        facet_labels=["Positive copying scores", "Negative copying scores"],
        title="Copying scores of attention heads' OV circuits",
        width=900,
    )

    heads = {"name mover": [(9, 9), (10, 0), (9, 6)], "negative name mover": [(10, 7), (11, 10)]}

    for i, name in enumerate(["name mover", "negative name mover"]):
        make_table(
            title=f"Copying Scores ({name} heads)",
            colnames=["Head", "Score"],
            cols=[
                list(map(str, heads[name])) + ["[dark_orange bold]Average"],
                [f"{copying_results[i, layer - 1, head]:.2%}" for (layer, head) in heads[name]]
                + [f"[dark_orange bold]{copying_results[i].mean():.2%}"],
            ],
        )


# %%

# === Early Head Validation (Duplicate, Previous, Induction) ===

def generate_repeated_tokens(
    model: StandardizedTransformer, seq_len: int, batch: int = 1
) -> Float[Tensor, "batch 2*seq_len"]:
    """Generates a sequence of repeated random tokens (no start token)."""
    rep_tokens_half = t.randint(0, tokenizer.vocab_size, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens


def get_attn_scores(
    model: StandardizedTransformer,
    seq_len: int,
    batch: int,
    head_type: Literal["duplicate", "prev", "induction"],
) -> Float[Tensor, "n_layers n_heads"]:
    """Returns attention scores for sequence of duplicated tokens, for every head."""
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)

    if head_type == "duplicate":
        src_indices = range(seq_len)
        dest_indices = range(seq_len, 2 * seq_len)
    elif head_type == "prev":
        src_indices = range(seq_len)
        dest_indices = range(1, seq_len + 1)
    elif head_type == "induction":
        dest_indices = range(seq_len, 2 * seq_len)
        src_indices = range(1, seq_len + 1)

    results = t.zeros(N_LAYERS, N_HEADS, device=device, dtype=t.float32)
    for layer in range(N_LAYERS):
        with model.trace(rep_tokens):
            attn_scores = model.attention_probabilities[layer].save()
        for head in range(N_HEADS):
            avg_attn = attn_scores[:, head, dest_indices, src_indices].mean().item()
            results[layer, head] = avg_attn

    return results


def plot_early_head_validation_results(seq_len: int = 50, batch: int = 50):
    """Produces a plot that looks like Figure 18 in the paper."""
    head_types = ["duplicate", "prev", "induction"]

    results = t.stack(
        [get_attn_scores(model, seq_len, batch, head_type=head_type) for head_type in head_types]
    )

    imshow(
        results,
        facet_col=0,
        facet_labels=[
            f"{head_type.capitalize()} token attention prob.<br>on sequences of random tokens"
            for head_type in head_types
        ],
        labels={"x": "Head", "y": "Layer"},
        width=1300,
    )


if MAIN:
    plot_early_head_validation_results()


# %%

# === Circuit Definition ===

CIRCUIT = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative name mover": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}

SEQ_POS_TO_KEEP = {
    "name mover": "end",
    "backup name mover": "end",
    "negative name mover": "end",
    "s2 inhibition": "end",
    "induction": "S2",
    "duplicate token": "S2",
    "previous token": "S1+1",
}


# %%

# === Mean Ablation for Circuit Evaluation ===

def get_heads_and_posns_to_keep(
    means_dataset: IOIDataset,
    circuit: dict[str, list[tuple[int, int]]],
    seq_pos_to_keep: dict[str, str],
) -> dict[int, Bool[Tensor, "batch seq head"]]:
    """
    Returns a dictionary mapping layers to a boolean mask giving the indices of the z output which
    should NOT be mean-ablated.
    """
    heads_and_posns_to_keep = {}
    batch_size, seq_len = len(means_dataset), means_dataset.max_len

    for layer in range(N_LAYERS):
        mask = t.zeros(size=(batch_size, seq_len, N_HEADS))

        for head_type, head_list in circuit.items():
            seq_pos = seq_pos_to_keep[head_type]
            indices = means_dataset.word_idx[seq_pos]
            for layer_idx, head_idx in head_list:
                if layer_idx == layer:
                    mask[range(batch_size), indices, head_idx] = 1

        heads_and_posns_to_keep[layer] = mask.bool()

    return heads_and_posns_to_keep


def compute_means_by_template(
    means_dataset: IOIDataset, model: StandardizedTransformer
) -> Float[Tensor, "layer batch seq head_idx d_head"]:
    """
    Returns the mean of each head's output (z vector, i.e. c_proj input) over the means dataset.
    Computed separately for each group of prompts with the same template.
    """
    # Cache z outputs for all layers
    z_cache = {}
    for layer in range(N_LAYERS):
        with model.trace(means_dataset.toks.long()):
            z_cache[layer] = model.layers[layer].self_attn.c_proj.input.save()

    batch = len(means_dataset)
    seq_len = means_dataset.max_len
    means = t.zeros(size=(N_LAYERS, batch, seq_len, N_HEADS, D_HEAD), device=device)

    for layer in range(N_LAYERS):
        z_layer = z_cache[layer]  # [batch, seq, d_model]
        # Reshape to [batch, seq, n_heads, d_head]
        z_layer_heads = z_layer.view(batch, seq_len, N_HEADS, D_HEAD)

        for template_group in means_dataset.groups:
            z_for_template = z_layer_heads[template_group]
            z_means = einops.reduce(
                z_for_template, "batch seq head d_head -> seq head d_head", "mean"
            )
            means[layer, template_group] = z_means

    return means


def run_with_mean_ablation(
    model: StandardizedTransformer,
    tokens: Float[Tensor, "batch seq"],
    means: Float[Tensor, "layer batch seq head_idx d_head"],
    heads_and_posns_to_keep: dict[int, Bool[Tensor, "batch seq head"]],
) -> Float[Tensor, "batch seq d_vocab"]:
    """
    Run the model on tokens, but replace z values with means except where the mask says to keep.
    """
    with model.trace(tokens):
        for layer in range(N_LAYERS):
            # Get the c_proj input (z) and reshape
            z = model.layers[layer].self_attn.c_proj.input  # [batch, seq, d_model]
            z_heads = z.reshape(z.shape[0], z.shape[1], N_HEADS, D_HEAD)

            # Apply mask: keep original where mask=True, replace with means where mask=False
            mask = heads_and_posns_to_keep[layer].unsqueeze(-1).to(z.device)  # [batch, seq, head, 1]
            z_new = t.where(mask, z_heads, means[layer].to(z.device))

            # Write back
            model.layers[layer].self_attn.c_proj.input[:] = z_new.reshape(z.shape)

        logits = model.logits.save()

    return logits


def add_mean_ablation_and_run(
    model: StandardizedTransformer,
    ioi_tokens: Float[Tensor, "batch seq"],
    means_dataset: IOIDataset,
    circuit: dict[str, list[tuple[int, int]]] = CIRCUIT,
    seq_pos_to_keep: dict[str, str] = SEQ_POS_TO_KEEP,
) -> Float[Tensor, "batch seq d_vocab"]:
    """
    Runs the model with mean ablation according to the circuit specification.
    """
    means = compute_means_by_template(means_dataset, model)
    heads_and_posns_to_keep = get_heads_and_posns_to_keep(means_dataset, circuit, seq_pos_to_keep)
    return run_with_mean_ablation(model, ioi_tokens, means, heads_and_posns_to_keep)


# %%

if MAIN:
    ioi_logits_minimal = add_mean_ablation_and_run(
        model,
        ioi_dataset.toks,
        means_dataset=abc_dataset,
        circuit=CIRCUIT,
        seq_pos_to_keep=SEQ_POS_TO_KEEP,
    )

    print(f"""
    Avg logit diff (IOI dataset, using entire model): {logits_to_ave_logit_diff_2(ioi_logits_original):.4f}
    Avg logit diff (IOI dataset, only using circuit): {logits_to_ave_logit_diff_2(ioi_logits_minimal):.4f}
    """)


# %%

# === Minimality Testing ===

K_FOR_EACH_COMPONENT = {
    (9, 9): set(),
    (10, 0): {(9, 9)},
    (9, 6): {(9, 9), (10, 0)},
    (10, 7): {(11, 10)},
    (11, 10): {(10, 7)},
    (8, 10): {(7, 9), (8, 6), (7, 3)},
    (7, 9): {(8, 10), (8, 6), (7, 3)},
    (8, 6): {(7, 9), (8, 10), (7, 3)},
    (7, 3): {(7, 9), (8, 10), (8, 6)},
    (5, 5): {(5, 9), (6, 9), (5, 8)},
    (5, 9): {(11, 10), (10, 7)},
    (6, 9): {(5, 9), (5, 5), (5, 8)},
    (5, 8): {(11, 10), (10, 7)},
    (0, 1): {(0, 10), (3, 0)},
    (0, 10): {(0, 1), (3, 0)},
    (3, 0): {(0, 1), (0, 10)},
    (4, 11): {(2, 2)},
    (2, 2): {(4, 11)},
    (11, 2): {(9, 9), (10, 0), (9, 6)},
    (10, 6): {(9, 9), (10, 0), (9, 6), (11, 2)},
    (10, 10): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6)},
    (10, 2): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6), (10, 10)},
    (9, 7): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6), (10, 10), (10, 2)},
    (10, 1): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6), (10, 10), (10, 2), (9, 7)},
    (11, 9): {(9, 9), (10, 0), (9, 6), (9, 0)},
    (9, 0): {(9, 9), (10, 0), (9, 6), (11, 9)},
}


# %%


def plot_minimal_set_results(minimality_scores: dict[tuple[int, int], float]):
    """Plots the minimality results, in a way resembling figure 7 in the paper."""
    CIRCUIT_reversed = {head: k for k, v in CIRCUIT.items() for head in v}
    colors = [CIRCUIT_reversed[head].capitalize() + " head" for head in minimality_scores.keys()]
    color_sequence = [px.colors.qualitative.Dark2[i] for i in [0, 1, 2, 5, 3, 6]] + ["#BAEA84"]

    bar(
        list(minimality_scores.values()),
        x=list(map(str, minimality_scores.keys())),
        labels={"x": "Attention head", "y": "Change in logit diff", "color": "Head type"},
        color=colors,
        template="ggplot2",
        color_discrete_sequence=color_sequence,
        bargap=0.02,
        yaxis_tickformat=".0%",
        legend_title_text="",
        title="Plot of minimality scores (as percentages of full model logit diff)",
        width=800,
        hovermode="x unified",
    )


# %%

def get_score(
    model: StandardizedTransformer,
    ioi_dataset: IOIDataset,
    abc_dataset: IOIDataset,
    K: set[tuple[int, int]],
    C: dict[str, list[tuple[int, int]]],
) -> float:
    """
    Returns the value F(C \\ K), where F is the logit diff, C is the core circuit, and K is the set
    of circuit components to remove.
    """
    C_excl_K = {k: [head for head in v if head not in K] for k, v in C.items()}
    logits = add_mean_ablation_and_run(model, ioi_dataset.toks, abc_dataset, C_excl_K, SEQ_POS_TO_KEEP)
    score = logits_to_ave_logit_diff_2(logits, ioi_dataset).item()
    return score


def get_minimality_score(
    model: StandardizedTransformer,
    ioi_dataset: IOIDataset,
    abc_dataset: IOIDataset,
    v: tuple[int, int],
    K: set[tuple[int, int]],
    C: dict[str, list[tuple[int, int]]] = CIRCUIT,
) -> float:
    """
    Returns |F(C \\ K_union_v) - F(C \\ K)|
    """
    assert v not in K
    K_union_v = K | {v}
    C_excl_K_score = get_score(model, ioi_dataset, abc_dataset, K, C)
    C_excl_Kv_score = get_score(model, ioi_dataset, abc_dataset, K_union_v, C)
    return abs(C_excl_K_score - C_excl_Kv_score)


if MAIN:

    def get_all_minimality_scores(
        model: StandardizedTransformer,
        ioi_dataset: IOIDataset = ioi_dataset,
        abc_dataset: IOIDataset = abc_dataset,
        k_for_each_component: dict = K_FOR_EACH_COMPONENT,
    ) -> dict[tuple[int, int], float]:
        """
        Returns dict of minimality scores for every head in the circuit.
        """
        # Get full model score
        with model.trace(ioi_dataset.toks):
            full_logits = model.logits.save()
        full_circuit_score = logits_to_ave_logit_diff_2(full_logits, ioi_dataset).item()

        minimality_scores = {}
        for v, K in tqdm(k_for_each_component.items()):
            score = get_minimality_score(model, ioi_dataset, abc_dataset, v, K)
            minimality_scores[v] = score / full_circuit_score

        return minimality_scores

    minimality_scores = get_all_minimality_scores(model)
    plot_minimal_set_results(minimality_scores)


# %%

# === Induction Head Validation ===

if MAIN:
    attn_heads = [(5, 5), (6, 9)]

    batch = 1
    seq_len = 15
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)

    attn_patterns = []
    for layer, head in attn_heads:
        with model.trace(rep_tokens):
            ap = model.attention_probabilities[layer].save()
        attn_patterns.append(ap[0, head])

    attn = t.stack(attn_patterns)
    str_tokens = [tokenizer.decode(t_id) for t_id in rep_tokens[0]]
    cv.attention.attention_heads(
        tokens=str_tokens,
        attention=attn,
        attention_head_names=[f"{layer}.{head}" for (layer, head) in attn_heads],
    )


# %%

# === Path Patching: Induction Head Keys ===

if MAIN:
    induction_head_key_path_patching_results = get_path_patch_head_to_heads(
        receiver_heads=[(5, 5), (6, 9)],
        receiver_input="k",
        model=model,
        patching_metric=ioi_metric_2,
    )

    imshow(
        100 * induction_head_key_path_patching_results,
        title="Direct effect on Induction Heads' keys",
        labels={"x": "Head", "y": "Layer", "color": "Logit diff.<br>variation"},
        coloraxis=dict(colorbar_ticksuffix="%"),
        width=600,
    )


# %%

# === Backup Name Mover Analysis ===

if MAIN:
    # Get fresh logits and per-head logit diffs
    with model.trace(ioi_dataset.toks):
        ioi_logits = model.logits.save()
    original_average_logit_diff_2 = logits_to_ave_logit_diff_2(ioi_logits)

    W_U = model._model.lm_head.weight
    s_unembeddings = W_U[ioi_dataset.s_tokenIDs]
    io_unembeddings = W_U[ioi_dataset.io_tokenIDs]
    logit_diff_directions_2 = io_unembeddings - s_unembeddings  # [batch, d_model]

    # Compute per-head logit diffs using the IOI dataset
    per_head_logit_diffs_2 = t.zeros(N_LAYERS, N_HEADS, device=device)

    for layer in tqdm(range(N_LAYERS), desc="Per-head logit diffs (IOI)"):
        head_output = get_per_head_output(model, ioi_dataset.toks, layer)
        # Get output at end token positions
        end_positions = ioi_dataset.word_idx["end"]
        head_output_end = head_output[t.arange(len(ioi_dataset)), end_positions]  # [batch, n_heads, d_model]
        for head in range(N_HEADS):
            per_head_logit_diffs_2[layer, head] = residual_stack_to_logit_diff(
                head_output_end[:, head, :].unsqueeze(0),
                logit_diff_directions_2,
            ).item()

    top_layer, top_head = topk_of_Nd_tensor(per_head_logit_diffs_2, k=1)[0]
    print(f"Top Name Mover to ablate: {top_layer}.{top_head}")

    # Ablate the top name mover head
    abc_means = compute_means_by_template(abc_dataset, model)[top_layer]

    with model.trace(ioi_dataset.toks):
        # Get the z values and ablate the top head at end positions
        z = model.layers[top_layer].self_attn.c_proj.input
        z_heads = z.reshape(z.shape[0], z.shape[1], N_HEADS, D_HEAD)
        # Replace the top head's values at end positions with means
        for b_idx in range(len(ioi_dataset)):
            end_pos = ioi_dataset.word_idx["end"][b_idx]
            z_heads[b_idx, end_pos, top_head] = abc_means[b_idx, end_pos, top_head]
        model.layers[top_layer].self_attn.c_proj.input[:] = z_heads.reshape(z.shape)
        ablated_logits = model.logits.save()

    rprint(
        "\n".join(
            [
                f"{original_average_logit_diff_2:.4f} = Original logit diff",
                f"{per_head_logit_diffs_2[top_layer, top_head]:.4f} = Direct Logit Attribution of top name mover head",
                f"{original_average_logit_diff_2 - per_head_logit_diffs_2[top_layer, top_head]:.4f} = Naive prediction of post ablation logit diff",
                f"{logits_to_ave_logit_diff_2(ablated_logits):.4f} = Logit diff after ablating L{top_layer}H{top_head}",
            ]
        )
    )


# %%

# === Backup Head Analysis: Compare pre/post ablation per-head logit diffs ===

if MAIN:
    # Re-compute per-head logit diffs after ablation
    per_head_ablated_logit_diffs = t.zeros(N_LAYERS, N_HEADS, device=device)

    for layer in tqdm(range(N_LAYERS), desc="Per-head logit diffs (ablated)"):
        # We need to run the model with the ablation hook active and get per-head outputs
        # Since we can't easily compose hooks, we'll rerun the ablated model and collect outputs
        with model.trace(ioi_dataset.toks):
            # Apply ablation to top head
            z_top = model.layers[top_layer].self_attn.c_proj.input
            z_top_heads = z_top.reshape(z_top.shape[0], z_top.shape[1], N_HEADS, D_HEAD)
            for b_idx in range(len(ioi_dataset)):
                end_pos = ioi_dataset.word_idx["end"][b_idx]
                z_top_heads[b_idx, end_pos, top_head] = abc_means[b_idx, end_pos, top_head]
            model.layers[top_layer].self_attn.c_proj.input[:] = z_top_heads.reshape(z_top.shape)

            # Now collect the c_proj input for the current layer
            z_current = model.layers[layer].self_attn.c_proj.input.save()

        z_reshaped = z_current.view(z_current.shape[0], z_current.shape[1], N_HEADS, D_HEAD)
        W_O_layer = get_W_O(model, layer)

        for head in range(N_HEADS):
            head_out = einops.einsum(
                z_reshaped[:, :, head, :], W_O_layer[head],
                "batch seq d_head, d_head d_model -> batch seq d_model"
            )
            end_positions = ioi_dataset.word_idx["end"]
            head_out_end = head_out[t.arange(len(ioi_dataset)), end_positions]
            per_head_ablated_logit_diffs[layer, head] = residual_stack_to_logit_diff(
                head_out_end.unsqueeze(0), logit_diff_directions_2
            ).item()

    imshow(
        t.stack(
            [
                per_head_logit_diffs_2,
                per_head_ablated_logit_diffs,
                per_head_ablated_logit_diffs - per_head_logit_diffs_2,
            ]
        ),
        title="Direct logit contribution by head, pre / post ablation",
        labels={"x": "Head", "y": "Layer"},
        facet_col=0,
        facet_labels=["No ablation", f"{top_layer}.{top_head} is ablated", "Change in head contribution post-ablation"],
        width=1200,
    )

    head_labels = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]
    scatter(
        y=per_head_logit_diffs_2.flatten(),
        x=per_head_ablated_logit_diffs.flatten(),
        hover_name=head_labels,
        range_x=(-1, 1),
        range_y=(-2, 2),
        labels={"x": "Ablated", "y": "Original"},
        title="Original vs Post-Ablation Direct Logit Attribution of Heads",
        width=600,
        add_line="y=x",
    )


# %%

# === S2 Inhibition Head Analysis: Token vs Position Signal ===

if MAIN:
    datasets: list[tuple[tuple, str, IOIDataset]] = [
        ((0, 0), "original", ioi_dataset),
        ((1, 0), "random token", ioi_dataset.gen_flipped_prompts("ABB->CDD, BAB->DCD")),
        ((2, 0), "inverted token", ioi_dataset.gen_flipped_prompts("ABB->BAA, BAB->ABA")),
        ((0, 1), "inverted position", ioi_dataset.gen_flipped_prompts("ABB->BAB, BAB->ABB")),
        (
            (1, 1),
            "inverted position, random token",
            ioi_dataset.gen_flipped_prompts("ABB->DCD, BAB->CDD"),
        ),
        (
            (2, 1),
            "inverted position, inverted token",
            ioi_dataset.gen_flipped_prompts("ABB->ABA, BAB->BAA"),
        ),
    ]

    results_s2 = t.zeros(3, 2).to(device)

    s2_inhibition_heads = CIRCUIT["s2 inhibition"]

    for (row, col), desc, dataset in datasets:
        # Get clean z values from the modified dataset
        modified_z_cache = {}
        for layer in set(l for l, h in s2_inhibition_heads):
            with model.trace(dataset.toks):
                modified_z_cache[layer] = model.layers[layer].self_attn.c_proj.input.save()

        # Run on IOI dataset, patching S-inhibition heads from modified dataset
        with model.trace(ioi_dataset.toks):
            for layer in set(l for l, h in s2_inhibition_heads):
                z = model.layers[layer].self_attn.c_proj.input
                z_heads = z.reshape(z.shape[0], z.shape[1], N_HEADS, D_HEAD)
                mod_z_heads = modified_z_cache[layer].reshape(z.shape[0], z.shape[1], N_HEADS, D_HEAD)
                heads_to_patch = [h for l, h in s2_inhibition_heads if l == layer]
                for h in heads_to_patch:
                    z_heads[:, :, h] = mod_z_heads[:, :, h]
                model.layers[layer].self_attn.c_proj.input[:] = z_heads.reshape(z.shape)
            patched_logits = model.logits.save()

        results_s2[row, col] = logits_to_ave_logit_diff_2(patched_logits, ioi_dataset)

    imshow(
        results_s2,
        labels={"x": "Positional signal", "y": "Token signal"},
        x=["Original", "Inverted"],
        y=["Original", "Random", "Inverted"],
        title="Logit diff after changing all S2 inhibition heads' output signals via patching",
        text_auto=".2f",
        width=700,
    )


# %%

# === S2 Inhibition: Per-Head Token vs Position ===

if MAIN:
    results_per_s2_head = t.zeros(len(CIRCUIT["s2 inhibition"]), 3, 2).to(device)

    for i, (s2_layer, s2_head) in enumerate(CIRCUIT["s2 inhibition"]):
        for (row, col), desc, dataset in datasets:
            # Get z values from modified dataset for this layer
            with model.trace(dataset.toks):
                modified_z = model.layers[s2_layer].self_attn.c_proj.input.save()

            # Patch just this one head
            with model.trace(ioi_dataset.toks):
                z = model.layers[s2_layer].self_attn.c_proj.input
                z_heads = z.reshape(z.shape[0], z.shape[1], N_HEADS, D_HEAD)
                mod_z = modified_z.reshape(z.shape[0], z.shape[1], N_HEADS, D_HEAD)
                z_heads[:, :, s2_head] = mod_z[:, :, s2_head]
                model.layers[s2_layer].self_attn.c_proj.input[:] = z_heads.reshape(z.shape)
                patched_logits = model.logits.save()

            results_per_s2_head[i, row, col] = logits_to_ave_logit_diff_2(patched_logits, ioi_dataset)

    imshow(
        (results_per_s2_head - results_per_s2_head[0, 0, 0]) / results_per_s2_head[0, 0, 0],
        labels={"x": "Positional signal", "y": "Token signal"},
        x=["Original", "Inverted"],
        y=["Original", "Random", "Inverted"],
        title="Logit diff after patching individual S2 inhibition heads (as proportion of clean logit diff)",
        facet_col=0,
        facet_labels=[f"{layer}.{head}" for (layer, head) in CIRCUIT["s2 inhibition"]],
        facet_col_spacing=0.08,
        width=1100,
        text_auto=".2f",
    )

# %%
