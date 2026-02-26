"""
Tests for ARENA 1.4.2 - Balanced Bracket Classifier (NNsight version)
"""

import json
import sys
from pathlib import Path
from typing import Callable

import einops
import torch as t
from jaxtyping import Float
from nnterp import StandardizedTransformer
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brackets_datasets import BracketsDataset, SimpleTokenizer

device = t.device("cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu")
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"


def test_get_post_final_ln_dir(get_post_final_ln_dir: Callable, model: StandardizedTransformer):
    """Test that get_post_final_ln_dir returns W_U[:, 0] - W_U[:, 1]."""
    import solutions

    actual = get_post_final_ln_dir(model)
    expected = solutions.get_post_final_ln_dir(model)
    t.testing.assert_close(actual, expected, atol=1e-4, rtol=0.0)
    print("All tests in `test_get_post_final_ln_dir` passed!")


def test_get_ln_fit(get_ln_fit: Callable, model: StandardizedTransformer, data: BracketsDataset):
    """Test that layernorm fitting produces correct linear regression coefficients."""
    import solutions

    for seq_pos in [1, 0, None]:
        fit_actual = get_ln_fit(model, data, "ln_final", seq_pos)
        fit_expected = solutions.get_ln_fit(model, data, "ln_final", seq_pos)
        t.testing.assert_close(
            t.from_numpy(fit_actual[0].coef_),
            t.from_numpy(fit_expected[0].coef_),
            atol=1e-4,
            rtol=0.0,
        )

    print("All tests in `test_get_ln_fit` passed!")


def test_get_pre_final_ln_dir(get_pre_final_ln_dir: Callable, model: StandardizedTransformer, data: BracketsDataset):
    """Test that the pre-final-LN unbalanced direction is correct."""
    import solutions

    dir_actual = get_pre_final_ln_dir(model, data)
    dir_expected = solutions.get_pre_final_ln_dir(model, data)
    t.testing.assert_close(dir_actual, dir_expected, atol=1e-4, rtol=0.0)
    print("All tests in `test_get_pre_final_ln_dir` passed!")


def test_get_out_by_components(get_out_by_components: Callable, model: StandardizedTransformer, data: BracketsDataset):
    """Test that output-by-components sums correctly."""
    import solutions

    dir_actual = get_out_by_components(model, data)
    dir_expected = solutions.get_out_by_components(model, data)
    t.testing.assert_close(dir_actual.sum(1), dir_expected.sum(1), atol=1e-4, rtol=0.0)
    print("All tests in `test_get_out_by_components` passed!")


def test_out_by_component_in_unbalanced_dir(
    out_by_component_in_unbalanced_dir: Float[Tensor, "comp batch"],
    model: StandardizedTransformer,
    data: BracketsDataset,
):
    """Test the per-component projection onto the unbalanced direction."""
    import solutions

    out_by_components_seq0 = solutions.get_out_by_components(model, data)[:, :, 0, :]
    pre_final_ln_dir = solutions.get_pre_final_ln_dir(model, data)

    expected = einops.einsum(
        out_by_components_seq0,
        pre_final_ln_dir,
        "comp batch d_model, d_model -> comp batch",
    )
    expected -= expected[:, data.isbal].mean(dim=1).unsqueeze(1)

    t.testing.assert_close(
        out_by_component_in_unbalanced_dir, expected, atol=1e-4, rtol=0.0
    )
    print("All tests in `test_out_by_component_in_unbalanced_dir` passed!")


def test_total_elevation_and_negative_failures(
    data: BracketsDataset,
    total_elevation_failure: Float[Tensor, "batch"],
    negative_failure: Float[Tensor, "batch"],
):
    """Test the vectorized failure-type classification."""
    import solutions

    tef_expected, nf_expected = solutions.is_balanced_vectorized_return_both(data.toks)

    t.testing.assert_close(negative_failure, nf_expected)
    t.testing.assert_close(
        total_elevation_failure,
        tef_expected,
        msg="total_elevation_failure is not correct. Did you remember to read the sequence from right to left?",
    )
    print("All tests in `test_total_elevation_and_negative_failures` passed!")


def test_get_attn_probs(get_attn_probs: Callable, model: StandardizedTransformer, data: BracketsDataset):
    """Test attention probability extraction."""
    import solutions

    probs_actual = get_attn_probs(model, data, 2, 0)
    probs_expected = solutions.get_attn_probs(model, data, 2, 0)
    t.testing.assert_close(probs_actual, probs_expected, atol=1e-4, rtol=0.0)
    print("All tests in `test_get_attn_probs` passed!")


def test_get_WOV(get_WOV: Callable, model: StandardizedTransformer):
    """Test W_OV matrix computation."""
    W_OV_00 = get_WOV(model, 0, 0)
    W_V = model._model.blocks[0].attn.W_V[0]
    W_O = model._model.blocks[0].attn.W_O[0]
    W_OV_00_expected = W_V @ W_O

    t.testing.assert_close(W_OV_00, W_OV_00_expected, atol=1e-4, rtol=0.0)
    print("All tests in `test_get_WOV` passed!")


def test_get_pre_20_dir(get_pre_20_dir: Callable, model: StandardizedTransformer, data: BracketsDataset):
    """Test the pre-head-2.0 unbalanced direction."""
    import solutions

    dir_actual = get_pre_20_dir(model, data)
    dir_expected = solutions.get_pre_20_dir(model, data)
    t.testing.assert_close(dir_actual, dir_expected, atol=1e-4, rtol=0.0)
    print("All tests in `test_get_pre_20_dir` passed!")


def test_out_by_component_in_pre_20_unbalanced_dir(
    out_by_component_in_pre_20_unbalanced_dir: Float[Tensor, "comp batch"],
    model: StandardizedTransformer,
    data: BracketsDataset,
):
    """Test component projections onto the pre-2.0 unbalanced direction."""
    import solutions

    out_by_components_seq1 = solutions.get_out_by_components(model, data)[:, :, 1, :]
    pre_layer2_outputs_seqpos1 = out_by_components_seq1[:-3]
    expected = einops.einsum(
        pre_layer2_outputs_seqpos1,
        solutions.get_pre_20_dir(model, data),
        "comp batch emb, emb -> comp batch",
    )
    expected -= expected[:, data.isbal].mean(-1, keepdim=True)

    t.testing.assert_close(
        out_by_component_in_pre_20_unbalanced_dir, expected, atol=1e-4, rtol=0.0
    )
    print("All tests in `test_out_by_component_in_pre_20_unbalanced_dir` passed!")


def test_get_out_by_neuron(get_out_by_neuron: Callable, model: StandardizedTransformer, data: BracketsDataset):
    """Test per-neuron output computation."""
    import solutions

    out = get_out_by_neuron(model, data, layer=0, seq=1)
    out_expected = solutions.get_out_by_neuron(model, data, layer=0, seq=1)
    t.testing.assert_close(out, out_expected, atol=1e-4, rtol=0.0)
    print("All tests in `test_get_out_by_neuron` passed!")


def test_get_out_by_neuron_in_20_dir(
    get_out_by_neuron_in_20_dir: Callable, model: StandardizedTransformer, data: BracketsDataset
):
    """Test neuron projections onto the pre-2.0 unbalanced direction."""
    import solutions

    out = get_out_by_neuron_in_20_dir(model, data, layer=0)
    out_expected = solutions.get_out_by_neuron_in_20_dir(model, data, layer=0)
    t.testing.assert_close(out, out_expected, atol=1e-4, rtol=0.0)
    print("All tests in `test_get_out_by_neuron_in_20_dir` passed!")


def test_get_out_by_neuron_in_20_dir_less_memory(
    get_out_by_neuron_in_20_dir_less_memory: Callable,
    model: StandardizedTransformer,
    data: BracketsDataset,
):
    """Test memory-efficient neuron projection computation."""
    import solutions

    out = get_out_by_neuron_in_20_dir_less_memory(model, data, layer=0)
    out_expected = solutions.get_out_by_neuron_in_20_dir_less_memory(model, data, layer=0)
    t.testing.assert_close(out, out_expected, atol=1e-4, rtol=0.0)
    print("All tests in `test_get_out_by_neuron_in_20_dir_less_memory` passed!")


def test_get_q_and_k_for_given_input(get_q_and_k_for_given_input, model, tokenizer):
    """Test query and key extraction for a given input."""
    import solutions

    parens = "()"
    q, k = get_q_and_k_for_given_input(model, tokenizer, parens, 0)
    q_expected, k_expected = solutions.get_q_and_k_for_given_input(model, tokenizer, parens, 0)
    t.testing.assert_close(q, q_expected, atol=1e-4, rtol=0.0)
    t.testing.assert_close(k, k_expected, atol=1e-4, rtol=0.0)
    print("All tests in `test_get_q_and_k_for_given_input` passed!")


# ── Run all tests ──

if MAIN:
    from solutions import (
        BracketClassifier,
        get_attn_probs,
        get_ln_fit,
        get_out_by_components,
        get_out_by_neuron,
        get_out_by_neuron_in_20_dir,
        get_out_by_neuron_in_20_dir_less_memory,
        get_post_final_ln_dir,
        get_pre_20_dir,
        get_pre_final_ln_dir,
        get_q_and_k_for_given_input,
        get_WOV,
        is_balanced_vectorized_return_both,
        run_model,
        wrap_bracket_model,
    )

    section_dir = Path(__file__).resolve().parent

    # Load model
    raw_model = BracketClassifier().to(device)
    state_dict = t.load(section_dir / "bracket_classifier_converted.pt", map_location=device, weights_only=True)
    raw_model.load_state_dict(state_dict)
    raw_model.eval()
    model = wrap_bracket_model(raw_model)

    tokenizer = SimpleTokenizer("()")

    N_SAMPLES = 5000
    with open(section_dir / "brackets_data.json") as f:
        data_tuples = json.load(f)[:N_SAMPLES]
    data = BracketsDataset(data_tuples).to(device)
    data_mini = BracketsDataset(data_tuples[:100]).to(device)

    print("Running all tests...")
    print()

    test_get_post_final_ln_dir(get_post_final_ln_dir, model)
    test_get_ln_fit(get_ln_fit, model, data_mini)
    test_get_pre_final_ln_dir(get_pre_final_ln_dir, model, data_mini)
    test_get_out_by_components(get_out_by_components, model, data_mini)
    test_get_attn_probs(get_attn_probs, model, data_mini)
    test_get_WOV(get_WOV, model)
    test_get_pre_20_dir(get_pre_20_dir, model, data_mini)
    test_get_out_by_neuron(get_out_by_neuron, model, data_mini)
    test_get_out_by_neuron_in_20_dir(get_out_by_neuron_in_20_dir, model, data_mini)
    test_get_out_by_neuron_in_20_dir_less_memory(get_out_by_neuron_in_20_dir_less_memory, model, data_mini)
    test_get_q_and_k_for_given_input(get_q_and_k_for_given_input, model, tokenizer)

    # Test failure type classification
    total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(data.toks)
    test_total_elevation_and_negative_failures(data, total_elevation_failure, negative_failure)

    print()
    print("All tests passed!")
