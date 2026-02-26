"""
Tests for Section 1.2: Intro to Mech Interp.

These tests validate student implementations against golden values
computed from the solutions. NO TransformerLens dependency.
"""

import sys
from pathlib import Path
from typing import Callable

import torch as t
from jaxtyping import Float
from torch import Tensor

# Ensure parent dirs are importable
arena_ch1_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arena_ch1_dir))
sys.path.insert(0, str(arena_ch1_dir / "_conversion"))


def test_get_log_probs(get_log_probs: Callable):
    """Test that get_log_probs correctly indexes into log-softmax output."""
    logits = t.randn(2, 5, 100)
    tokens = t.randint(0, 100, (2, 5))
    result = get_log_probs(logits, tokens)
    assert result.shape == (2, 4), f"Expected shape (2, 4), got {result.shape}"

    # Verify against manual computation
    logprobs = logits.log_softmax(dim=-1)
    for b in range(2):
        for s in range(4):
            expected = logprobs[b, s, tokens[b, s + 1]]
            t.testing.assert_close(result[b, s], expected)

    print("All tests in `test_get_log_probs` passed!")


def test_logit_attribution(logit_attribution: Callable, model, tokens: Tensor):
    """Test logit attribution sums to correct token logits."""
    import part2_intro_to_mech_interp.solutions as solutions

    embed, head_results = solutions.get_attn_head_results(model, tokens)
    l1_results = head_results[0]
    l2_results = head_results[1]

    W_U = model._model.unembed.weight.T
    logit_attr = logit_attribution(embed, l1_results, l2_results, W_U, tokens.squeeze())

    # Check shape
    seq_len = tokens.shape[1]
    n_heads = model._model.n_heads
    expected_cols = 1 + 2 * n_heads
    assert logit_attr.shape == (
        seq_len - 1,
        expected_cols,
    ), f"Expected shape ({seq_len - 1}, {expected_cols}), got {logit_attr.shape}"

    # Check that sum matches correct token logits
    with model.trace(tokens):
        logits = model.lm_head.output.save()
    correct_token_logits = logits[0, t.arange(seq_len - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)

    print("All tests in `test_logit_attribution` passed!")


def test_get_ablation_scores(
    ablation_scores: Float[Tensor, "layer head"],
    model,
    rep_tokens: Float[Tensor, "batch seq"],
):
    """Test ablation scores against solutions."""
    import part2_intro_to_mech_interp.solutions as solutions

    ablation_scores_expected = solutions.get_ablation_scores(model, rep_tokens)
    t.testing.assert_close(ablation_scores, ablation_scores_expected, atol=1e-4, rtol=1e-4)

    print("All tests in `test_get_ablation_scores` passed!")


def test_full_OV_circuit(
    full_OV_circuit: Tensor, model, layer: int, head: int
):
    """Test the full OV circuit computation."""
    raw = model._model
    W_E = raw.embed.weight
    W_V = raw.blocks[layer].attn.W_V[head]
    W_O = raw.blocks[layer].attn.W_O[head]
    W_U = raw.unembed.weight.T

    OV = W_V @ W_O
    expected = W_E @ OV @ W_U

    t.testing.assert_close(
        full_OV_circuit[:20, :20], expected[:20, :20], atol=1e-4, rtol=1e-4
    )

    print("All tests in `test_full_OV_circuit` passed!")


def test_decompose_attn_scores(
    decompose_attn_scores: Callable, q: Tensor, k: Tensor, model
):
    """Test attention score decomposition."""
    import part2_intro_to_mech_interp.solutions as solutions

    decomposed_scores = decompose_attn_scores(q, k, model)
    decomposed_scores_expected = solutions.decompose_attn_scores(q, k, model)

    t.testing.assert_close(decomposed_scores, decomposed_scores_expected, atol=1e-4, rtol=1e-4)

    print("All tests in `test_decompose_attn_scores` passed!")


def test_find_K_comp_full_circuit(find_K_comp_full_circuit: Callable, model):
    """Test K-composition full circuit."""
    import part2_intro_to_mech_interp.solutions as solutions

    Q, K_T = find_K_comp_full_circuit(model, 7, 4)
    Q_exp, K_T_exp = solutions.find_K_comp_full_circuit(model, 7, 4)

    # Compare Q and K_T directly (avoid OOM from d_vocab x d_vocab matmul)
    t.testing.assert_close(Q[:20], Q_exp[:20], atol=1e-4, rtol=1e-4)
    t.testing.assert_close(K_T[:, :20], K_T_exp[:, :20], atol=1e-4, rtol=1e-4)

    print("All tests in `test_find_K_comp_full_circuit` passed!")


def test_get_comp_score(get_comp_score: Callable):
    """Test composition score computation."""
    W_A = t.rand(3, 4)
    W_B = t.rand(4, 5)

    import part2_intro_to_mech_interp.solutions as solutions

    comp_score = get_comp_score(W_A, W_B)
    comp_score_expected = solutions.get_comp_score(W_A, W_B)

    t.testing.assert_close(
        t.tensor(comp_score), t.tensor(comp_score_expected), atol=1e-4, rtol=1e-4
    )

    print("All tests in `test_get_comp_score` passed!")
