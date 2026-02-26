"""
Tests for Section 1.5: Grokking & Modular Arithmetic.
Adapted from ARENA 3.0 for NNsight (no TransformerLens dependency).
"""

import torch as t
import numpy as np
import einops

device = t.device("cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu")
p = 113


def test_make_fourier_basis(make_fourier_basis):
    fourier_basis, fourier_names = make_fourier_basis(p)

    # Check shape
    assert fourier_basis.shape == (p, p), f"Expected shape ({p}, {p}), got {fourier_basis.shape}"

    # Check orthonormality
    inner_products = fourier_basis @ fourier_basis.T
    identity = t.eye(p, device=fourier_basis.device)
    assert t.allclose(inner_products, identity, atol=1e-4), "Fourier basis is not orthonormal"

    # Check naming
    assert fourier_names[0].lower() == "const"
    assert len(fourier_names) == p

    print("All tests in `test_make_fourier_basis` passed!")


def test_fft1d(fft1d):
    from part52_grokking_and_modular_arithmetic.solutions import make_fourier_basis

    fourier_basis, _ = make_fourier_basis(p)

    x = t.randn(p).to(device)
    actual = fft1d(x)
    expected = x @ fourier_basis.T
    t.testing.assert_close(actual, expected, msg="Tests failed for `fft1d` with a vector input.")
    print("Tests passed for `fft1d` with a vector input!")

    x = t.randn(3, p).to(device)
    actual = fft1d(x)
    expected = x @ fourier_basis.T
    t.testing.assert_close(actual, expected, msg="Tests failed for `fft1d` with a batch of vectors.")
    print("Tests passed for `fft1d` with a batch of vectors!")


def test_fourier_2d_basis_term(fourier_2d_basis_term):
    from part52_grokking_and_modular_arithmetic.solutions import make_fourier_basis

    fourier_basis, _ = make_fourier_basis(p)
    (i, j) = (3, 5)
    actual = fourier_2d_basis_term(i, j)
    expected = fourier_basis[i][:, None] * fourier_basis[j][None, :]

    t.testing.assert_close(actual, expected)
    print("All tests in `test_fourier_2d_basis_term` passed!")


def test_fft2d(fft2d):
    from part52_grokking_and_modular_arithmetic.solutions import make_fourier_basis

    fourier_basis, _ = make_fourier_basis(p)

    x = t.randn(p, p).to(device)
    actual = fft2d(x)
    expected = einops.einsum(x, fourier_basis, fourier_basis, "px py, i px, j py -> i j")
    t.testing.assert_close(actual, expected, msg="Tests failed for `fft2d` with a single input.")
    print("Tests passed for `fft2d` with a single input!")

    x = t.randn(p, p, 3).to(device)
    actual = fft2d(x)
    expected = einops.einsum(x, fourier_basis, fourier_basis, "px py k, i px, j py -> i j k")
    t.testing.assert_close(actual, expected, msg="Tests failed for `fft2d` with a batch of inputs.")
    print("Tests passed for `fft2d` with a batch of inputs!")


def test_project_onto_direction(project_onto_direction):
    v = t.randn(3).to(device)
    batch_vecs = t.randn(3, 4).to(device)
    actual = project_onto_direction(batch_vecs, v)

    # Manual computation
    components = einops.einsum(batch_vecs, v, "n k, n -> k")
    expected = einops.einsum(components, v, "k, n -> n k")

    t.testing.assert_close(actual, expected)
    print("All tests in `test_project_onto_direction` passed!")


def test_project_onto_frequency(project_onto_frequency):
    from part52_grokking_and_modular_arithmetic.solutions import (
        fourier_2d_basis_term,
        project_onto_direction,
    )

    freq = 7
    batch_vecs = t.randn(p * p, 4).to(device)
    actual = project_onto_frequency(batch_vecs, freq)

    # Manual computation
    expected = sum([
        project_onto_direction(batch_vecs, fourier_2d_basis_term(i, j).flatten())
        for i in [0, 2 * freq - 1, 2 * freq]
        for j in [0, 2 * freq - 1, 2 * freq]
    ])

    t.testing.assert_close(actual, expected)
    print("All tests in `test_project_onto_frequency` passed!")


def test_get_trig_sum_directions(get_trig_sum_directions):
    from part52_grokking_and_modular_arithmetic.solutions import fourier_2d_basis_term

    k = 3
    actual_cos, actual_sin = get_trig_sum_directions(k)

    cosx_cosy = fourier_2d_basis_term(2 * k - 1, 2 * k - 1)
    sinx_siny = fourier_2d_basis_term(2 * k, 2 * k)
    sinx_cosy = fourier_2d_basis_term(2 * k, 2 * k - 1)
    cosx_siny = fourier_2d_basis_term(2 * k - 1, 2 * k)

    expected_cos = (cosx_cosy - sinx_siny) / np.sqrt(2)
    expected_sin = (sinx_cosy + cosx_siny) / np.sqrt(2)

    t.testing.assert_close(actual_cos, expected_cos)
    t.testing.assert_close(actual_sin, expected_sin)
    print("All tests in `test_get_trig_sum_directions` passed!")


def test_excl_loss(excl_loss, model, key_freqs):
    excl_loss_list = excl_loss(model, key_freqs)
    t.testing.assert_close(
        excl_loss_list,
        [0.000282, 0.000633, 0.00161, 0.07195, 0.032879],
        atol=1e-4,
        rtol=1e-4,
    )
    print("All tests in `test_excl_loss` passed!")


def test_fourier_embed(fourier_embed, model):
    out = fourier_embed(model)
    t.testing.assert_close(out[:2].cpu(), t.tensor([0.1017279, 0.083272]), atol=1e-4, rtol=1e-4)
    print("All tests in `test_fourier_embed` passed!")


def test_embed_SVD(embed_SVD, model):
    S = embed_SVD(model)
    t.testing.assert_close(S[:2].cpu(), t.tensor([4.1993327, 4.0926037]), atol=1e-4, rtol=1e-4)
    print("All tests in `test_embed_SVD` passed!")
