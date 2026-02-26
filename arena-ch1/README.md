# ARENA 3.0 Chapter 1: Mechanistic Interpretability (NNsight + nnterp)

This is an adaptation of [ARENA 3.0](https://arena.education) Chapter 1 (Transformer Interpretability) that replaces TransformerLens with **NNsight** and **nnterp** as the sole interpretability libraries.

**Original course material** by [Callum McDougall](https://github.com/callummcdougall/ARENA_3.0).
**Adapted for NNsight + nnterp** by the [NDIF team](https://ndif.us/).

---

## Sections

| Section | Directory | Description |
|---|---|---|
| 1.2 | `part2_intro_to_mech_interp/` | Introduction to mechanistic interpretability. NNsight fundamentals, GPT-2 induction heads, attention patterns, and direct logit attribution using raw NNsight. Concludes with an nnterp introduction. |
| 1.3.1 | `part31_superposition_and_saes/` | Toy models of superposition. Feature geometry, superposition theory, and toy model training with plain PyTorch and minimal NNsight. |
| 1.3.2 | `part32_interp_with_saes/` | Interpretability with sparse autoencoders. SAE feature analysis on language models using nnterp. |
| 1.4.1 | `part41_indirect_object_identification/` | Indirect Object Identification circuit in GPT-2. Activation patching, path patching, and circuit discovery using nnterp. |
| 1.4.2 | `part42_function_vectors_and_model_steering/` | Function vectors and model steering. Computing function vectors from in-context learning examples and steering GPT-2 behavior using nnterp. |
| 1.5.1 | `part51_balanced_bracket_classifier/` | Balanced bracket classifier. Interpretability on a custom-trained bracket classification transformer using raw NNsight. |
| 1.5.2 | `part52_grokking_and_modular_arithmetic/` | Grokking and modular arithmetic. Fourier analysis of learned representations and progress measures using raw NNsight. |
| 1.5.3 | `part53_othellogpt/` | OthelloGPT. Probing for board state representations in a model trained on Othello games using raw NNsight. |

---

## Setup

The notebooks are designed for Google Colab (free tier, T4 GPU). Each notebook includes its own setup cell, but the core dependencies are:

```bash
# NNsight (pinned version)
pip install git+https://github.com/ndif-team/nnsight.git@v0.5.16

# nnterp (installed with --no-deps due to unreleased nnsight version requirement)
pip install --no-deps nnterp

# Visualization and utility packages
pip install circuitsvis plotly ipywidgets jaxtyping einops
```

`torch` and `transformers` are pre-installed on Colab and should not be upgraded.

---

## How to Use

1. Open `exercises.ipynb` for any section in Google Colab.
2. Run the setup cell at the top to install dependencies.
3. Work through the exercises. Each exercise includes a docstring describing the expected behavior and a test cell to check your implementation.
4. If you get stuck, refer to `solutions.ipynb` in the same directory for worked answers. Standalone function implementations are also available in `solutions.py`.

---

## Why NNsight Instead of TransformerLens?

TransformerLens provides a clean API for interpretability research, but it does so by reimplementing transformer architectures from scratch. This introduces numerical divergence from the original HuggingFace models (due to features like `fold_ln`, `center_unembed`, and `center_writing_weights`) and requires manual adaptation for each new architecture.

NNsight takes a different approach: it wraps the original model without reimplementing it, preserving exact numerical behavior while providing full access to internal activations via a tracing API. nnterp builds on NNsight to provide a standardized interface across architectures (unified layer access, attention probability extraction, built-in logit lens, and more).

As noted in the nnterp paper:

> "Custom implementations like TransformerLens ensure consistent interfaces but require coding a manual adaptation for each architecture, introducing numerical mismatch with the original models, while direct HuggingFace access through NNsight preserves exact behavior."
>
> -- Dumas, C. (2025). "nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers." arXiv:2511.14465.

---

## Links

- [NNsight documentation](https://nnsight.net/)
- [nnterp documentation](https://ndif-team.github.io/nnterp/)
- [ARENA 3.0 original course](https://arena.education)
- [ARENA 3.0 GitHub repository](https://github.com/callummcdougall/ARENA_3.0)
- [NNsight paper (ICLR 2025)](https://arxiv.org/abs/2407.14561)
- [nnterp paper](https://arxiv.org/abs/2511.14465)
