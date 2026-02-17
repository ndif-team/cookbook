# ARENA 1.2: Intro to Mechanistic Interpretability — nnsight Implementation

Replication of [ARENA 1.2: Intro to Mech Interp](https://arena3-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp) exercises using nnsight and NDIF for remote execution.

## Files

```
arena_1_2/
├── README.md                              # This file
├── arena_nnsight_1_2.ipynb                # Solutions notebook (with explanations)
├── arena_nnsight_1_2_exercises.ipynb      # Exercises notebook (stubs only)
```

- **Solutions notebook**: Full implementations with ARENA-style explanatory text, learning objectives, hints, and solutions.
- **Exercises notebook**: Same structure and explanations, but exercise code cells contain only function stubs (`def fn(): """docstring""" pass`) for students to fill in.

## Implemented Scope

### Implemented
- All ARENA 1.2 exercises reimplemented with nnsight
- **Section 1 — Introduction to nnsight**: Loading models, tokenization, caching activations, visualising attention heads
- **Section 2 — Finding Induction Heads**: Attention pattern detectors, repeated token experiments, induction head detection
- **Section 3 — nnsight Interventions**: Induction scores via trace context, direct logit attribution, zero and mean ablation
- **Section 4 — Reverse-engineering Induction Circuits**: OV copying circuit, QK prev-token circuit, K-composition analysis, composition scores, targeted ablation

### Core Results Reproduced
| Metric | Original (TransformerLens) | This Implementation (nnsight) | Match |
|--------|---------------------------|-------------------------------|-------|
| Previous-token head identified | L0H7 | L0H7 | Yes |
| Induction head identified | L1H4 | L1H4 | Yes |
| Baseline induction score | ~0.55-0.60 | 0.560 | Yes |
| L0H7 ablation impact | Large negative | -0.539 | Yes |
| K-composition pathway | L0H7 → L1H4 | L0H7 → L1H4 | Yes |

### Not Implemented
- None — this is a complete implementation of ARENA 1.2

## Setup

### Requirements
- Python 3.10+
- NDIF API key (sign up at https://login.ndif.us/)
- Google Colab (recommended) or local Jupyter environment

### Installation
```bash
pip install nnsight einops circuitsvis plotly jaxtyping huggingface_hub transformers eindex-callum torch
```

### Configuration
Replace the API key placeholder in the setup cell with your NDIF key:
```python
CONFIG.set_default_api_key("YOUR_NDIF_API_KEY")
```

## Usage

1. Open either notebook in Google Colab or Jupyter
2. Set your NDIF API key in the setup cell
3. Run all cells sequentially
4. The notebook uses NDIF remote execution — no local GPU required
5. All visualizations render inline using circuitsvis and plotly

## Model & Dataset

- **Model**: `attn-only-2l` (2-layer attention-only transformer with shortformer positional embeddings)
- **Dataset**: Synthetic repeated token sequences for induction testing
  - Format: `[A] [B] ... [A]` → predict `[B]`
  - 50 sequences, each 61 tokens long
- **Model matches original**: Yes — same architecture as ARENA exercises

## Key nnsight Patterns

This implementation required several nnsight-specific patterns not present in the original TransformerLens version:

1. **Variable scope**: Variables must be initialized *before* `with model.trace()` blocks
2. **Forward-order access**: Module access must follow forward-pass order
3. **NDIF remote execution**: All model operations run on remote GPUs via NDIF API

## Deviations from Original

None — this is a faithful reimplementation. The only differences are:
- Library: nnsight (with NDIF remote execution) vs TransformerLens (local execution)
- Syntax: `model.trace()` vs `model.run_with_cache()`, `.output.save()` vs hooks
