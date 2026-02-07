# ARENA 1.2: Induction Heads - nnsight Implementation

Replication of [ARENA 1.2: Induction Heads](https://arena3-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp) exercises using nnsight and NDIF for remote execution.

## Implemented Scope

### ✅ Implemented
- **Exercise 1-8**: Reverse-engineering induction circuits in attention-only transformers
- **Attention pattern visualization**: Previous-token head (L0H7) and induction head (L1H4) patterns
- **Circuit analysis**:
  - OV circuit (what heads write to residual stream)
  - QK circuit (what patterns heads attend to)
  - K-composition analysis (L0→L1 information flow)
- **Ablation studies**:
  - Mean ablation across all heads
  - Targeted ablation to identify critical heads
- **Key figures**:
  - Attention pattern heatmaps for both layers
  - Logit difference plots showing induction behavior
  - Full QK circuit visualization (L0 → L1 composition)

### Core Results Reproduced
- Identified **L0H7** as the critical previous-token head
- Identified **L1H4** as the induction head
- Demonstrated K-composition pathway: L0H7 → L1H4
- Quantified impact: Ablating L0H7 reduces induction score by **-0.539** (largest negative change)

### ⚠️ Not Implemented
- None - this is a complete implementation of ARENA 1.2 exercises

## Setup

### Requirements
- Python 3.10+
- NDIF API key (sign up at https://login.ndif.us/)
- Google Colab (recommended) or local Jupyter environment

### Installation
```bash
pip install nnsight einops circuitsvis plotly jaxtyping huggingface_hub transformers eindex-callum torch nbformat
```

### Configuration
Replace the API key in Cell 2 with your NDIF key:
```python
CONFIG.set_default_api_key("your-api-key-here")
```

## Usage

1. Open the notebook in Google Colab or Jupyter
2. Run all cells sequentially
3. The notebook uses NDIF remote execution - no local GPU required
4. All visualizations render inline using circuitsvis and plotly

**Colab Link**: [View-only run through](#) _(to be added)_

## Model & Dataset

- **Model**: `attn-only-2l` (2-layer attention-only transformer with shortformer positional embeddings)
- **Dataset**: Synthetic repeated token sequences for induction testing
  - Format: `[A] [B] ... [A]` → predict `[B]`
  - 50 sequences, each 61 tokens long
- **Model matches original**: ✅ Yes - same architecture as ARENA exercises

## Key Implementation Details

### nnsight-specific patterns
This implementation required several nnsight-specific patterns not present in the original TransformerLens version:

1. **Variable scope quirk**: Variables must be initialized *before* `with model.trace()` blocks
   ```python
   patterns = {}  # Initialize outside
   with model.trace(tokens):
       patterns[0] = model.blocks[0].attn.hook_pattern.output.save()
   ```

2. **Forward-order access**: Module access must follow forward-pass order (embed → blocks[0] → blocks[1] → output)

3. **NDIF remote execution**: All model operations run on remote GPUs via NDIF API

### Ablation methodology
- **Mean ablation**: Replace attention pattern with mean across all positions
- **Targeted ablation**: Systematically ablate each head individually to identify critical components

## Results Comparison

| Metric | Original (TransformerLens) | This Implementation (nnsight) | Match |
|--------|---------------------------|-------------------------------|-------|
| Previous-token head identified | L0H7 | L0H7 | ✅ |
| Induction head identified | L1H4 | L1H4 | ✅ |
| Baseline induction score | ~0.55-0.60 | 0.560 | ✅ |
| L0H7 ablation impact | Large negative | -0.539 | ✅ |
| K-composition pathway | L0H7 → L1H4 | L0H7 → L1H4 | ✅ |

## Deviations from Original

**None** - This is a faithful reimplementation using nnsight's API. All core results, methodologies, and conclusions match the original ARENA 1.2 exercises.

The main differences are:
- Library choice: nnsight (with NDIF remote execution) vs TransformerLens (local execution)
- Syntax differences due to different APIs (e.g., `model.trace()` vs `model.run_with_cache()`)
- Variable scoping patterns specific to nnsight's deferred execution model

## Files

```
arena_1_2/
├── README.md           # This file
├── arena_1_2.ipynb     # Complete notebook implementation
```

## Notes

This implementation demonstrates:
- ✅ Uses nnsight for all model interactions
- ✅ Uses NDIF remote execution (no local GPU required)
- ✅ Fixed seeds for reproducibility
- ✅ All cells run without errors
- ✅ Comprehensive inline documentation
- ✅ Visualizations saved and rendered inline
