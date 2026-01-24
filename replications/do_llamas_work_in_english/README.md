# Do Llamas Work in English?

Replication of experiments from:

> **"Do Llamas Work in English? On the Latent Language of Multilingual Transformers"**
> Wendler, Veselovsky, Monea, West (2024)
> [arXiv:2402.10588](https://arxiv.org/abs/2402.10588)

## Overview

This notebook investigates whether multilingual LLMs use English as an internal "pivot language" when processing non-English text. Using the logit lens technique, we track token probabilities across all 32 layers of Llama-2-7B to observe when English vs. target language tokens become most probable.

## Setup

### Requirements
- Python 3.12+
- NDIF API key from https://login.ndif.us
- HuggingFace token for Llama-2-7B access

### Installation
```bash
pip install torch nnsight transformers pandas numpy matplotlib seaborn tqdm
```

### Running
1. Open `do_llamas_work_in_english.ipynb` in Google Colab or Jupyter
2. Add your NDIF and HuggingFace tokens in Cell 1.2
3. Run all cells in sequence

## Implemented Experiments

### Translation Task (French → Chinese)
- Few-shot translation prompts using word pairs
- Tracks English and Chinese token probabilities across layers
- Computes entropy and energy metrics

### Cloze Task (French)
- Fill-in-the-blank prompts in French
- Tracks English and French token probabilities
- Same metrics as translation experiment

## Key Findings Replicated

| Metric | Translation (fr→zh) | Cloze (fr) |
|--------|---------------------|------------|
| English peaks at | Layer 23 | Layer 23 |
| Target lang peaks at | Layer 32 | Layer 32 |
| Max English prob | ~70% | ~46% |
| Max target prob | ~63% | ~49% |

**Result:** English token probability peaks in middle layers before target language, supporting the latent language hypothesis.

## Not Implemented

- Other language pairs (de→zh, ru→zh)
- Larger models (Llama-2-13B, 70B)
- Intervention experiments (Section 5 of paper)

**Reasoning:** Focus on demonstrating core methodology with NDIF remote execution. Language pairs can be extended by modifying configuration.

## Model and Dataset

- **Model:** meta-llama/Llama-2-7B-hf via NDIF remote execution
- **Dataset:** Word pairs from paper's supplementary materials (118 words for translation, 118 for cloze)

## Deviations from Original

1. **Remote execution:** Uses NDIF instead of local GPU, requiring stack-inside-trace pattern for proper proxy handling
2. **Subset of experiments:** Only French→Chinese translation and French cloze implemented
3. **Results saving:** Added `.pt` file export for reproducibility

## Comparison to Original Results

Our replication shows:
- English peaks at layer 23 (paper: ~15-20)
- Target language peaks at layer 32 (paper: final layers)
- Qualitative pattern matches: English as pivot language confirmed

The slight difference in peak layer may be due to prompt formatting variations.

## Output Files

After running:
- `results/translation_results.pt` - Translation experiment metrics
- `results/cloze_results.pt` - Cloze experiment metrics
- `results/all_results.pt` - Combined results with metadata
