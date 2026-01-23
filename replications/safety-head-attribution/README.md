# Safety Head Attribution with nnsight

An nnsight reimplementation of "On the Role of Attention Heads in Large Language Model Safety" ([arXiv:2410.13708](https://arxiv.org/abs/2410.13708)).

## Overview

This notebook demonstrates how to identify and ablate safety-critical attention heads in large language models using [nnsight](https://nnsight.net/) and [NDIF](https://ndif.us/) remote execution.

**Key finding from the paper:** Ablating a single attention head (0.006% of parameters) allows aligned models to respond to 16x more harmful queries.

## Methods Implemented

1. **SHIPS** (Safety Head ImPortant Score) - Query-level safety head identification via KL divergence
2. **Sahara** (Safety Attention Head AttRibution Algorithm) - Dataset-level attribution via hidden state subspace analysis
3. **Surgery** - Model ablation and safety evaluation

## Quick Start

### Prerequisites

1. **NDIF API Key**: Get your free API key from https://login.ndif.us/
2. **HuggingFace Account**: Accept the Llama 2 license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
3. **HuggingFace Token**: Get your token from https://huggingface.co/settings/tokens

### Running the Notebook

1. Open `demo.ipynb` in Jupyter or Google Colab
2. In Cell 2, replace `YOUR_NDIF_API_KEY_HERE` with your NDIF API key
3. In Cell 3, replace `YOUR_HF_TOKEN_HERE` with your HuggingFace token
4. Run all cells in order

### Quick Mode vs Full Mode

Set `QUICK_MODE = True` in Cell 4 for fast testing, or `False` for full analysis.

| Component | QUICK_MODE=True | QUICK_MODE=False |
|-----------|-----------------|------------------|
| SHIPS | ~64 heads (sampled) | All 1024 heads |
| Sahara | 2 queries, ~16 heads | 100 queries, all heads |
| Evaluation | 5 queries, 32 tokens | 100 queries, 64 tokens |

## Files

- `demo.ipynb` - Main notebook with all three methods
- `maliciousinstruct.csv` - Dataset of 100 harmful queries for evaluation
- Additional datasets: `jailbreakbench.csv`, `advbench.csv`, `harmful_behaviors.csv`, etc.

## Output Files

After running the notebook, the following data files are saved:
- `ships_data.pt` - SHIPS scores for all tested heads
- `sahara_data.pt` - Sahara subspace shift scores
- `eval_data.pt` - Evaluation results (baseline vs ablated)
- `experiment_data.pt` - Compiled data from all experiments

These files serve as proof of a complete experimental run and can be loaded for further analysis.

## Model

This implementation uses **Llama-2-7b-chat-hf** (32 layers x 32 heads = 1024 attention heads), the same model as the original paper.

## References

- **Paper**: [On the Role of Attention Heads in Large Language Model Safety](https://arxiv.org/abs/2410.13708)
- **Original Repository**: https://github.com/ydyjya/safetyheadattribution
- **nnsight**: https://nnsight.net/
- **NDIF**: https://ndif.us/
