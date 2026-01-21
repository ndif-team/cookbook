# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains implementations/replications of mechanistic interpretability papers. The project uses NDIF (National Deep Inference Fabric) infrastructure for remote GPU execution.

## Required Tools

- **nnsight**: Library for interpreting and manipulating internal states of deep learning models via deferred execution tracing
- **nnterp**: Higher-level interpretability utilities (logit lens, attention_probabilities, etc.) - bundled with nnsight
- **NDIF API key**: Required for remote execution - sign up at https://login.ndif.us/

## Repository Structure

```
replications/
├── paper_name/
│   ├── README.md           # MMVE results, setup instructions, link to Colab
│   ├── paper_name.ipynb    # Main Jupyter notebook
│   ├── paper_name.py       # Core script
│   ├── utils.py            # Helper functions (if needed)
│   ├── config.json         # Experiment hyperparameters
│   ├── figures/            # Generated plots/graphs
│   └── results/            # Outputs and logs
```

## Implementation Requirements

When implementing paper replications:

1. **Use nnsight/nnterp exclusively** for all model interventions and interpretability methods
2. **Use NDIF remote execution** - notebooks must run in Google Colab on free CPU mode
3. **Document scope clearly** - which experiments/figures/claims are implemented vs not implemented (with reasons)
4. **No hardcoded paths or credentials** - use fixed seeds for reproducibility
5. **Save all artifacts** - figures, tables, logs, serialized results
6. **Include comparison tables** showing replicated vs original paper results
7. **Explain deviations** - different models, datasets, or approaches must be justified

## Key nnsight Patterns

### Basic tracing
```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

with model.trace("input text"):
    hidden_states = model.transformer.h[-1].output[0].save()
```

### Remote execution (NDIF)
```python
with model.trace("input", remote=True):
    output = model.lm_head.output.save()
```

### Cross-invoke activation patching
```python
with model.trace() as tracer:
    with tracer.invoke("clean prompt"):
        clean_hs = model.transformer.h[5].output[0][:, -1, :].save()
    with tracer.invoke("corrupt prompt"):
        model.transformer.h[5].output[0][:, -1, :] = clean_hs
        patched_logits = model.lm_head.output.save()
```

### Critical gotchas
- Always call `.save()` on values you need to access after the trace
- Access modules in forward-pass execution order within a single invoke
- Use `tracer.invoke()` with no arguments to operate on the entire batch
- For generation, code after unbounded `tracer.iter[:]` blocks never executes - use separate invokes

## References

- [nnsight documentation](https://www.nnsight.net)
- [NDIF forum](https://discuss.ndif.us/)
- See `llms.md` for comprehensive nnsight API guide
- See `NNsight.md` for deep technical architecture documentation
