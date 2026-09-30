# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains implementations/replications of mechanistic interpretability papers. The project uses NDIF (National Deep Inference Fabric) infrastructure for remote GPU execution.

## Required Tools

Target **nnsight 0.8.0rc1** and **transformers 5.x**. Install `nnsight==0.8.0rc1` explicitly; a plain stable install currently selects 0.7.0. Use `TransformersModel` for HuggingFace pipelines; `LanguageModel` is a deprecated alias.

- **nnsight**: Library for interpreting and manipulating internal states of deep learning models via deferred execution with cooperative greenlets
- **nnterp**: Higher-level interpretability utilities (logit lens, attention_probabilities, etc.) - released nnterp 1.2/1.3 requires the course `arena-ch1/nnterp_compat.py` adapter with nnsight 0.8; import `StandardizedTransformer` from it
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
2. **Use NDIF remote execution** for paper replications when a compatible deployment is available - notebooks should run in Google Colab on free CPU mode. During the 0.8 prerelease, NDIF remains on 0.7; validate 0.8 locally or with `remote="local"` and document this temporary limitation. ARENA notebooks use local CPU/GPU execution.
3. **Document scope clearly** - which experiments/figures/claims are implemented vs not implemented (with reasons)
4. **No hardcoded paths or credentials** - use fixed seeds for reproducibility
5. **Save all artifacts** - figures, tables, logs, serialized results
6. **Include comparison tables** showing replicated vs original paper results
7. **Explain deviations** - different models, datasets, or approaches must be justified

## Key nnsight Patterns

### Basic tracing
```python
from nnsight import TransformersModel
model = TransformersModel("openai-community/gpt2", task="text-generation", device_map="auto", dispatch=True)

with model.trace("input text"):
    hidden_states = model.transformer.h[-1].output.save()
```

### Remote serialization check (offline)
```python
# Use remote=True only on an NDIF deployment compatible with 0.8.
with model.trace("input", remote="local"):
    output = model.lm_head.output.save()
```

### Cross-invoke activation patching
```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("clean prompt"):
        clean_hs = model.transformer.h[5].output[:, -1, :].save()
        barrier()
    with tracer.invoke("corrupt prompt"):
        barrier()
        model.transformer.h[5].output[:, -1, :] = clean_hs
        patched_logits = model.lm_head.output.save()
```

### Critical gotchas
- Always call `.save()` on values you need to access after the trace
- Access modules in forward-pass execution order within a single invoke
- Use `tracer.invoke()` with no arguments to operate on the entire batch
- For generation, code after unbounded `tracer.iter[:]` loops never executes - use a separate empty invoke for the final result, or a bounded loop with `min_new_tokens` matching the bound

## References

- [nnsight documentation](https://www.nnsight.net)
- [NDIF forum](https://discuss.ndif.us/)
- See `llms.md` for comprehensive nnsight API guide
- See `NNsight.md` for deep technical architecture documentation
