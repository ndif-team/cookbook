# cookbook

Replications of mechanistic interpretability papers using [nnsight](https://nnsight.net) and [NDIF](https://ndif.us) for remote GPU execution.

**Status:** Replications are in progress. Each implementation lives in `replications/paper_name/` with a notebook that runs in Google Colab.

## Contributing

Claim a paper from the spreadsheet, implement it using nnterp/nnsight, and submit a PR. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full process and requirements.

## Getting Started

```bash
# NNsight 0.8 prerelease; Python 3.10–3.14
pip install nnsight==0.8.0rc1 accelerate
```

NNsight 0.8 is currently a prerelease. Use `TransformersModel` for HuggingFace models. Local execution and `remote="local"` serialization checks are available now; NDIF remote execution remains on 0.7 until its 0.8 rollout. Sign up for an NDIF API key at https://login.ndif.us/ for compatible remote deployments.

Released nnterp 1.2/1.3 needs the course adapter on NNsight 0.8; installing nnterp alone does not make it compatible. The [ARENA chapter 1 notebooks](arena-ch1/README.md) use local CPU/GPU execution; follow their requirements and import `StandardizedTransformer` from `nnterp_compat`.

## Documentation

- [llms.md](llms.md) - nnsight API guide for AI agents
- [NNsight.md](NNsight.md) - Deep technical architecture docs
- [nnsight.net](https://nnsight.net) - Official documentation

## Checking documentation examples

Install `tests/requirements-markdown.txt`, then run `python -m pytest tests/test_markdown_examples.py`. The checks execute each Python and shell block with its stated prerequisites and report illustrative output separately. NDIF client examples use an isolated HTTP/websocket fixture with real serialization and local inference; this does not validate the live service. The eight vLLM blocks require [the separate GPU validation notebook](tests/nnsight_080_markdown_gpu_checks.ipynb) and matching CUDA wheels.
