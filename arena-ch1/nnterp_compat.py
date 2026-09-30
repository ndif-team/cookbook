"""Small adapter for nnterp 1.2/1.3 and NNsight 0.8.

Import the course's StandardizedTransformer here until nnterp supports the
new public Object import, module storage, and source-operation numbering.
"""

import sys
from types import ModuleType

import torch
from nnsight import Envoy, Object
from transformers import GPT2LMHeadModel, PreTrainedModel

# nnterp still imports this old internal path for a type annotation only.
_legacy_globals = ModuleType("nnsight.intervention.tracing.globals")
_legacy_globals.Object = Object
sys.modules.setdefault(_legacy_globals.__name__, _legacy_globals)

from nnterp import StandardizedTransformer as _StandardizedTransformer
from nnterp.rename_utils import AttnProbFunction, RenameConfig


class GPT2AttentionProbFunction(AttnProbFunction):
    def get_attention_prob_source(self, attention_module, return_module_source=False):
        # The assignment to attention_interface is operation 0; its call is 1.
        source = attention_module.source.attention_interface_1.source
        if return_module_source:
            return source
        if hasattr(source, "nn_functional_dropout_0"):
            return source.nn_functional_dropout_0
        # Earlier transformers releases use the attention module's Dropout.
        return attention_module.attn_dropout


class StandardizedTransformer(_StandardizedTransformer):
    """Keep the course's nnterp interface on the NNsight 0.8 runtime."""

    def __init__(self, model, *args, **kwargs):
        if kwargs.get("enable_attention_probs") and kwargs.get("rename_config") is None:
            if isinstance(model, GPT2LMHeadModel) or (
                isinstance(model, str) and "gpt2" in model.lower()
            ):
                kwargs["rename_config"] = RenameConfig(
                    attn_prob_source=GPT2AttentionProbFunction()
                )
        super().__init__(model, *args, **kwargs)

    @property
    def _model(self):
        return self._module

    @property
    def device(self):
        return next(self._module.parameters()).device

    def _wrap(self, module, *args, **kwargs):
        self._arena_custom_model = not isinstance(module, PreTrainedModel)
        if self._arena_custom_model:
            # These models accept tensors directly and have no HF pipeline.
            return module
        if kwargs.get("attn_implementation") is not None:
            module.set_attn_implementation(kwargs["attn_implementation"])
        return super()._wrap(module, *args, **kwargs)

    def _batch(self, invokes, fn):
        if getattr(self, "_arena_custom_model", False):
            return Envoy._batch(self, invokes, fn)
        return super()._batch(invokes, fn)

    @property
    def logits(self):
        output = self.output
        return output if isinstance(output, torch.Tensor) else output.logits
