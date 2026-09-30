# llms.md - NNsight AI Agent Guide

Verified against **nnsight 0.8.0rc1** and **transformers 5.x**. Install with `pip install nnsight==0.8.0rc1` (or `pip install --pre nnsight`). The latest stable release is still 0.7.0 as of September 29, 2026. `LanguageModel` and `VisionLanguageModel` are deprecated aliases; use `TransformersModel` in new code.

Unless stated otherwise, examples use the GPT-2 `model`, imports, and `REMOTE` setting from Quick Reference. Each example starts with a fresh model; examples that demonstrate errors are checked for those errors. `REMOTE = "local"` exercises serialization offline without an API key. NDIF still runs 0.7 during the 0.8 prerelease; `remote=True`, non-blocking jobs, and async NDIF jobs need a compatible service before they can be validated live. See [the example checks](tests/test_markdown_examples.py).

This document provides comprehensive guidance for AI agents working with the `nnsight` library. NNsight enables interpreting and manipulating the internals states of deep learning models through a deferred execution tracing system.

### Related Resources

- **[NNsight.md](./NNsight.md)** - Deep technical documentation covering nnsight's internal architecture (tracing, interleaving, Envoy system, vLLM integration, etc.)
- **[Documentation](https://www.nnsight.net)** - Official docs with tutorials, guides, and API reference
- **[Forum](https://discuss.ndif.us/)** - Community forum for questions, discussions, and troubleshooting

---

## Quick Reference

```python
from nnsight import NNsight, TransformersModel
import nnsight
import torch

REMOTE = "local"  # Offline round trip; True requires an NDIF 0.8 deployment
model = TransformersModel("openai-community/gpt2", task="text-generation", device_map="cpu", dispatch=True, attn_implementation="eager")
with model.trace("input text"):
    model.transformer.h[0].output[:] = 0
    hidden_states = model.transformer.h[-1].output.save()
```

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [NNsight vs TransformersModel](#nnsight-vs-transformersmodel)
3. [Tracing Context](#tracing-context)
4. [Accessing Activations](#accessing-activations)
5. [Modifying Activations (Interventions)](#modifying-activations-interventions)
6. [Batching with Invokers](#batching-with-invokers)
7. [Multi-Token Generation](#multi-token-generation)
8. [Gradients and Backpropagation](#gradients-and-backpropagation)
9. [Conditionals and Iteration](#conditionals-and-iteration)
10. [Model Editing](#model-editing)
11. [Scanning and Validation](#scanning-and-validation)
12. [Caching Activations](#caching-activations)
13. [Source Tracing](#source-tracing)
14. [Module Skipping](#module-skipping)
15. [vLLM Integration](#vllm-integration)
16. [Remote Execution (NDIF)](#remote-execution-ndif)
17. [Sessions](#sessions)
18. [Common Patterns](#common-patterns)
19. [Critical Gotchas](#critical-gotchas)
20. [Debugging Tips](#debugging-tips)
21. [Configuration](#configuration)
22. [Other NNsight 0.8 Features](#other-nnsight-08-features)

---

## Core Concepts

### Deferred Execution Model

NNsight uses a **deferred execution** paradigm with **greenlet-based cooperative synchronization**. Here's how it works:

1. **Code extraction**: When you enter a `with model.trace(...)` block, nnsight immediately exits the block (before your code runs) and extracts/compiles your code using AST parsing
2. **Greenlet execution**: Your intervention code runs in a separate worker greenlet
3. **Value synchronization**: When your code accesses `.output` or `.input`, the greenlet **parks and waits** until that value is available from the model
4. **Hook-based injection**: The model's forward pass uses PyTorch hooks to provide values to parked greenlets
5. **Coordinated execution**: After providing a value, the main thread waits for the worker greenlet to either request another value or finish

```python
with model.trace("Hello"):
    # This code is extracted, compiled, and run in a worker greenlet
    # When we access .output, the greenlet waits until the model provides it
    hs = model.transformer.h[-1].output
    
    # .save() marks the value to persist after the context exits
    hs = hs.save()

# After exiting, hs contains the actual tensor
print(hs.shape)  # torch.Size([1, 1, 768])
```

**Key insight:** Your code runs directly. When you write `torch.sum(module.output)`, that's real PyTorch code executing in a greenlet — it waits for `module.output` to be available first.

### Greenlets and Invokers

Each **invoke** runs its intervention code in a separate worker greenlet. This is a critical architectural concept:

1. **Each invoke is a greenlet** - When you call `tracer.invoke(...)`, you're creating a worker greenlet that runs your intervention code
2. **Greenlets cooperate on one thread** - Only one greenlet executes at a time
3. **Invokes resume at requested model locations** - Invokes are batched row groups, not separate forward passes; barriers order cross-invoke dependencies
4. **Greenlets wait for values** - When your code accesses `.input`, `.output`, or `.source`, the greenlet parks until the model provides that value via hooks
5. **Access modules in execution order** - Within an invoke, you MUST access modules in forward-pass order. Requesting layer 5's output then layer 2's output raises OutOfOrderError (layer 2 already ran)

```python
with model.trace() as tracer:
    # Invoke 1: Worker greenlet for the first row group
    with tracer.invoke("Hello"):
        # Access modules in execution order ONLY
        layer_2 = model.transformer.h[2].output.save()  # Worker waits here (OK)
        layer_5 = model.transformer.h[5].output.save()  # Then waits here (OK)

    # Invoke 2: Worker greenlet for the second row group
    with tracer.invoke("World"):
        layer_0 = model.transformer.h[0].output.save()
```
/*
!IMPORTANT:
You MUST access modules (and their .output, .input, etc.) in the order they execute in the model's forward pass.
Accessing layer 5's output before layer 2 raises OutOfOrderError!
*/

### Prompt-less Invokers

Call `.invoke()` with **no arguments** to run intervention code on the **entire batch** from all previous invokes:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out_1 = model.lm_head.output[:, -1].save()  # Shape: [1, vocab]
    
    with tracer.invoke(["World", "Test"]):
        out_2 = model.lm_head.output[:, -1].save()  # Shape: [2, vocab]
    
    # No-arg invoke: operates on ALL 3 inputs batched together
    with tracer.invoke():
        out_all = model.lm_head.output[:, -1].save()  # Shape: [3, vocab]
    
    # Another no-arg invoke: same batch
    with tracer.invoke():
        out_all_2 = model.lm_head.output[:, -1].save()  # Shape: [3, vocab]

# out_all contains the same data as concatenating out_1 and out_2
```

This is useful for:
- Running different intervention logic on the same batch
- Accessing the combined batch after setting up individual invokes
- Comparing interventions across the full batch

### Key Properties

Every module wrapped by NNsight has these special properties. Accessing them causes the worker greenlet to **wait** until the value is available:

| Property | Description |
|----------|-------------|
| `.output` | The module's forward pass output (greenlet waits for hook) |
| `.input` | The first positional argument to the module |
| `.inputs` | All inputs as `(tuple(args), dict(kwargs))` |

**Note on `.grad`:** Gradients are accessed on **tensors** (not modules), and only inside a `with tensor.backward():` context. See [Gradients and Backpropagation](#gradients-and-backpropagation).

---

## NNsight vs TransformersModel

### NNsight (Base Class)

Use `NNsight` for any PyTorch model:

```python
from nnsight import NNsight
import torch

net = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.Linear(10, 2)
)

model = NNsight(net)

with model.trace(torch.rand(1, 5)):
    output = model.output.save()
```

### TransformersModel (HuggingFace Integration)

Use `TransformersModel` for HuggingFace transformers with automatic tokenization:

```python
from nnsight import TransformersModel

# Loads model + tokenizer automatically
model = TransformersModel("openai-community/gpt2", task="text-generation", device_map="auto", dispatch=True)

# Can pass strings directly - tokenization is handled
with model.trace("The Eiffel Tower is in"):
    hidden_states = model.transformer.h[-1].output.save()
```

**How TransformersModel Works:**

`TransformersModel` is backed by a HuggingFace `transformers.pipeline`. The task selects the model class and preprocessing, so the same wrapper handles text generation, fill-mask, classification, vision, and audio. This means:

1. **Keyword arguments are forwarded** - Loading options such as `dtype`, `device_map`, and `attn_implementation` are routed through the pipeline; preprocessing and generation have their own arguments
2. **Same model, enhanced interface** - The wrapped model is identical to what you'd get from `transformers`, but with NNsight's intervention capabilities added
3. **Tokenizer included** - The appropriate tokenizer is loaded automatically alongside the model

```python
# These kwargs are passed directly to AutoModelForCausalLM.from_pretrained()
model = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    device_map="auto",           # HuggingFace accelerate device mapping
    dtype=torch.float16,   # Precision setting
    trust_remote_code=True,      # For custom model code
    attn_implementation="eager",  # Attention backend (portable on CPU)
)
```

**Important `TransformersModel` parameters:**

| Parameter | Description |
|-----------|-------------|
| `device_map` | Device placement - `"auto"` distributes layers across available GPUs (and CPU if needed). Uses HuggingFace Accelerate. Other options: `"cuda"`, `"cpu"`, or a custom dict |
| `dispatch=True` | Load model weights into memory immediately. Default is lazy loading (meta tensors); the first local trace dispatches automatically |
| `dtype` | Model precision (e.g., `torch.float16`, `torch.bfloat16`). Forwarded to HuggingFace |
| `rename={...}` | Create module aliases (see [Module Renaming](#module-renaming)) |

**Note on `device_map="auto"`:** This tells HuggingFace Accelerate to automatically distribute model layers across all available GPUs. If the model doesn't fit on GPUs, it will offload to CPU. This is the recommended setting for large models.

**Wrapping pre-loaded models:**

You can wrap an existing HuggingFace model. Pass its tokenizer explicitly, or let the pipeline load it from the model's `name_or_path`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model yourself
hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Explicit tokenizer avoids relying on name_or_path inference
model = TransformersModel(hf_model, tokenizer=tokenizer)
```

A model constructed from a config may have no usable `name_or_path`; provide `tokenizer=` in that case. A pretrained GPT-2 model with its checkpoint name still attached can load the tokenizer automatically.

```
pretrained model with usable name_or_path -> tokenizer can be inferred
model without usable name_or_path -> provide tokenizer explicitly
```

---

## Tracing Context

### Basic Tracing

```python
# Single input - requires at least one positional argument
with model.trace("Hello World"):
    output = model.output.save()

# With generation (for multi-token output)
with model.generate("Hello", max_new_tokens=5, min_new_tokens=5) as tracer:
    output = tracer.result.save()
```

### Trace With vs Without Invokes

**Important:** When using `.trace()` without explicit invokes, you **must** provide at least one positional argument:

```python
# CORRECT - positional argument provided
with model.trace("Hello"):
    output = model.output.save()

# CORRECT - using explicit invokes (no arg to trace needed)
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        output = model.output.save()

# WRONG - no positional arg and no invokes
try:
    with model.trace():
        output = model.output.save()
except ValueError:
    print("Provide an input or an explicit invoke")
```

When you provide an argument to `.trace()`, an implicit invoke is created for you. This is equivalent to:

```python
# These are equivalent:
with model.trace("Hello"):
    output = model.output.save()

with model.trace() as tracer:
    with tracer.invoke("Hello"):
        output = model.output.save()
```

### Context Objects

The tracing context returns a `tracer` object with useful methods:

```python
with model.trace("Hello") as tracer:
    print("Debug:", model.transformer.h[0].output.shape)  # Print works normally
    tracer.stop()  # Early termination
```

---

## Accessing Activations

### Module Hierarchy

Print the model to see its structure:

```python
print(model)
# GPT2LMHeadModel(
#   (transformer): GPT2Model(
#     (h): ModuleList(
#       (0-11): 12 x GPT2Block(
#         (attn): GPT2Attention(...)
#         (mlp): GPT2MLP(...)
#       )
#     )
#   )
#   (lm_head): Linear(...)
# )
```

### Accessing Outputs

```python
with model.trace("Hello"):
    attn_out = model.transformer.h[0].attn.output[0].save()
    mlp_out = model.transformer.h[0].mlp.output.save()
    layer_5_out = model.transformer.h[5].output.save()
    logits = model.lm_head.output.save()
```

### Accessing Inputs

```python
with model.trace("Hello"):
    # First positional argument
    layer_input = model.transformer.h[0].input.save()
    
    # All inputs: (args_tuple, kwargs_dict)
    all_inputs = model.transformer.h[0].inputs.save()
```

### The `.save()` Method

**Critical:** Trace-local variable bindings are only exported when saved. `.save()` must be called inside a trace and assigned to a variable:

```python
with model.trace("Hello"):
    # WRONG - value will be lost
    output = model.transformer.h[-1].output

    # CORRECT - value persists after context
    output = model.transformer.h[-1].output.save()
```

**Two ways to save:**

```python
import nnsight
with model.trace("Hello"):
    output = model.transformer.h[-1].output.save()

with model.trace("Hello"):
    output = nnsight.save(model.transformer.h[-1].output)
```

---

## Modifying Activations (Interventions)

### In-Place Modification

Use slice assignment for in-place modifications:

```python
with model.trace("Hello"):
    # Zero out all activations
    model.transformer.h[0].output[:] = 0
    
    # Modify specific positions
    model.transformer.h[0].output[:, -1, :] = 0  # Last token only
    model.transformer.h[0].output[:, :, 0] = 1   # First hidden dim
```

### Replacement

Use direct assignment to replace the entire output:

```python
with model.trace("Hello"):
    model.transformer.wte.output = model.transformer.wte.output * 0.5
    hs = model.transformer.h[0].output.clone()
    model.transformer.h[0].output = hs * 2
```

### Tuple Outputs

Many modules return tuples. Handle carefully:

```python
with model.trace("Hello"):
    # GPT-2 attention returns a tuple; GPT-2 blocks return a Tensor in transformers 5.x.
    full_output = model.transformer.h[0].attn.output
    model.transformer.h[0].attn.output = (
        torch.zeros_like(full_output[0]),
    ) + full_output[1:]
```

### Clone Before Saving Modified Values

If you modify in-place and want to see the "before" state:

```python
with model.trace("Hello"):
    # Clone BEFORE the modification
    before = model.transformer.h[0].output.clone().save()
    
    model.transformer.h[0].output[:] = 0
    
    after = model.transformer.h[0].output.save()
```

---

## Batching with Invokers

Process multiple inputs in a single forward pass using invokers. Remember: **each invoke is a separate logical greenlet** that runs serially in definition order.

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    # First invoke: first row group
    with tracer.invoke("The Eiffel Tower is in"):
        embeddings = model.transformer.wte.output
        barrier()
        output1 = model.lm_head.output.save()
    
    # Second invoke: second row group
    # Can reference values from first invoke
    with tracer.invoke("_ _ _ _ _ _ _"):
        barrier()
        model.transformer.wte.output = embeddings  # Cross-invoke intervention
        output2 = model.lm_head.output.save()
```

### Prompt-less Invokers for Batch-Wide Operations

Use `.invoke()` with no arguments to run intervention code on the **entire batch**:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        pass
    
    with tracer.invoke("World"):
        pass
    
    # No-arg invoke: worker that sees the combined batch
    with tracer.invoke():
        # This worker must also access modules in execution order
        all_outputs = model.lm_head.output.save()  # Shape: [2, seq, vocab]
```

### Barriers for Cross-Invoke Synchronization

When you need values from one invoke before another proceeds:

```python
with model.generate(max_new_tokens=3, min_new_tokens=3) as tracer:
    barrier = tracer.barrier(2)  # Create barrier for 2 invokes
    
    with tracer.invoke("Madison Square Garden is in the city of"):
        embeddings = model.transformer.wte.output
        barrier()  # Wait here
        output1 = tracer.result.save()
    
    with tracer.invoke("_ _ _ _ _ _ _ _ _"):
        barrier()  # Wait here
        model.transformer.wte.output = embeddings  # Now safe to use
        output2 = tracer.result.save()
```

### Batched Inputs

```python
with model.trace() as tracer:
    # Single prompt
    with tracer.invoke("Hello"):
        out1 = model.lm_head.output[:, -1].save()  # Shape: [1, vocab]
    
    # Multiple prompts batched
    with tracer.invoke(["Hello", "World"]):
        out2 = model.lm_head.output[:, -1].save()  # Shape: [2, vocab]
```

---

## Multi-Token Generation

### Using `.generate()`

```python
with model.generate("Hello", max_new_tokens=5, min_new_tokens=5, do_sample=False) as tracer:
    # Without an iteration loop, interventions target the first occurrence.
    hidden_states = model.transformer.h[-1].output.save()
    output = tracer.result.save()
decoded = model.tokenizer.decode(output[0])
```

### Iterating Over Generation Steps

Use `tracer.iter[...]` to access specific generation steps. The `iter` property accepts:
- **Slice**: `tracer.iter[:]` (all steps), `tracer.iter[1:3]` (steps 1-2)
- **Int**: `tracer.iter[2]` (step 2 only)
- **List**: `tracer.iter[[0, 2, 4]]` (specific steps)

Each iteration moves a cursor that selects the module occurrence requested by the mediator. The `for` form is preferred; the older `with tracer.iter[...]` form is deprecated.

```python
with model.generate("Hello", max_new_tokens=5, min_new_tokens=5) as tracer:
    logits = list().save()
    
    # All steps (slice)
    for step in tracer.iter[:]:
        logits.append(model.lm_head.output[0][-1].argmax(dim=-1))
    
# Or specific steps
with model.generate("Hello", max_new_tokens=5, min_new_tokens=5) as tracer:
    logits = list().save()
    
    # Steps 1-3 only (slice)
    for step in tracer.iter[1:3]:
        logits.append(model.lm_head.output)

# Single step (int)
with model.generate("Hello", max_new_tokens=5, min_new_tokens=5) as tracer:
    for step in tracer.iter[0]:
        first_logits = model.lm_head.output.save()

# Specific steps (list)
with model.generate("Hello", max_new_tokens=5, min_new_tokens=5) as tracer:
    logits = list().save()
    for step in tracer.iter[[0, 2, 4]]:
        logits.append(model.lm_head.output)
```

### Conditional Interventions Per Step

```python
with model.generate("Hello", max_new_tokens=5, min_new_tokens=5) as tracer:
    outputs = list().save()
    
    for step_idx in tracer.iter[:]:
        if step_idx == 2:
            # Only intervene on step 2
            model.transformer.h[0].output[:] = 0
        
        outputs.append(model.transformer.h[-1].output)
```

### Using `.all()` for Recursive Application

```python
with model.generate("Hello", max_new_tokens=3, min_new_tokens=3) as tracer:
    hidden_states = list().save()
    
    # Apply to all generation steps for all descendants
    for step in tracer.all():
        model.transformer.h[0].output[:] = 0
        hidden_states.append(model.transformer.h[-1].output)
```

### Using `.next()` for Manual Stepping

The `.next()` API was removed in 0.8. Select specific occurrences with `tracer.iter`:

```python
with model.generate("Hello", max_new_tokens=3, min_new_tokens=3, do_sample=False) as tracer:
    hidden_states = nnsight.save([])
    for step in tracer.iter[[0, 1, 2]]:
        hidden_states.append(model.transformer.h[-1].output)
    output = tracer.result.save()
```

### ⚠️ Critical Footgun: Unbounded Iteration

An open-ended `tracer.iter[:]` or `tracer.all()` loop waits for a location after the last generated step. Values saved inside the loop survive, but code after the loop does not execute. A bounded loop can also be cut short by EOS; use `min_new_tokens` when the example requires a fixed count.

```python
with model.generate("Hello", max_new_tokens=3, min_new_tokens=3) as tracer:
    for step in tracer.iter[:]:
        hidden = model.transformer.h[-1].output.save()
    final_logits = tracer.result.save()  # Not reached

assert "final_logits" not in locals()  # Expected missing binding
```

Use a bounded loop when you know the count:

```python
with model.generate("Hello", max_new_tokens=3, min_new_tokens=3) as tracer:
    hidden_states = nnsight.save([])
    for step in tracer.iter[:3]:
        hidden_states.append(model.transformer.h[-1].output)
    output = tracer.result.save()
```

For an open-ended loop, give the result its own empty invoke. Set up the input through an explicit invoke; nesting `tracer.invoke()` inside a trace that already has an implicit input invoke is invalid:

```python
with model.generate(max_new_tokens=3, min_new_tokens=3) as tracer:
    with tracer.invoke("Hello"):
        for step in tracer.iter[:]:
            hidden = model.transformer.h[-1].output.save()
    with tracer.invoke():
        output = tracer.result.save()
```

## Gradients and Backpropagation

**Important:** Gradients are accessed on **tensors** (not modules), and only inside a `with tensor.backward():` context.

### How Backward Works

When you use `with tensor.backward():`, nnsight creates a **completely separate interleaving session**:

1. When you `import nnsight`, it monkey-patches `torch.Tensor.backward` to check if it's being used as a trace
2. Inside the backward context, a new interleaving session runs with tensor gradient hooks
3. You can ONLY access `.grad` on tensors inside this context - not `.output`, `.input`, etc.
4. After the backward context exits, the original forward trace resumes

This means: **get any `.output` values you need BEFORE entering the backward context**.

### Gradient Access Order (Reverse of Forward Pass)

Just like the forward pass requires accessing modules in execution order, **the backward pass requires accessing gradients in reverse order**. This is because gradients flow backwards through the model:

```
Forward pass order:  layer0 → layer1 → ... → layer11 → lm_head → loss
Backward pass order: loss → lm_head → layer11 → ... → layer1 → layer0
```

If you accessed `layer5.output` and `layer10.output` during the forward pass, you must access their gradients in reverse: `layer10.grad` first, then `layer5.grad`.

### Accessing Gradients

```python
with model.trace("Hello"):
    # Get the tensor and enable gradients
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)
    
    logits = model.lm_head.output
    loss = logits.sum()
    
    # Access gradients ONLY inside a backward context
    # This is a SEPARATE interleaving session
    with loss.backward():
        grad = hs.grad.save()

print(grad.shape)  # Now contains the gradient
```

### Modifying Gradients

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)
    
    logits = model.lm_head.output
    
    # Use backward as a context to access and modify gradients
    with logits.sum().backward():
        # Access gradient on the tensor
        hs_grad = hs.grad.save()
        
        # Modify gradient on the tensor
        hs.grad[:] = 0
```

### Retain Graph for Multiple Backward Passes

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)
    logits = model.lm_head.output
    
    with logits.sum().backward(retain_graph=True):
        grad1 = hs.grad.save()
    
    # Second backward pass
    modified_logits = logits * 2
    with modified_logits.sum().backward():
        grad2 = hs.grad.save()
```

### Standalone Backward (Outside a Trace)

You can also use backward tracing on its own, without wrapping it in a `model.trace()`:

```python
# First, run a forward pass to get tensors
with model.trace("Hello"):
    hs = model.transformer.h[-1].output
    hs.requires_grad_(True)
    hs = hs.save()  # Save the tensor for later
    logits = model.lm_head.output.save()

# Then, trace the backward pass separately
loss = logits.sum()
with loss.backward():
    grad = hs.grad.save()

print(grad.shape)
```

This is useful when you want to compute gradients after inspecting forward pass results.

---

## Conditionals and Iteration

### Python Conditionals (v0.5+ Pattern)

Standard Python `if` statements work inside tracing contexts:

```python
with model.trace("Hello") as tracer:
    output = model.transformer.h[0].output
    
    # Python conditionals work with real tensor values
    if torch.all(output < 100000):
        model.transformer.h[-1].output[:] = 0
    
    result = model.transformer.h[-1].output.save()
```

### Session-Level Conditionals

```python
with model.session() as session:
    with model.trace("Hello"):
        if torch.all(model.transformer.h[5].output < 100000):
            model.transformer.h[-1].output[:] = 0
        
        output = model.transformer.h[-1].output.save()
```

### Python Loops

Standard Python `for` loops work in session contexts:

```python
with model.session() as session:
    results = list().save()
    
    for prompt in ["Hello", "World", "Test"]:
        with model.trace(prompt):
            results.append(model.lm_head.output.argmax(dim=-1))
```

---

## Model Editing

Create persistently modified versions of a model:

```python
# Non-inplace editing (creates a new model reference)
with model.edit() as (tracer, model_edited):
    model_edited.transformer.h[1].output[:] = 0

# Use original model
with model.trace("Hello"):
    out1 = model.transformer.h[1].output.save()

# Use edited model
with model_edited.trace("Hello"):
    out2 = model_edited.transformer.h[1].output.save()
```

### In-Place Editing

```python
with model.edit(inplace=True):
    model.transformer.h[1].output[:] = 0

# Now ALL traces use the edited model
with model.trace("Hello"):
    output = model.transformer.h[1].output.save()  # Will be zeros
```

### Clearing Edits

```python
# Remove all in-place edits
model.clear_edits()
```

---

## Scanning and Validation

### Scan Mode

Get shapes and types without running the full model:

```python
with model.scan("Hello"):
    # Access shape information
    dim = nnsight.save(model.transformer.h[0].output.shape[-1])
    
print(dim)  # e.g., 768
```

### Validation Mode

Use `.scan()` with fake tensors to check shapes. This deliberately invalid position is caught before real computation:

```python
try:
    with model.scan("Hello"):
        model.transformer.h[0].output[:, 10] = 0
except IndexError:
    print("Invalid sequence position caught")
```

## Caching Activations

Automatically cache outputs from multiple modules:

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()  # Cache all modules

# Access cached values
layer0_out = cache['model.transformer.h.0'].output
```

### Cache Specific Modules

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache(modules=[
        model.transformer.h[0],
        model.transformer.h[1],
        model.lm_head
    ])
```

### Cache Inputs Too

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache(include_inputs=True)

# Access inputs
layer1_input = cache['model.transformer.h.1'].inputs
```

### Cache with Interventions

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()  # Must call BEFORE interventions
    
    model.transformer.h[0].output[:] = 0

# Cache contains the modified values
assert torch.all(cache['model.transformer.h.0'].output[0] == 0)
```

### Attribute-Style Access

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()

# Both work:
out1 = cache['model.transformer.h.0'].output
out2 = cache.transformer.h[0].output
```

---

## Source Tracing

`.source` enables access to **intermediate operations** within a module's forward pass. When you access `.source`, nnsight **rewrites the module's forward method** to hook into all operations (function calls, method calls, etc.) so you can intercept their inputs and outputs.

### How Source Works

1. **Forward Rewriting**: Accessing `.source` injects hooks into every operation in the module's forward method
2. **Operation Discovery**: Print `.source` to see all available operations with their names and line numbers
3. **Operation Access**: Use `.source.<operation_name>` to access a specific operation
4. **Standard Access**: Operations have `.input`, `.inputs`, and `.output` just like modules

### Discovering Available Operations

Print `.source` before tracing to discover the names in your installed transformers version:

```python
print(model.transformer.h[0].attn.source)
```

### Viewing a Specific Operation

In transformers 5.x, `attention_interface_0` selects the function and `attention_interface_1` calls it:

```python
print(model.transformer.h[0].attn.source.attention_interface_1)
```

### Accessing Operation Values

Read an operation's inputs before its output, just as you would for a module:

```python
with model.trace("Hello"):
    attn_args, attn_kwargs = model.transformer.h[0].attn.source.attention_interface_1.inputs
    attn_out = model.transformer.h[0].attn.source.attention_interface_1.output.save()
    model.transformer.h[0].attn.source.self_c_proj_0.output[:] = 0
```

### Recursive Source Tracing

You can trace into operations that call other functions. With eager GPT-2 attention, probabilities pass through functional dropout:

```python
with model.trace("Hello"):
    attention_weights = (
        model.transformer.h[0].attn.source.attention_interface_1
        .source.nn_functional_dropout_0.output.save()
    )
```

Access a submodule's source directly through its Envoy. Operation names depend on the installed model source:

```python
print(model.transformer.h[0].attn.c_proj.source)
```

## Module Skipping

Skip a module's computation entirely, supplying a replacement output of the correct structure:

```python
with model.trace("Hello"):
    layer0_out = model.transformer.h[0].output.save()
    model.transformer.h[1].skip(layer0_out)
    layer1_out = model.transformer.h[1].output.save()
assert torch.equal(layer0_out, layer1_out)
```

Inner modules of a skipped module never run. Skips respect execution order and affect the full batched module call; do not treat a skip as an independent per-row forward.

## vLLM Integration

Use a vLLM wheel and PyTorch build compiled for the same CUDA version; the unqualified extra can select a wheel that does not match an existing Colab environment. Follow the [official GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

Install `nnsight[vllm]==0.8.0rc1` on a supported Linux CUDA machine. These examples use one GPU and a small public model. vLLM activations have no batch axis: each worker sees its request's rows in a flat token slab. Clone saved activations because vLLM reuses buffers. In a Python script, construct and run the engine under an `if __name__ == "__main__":` guard; notebook cells do not need that guard.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM(
    "openai-community/gpt2", tensor_parallel_size=1, dispatch=True,
    dtype="float16", gpu_memory_utilization=0.3, max_model_len=64,
)
```

### Basic vLLM Tracing

`logits` and `samples` are hookable values, not modules: use `model.logits`, not `model.logits.output`.

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    logits = model.logits.clone().save()
next_token = model.tokenizer.decode(logits.argmax(dim=-1))
```

### vLLM Multi-Token Generation

```python
with model.trace("Hello", max_tokens=3, min_tokens=3, temperature=0.0) as tracer:
    logits = nnsight.save([])
    for step in tracer.iter[:3]:
        logits.append(model.logits.clone())
```

### vLLM Interventions

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    hidden = model.transformer.h[8].output.clone().save()
    model.transformer.h[8].output[:] = 0
    logits = model.logits.clone().save()
```

### vLLM Sampling

```python
with model.trace(max_tokens=3, min_tokens=3) as tracer:
    with tracer.invoke("Hello", temperature=0.8, top_p=0.95):
        samples = nnsight.save([])
        for step in tracer.iter[:3]:
            samples.append(model.samples.item())
```

### CUDA Graph Taps

An eager engine exposes every location. `taps=` enables breakable CUDA graphs and exposes only the declared locations during replay; it requires a vLLM version with breakable-graph support. Keep the tap set small. Source-operation paths can also be tapped. The GPU checks use vLLM 0.26.0 with its official CUDA 12.9 wheel, matching PyTorch 2.11.0, torchvision, torchaudio, and TorchCodec builds on a Colab T4.

```python
tapped_model = VLLM(
    "openai-community/gpt2", dispatch=True,
    dtype="float16", gpu_memory_utilization=0.3, max_model_len=64,
    taps=["transformer.h.8.output"],
)
with tapped_model.trace("Hello", max_tokens=1, temperature=0.0):
    hidden = tapped_model.transformer.h[8].output.clone().save()
```

### Persistent Engine Edits

vLLM edits are installed on the engine and apply to later ordinary requests. Saves arrive on each finished request's `.saves` dictionary.

```python
with model.edit() as (tracer, edit):
    hidden = model.transformer.h[8].output.clone().save()
outputs = model.generate(["Hello", "The capital of Japan is"], max_tokens=1, temperature=0.0)
captured_hidden = outputs[1].saves["hidden"]
model.clear_edits()
```

For streaming, construct `VLLM(..., mode="async")` and consume `async for output in tracer.backend` (an attribute, not a call). Finished outputs carry their saves. The `serve` extra supplies `nnsight-serve`, allowing clients without a GPU to trace an engine over HTTP. See the [vLLM guide](https://nnsight.net/models/vllm/) for tensor parallelism, async execution, serving, and architecture-specific residual tuples.

---

## Remote Execution (NDIF)

Run interventions on NDIF's remote infrastructure without local GPU resources.

### Setup

```python
from nnsight import CONFIG
import nnsight
import os

# For NDIF, set NDIF_API_KEY in your environment (or use `nnsight login`).
# Offline examples require no key.
if os.environ.get("NDIF_API_KEY"):
    CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
```

### Basic Remote Tracing

```python
# Model loads on 'meta' device - no local GPU memory used
model = TransformersModel("openai-community/gpt2", task="text-generation")
print(model.device)  # "meta"

# REMOTE="local" validates serialization and runs locally; True uses NDIF
with model.trace("The Eiffel Tower is in the city of", remote=REMOTE):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))  # "Paris"
```

### Remote Generation

```python
with model.generate("Hello", max_new_tokens=5, min_new_tokens=5, remote=REMOTE) as tracer:
    logits = nnsight.save([])
    for step in tracer.iter[:5]:
        logits.append(model.lm_head.output[0, -1].argmax(-1))
    output = tracer.result.save()
print(model.tokenizer.decode(output[0]))
```

### Request Lifecycle

With `remote=True` on a compatible NDIF deployment, you'll see status updates. The offline `remote="local"` simulator does not create an NDIF job:

| Status | Description |
|--------|-------------|
| `RECEIVED` | Request validated with API key |
| `QUEUED` | Waiting in model's queue |
| `DISPATCHED` | Forwarded to model deployment |
| `RUNNING` | Executing your intervention |
| `LOG` | Print statements from your code |
| `COMPLETED` | Results ready for download |

### Print Statements in Remote Traces

On NDIF, print statements inside remote traces are captured and sent back as `LOG` status. With `REMOTE = "local"`, they print directly in your local process:

```python
with model.trace("Hello", remote=REMOTE):
    hidden = model.transformer.h[0].output
    print(f"Hidden shape: {hidden.shape}")  # Local stdout here; LOG on NDIF
    print(f"Hidden mean: {hidden.mean()}")  # Local stdout here; LOG on NDIF
    output = model.lm_head.output.save()
```

### Saving Results Remotely

**Critical:** `.save()` is how values are transmitted back from NDIF. Mutating a list created outside the trace does not return that list from the server. The offline simulator runs in your process and can mutate it, so that behavior does not demonstrate remote process isolation.

```python
# Not portable to NDIF: an external list is only mutated by the offline run
my_list = list()
with model.trace("Hello", remote=REMOTE):
    my_list.append(model.output.save())  # Offline: updated; live NDIF: stays empty

# CORRECT: Create and save inside the trace
with model.trace("Hello", remote=REMOTE):
    my_list = list().save()  # Create inside trace
    my_list.append(model.output)

# Return a detached CPU tensor for local analysis; .cpu() does not reduce its size
with model.trace("Hello", remote=REMOTE):
    hidden = model.transformer.h[0].output.detach().cpu().save()
```

### Non-Blocking Execution

Submit jobs without waiting:

```python
# Requires an NDIF deployment compatible with 0.8 and an API key.
with model.trace("Hello", remote=True, blocking=False) as tracer:
    output = model.lm_head.output.save()
backend = tracer.backend
print(backend.job_id, backend.status)

import time
while True:
    result = backend()
    if result is not None:
        break
    time.sleep(1)
print(result["output"].shape)
```

### Disable Remote Logging

```python
CONFIG.APP.REMOTE_LOGGING = False
```

---

## Sessions

Group multiple traces for shared state and efficiency.

### Local Sessions

```python
with model.session() as session:
    # First trace
    with model.trace("Hello"):
        hs1 = model.transformer.h[0].output.save()
    
    # Second trace - can reference values from first
    with model.trace("World"):
        model.transformer.h[0].output[:] = hs1
        hs2 = model.transformer.h[0].output.save()
```

### Remote Sessions

This example runs as a local session. Use `remote="local"` on the session to check serialization offline, or `remote=True` on a compatible NDIF service to send the traces as one request.

Sessions are especially powerful for remote execution — they bundle multiple traces into a **single request**:

```python
# Local validation; add remote=True only on a compatible NDIF service.
with model.session():
    # First trace: capture hidden states
    with model.trace("Megan Rapinoe plays the sport of"):
        hs = model.transformer.h[5].output[:, -1, :]  # No .save() needed!
    
    # Second trace: clean baseline
    with model.trace("Shaquille O'Neal plays the sport of"):
        clean = model.lm_head.output[0][-1].argmax(dim=-1).save()
    
    # Third trace: patch the hidden states
    with model.trace("Shaquille O'Neal plays the sport of"):
        model.transformer.h[5].output[:, -1, :] = hs  # Direct reference works!
        patched = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(f"Clean: {model.tokenizer.decode(clean)}")
print(f"Patched: {model.tokenizer.decode(patched)}")
```

**Benefits of remote sessions:**
- Single queue wait for all traces
- Values from earlier traces accessible directly (no `.save()` needed)
- For NDIF, put `remote=True` on the session, not inner traces

---

## Module Renaming

Create aliases for easier access:

```python
model = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    rename={
        "transformer.h": "layers",           # Mount at new path
        "mlp": "feedforward",                # Rename all MLPs
        ".transformer": ["model", "backbone"] # Multiple aliases
    }
)

# Now both work:
with model.trace("Hello"):
    out1 = model.layers[0].feedforward.output.save()
    out2 = model.transformer.h[0].mlp.output.save()  # Original still works
```

---

## Common Patterns

### Activation Patching

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    # Clean run
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[:, -1, :].save()
        barrier()
    
    # Patched run
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[:, -1, :] = clean_hs
        patched_logits = model.lm_head.output.save()
```

### Logit Lens

```python
with model.trace("The Eiffel Tower is in"):
    # Apply final layer norm and lm_head to intermediate layers
    for i in range(12):
        hs = model.transformer.h[i].output
        logits = model.lm_head(model.transformer.ln_f(hs))
        tokens = logits.argmax(dim=-1).save()
        print(f"Layer {i}:", model.tokenizer.decode(tokens[0][-1]))
```

**Why this works:** When you call `model.lm_head(...)` inside a trace, it uses `.forward()` instead of `__call__()`, bypassing interleaving hooks. This means the module's computation runs without triggering `.input`/`.output` hooks.

### Using Auxiliary Modules (SAEs, LoRA, etc.)

When you add auxiliary modules to a model (like SAEs or LoRA adapters), use `hook=True` to enable `.input`/`.output` access on them:

```python
# Attach a small auxiliary module as a runnable SAE stand-in.
model.sae = torch.nn.Identity()
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        hidden = model.transformer.h[5].output
        reconstructed = model.sae(hidden, hook=True)
        model.transformer.h[5].output = reconstructed
    with tracer.invoke("Hello"):
        sae_activations = model.sae.output.save()
```

### Ablation Study

```python
with model.trace() as tracer:
    with tracer.invoke("Hello World"):
        # Baseline
        baseline = model.lm_head.output[:, -1].save()
    
    with tracer.invoke("Hello World"):
        # Ablate specific layer
        model.transformer.h[5].mlp.output[:] = 0
        ablated = model.lm_head.output[:, -1].save()

diff = (baseline - ablated).abs().mean()
```

### Attention Pattern Extraction

```python
with model.trace("Hello World"):
    # Eager attention exposes the probabilities after dropout.
    attn_weights = (
        model.transformer.h[0].attn.source.attention_interface_1
        .source.nn_functional_dropout_0.output.save()
    )
```

### Steering with Added Vectors

```python
steering_vector = torch.randn(model.config.n_embd)  # Pre-computed direction

with model.trace("Hello"):
    model.transformer.h[10].output[:, -1, :] += steering_vector.to(model.transformer.h[10].output) * 0.5
    output = model.lm_head.output.save()
```

---

## Critical Gotchas

### 1. Forgetting `.save()`

```python
# WRONG - binding is not exported
with model.trace("Hello"):
    output = model.transformer.h[-1].output
# output is now useless

# CORRECT
with model.trace("Hello"):
    output = model.transformer.h[-1].output.save()
```

### 2. In-Place vs Replacement

```python
with model.trace("Hello"):
    model.transformer.h[0].output[:] = 0
    model.transformer.h[0].output = torch.zeros_like(model.transformer.h[0].output)
```

### 3. Tuple Outputs

```python
with model.trace("Hello"):
    full_output = model.transformer.h[0].attn.output
    # Mutate the Tensor inside the tuple.
    full_output[0][:] = 0
    # To replace a tuple element, construct and assign a new tuple.
    model.transformer.h[0].attn.output = (torch.ones_like(full_output[0]),) + full_output[1:]
```

### 4. Clone Before In-Place Modification

```python
# WRONG - modifying and trying to see original
with model.trace("Hello"):
    before = model.transformer.h[0].output.save()  # Points to same tensor!
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()
# before == after because both point to modified tensor

# CORRECT
with model.trace("Hello"):
    before = model.transformer.h[0].output.clone().save()
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()
```

### 5. Module Access Must Be in Execution Order

Within one invoke, read locations in the order the model reaches them. An already-passed location raises `OutOfOrderError`.

```python
from nnsight.intervention.interleaver import OutOfOrderError

with model.trace("Hello"):
    out1 = model.transformer.h[1].output.save()
    out5 = model.transformer.h[5].output.save()

try:
    with model.trace("Hello"):
        out5 = model.transformer.h[5].output.save()
        out1 = model.transformer.h[1].output.save()
except OutOfOrderError:
    print("Layer 1 was already passed")
```

Separate traces let you inspect locations in another order with a fresh forward pass:

```python
with model.trace("Hello"):
    out5 = model.transformer.h[5].output.save()
with model.trace("Hello"):
    out1 = model.transformer.h[1].output.save()
```

### 6. Trace Requires Input Without Invokes

```python
# WRONG - no input and no invokes
try:
    with model.trace():
        output = model.output.save()
except ValueError:
    print("Provide an input or an explicit invoke")

# CORRECT - provide input
with model.trace("Hello"):
    output = model.output.save()

# CORRECT - use explicit invokes
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        output = model.output.save()
```

### 7. Values Are Real Tensors (Not Proxies)

In nnsight's greenlet-based architecture, reading `.output` parks the worker and receives the **actual tensor**:

```python
with model.trace("Hello"):
    # Worker waits here and gets the REAL tensor
    hs = model.transformer.h[0].output
    
    # This is a real shape, real operations work directly
    shape = hs.shape  # torch.Size([1, 1, 768])
    zeros = torch.zeros(shape)  # Real tensor operation
    
    # Printing works normally
    print(shape)  # torch.Size([1, 1, 768])
```

Use `.scan()` if you need shapes **without** running the model:

```python
with model.scan("Hello"):
    shape = nnsight.save(model.transformer.h[0].output.shape)  # Export the shape
```

### 8. Generation vs Trace

```python
# Use .trace() for single forward pass
with model.trace("Hello"):
    output = model.output.save()

# Use .generate() for multi-token generation
with model.generate("Hello", max_new_tokens=5, min_new_tokens=5) as tracer:
    output = tracer.result.save()
```

### 9. Device Placement

```python
# Tensors must be on the correct device
with model.trace("Hello"):
    device = model.transformer.h[0].output.device
    noise = torch.randn(model.config.n_embd).to(device)  # Match device!
    model.transformer.h[0].output[:, -1, :] += noise
```

---

## Understanding Exceptions in NNsight

Debugging in NNsight can be tricky because of **deferred execution**. Your intervention code is captured, compiled into a function, and run in a worker greenlet — not where you wrote it. Without special handling, exceptions would point to internal NNsight code instead of your original trace.

### How NNsight Fixes This

NNsight **reconstructs exception tracebacks** to show your original code and line numbers. When an exception occurs:

1. NNsight catches it at the trace boundary
2. Maps the internal line numbers back to your source file
3. Rebuilds the traceback to look like a normal Python exception

**What you see:**
```
Traceback (most recent call last):
  File "my_experiment.py", line 6, in <module>
    hidden = model.transformer.h[100].output.save()
AttributeError: invalid module index 100
```

This points directly to your code, even though it actually ran in a compiled worker greenlet.

### Exception Type Preservation

In local traces, NNsight preserves the original exception type, so you can catch specific exceptions. Deferred worker errors crossing a process boundary, such as vLLM engine errors, can instead arrive as `RuntimeError` with the original type name and traceback:

```python
try:
    with model.trace("Hello"):
        hidden = model.transformer.h[100].output.save()
except (IndexError, AttributeError):
    print("Caught the invalid module index!")  # This works!
```

### DEBUG Mode

By default, NNsight hides its internal frames from tracebacks. If the default traceback isn't helpful, enable DEBUG mode to see the full execution path:

```python
from nnsight import CONFIG
CONFIG.APP.DEBUG = True
```

This shows internal NNsight frames, which can help:
- Understand where in the pipeline an error occurred
- Provide more context when the user-facing traceback is unclear

### Common Exceptions

| Exception | Cause | Fix |
|-----------|-------|-----|
| `OutOfOrderError: ... was requested but the model already ran past it` | Accessed modules in wrong order within an invoke | Access modules in forward-pass order |
| `OutOfOrderError` | Mediator still waiting - module never called or gradient order wrong | Check module path exists; access gradients in reverse order |
| `ValueError: trace() needs an input, or at least one ... block` | Trace has no input or invokes | Provide input to `.trace(input)` or use `.invoke()` |
| `ValueError: Cannot access ... outside of interleaving` | Activation read outside an intervention worker | Read inside a trace/invoke and save the result |
| `ValueError: Cannot invoke while the model is already running` | Tried to create invoke inside another invoke | Use sequential, non-nested invokes |
| `OutOfOrderError` | Requested a forward activation during a backward session | Capture tensors before backward, access their `.grad` inside |
| `AttributeError: ... has no attribute X` | Nonexistent module accessed | Use `print(model)` to see available modules |
| Tokenizer loading error | Pre-loaded model has no usable checkpoint name or tokenizer | Provide `tokenizer=` when wrapping |
| `NotImplementedError: ... does not support batching multiple invokes` | Base wrapper cannot combine multiple invokes | Use a batching-capable wrapper or override `_batch(invokes, fn)` |

**The `.i0` suffix** in error messages indicates iteration 0 (first call). In generation, you'd see `.i1`, `.i2`, etc.

**Gradient errors show tensor IDs** (like `139820463417744.grad`) instead of module paths because gradients are tracked per-tensor, not per-module.

---

## Debugging Tips

### 1. Use Print and Breakpoints

Print works normally inside traces:

```python
with model.trace("Hello"):
    out = model.transformer.h[0].output
    print("Shape:", out.shape)
    print("Mean:", out.mean())
```

Use `breakpoint()` for interactive debugging:

```python
with model.trace("Hello"):
    out = model.transformer.h[0].output
    breakpoint()  # Drops into pdb
```

### 2. Validate with Scan

```python
# Expected failure: "Hello" has fewer than 1001 token positions.
try:
    with model.scan("Hello"):
        model.transformer.h[0].output[:, 1000] = 0
except IndexError:
    print("Invalid sequence-position index caught during scan")
```

### 3. Check Module Structure

```python
# Print full model structure
print(model)

# Check specific module
print(model.transformer.h[0])
```

### 4. Check Shapes with Scan

Use `.scan()` to get shapes without running the full model:

```python
with model.scan("Hello"):
    print(model.transformer.h[0].output.shape)  # (1, seq_len, hidden_dim)
    print(model.lm_head.output.shape)  # (1, seq_len, vocab_size)
```

### 5. Debug Configuration

```python
import nnsight

# Enable detailed error messages
nnsight.CONFIG.APP.DEBUG = True
nnsight.CONFIG.save()
```

---

## Configuration

NNsight has several configuration options accessible via `nnsight.CONFIG`:

```python
from nnsight import CONFIG
CONFIG.APP.DEBUG = True
CONFIG.APP.REMOTE_LOGGING = True
CONFIG.APP.PYMOUNT = True
# Linux/x86-64 guard for torch C++ exceptions in greenlets.
CONFIG.APP.DISABLE_CPP_BACKTRACE = True
CONFIG.save()
```

### Cross-Invoker Variable Sharing

Variables from sibling invokes share a captured scope. Because workers resume at model locations rather than running one whole invoke at a time, synchronize value dependencies with a barrier:

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("Hello"):
        embeddings = model.transformer.wte.output  # Captured here
        barrier()
    
    with tracer.invoke("World"):
        barrier()
        model.transformer.wte.output = embeddings  # Used here (shared after the barrier)
```

The old configuration switch was removed:

```python
# Sharing uses the captured Scope in 0.8; there is no CROSS_INVOKER setting.
assert "CROSS_INVOKER" not in CONFIG.APP.model_fields
```

### Barrier Synchronization

When **both invokes access the same module**, you need to synchronize them so variables can be shared at that point:

```python
with llm.trace() as tracer:
    
    barrier = tracer.barrier(2)  # Create barrier for 2 participants
    
    with tracer.invoke("The Eiffel Tower is in"):
        paris_embeddings = llm.transformer.wte.output
        barrier()  # Wait here
    
    with tracer.invoke("_ _ _ _ _"):
        barrier()  # Synchronize
        llm.transformer.wte.output = paris_embeddings  # Now available!
        patched_output = llm.lm_head.output[:, -1].save()
```

Without the barrier, the second invoke would fail with `paris_embeddings is not defined` because the consuming worker may read the shared name before the producer has received its activation.

### Re-wrapping the Same Model

If you wrap the same PyTorch model with `NNsight` multiple times, hooks are properly cleaned up and re-applied (not stacked):

```python
model1 = NNsight(my_pytorch_model)
model2 = NNsight(my_pytorch_model)  # Safe - hooks replaced, not duplicated
```

---

## API Quick Reference

### Context Managers

| Method | Description |
|--------|-------------|
| `model.trace(input)` | Single forward pass with interventions |
| `model.generate(input, max_new_tokens=N)` | Multi-token generation |
| `model.scan(input)` | Get shapes without full execution |
| `model.edit()` | Create persistent model modifications |
| `model.session()` | Group multiple traces |

### Tracer Methods

| Method | Description |
|--------|-------------|
| `tracer.invoke(input)` | Add input to batch |
| `tracer.barrier(n)` | Synchronization barrier |
| `tracer.cache(...)` | Activation caching |
| `tracer.stop()` | Early termination |
| `tracer.iter[slice]` | Iterate generation steps |
| `tracer.all()` | Apply to all steps |
| `tracer.result` | Get final output of traced function |

### Module Properties

| Property | Description |
|----------|-------------|
| `.output` | Module output |
| `.input` | First positional input |
| `.inputs` | All inputs `(args, kwargs)` |
| `.source` | Internal operation tracing |
| `.next()` | Removed in 0.8; select steps with `tracer.iter` |
| `.skip(value)` | Skip module with given output |


**Note on `.grad`:** Access gradients on **tensors** only inside a `with tensor.backward():` context. See [Gradients and Backpropagation](#gradients-and-backpropagation).

---

## Other NNsight 0.8 Features

### Pipelines Beyond Causal Language Models

`TransformersModel` supports any task the HuggingFace pipeline factory can build, including fill-mask, classification, image-text-to-text, and speech recognition. Preprocessing is task-specific. This small public checkpoint demonstrates fill-mask without downloading a large model:

```python
masked_model = TransformersModel(
    "hf-internal-testing/tiny-random-BertForMaskedLM",
    task="fill-mask", dispatch=True,
)
with masked_model.pipe("The capital of France is [MASK].", top_k=2) as tracer:
    records = tracer.result.save()
assert len(records) == 2
```

### Generate Versus Pipe

`generate()` returns token IDs for causal language models; `pipe()` returns task pipeline records, such as decoded text or labels:

```python
with model.generate("Hello", max_new_tokens=2, min_new_tokens=2, do_sample=False) as tracer:
    token_ids = tracer.result.save()
with model.pipe("Hello", max_new_tokens=2, min_new_tokens=2, do_sample=False) as tracer:
    records = tracer.result.save()
assert "generated_text" in records[0]
```

### Diffusion Pipelines

Install `diffusers` separately. `DiffusionModel` wraps UNet- and transformer-based pipelines, supports per-invoke batching, accepts `seed=`, and exposes denoising iterations. `automodel=` selects the loading class. The tiny checkpoint below tests the API; its images are not representative of a trained production model.

```python
from nnsight import DiffusionModel

diffusion = DiffusionModel("hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None)
with diffusion.generate("a cat", num_inference_steps=2, seed=0) as tracer:
    noise_predictions = nnsight.save([])
    for step in tracer.iter[:2]:
        noise_predictions.append(diffusion.unet.output[0].clone())
    images = tracer.result.save()
assert len(noise_predictions) == 2
assert len(images.images) == 1
```

### Large-Model Loading

- **Quantization:** `dtype="nf4"` and the other supported quantization names select loader configurations. Bitsandbytes formats require `bitsandbytes` and `accelerate`, plus supported hardware. See [quantization](https://github.com/ndif-team/nnsight/blob/main/docs/models/quantization.md).
- **Tensor parallelism:** Transformers models support `distributed_config=DistributedConfig(tp_size=N)` under `torchrun`, with sharded activations gathered for the trace. This is distinct from `device_map="auto"` layer placement and requires multiple supported GPUs. See [tensor parallelism](https://github.com/ndif-team/nnsight/blob/main/docs/models/tensor-parallel.md).
- **PEFT:** `TransformersModel(..., peft=adapter_repo_id)` applies adapters at load time; the remote environment can carry per-request adapters. Install `peft` for this optional feature.
- **Custom hookable values:** `@eproperty` is the public descriptor behind `.input`, `.output`, and `tracer.result`; `envoys=` mounts custom Envoy classes by module type or path suffix. See the runnable examples in [NNsight.md](NNsight.md#10-extending-nnsight).
- **Remote tooling:** `nnsight login` stores and verifies an NDIF key; `nnsight.status()`, `nnsight.compare()`, and `nnsight.register(module)` support service inspection, environment comparison, and shipping local code by value. Live requests require a service compatible with the client.

---

## Version Notes

This guide is verified against **nnsight 0.8.0rc1** and **transformers 5.x**. Key changes from earlier versions:

- Standard Python `if`/`for` statements now work inside tracing contexts (replaces `nnsight.cond()`, `session.iter()`)
- `nnsight.apply()` is deprecated - use functions directly
- `nnsight.list()`, `nnsight.dict()`, etc. are deprecated - use standard Python types with `.save()`
- `nnsight.local()` and `nnsight.trace` decorator are deprecated

---

## File Structure Reference

```
nnsight/
├── src/nnsight/
│   ├── __init__.py          # Main exports
│   ├── modeling/
│   │   ├── base.py          # NNsight class
│   │   ├── transformers.py  # TransformersModel and pipeline integration
│   │   └── vllm/            # vLLM integration
│   └── intervention/
│       ├── envoy.py         # Core Envoy wrapper
│       ├── interleaver.py   # Execution interleaving
│       ├── tracer.py        # Intervention tracing
│       └── eproperty.py     # Hookable value descriptors
└── tests/
    ├── test_tiny.py         # Basic NNsight tests
    ├── test_lm.py           # TransformersModel tests
    └── test_vllm.py         # vLLM tests
```

