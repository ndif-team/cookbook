"""CPU regressions for the NNsight 0.8 migration; no model downloads."""
import importlib
import sys
from pathlib import Path

import pytest
import torch as t
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from nnterp_compat import AttnProbFunction, RenameConfig, StandardizedTransformer

ROOT = Path(__file__).resolve().parent


def section(name):
    sys.path.insert(0, str(ROOT / name))
    return importlib.import_module(f"{name}.solutions")


@pytest.fixture(scope="module")
def gpt2():
    t.manual_seed(0)
    backend = Tokenizer(WordLevel(
        {"<pad>": 0, "<eos>": 1, "<unk>": 2, "a": 3, "b": 4, "c": 5},
        unk_token="<unk>",
    ))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
        bos_token="<eos>", unk_token="<unk>", padding_side="left",
    )
    raw = GPT2LMHeadModel(GPT2Config(
        n_layer=2, n_head=2, n_embd=32, n_positions=64, vocab_size=6,
        pad_token_id=0, eos_token_id=1, bos_token_id=1,
    )).eval()
    return StandardizedTransformer(
        raw, tokenizer=tokenizer, device="cpu", device_map=None, enable_attention_probs=True,
    )


@pytest.mark.parametrize("name", [p.name for p in sorted(ROOT.glob("part*")) if p.is_dir()])
def test_course_imports(name):
    section(name)
    importlib.import_module(f"{name}.tests")


@pytest.mark.parametrize("name,cls,wrap", [
    ("part51_balanced_bracket_classifier", "BracketClassifier", "wrap_bracket_model"),
    ("part52_grokking_and_modular_arithmetic", "GrokkingTransformer", "wrap_grokking_model"),
    ("part53_othellogpt", "OthelloGPT", "wrap_othello_model"),
])
def test_custom_models_match_pytorch(name, cls, wrap):
    s = section(name)
    raw = getattr(s, cls)().eval()
    t.manual_seed(0)
    for param in raw.parameters():
        t.nn.init.normal_(param, std=0.02)
    model = getattr(s, wrap)(raw)
    tokens = t.tensor([[0, 3, 4], [0, 4, 3]])
    expected = raw(tokens)
    with model.trace(tokens):
        hidden = model.layers_output[0].save()
        logits = model.logits.save()
    assert hidden.shape[:2] == tokens.shape
    t.testing.assert_close(logits, expected)


def test_attention_only_model():
    s = section("part2_intro_to_mech_interp")
    from convert_2L_attn_only import AttnOnly2L
    raw = AttnOnly2L(n_heads=2, d_model=32, d_head=16, d_vocab=16, n_ctx=8).eval()
    for param in raw.parameters():
        t.nn.init.normal_(param, std=0.02)
    raw.config = s.FakeConfig(num_attention_heads=2, hidden_size=32, vocab_size=16)
    model = StandardizedTransformer(raw, device_map=None, check_renaming=False,
        rename_config=RenameConfig(layers_name="blocks", attn_name="attn", lm_head_name="unembed",
            ignore_mlp=True, attn_prob_source=s.SourceSoftmaxAttnProbFunction()))
    model.attention_probabilities.enabled = True
    tokens = t.tensor([[1, 2, 3]])
    expected = raw(tokens)
    with model.trace(tokens):
        pattern = model.attention_probabilities[0].save()
        logits = model.logits.save()
    t.testing.assert_close(pattern.sum(-1), t.ones_like(pattern.sum(-1)))
    t.testing.assert_close(logits, expected)


def test_trained_bracket_attention_and_components():
    s = section("part51_balanced_bracket_classifier")
    from brackets_datasets import BracketsDataset
    raw = s.BracketClassifier().eval()
    raw.load_state_dict(t.load(ROOT / "part51_balanced_bracket_classifier/bracket_classifier_converted.pt",
        map_location="cpu", weights_only=True))
    model = s.wrap_bracket_model(raw)
    data = BracketsDataset([("()", True), (")(", False), ("(())", True)])
    mask = data.toks == 1
    expected = raw(data.toks, padding_mask=mask)
    components = s.get_out_by_components(model, data)
    with model.trace(data.toks, padding_mask=mask):
        ln_input = model.ln_final.input.save()
        logits = model.logits.save()
    t.testing.assert_close(logits, expected)
    biases = sum(block.attn.b_O for block in raw.blocks)
    t.testing.assert_close(components.sum(0) + biases, ln_input)
    with model.trace(data.toks, padding_mask=mask):
        pattern = model.attention_probabilities[0]
        model.attention_probabilities[0] = t.zeros_like(pattern)
        ablated = model.logits.save()
    assert not t.allclose(ablated, expected)
    # An intervention must not persist into the next trace.
    with model.trace(data.toks, padding_mask=mask):
        restored = model.logits.save()
    t.testing.assert_close(restored, expected)


def test_gpt2_attention_and_cross_invoke(gpt2):
    encoded = gpt2.tokenizer(["a b", "a"], padding=True, return_tensors="pt")
    expected = gpt2._model(**encoded).logits
    with gpt2.trace(encoded):
        probs = gpt2.attention_probabilities[0].save()
        hidden = gpt2.layers_output[0].save()
        logits = gpt2.logits.save()
    assert probs.shape == (2, 2, 2, 2)
    assert hidden.shape == (2, 2, 32)
    # v0.8 adds mask-derived position IDs for left-padded prompts.
    positions = encoded.attention_mask.long().cumsum(-1) - 1
    positions.masked_fill_(encoded.attention_mask == 0, 0)
    expected = gpt2._model(**encoded, position_ids=positions).logits
    t.testing.assert_close(logits, expected)
    with gpt2.trace() as tracer:
        with tracer.invoke("a b"):
            clean = gpt2.layers_output[0].clone()
        with tracer.invoke("a c"):
            target = gpt2.layers_output[0]
            target[:] = clean
            patched = gpt2.logits.save()
    expected = gpt2._model(gpt2.tokenizer("a b", return_tensors="pt").input_ids).logits
    t.testing.assert_close(patched, expected)


def test_function_vector_generation(gpt2):
    s = section("part42_function_vectors_and_model_steering")
    baseline, zero_vector = s.intervene_with_fn_vector(gpt2, "a", 0, t.zeros(32), "a {x}", 2)
    assert baseline == zero_vector
    assert baseline.startswith("a a")
    unsteered, steered = s.calculate_and_apply_steering_vector(
        gpt2, "a b", [(0, 0.2, "a b"), (0, -0.2, "a c")],
        n_tokens=2, n_comparisons=2, use_bos=False,
    )
    assert len(unsteered) == len(steered) == 2


def test_model_id_loading_and_dispatch(gpt2, tmp_path):
    gpt2._model.save_pretrained(tmp_path)
    gpt2.tokenizer.save_pretrained(tmp_path)
    model = StandardizedTransformer(str(tmp_path), device="cpu", device_map=None, enable_attention_probs=True,
        rename_config=RenameConfig(attn_prob_source=__import__("nnterp_compat").GPT2AttentionProbFunction()))
    tokens = t.tensor([[3, 4]])
    expected = gpt2._model(tokens).logits
    with model.trace(tokens):
        logits = model.logits.save()
    t.testing.assert_close(logits, expected)


def test_function_vector_extraction_and_intervention(gpt2, monkeypatch):
    s = section("part42_function_vectors_and_model_steering")
    for key, value in {"N_HEADS": 2, "D_HEAD": 16}.items():
        monkeypatch.setattr(s, key, value, raising=False)
    pairs = [["a", "b"], ["b", "c"], ["c", "a"]]
    dataset = s.ICLDataset(pairs, size=2, n_prepended=1)
    zero_shot = s.ICLDataset(pairs, size=2, n_prepended=0)
    completions, h = s.calculate_h(gpt2, dataset, layer=0)
    assert len(completions) == 2 and h.shape == (32,)
    expected = s.intervene_with_h(gpt2, zero_shot, h, layer=0)
    assert s.calculate_h_and_intervene(gpt2, dataset, zero_shot, layer=0) == expected
    fn_vector = s.calculate_fn_vector(gpt2, dataset, [(0, 0), (1, 1)])
    assert fn_vector.shape == (32,) and t.isfinite(fn_vector).all()
    scores = s.calculate_fn_vectors_and_intervene(gpt2, dataset, layers=[0, 1])
    assert scores.shape == (2, 2) and t.isfinite(scores).all()


def test_ioi_residual_patching(gpt2, monkeypatch):
    s = section("part41_indirect_object_identification")
    monkeypatch.setattr(s, "N_LAYERS", 2)
    clean = t.tensor([[3, 4]])
    corrupt = t.tensor([[3, 5]])
    metric = lambda logits: logits[0, -1, 4]
    results = s.get_act_patch_resid_pre(gpt2, corrupt, clean, metric)
    # Independently patch raw PyTorch modules, with the same intervention sites.
    expected = t.zeros_like(results)
    for layer, module in enumerate([gpt2._model.transformer.wte, gpt2._model.transformer.h[0]]):
        captured = []
        handle = module.register_forward_hook(lambda mod, args, out: captured.append(out.clone()))
        gpt2._model(clean)
        handle.remove()
        for pos in range(2):
            def patch(mod, args, out):
                out = out.clone()
                out[:, pos] = captured[0][:, pos]
                return out
            handle = module.register_forward_hook(patch)
            expected[layer, pos] = metric(gpt2._model(corrupt).logits)
            handle.remove()
    t.testing.assert_close(results, expected)


@pytest.fixture(scope="module")
def grokking_model():
    s = section("part52_grokking_and_modular_arithmetic")
    t.manual_seed(0)
    model = s.GrokkingTransformer().eval()
    for param in model.parameters():
        t.nn.init.normal_(param, std=0.02)
    return model


def run_grokking_exercise_cells(model, filename, tmp_path):
    """Execute the actual four exercise blocks, without downloading training history."""
    import ast
    import json

    s = section("part52_grokking_and_modular_arithmetic")
    tests = importlib.import_module("part52_grokking_and_modular_arithmetic.tests")
    path = ROOT / "part52_grokking_and_modular_arithmetic" / filename
    if path.suffix == ".ipynb":
        cells = ["".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]
                 if cell["cell_type"] == "code"]
    else:
        source = path.read_text()
        cells = [ast.get_source_segment(source, node) for node in ast.parse(source).body
                 if isinstance(node, ast.If)]
    markers = ["W_O = model.blocks[0].attn.W_O", "original_logits_full = wrapped.lm_head.output.save()",
               "tests.test_cache_activations(", "tests.test_effective_weights("]
    selected = []
    for marker in markers:
        matches = [cell for cell in cells if marker in cell]
        assert len(matches) == 1, (filename, marker)
        selected.append(matches[0])
    namespace = dict(vars(s), MAIN=True, model=model, device=t.device("cpu"), tests=tests,
                     all_data=s.all_data.cpu(), labels=s.labels.cpu())
    # NNsight inspects Python source when entering a trace.
    script = tmp_path / "grokking_exercise_cells.py"
    script.write_text("\n\n".join(selected))
    exec(compile(script.read_text(), str(script), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("filename", ["solutions.py", "solutions.ipynb", "exercises.ipynb"])
def test_restored_grokking_exercise_checks(grokking_model, filename, tmp_path):
    result = run_grokking_exercise_cells(grokking_model, filename, tmp_path)
    # Check the NNsight captures against a separate plain PyTorch forward pass.
    captured = {}
    modules = {("pattern", 0): grokking_model.blocks[0].attn.hook_pattern,
               ("pre", 0): grokking_model.blocks[0].mlp.hook_pre,
               ("post", 0): grokking_model.blocks[0].mlp.hook_post}
    handles = [module.register_forward_hook(
        lambda module, args, output, key=key: captured.__setitem__(key, output.detach().clone()))
        for key, module in modules.items()]
    try:
        grokking_model(result["all_data"])
    finally:
        for handle in handles:
            handle.remove()
    for key in modules:
        t.testing.assert_close(result["cache"][key], captured[key])


@pytest.mark.parametrize("check,index", [("cache_activations", i) for i in range(3)] +
                         [("effective_weights", i) for i in range(3)])
def test_restored_grokking_checks_reject_wrong_values(grokking_model, check, index, tmp_path):
    result = run_grokking_exercise_cells(grokking_model, "solutions.py", tmp_path)
    if check == "cache_activations":
        args = [result[name] for name in ("attn_mat", "neuron_acts_post", "neuron_acts_pre")]
        reference = result["cache"]
    else:
        args = [result[name] for name in ("W_logit", "W_neur", "W_attn")]
        reference = grokking_model
    args[index] = args[index] + 1
    with pytest.raises(AssertionError):
        getattr(result["tests"], f"test_{check}")(*args, reference)


@pytest.mark.parametrize("path", sorted([*ROOT.glob("part*/exercises.ipynb"), *ROOT.glob("part*/solutions.ipynb")]))
def test_notebook_code_compiles(path):
    """Student placeholders must still be valid Python before answers are filled in."""
    import json
    from IPython.core.inputtransformer2 import TransformerManager

    transform = TransformerManager()
    for index, cell in enumerate(json.loads(path.read_text())["cells"]):
        if cell["cell_type"] == "code":
            source = transform.transform_cell("".join(cell["source"]))
            compile(source, f"{path.name}:cell {index}", "exec")


@pytest.mark.parametrize("filename,cell_index", [("exercises.ipynb", 227), ("solutions.ipynb", 228)])
@pytest.mark.parametrize("top_layer", [0, 1])
def test_ioi_backup_head_capture_matches_pytorch(gpt2, filename, cell_index, top_layer, tmp_path):
    """Check the actual notebook trace before, at, and after the ablated layer."""
    import ast
    import json

    notebook = json.loads((ROOT / "part41_indirect_object_identification" / filename).read_text())
    tree = ast.parse("".join(notebook["cells"][cell_index]["source"]))
    trace = next(node for node in ast.walk(tree) if isinstance(node, ast.With))
    trace_file = tmp_path / "backup_head_trace.py"
    trace_file.write_text(ast.unparse(trace))
    code = compile(trace_file.read_text(), str(trace_file), "exec")
    tokens = t.tensor([[3, 4, 5], [5, 4, 3]])

    class Dataset:
        toks = tokens
        word_idx = {"end": t.tensor([2, 2])}

        def __len__(self):
            return 2

    expected = {}

    def ablate(module, args):
        z = args[0].clone()
        z[:, -1, :16] = 0
        return (z, *args[1:])

    handles = [gpt2._model.transformer.h[top_layer].attn.c_proj.register_forward_pre_hook(ablate)]
    for layer, block in enumerate(gpt2._model.transformer.h):
        def capture(module, args, index=layer):
            expected[index] = args[0].detach().clone()
        handles.append(block.attn.c_proj.register_forward_pre_hook(capture))
    try:
        gpt2._model(tokens)
    finally:
        for handle in handles:
            handle.remove()

    for layer in range(2):
        namespace = dict(model=gpt2, ioi_dataset=Dataset(), layer=layer, top_layer=top_layer,
            top_head=0, N_HEADS=2, D_HEAD=16, abc_means=t.zeros(2, 3, 2, 16))
        exec(code, namespace)
        t.testing.assert_close(namespace["z_current"], expected[layer])


def test_function_vector_notebook_matches_python(gpt2, monkeypatch, tmp_path):
    """Execute the notebook definition as written, including dictionary saving."""
    import json

    s = section("part42_function_vectors_and_model_steering")
    monkeypatch.setattr(s, "N_HEADS", 2, raising=False)
    monkeypatch.setattr(s, "D_HEAD", 16, raising=False)
    notebook = json.loads((ROOT / "part42_function_vectors_and_model_steering/solutions.ipynb").read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def calculate_fn_vectors_and_intervene(" in "".join(cell["source"]))
    source_file = tmp_path / "function_vector_notebook.py"
    source_file.write_text(source)
    namespace = dict(vars(s))
    exec(compile(source, str(source_file), "exec"), namespace)
    dataset = s.ICLDataset([["a", "b"], ["b", "c"], ["c", "a"]], size=2, n_prepended=1)
    expected = s.calculate_fn_vectors_and_intervene(gpt2, dataset, layers=[0, 1])
    actual = namespace["calculate_fn_vectors_and_intervene"](gpt2, dataset, layers=[0, 1])
    t.testing.assert_close(actual, expected)
