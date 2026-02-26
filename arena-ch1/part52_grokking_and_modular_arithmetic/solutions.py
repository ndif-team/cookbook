# %%

import sys
from functools import partial
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenameConfig, AttnProbFunction

# Setup paths
section_dir = Path(__file__).resolve().parent
conversion_dir = section_dir.parent / "_conversion"
sys.path.insert(0, str(section_dir))
sys.path.insert(0, str(section_dir.parent))

from convert_grokking_model import GrokkingTransformer, load_model
from plotly_utils import to_numpy
import part52_grokking_and_modular_arithmetic.utils as utils
import part52_grokking_and_modular_arithmetic.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu")

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"


class FakeConfig:
    """nnterp calls model.config with __contains__ checks."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)
    def __contains__(self, item): return hasattr(self, item)
    def __getitem__(self, item): return getattr(self, item)


class HookPointAttnProbFunction(AttnProbFunction):
    """For models with HookPoint modules (like GrokkingTransformer)."""
    def get_attention_prob_source(self, attention_module, return_module_source=False):
        if return_module_source: return attention_module.hook_pattern
        return attention_module.hook_pattern.output


def wrap_grokking_model(model):
    """Wrap a GrokkingTransformer with StandardizedTransformer."""
    model.config = FakeConfig(num_attention_heads=4, hidden_size=128, vocab_size=114)
    rename_config = RenameConfig(
        layers_name="blocks", attn_name="attn", mlp_name="mlp",
        ln_final_name=None, lm_head_name="unembed",
        attn_prob_source=HookPointAttnProbFunction(),
    )
    wrapped = StandardizedTransformer(
        model, rename_config=rename_config,
        device_map=None, check_renaming=False,
    )
    wrapped.attention_probabilities.enabled = True
    return wrapped


# %%

p = 113

# Module-level data (needed by metric functions even when imported)
all_data = t.tensor([(i, j, p) for i in range(p) for j in range(p)]).to(device)
labels = t.tensor([utils.target_fn(i, j) for i, j, _ in all_data]).to(device)

# Model config (matching the converted model)
n_layers = 1
d_model = 128
d_mlp = 4 * 128
n_heads = 4
d_head = d_model // n_heads
d_vocab = p + 1
n_ctx = 3

# %%

def _download_grokking_file(filename: str, local_dir: Path | None = None) -> str:
    """Download a grokking checkpoint file, using local cache if available."""
    if local_dir is not None:
        local_path = local_dir / filename
        if local_path.exists():
            return str(local_path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="woog/arena-grokking-checkpoints", filename=filename)


if MAIN:
    # Load the converted model
    model = load_model(_download_grokking_file("grokking_model_converted.pt", conversion_dir), device=str(device))
    model.eval()

# %%

if MAIN:
    # Load full run data
    full_run_data_meta = t.load(
        _download_grokking_file("grokking_model_full_run_data.pt", conversion_dir),
        map_location="cpu", weights_only=True,
    )

    # Load all converted state dicts
    all_state_dicts = t.load(
        _download_grokking_file("grokking_model_all_state_dicts.pt", conversion_dir),
        map_location="cpu", weights_only=True,
    )

    full_run_data = {
        "train_losses": full_run_data_meta["train_losses"],
        "test_losses": full_run_data_meta["test_losses"],
        "epochs": full_run_data_meta["epochs"],
        "state_dicts": all_state_dicts,
    }

# %%

if MAIN:
    utils.lines(
        lines_list=[full_run_data["train_losses"][::10], full_run_data["test_losses"]],
        labels=["train loss", "test loss"],
        title="Grokking Training Curve",
        x=np.arange(5000) * 10,
        xaxis="Epoch",
        yaxis="Loss",
        log_y=True,
        width=900,
        height=450,
    )

# %%

if MAIN:
    # Helper variables - extract weights from model directly
    # Our model stores weights in TL's internal layout:
    #   W_Q, W_K, W_V: [n_heads, d_model, d_head]
    #   W_O: [n_heads, d_head, d_model]
    #   W_in: [d_model, d_mlp]
    #   W_out: [d_mlp, d_model]
    #   embed.weight: [d_vocab, d_model]
    #   pos_embed.weight: [n_ctx, d_model]
    #   unembed.weight: [d_vocab, d_model] (nn.Linear)

    W_O = model.blocks[0].attn.W_O.data       # [n_heads, d_head, d_model]
    W_K = model.blocks[0].attn.W_K.data       # [n_heads, d_model, d_head]
    W_Q = model.blocks[0].attn.W_Q.data       # [n_heads, d_model, d_head]
    W_V = model.blocks[0].attn.W_V.data       # [n_heads, d_model, d_head]
    W_in = model.blocks[0].mlp.W_in.data      # [d_model, d_mlp]
    W_out = model.blocks[0].mlp.W_out.data    # [d_mlp, d_model]
    W_pos = model.pos_embed.weight.data        # [n_ctx, d_model]
    W_E = model.embed.weight.data[:-1]         # [p, d_model] (exclude = token)
    # For final position residual initial: embedding of "=" token + positional embedding at position 2
    final_pos_resid_initial = model.embed.weight.data[-1] + W_pos[2]
    # W_U: unembed weight is [d_vocab, d_model] in nn.Linear, but TL stores as [d_model, d_vocab]
    # We need W_U in TL convention: [d_model, d_vocab-1] (exclude = token)
    W_U = model.unembed.weight.data.T[:, :-1]  # [d_model, p]

    print("W_O  ", tuple(W_O.shape))
    print("W_K  ", tuple(W_K.shape))
    print("W_Q  ", tuple(W_Q.shape))
    print("W_V  ", tuple(W_V.shape))
    print("W_in ", tuple(W_in.shape))
    print("W_out", tuple(W_out.shape))
    print("W_pos", tuple(W_pos.shape))
    print("W_E  ", tuple(W_E.shape))
    print("W_U  ", tuple(W_U.shape))

# %%

if MAIN:
    # Run model to get logits and cached activations using NNsight
    # NOTE: NNsight requires accessing modules in forward-pass order
    wrapped = wrap_grokking_model(model)
    with wrapped.trace(all_data):
        attn_pattern = wrapped.blocks[0].attn.hook_pattern.output.save()
        neuron_acts_pre_full = wrapped.blocks[0].mlp.hook_pre.output.save()
        neuron_acts_post_full = wrapped.blocks[0].mlp.hook_post.output.save()
        original_logits_full = wrapped.lm_head.output.save()

    # Final position only, also remove the logits for `=`
    original_logits = original_logits_full[:, -1, :-1]

    # Get cross entropy loss
    original_loss = utils.cross_entropy_high_precision(original_logits, labels)
    print(f"Original loss: {original_loss.item():.3e}")

# %%

if MAIN:
    # Extract cache-equivalent activations
    attn_mat = attn_pattern[:, :, 2]   # [batch, n_heads, pos_k] - attention FROM position 2
    neuron_acts_post = neuron_acts_post_full[:, -1]  # [batch, d_mlp] - final position
    neuron_acts_pre = neuron_acts_pre_full[:, -1]    # [batch, d_mlp]

    # Test shapes
    assert attn_mat.shape == (p * p, n_heads, 3)
    assert neuron_acts_post.shape == (p * p, d_mlp)
    assert neuron_acts_pre.shape == (p * p, d_mlp)

    print("All cache activation shape tests passed!")

# %%

if MAIN:
    W_logit = W_out @ W_U

    W_OV = W_V @ W_O
    W_neur = W_E @ W_OV @ W_in

    W_QK = W_Q @ W_K.transpose(-1, -2)
    W_attn = final_pos_resid_initial @ W_QK @ W_E.T / (d_head**0.5)

    # Test shapes
    assert W_logit.shape == (d_mlp, d_vocab - 1)
    assert W_neur.shape == (n_heads, d_vocab - 1, d_mlp)
    assert W_attn.shape == (n_heads, d_vocab - 1)

    print("All effective weight shape tests passed!")

# %%

if MAIN:
    attn_mat = attn_mat[:, :, :2]  # we only care about attn from first 2 tokens to the "=" token
    attn_mat_sq = einops.rearrange(attn_mat, "(x y) head seq -> x y head seq", x=p)

    utils.inputs_heatmap(
        attn_mat_sq[..., 0],
        title="Attention score for heads at position 0",
        animation_frame=2,
        animation_name="head",
    )

# %%

if MAIN:
    neuron_acts_post_sq = einops.rearrange(neuron_acts_post, "(x y) d_mlp -> x y d_mlp", x=p)
    neuron_acts_pre_sq = einops.rearrange(neuron_acts_pre, "(x y) d_mlp -> x y d_mlp", x=p)

    top_k = 3
    utils.inputs_heatmap(
        neuron_acts_post_sq[..., :top_k],
        title=f"Activations for first {top_k} neurons",
        animation_frame=2,
        animation_name="Neuron",
    )

# %%

if MAIN:
    top_k = 5
    utils.animate_multi_lines(
        W_neur[..., :top_k],
        y_index=[f"head {hi}" for hi in range(4)],
        labels={"x": "Input token", "value": "Contribution to neuron"},
        snapshot="Neuron",
        title=f"Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attention)",
    )

# %%

if MAIN:
    utils.lines(
        W_attn,
        labels=[f"head {hi}" for hi in range(4)],
        xaxis="Input token",
        yaxis="Contribution to attn score",
        title="Contribution to attention score (pre-softmax) for each head",
    )

# %%


def make_fourier_basis(p: int) -> tuple[Tensor, list[str]]:
    """
    Returns a pair `fourier_basis, fourier_basis_names`, where `fourier_basis` is
    a `(p, p)` tensor whose rows are Fourier components and `fourier_basis_names`
    is a list of length `p` containing the names of the Fourier components (e.g.
    `["const", "cos 1", "sin 1", ...]`). You may assume that `p` is odd.
    """
    fourier_basis = t.ones(p, p)
    fourier_basis_names = ["Const"]
    for i in range(1, p // 2 + 1):
        fourier_basis[2 * i - 1] = t.cos(2 * t.pi * t.arange(p) * i / p)
        fourier_basis[2 * i] = t.sin(2 * t.pi * t.arange(p) * i / p)
        fourier_basis_names.extend([f"cos {i}", f"sin {i}"])
    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return fourier_basis.to(device), fourier_basis_names


if MAIN:
    tests.test_make_fourier_basis(make_fourier_basis)

# %%

fourier_basis, fourier_basis_names = make_fourier_basis(p)

if MAIN:
    utils.animate_lines(
        fourier_basis,
        snapshot_index=fourier_basis_names,
        snapshot="Fourier Component",
        title="Graphs of Fourier Components (Use Slider)",
    )

# %%

if MAIN:
    utils.imshow_fourier(fourier_basis @ fourier_basis.T, title="Fourier Basis Cosine Similarity Matrix")

# %%


def fft1d(x: Tensor) -> Tensor:
    """
    Returns the 1D Fourier transform of `x`, which can be a vector or a batch of vectors.

    x.shape = (..., p)
    """
    return x @ fourier_basis.T


if MAIN:
    tests.test_fft1d(fft1d)

# %%

if MAIN:
    v = sum([fourier_basis[4], fourier_basis[15] / 5, fourier_basis[67] / 10])

    utils.line(v, xaxis="Vocab basis", title="Example periodic function")
    utils.line(
        fft1d(v),
        xaxis="Fourier Basis",
        title="Fourier Transform of example function",
        hover=fourier_basis_names,
    )

# %%


def fourier_2d_basis_term(i: int, j: int) -> Float[Tensor, "p p"]:
    """
    Returns the 2D Fourier basis term corresponding to the outer product of the `i`-th component of
    the 1D Fourier basis in the `x` direction and the `j`-th component of the 1D Fourier basis in
    the `y` direction.

    Returns a 2D tensor of length `(p, p)`.
    """
    return fourier_basis[i][:, None] * fourier_basis[j][None, :]


if MAIN:
    tests.test_fourier_2d_basis_term(fourier_2d_basis_term)

# %%

if MAIN:
    x_term = 4
    y_term = 6

    utils.inputs_heatmap(
        fourier_2d_basis_term(x_term, y_term).T,
        title=f"2D Fourier Basis term {fourier_basis_names[x_term]}x {fourier_basis_names[y_term]}y",
    )

# %%


def fft2d(tensor: Tensor) -> Tensor:
    """
    Returns the components of `tensor` in the 2D Fourier basis.

    Assumes that the input has shape `(p, p, ...)`, where the
    last dimensions (if present) are the batch dims.
    Output has the same shape as the input.
    """
    return einops.einsum(tensor, fourier_basis, fourier_basis, "px py ..., i px, j py -> i j ...")


if MAIN:
    tests.test_fft2d(fft2d)

# %%

if MAIN:
    example_fn = sum([
        fourier_2d_basis_term(4, 6),
        fourier_2d_basis_term(14, 46) / 3,
        fourier_2d_basis_term(97, 100) / 6,
    ])

    utils.inputs_heatmap(example_fn.T, title="Example periodic function")

# %%

if MAIN:
    utils.imshow_fourier(fft2d(example_fn), title="Example periodic function in 2D Fourier basis")

# %%

if MAIN:
    attn_mat_fourier_basis = fft2d(attn_mat_sq)

    utils.imshow_fourier(
        attn_mat_fourier_basis[..., 0],
        title="Attention score for heads at position 0, in Fourier basis",
        animation_frame=2,
        animation_name="head",
    )

# %%

if MAIN:
    neuron_acts_post_fourier_basis = fft2d(neuron_acts_post_sq)

    top_k = 3
    utils.imshow_fourier(
        neuron_acts_post_fourier_basis[..., :top_k],
        title=f"Activations for first {top_k} neurons",
        animation_frame=2,
        animation_name="Neuron",
    )

# %%


def fft1d_given_dim(tensor: Tensor, dim: int) -> Tensor:
    """
    Performs 1D FFT along the given dimension (not necessarily the last one).
    """
    return fft1d(tensor.transpose(dim, -1)).transpose(dim, -1)


if MAIN:
    W_neur_fourier = fft1d_given_dim(W_neur, dim=1)

    top_k = 5
    utils.animate_multi_lines(
        W_neur_fourier[..., :top_k],
        y_index=[f"head {hi}" for hi in range(4)],
        labels={"x": "Fourier component", "value": "Contribution to neuron"},
        snapshot="Neuron",
        hover=fourier_basis_names,
        title=f"Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attn), in Fourier basis",
    )

# %%

if MAIN:
    utils.lines(
        fft1d(W_attn),
        labels=[f"Head {hi}" for hi in range(4)],
        xaxis="Input token",
        yaxis="Contribution to attn score",
        title="Contribution to attn score (pre-softmax) for each head, in Fourier Basis",
        hover=fourier_basis_names,
    )

# %%

if MAIN:
    utils.line(
        (fourier_basis @ W_E).pow(2).sum(1),
        hover=fourier_basis_names,
        title="Norm of embedding of each Fourier Component",
        xaxis="Fourier Component",
        yaxis="Norm",
    )

# %%

if MAIN:
    neuron_acts_centered = neuron_acts_post_sq - neuron_acts_post_sq.mean((0, 1), keepdim=True)
    neuron_acts_centered_fourier = fft2d(neuron_acts_centered)

    utils.imshow_fourier(
        neuron_acts_centered_fourier.pow(2).mean(-1),
        title="Norms of 2D Fourier components of centered neuron activations",
    )

# %%

if MAIN:
    from sklearn.linear_model import LinearRegression

    k = 42
    idx = 2 * k - 1
    vec = fourier_basis[idx]

    relu_func_values = F.relu(0.5 * (p**-0.5) + vec[None, :] + vec[:, None])

    data = t.stack(
        [fourier_2d_basis_term(i, j) for (i, j) in [(0, 0), (idx, 0), (0, idx), (idx, idx)]], dim=-1
    )

    data_np = to_numpy(data.reshape(p * p, 4))
    relu_func_values_np = to_numpy(relu_func_values.flatten())

    reg = LinearRegression(fit_intercept=False).fit(data_np, relu_func_values_np)
    coefs = reg.coef_
    r2 = reg.score(data_np, relu_func_values_np)
    print(
        "ReLU(0.5 + cos(wx) + cos(wy)) approx {:.3f}*const + {:.3f}*cos(wx) + {:.3f}*cos(wy) + {:.3f}*cos(wx)cos(wy)".format(
            *coefs
        )
    )
    print(f"r2: {r2:.3f}")

    data_np = data_np[:, :3]
    reg = LinearRegression().fit(data_np, relu_func_values_np)
    coefs = reg.coef_
    bias = reg.intercept_
    r2 = reg.score(data_np, relu_func_values_np)
    print(f"r2 (no quadratic term): {r2:.3f}")

# %%


def arrange_by_2d_freqs(tensor: Tensor) -> Tensor:
    """
    Takes a tensor of shape (p, p, *batch_dims), returns tensor of shape (p//2 - 1, 3, 3, *batch_dims)
    representing the Fourier coefficients sorted by frequency.
    """
    idx_2d_y_all = []
    idx_2d_x_all = []
    for freq in range(1, p // 2):
        idx_1d = [0, 2 * freq - 1, 2 * freq]
        idx_2d_x_all.append([idx_1d for _ in range(3)])
        idx_2d_y_all.append([[i] * 3 for i in idx_1d])
    return tensor[idx_2d_y_all, idx_2d_x_all]


def find_neuron_freqs(
    fourier_neuron_acts: Float[Tensor, "p p d_mlp"],
) -> tuple[Float[Tensor, "d_mlp"], Float[Tensor, "d_mlp"]]:
    """
    Returns the tensors `neuron_freqs` and `neuron_frac_explained`, containing the frequencies that
    explain the most variance of each neuron and the fraction of variance explained, respectively.
    """
    fourier_neuron_acts_by_freq = arrange_by_2d_freqs(fourier_neuron_acts)
    assert fourier_neuron_acts_by_freq.shape == (p // 2 - 1, 3, 3, utils.d_mlp)

    square_of_all_terms = einops.reduce(
        fourier_neuron_acts.pow(2), "x_coeff y_coeff neuron -> neuron", "sum"
    )

    square_of_each_freq = einops.reduce(
        fourier_neuron_acts_by_freq.pow(2), "freq x_coeff y_coeff neuron -> freq neuron", "sum"
    )

    neuron_variance_explained, neuron_freqs = square_of_each_freq.max(0)
    neuron_frac_explained = neuron_variance_explained / square_of_all_terms

    neuron_freqs += 1

    return neuron_freqs, neuron_frac_explained


if MAIN:
    neuron_freqs, neuron_frac_explained = find_neuron_freqs(neuron_acts_centered_fourier)
    key_freqs, neuron_freq_counts = t.unique(neuron_freqs, return_counts=True)

    assert key_freqs.tolist() == [14, 35, 41, 42, 52]

    print("All tests for `find_neuron_freqs` passed!")

# %%

if MAIN:
    # Get fraction of activations positive at position 2
    fraction_of_activations_positive_at_posn2 = (neuron_acts_pre > 0).float().mean(0)

    utils.scatter(
        x=neuron_freqs,
        y=neuron_frac_explained,
        xaxis="Neuron frequency",
        yaxis="Frac explained",
        colorbar_title="Frac positive",
        title="Fraction of neuron activations explained by key freq",
        color=to_numpy(fraction_of_activations_positive_at_posn2),
    )

# %%

if MAIN:
    neuron_freqs[neuron_frac_explained < 0.85] = -1.0
    key_freqs_plus = t.concatenate([key_freqs, -key_freqs.new_ones((1,))])

    for i, k in enumerate(key_freqs_plus):
        print(f"Cluster {i}: freq k={k}, {(neuron_freqs == k).sum()} neurons")

# %%

if MAIN:
    fourier_norms_in_each_cluster = []
    for freq in key_freqs:
        fourier_norms_in_each_cluster.append(
            einops.reduce(
                neuron_acts_centered_fourier.pow(2)[..., neuron_freqs == freq],
                "batch_y batch_x neuron -> batch_y batch_x",
                "mean",
            )
        )

    utils.imshow_fourier(
        t.stack(fourier_norms_in_each_cluster),
        title="Norm of 2D Fourier components of neuron activations in each cluster",
        facet_col=0,
        facet_labels=[f"Freq={freq}" for freq in key_freqs],
    )

# %%


def project_onto_direction(batch_vecs: Tensor, v: Tensor) -> Tensor:
    """
    Returns the component of each vector in `batch_vecs` in the direction of `v`.

    batch_vecs.shape = (n, ...)
    v.shape = (n,)
    """
    components_in_v_dir = einops.einsum(batch_vecs, v, "n ..., n -> ...")
    return einops.einsum(components_in_v_dir, v, "..., n -> n ...")


if MAIN:
    tests.test_project_onto_direction(project_onto_direction)

# %%


def project_onto_frequency(batch_vecs: Tensor, freq: int) -> Tensor:
    """
    Returns the projection of each vector in `batch_vecs` onto the
    2D Fourier basis directions corresponding to frequency `freq`.

    batch_vecs.shape = (p**2, ...)
    """
    assert batch_vecs.shape[0] == p**2

    return sum([
        project_onto_direction(
            batch_vecs,
            fourier_2d_basis_term(i, j).flatten(),
        )
        for i in [0, 2 * freq - 1, 2 * freq]
        for j in [0, 2 * freq - 1, 2 * freq]
    ])


if MAIN:
    tests.test_project_onto_frequency(project_onto_frequency)

# %%

if MAIN:
    logits_in_freqs = []

    for freq in key_freqs:
        filtered_neuron_acts = neuron_acts_post[:, neuron_freqs == freq]
        filtered_neuron_acts_in_freq = project_onto_frequency(filtered_neuron_acts, freq)
        logits_in_freq = filtered_neuron_acts_in_freq @ W_logit[neuron_freqs == freq]
        logits_in_freqs.append(logits_in_freq)

    logits_always_firing = neuron_acts_post[:, neuron_freqs == -1] @ W_logit[neuron_freqs == -1]
    logits_in_freqs.append(logits_always_firing)

    key_freq_loss = utils.test_logits(
        sum(logits_in_freqs), bias_correction=True, original_logits=original_logits
    )
    key_freq_loss_no_always_firing = utils.test_logits(
        sum(logits_in_freqs[:-1]), bias_correction=True, original_logits=original_logits
    )

    print(f"""Loss with neuron activations ONLY in key freq (including always firing cluster): {key_freq_loss:.3e}
    Loss with neuron activations ONLY in key freq (excluding always firing cluster): {key_freq_loss_no_always_firing:.3e}
    Original loss: {original_loss:.3e}
    """)

# %%

if MAIN:
    print("Loss with neuron activations excluding none:     {:.9f}".format(original_loss.item()))
    for c, freq in enumerate(key_freqs_plus):
        print(
            "Loss with neuron activations excluding freq={}:  {:.9f}".format(
                freq,
                utils.test_logits(
                    sum(logits_in_freqs) - logits_in_freqs[c],
                    bias_correction=True,
                    original_logits=original_logits,
                ),
            )
        )

# %%

if MAIN:
    utils.imshow_fourier(
        einops.reduce(neuron_acts_centered_fourier.pow(2), "y x neuron -> y x", "mean"),
        title="Norm of Fourier Components of Neuron Acts",
    )

    original_logits_sq = einops.rearrange(original_logits, "(x y) z -> x y z", x=p)
    original_logits_fourier = fft2d(original_logits_sq)
    utils.imshow_fourier(
        einops.reduce(original_logits_fourier.pow(2), "y x z -> y x", "mean"),
        title="Norm of Fourier Components of Logits",
    )

# %%


def get_trig_sum_directions(k: int) -> tuple[Float[Tensor, "p p"], Float[Tensor, "p p"]]:
    """
    Given frequency k, returns the normalized vectors in the 2D Fourier basis representing the
    two directions:

        cos(omega_k * (x + y))
        sin(omega_k * (x + y))

    respectively.
    """
    cosx_cosy_direction = fourier_2d_basis_term(2 * k - 1, 2 * k - 1)
    sinx_siny_direction = fourier_2d_basis_term(2 * k, 2 * k)
    sinx_cosy_direction = fourier_2d_basis_term(2 * k, 2 * k - 1)
    cosx_siny_direction = fourier_2d_basis_term(2 * k - 1, 2 * k)

    cos_xplusy_direction = (cosx_cosy_direction - sinx_siny_direction) / np.sqrt(2)
    sin_xplusy_direction = (sinx_cosy_direction + cosx_siny_direction) / np.sqrt(2)

    return cos_xplusy_direction, sin_xplusy_direction


if MAIN:
    tests.test_get_trig_sum_directions(get_trig_sum_directions)

# %%

if MAIN:
    trig_logits = []

    for k in key_freqs:
        cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(k)
        cos_xplusy_projection = project_onto_direction(
            original_logits, cos_xplusy_direction.flatten()
        )
        sin_xplusy_projection = project_onto_direction(
            original_logits, sin_xplusy_direction.flatten()
        )

        trig_logits.extend([cos_xplusy_projection, sin_xplusy_projection])

    trig_logits = sum(trig_logits)

    print(
        f"Loss with just x+y components: {utils.test_logits(trig_logits, True, original_logits):.4e}"
    )
    print(f"Original Loss: {original_loss:.4e}")

# %%

if MAIN:
    US = W_logit @ fourier_basis.T

    utils.imshow_div(
        US, x=fourier_basis_names, yaxis="Neuron index", title="W_logit in the Fourier Basis"
    )

# %%

if MAIN:
    US_sorted = t.concatenate([US[neuron_freqs == freq] for freq in key_freqs_plus])
    hline_positions = np.cumsum(
        [(neuron_freqs == freq).sum().item() for freq in key_freqs]
    ).tolist() + [d_mlp]

    utils.imshow_div(
        US_sorted,
        x=fourier_basis_names,
        yaxis="Neuron",
        title="W_logit in the Fourier Basis (rearranged by neuron cluster)",
        hline_positions=hline_positions,
        hline_labels=[f"Cluster: freq={freq}" for freq in key_freqs.tolist()] + ["No freq"],
    )

# %%

if MAIN:
    cos_components = []
    sin_components = []

    for k in key_freqs:
        sigma_u_sin = US[:, 2 * k]
        sigma_u_cos = US[:, 2 * k - 1]

        logits_in_cos_dir = neuron_acts_post_sq @ sigma_u_cos
        logits_in_sin_dir = neuron_acts_post_sq @ sigma_u_sin

        cos_components.append(fft2d(logits_in_cos_dir))
        sin_components.append(fft2d(logits_in_sin_dir))

    for title, components in zip(["Cosine", "Sine"], [cos_components, sin_components]):
        utils.imshow_fourier(
            t.stack(components),
            title=f"{title} components of neuron activations in Fourier basis",
            animation_frame=0,
            animation_name="Frequency",
            animation_labels=key_freqs.tolist(),
        )

# %%

# ============================================================
# Section 3: Training Dynamics / Progress Measures
# ============================================================

metric_cache = {}


def get_metrics(model, metric_cache, metric_fn, name, reset=False):
    """
    Define a metric (by metric_fn) and add it to the cache, with the name `name`.
    """
    if reset or (name not in metric_cache) or (len(metric_cache[name]) == 0):
        metric_cache[name] = []
        for sd in tqdm(full_run_data["state_dicts"]):
            model = utils.load_in_state_dict(model, sd)
            model = model.to(device)
            out = metric_fn(model)
            if isinstance(out, Tensor):
                out = to_numpy(out)
            metric_cache[name].append(out)
        # Restore the final model (epoch 400)
        model = utils.load_in_state_dict(model, full_run_data["state_dicts"][400])
        model = model.to(device)
        metric_cache[name] = t.tensor(np.array(metric_cache[name]))


def test_loss(model):
    logits = model(all_data)[:, -1, :-1]
    return utils.test_logits(logits, False, mode="test")


def train_loss(model):
    logits = model(all_data)[:, -1, :-1]
    return utils.test_logits(logits, False, mode="train")


if MAIN:
    epochs = full_run_data["epochs"]
    plot_metric = partial(utils.lines, x=epochs, xaxis="Epoch")

    get_metrics(model, metric_cache, test_loss, "test_loss")
    get_metrics(model, metric_cache, train_loss, "train_loss")

# %%


def excl_loss(model: GrokkingTransformer, key_freqs: list) -> list:
    """
    Returns the excluded loss (i.e. subtracting the components of logits corresponding to
    cos(w_k(x+y)) and sin(w_k(x+y)), for each frequency k in key_freqs.
    """
    excl_loss_list = []
    logits = model(all_data)[:, -1, :-1]
    for freq in key_freqs:
        cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(freq)

        logits_cos_xplusy = project_onto_direction(logits, cos_xplusy_direction.flatten())
        logits_sin_xplusy = project_onto_direction(logits, sin_xplusy_direction.flatten())
        logits_excl = logits - logits_cos_xplusy - logits_sin_xplusy

        loss = utils.test_logits(logits_excl, bias_correction=False, mode="train").item()
        excl_loss_list.append(loss)

    return excl_loss_list


if MAIN:
    tests.test_excl_loss(excl_loss, model, key_freqs)

# %%

if MAIN:
    get_metrics(model, metric_cache, partial(excl_loss, key_freqs=key_freqs), "excl_loss")

    plot_metric(
        t.concat([
            metric_cache["excl_loss"].T,
            metric_cache["train_loss"][None, :],
            metric_cache["test_loss"][None, :],
        ]),
        labels=[f"excl {freq}" for freq in key_freqs] + ["train", "test"],
        title="Excluded Loss for each trig component",
        log_y=True,
        yaxis="Loss",
    )

# %%


def fourier_embed(model: GrokkingTransformer):
    """
    Returns norm of Fourier transform of the model's embedding matrix.
    """
    W_E_fourier = fourier_basis.T @ model.embed.weight.data[:-1]
    return einops.reduce(W_E_fourier.pow(2), "vocab d_model -> vocab", "sum")


if MAIN:
    tests.test_fourier_embed(fourier_embed, model)

# %%

if MAIN:
    get_metrics(model, metric_cache, fourier_embed, "fourier_embed")

    utils.animate_lines(
        metric_cache["fourier_embed"][::2],
        snapshot_index=epochs[::2],
        snapshot="Epoch",
        hover=fourier_basis_names,
        animation_group="x",
        title="Norm of Fourier Components in the Embedding Over Training",
    )

# %%


def embed_SVD(model: GrokkingTransformer) -> Tensor:
    """
    Returns vector S, where W_E = U @ diag(S) @ V.T in singular value decomp.
    """
    U, S, V = t.svd(model.embed.weight.data[:, :-1])
    return S


if MAIN:
    tests.test_embed_SVD(embed_SVD, model)

# %%

if MAIN:
    get_metrics(model, metric_cache, embed_SVD, "embed_SVD")

    utils.animate_lines(
        metric_cache["embed_SVD"],
        snapshot_index=epochs,
        snapshot="Epoch",
        title="Singular Values of the Embedding During Training",
        xaxis="Singular Number",
        yaxis="Singular Value",
    )

# %%


def tensor_trig_ratio(model: GrokkingTransformer, mode: str):
    """
    Returns the fraction of variance of the (centered) activations which is explained by the Fourier
    directions corresponding to cos(omega(x+y)) and sin(omega(x+y)) for all the key frequencies.
    """
    # Get activations via NNsight (access in forward-pass order)
    wrapped = wrap_grokking_model(model)
    with wrapped.trace(all_data):
        pre_acts = wrapped.blocks[0].mlp.hook_pre.output.save()
        post_acts = wrapped.blocks[0].mlp.hook_post.output.save()
        logits_full = wrapped.lm_head.output.save()

    logits = logits_full[:, -1, :-1]

    if mode == "neuron_pre":
        tensor = pre_acts[:, -1]
    elif mode == "neuron_post":
        tensor = post_acts[:, -1]
    elif mode == "logit":
        tensor = logits
    else:
        raise ValueError(f"{mode} is not a valid mode")

    tensor_centered = tensor - einops.reduce(tensor, "xy index -> 1 index", "mean")
    tensor_var = einops.reduce(tensor_centered.pow(2), "xy index -> index", "sum")
    tensor_trig_vars = []

    for freq in key_freqs:
        cos_xplusy_direction, sin_xplusy_direction = get_trig_sum_directions(freq)
        cos_xplusy_projection_var = (
            project_onto_direction(tensor_centered, cos_xplusy_direction.flatten()).pow(2).sum(0)
        )
        sin_xplusy_projection_var = (
            project_onto_direction(tensor_centered, sin_xplusy_direction.flatten()).pow(2).sum(0)
        )
        tensor_trig_vars.extend([cos_xplusy_projection_var, sin_xplusy_projection_var])

    return to_numpy(sum(tensor_trig_vars) / tensor_var)


if MAIN:
    for mode in ["neuron_pre", "neuron_post", "logit"]:
        get_metrics(
            model,
            metric_cache,
            partial(tensor_trig_ratio, mode=mode),
            f"{mode}_trig_ratio",
            reset=True,
        )

    lines_list = []
    line_labels = []
    for mode in ["neuron_pre", "neuron_post", "logit"]:
        tensor = metric_cache[f"{mode}_trig_ratio"]
        lines_list.append(einops.reduce(tensor, "epoch index -> epoch", "mean"))
        line_labels.append(f"{mode}_trig_frac")

    plot_metric(
        lines_list,
        labels=line_labels,
        log_y=False,
        yaxis="Ratio",
        title="Fraction of logits and neurons explained by trig terms",
    )

# %%


def get_frac_explained(model: GrokkingTransformer) -> Tensor:
    # Get activations via NNsight
    wrapped = wrap_grokking_model(model)
    with wrapped.trace(all_data):
        pre_acts = wrapped.blocks[0].mlp.hook_pre.output.save()
        post_acts = wrapped.blocks[0].mlp.hook_post.output.save()

    returns = []

    for neuron_type, acts_full in [("pre", pre_acts), ("post", post_acts)]:
        neuron_acts = acts_full[:, -1].clone().detach()
        neuron_acts_centered = neuron_acts - neuron_acts.mean(0)
        neuron_acts_fourier = fft2d(
            einops.rearrange(neuron_acts_centered, "(x y) neuron -> x y neuron", x=p)
        )

        square_of_all_terms = einops.reduce(
            neuron_acts_fourier.pow(2), "x y neuron -> neuron", "sum"
        )

        frac_explained = t.zeros(utils.d_mlp).to(device)
        frac_explained_quadratic_terms = t.zeros(utils.d_mlp).to(device)

        for freq in key_freqs_plus:
            acts_fourier = arrange_by_2d_freqs(neuron_acts_fourier[..., neuron_freqs == freq])

            if freq == -1:
                squares_for_this_freq = squares_for_this_freq_quadratic_terms = einops.reduce(
                    acts_fourier[:, 1:, 1:].pow(2), "freq x y neuron -> neuron", "sum"
                )
            else:
                squares_for_this_freq = einops.reduce(
                    acts_fourier[freq - 1].pow(2), "x y neuron -> neuron", "sum"
                )
                squares_for_this_freq_quadratic_terms = einops.reduce(
                    acts_fourier[freq - 1, 1:, 1:].pow(2), "x y neuron -> neuron", "sum"
                )

            frac_explained[neuron_freqs == freq] = (
                squares_for_this_freq / square_of_all_terms[neuron_freqs == freq]
            )
            frac_explained_quadratic_terms[neuron_freqs == freq] = (
                squares_for_this_freq_quadratic_terms / square_of_all_terms[neuron_freqs == freq]
            )

        returns.extend([frac_explained, frac_explained_quadratic_terms])

    frac_active = (pre_acts[:, -1] > 0).float().mean(0)

    return t.nan_to_num(t.stack(returns + [neuron_freqs, frac_active], axis=0))


if MAIN:
    get_metrics(model, metric_cache, get_frac_explained, "get_frac_explained")

    frac_explained_pre = metric_cache["get_frac_explained"][:, 0]
    frac_explained_quadratic_pre = metric_cache["get_frac_explained"][:, 1]
    frac_explained_post = metric_cache["get_frac_explained"][:, 2]
    frac_explained_quadratic_post = metric_cache["get_frac_explained"][:, 3]
    neuron_freqs_ = metric_cache["get_frac_explained"][:, 4]
    frac_active = metric_cache["get_frac_explained"][:, 5]

    utils.animate_scatter(
        t.stack([frac_explained_quadratic_pre, frac_explained_quadratic_post], dim=1)[:200:5],
        color=neuron_freqs_[:200:5],
        color_name="freq",
        snapshot="epoch",
        snapshot_index=epochs[:200:5],
        xaxis="Quad ratio pre",
        yaxis="Quad ratio post",
        title="Fraction of variance explained by quadratic terms (up to epoch 20K)",
    )

    utils.animate_scatter(
        t.stack([neuron_freqs_, frac_explained_pre, frac_explained_post], dim=1)[:200:5],
        color=frac_active[:200:5],
        color_name="frac_active",
        snapshot="epoch",
        snapshot_index=epochs[:200:5],
        xaxis="Freq",
        yaxis="Frac explained",
        hover=list(range(utils.d_mlp)),
        title="Fraction of variance explained by this frequency (up to epoch 20K)",
    )

# %%


def avg_attn_pattern(model: GrokkingTransformer):
    wrapped = wrap_grokking_model(model)
    with wrapped.trace(all_data):
        pattern = wrapped.blocks[0].attn.hook_pattern.output.save()

    return to_numpy(
        einops.reduce(pattern[:, :, 2], "batch head pos -> head pos", "mean")
    )


if MAIN:
    get_metrics(model, metric_cache, avg_attn_pattern, "avg_attn_pattern")

    utils.imshow_div(
        metric_cache["avg_attn_pattern"][::5],
        animation_frame=0,
        animation_name="head",
        title="Avg attn by position and head, snapped every 100 epochs",
        xaxis="Pos",
        yaxis="Head",
        zmax=0.5,
        zmin=0.0,
        color_continuous_scale="Blues",
        text_auto=".3f",
    )

# %%

if MAIN:
    utils.lines(
        (metric_cache["avg_attn_pattern"][:, :, 0] - metric_cache["avg_attn_pattern"][:, :, 1]).T,
        labels=[f"Head {i}" for i in range(4)],
        x=epochs,
        xaxis="Epoch",
        yaxis="Average difference",
        title="Attention to pos 0 - pos 1 by head over training",
        width=900,
        height=450,
    )

# %%


def trig_loss(model: GrokkingTransformer, mode: str = "all"):
    logits = model(all_data)[:, -1, :-1]

    trig_logits = []
    for freq in key_freqs:
        cos_xplusy_dir, sin_xplusy_dir = get_trig_sum_directions(freq)
        cos_xplusy_proj = project_onto_direction(logits, cos_xplusy_dir.flatten())
        sin_xplusy_proj = project_onto_direction(logits, sin_xplusy_dir.flatten())
        trig_logits.extend([cos_xplusy_proj, sin_xplusy_proj])
    trig_logits = sum(trig_logits)

    return utils.test_logits(trig_logits, bias_correction=True, original_logits=logits, mode=mode)


if MAIN:
    get_metrics(model, metric_cache, trig_loss, "trig_loss")
    get_metrics(model, metric_cache, partial(trig_loss, mode="train"), "trig_loss_train")

    line_labels = ["test_loss", "train_loss", "trig_loss", "trig_loss_train"]
    plot_metric(
        [metric_cache[lab] for lab in line_labels],
        labels=line_labels,
        title="Different losses over training",
    )
    plot_metric(
        [metric_cache["test_loss"] / metric_cache["trig_loss"]],
        title="Ratio of trig and test loss",
    )

# %%


def sum_sq_weights(model):
    return [param.pow(2).sum().item() for name, param in model.named_parameters()]


if MAIN:
    get_metrics(model, metric_cache, sum_sq_weights, "sum_sq_weights")

    plot_metric(
        metric_cache["sum_sq_weights"].T,
        title="Sum of squared weights for each parameter",
        labels=[name.split(".")[-1] for name, _ in model.named_parameters()],
        log_y=False,
    )
    plot_metric(
        [einops.reduce(metric_cache["sum_sq_weights"], "epoch param -> epoch", "sum")],
        title="Total sum of squared weights",
        log_y=False,
    )

# %%
