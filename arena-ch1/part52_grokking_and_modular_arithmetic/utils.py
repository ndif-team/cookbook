"""
Utility functions for Section 1.5: Grokking & Modular Arithmetic.
Adapted from ARENA 3.0 (Callum McDougall) for NNsight.
"""

import random
from functools import partial
from typing import Callable

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch as t
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plotly_utils import to_numpy

device = t.device("cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu")

p = 113

# ============================================================
# Fourier basis (precomputed)
# ============================================================

def make_fourier_basis(p: int) -> tuple[t.Tensor, list[str]]:
    fourier_basis = t.ones(p, p)
    fourier_basis_names = ["Const"]
    for i in range(1, p // 2 + 1):
        fourier_basis[2 * i - 1] = t.cos(2 * t.pi * t.arange(p) * i / p)
        fourier_basis[2 * i] = t.sin(2 * t.pi * t.arange(p) * i / p)
        fourier_basis_names.extend([f"cos {i}", f"sin {i}"])
    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return fourier_basis.to(device), fourier_basis_names


fourier_basis, fourier_basis_names = make_fourier_basis(p)

# ============================================================
# Training config & data
# ============================================================

lr = 1e-3
weight_decay = 1.0
fn_name = "add"
frac_train = 0.3
num_epochs = 50000
save_models = False
save_every = 100
stopping_thresh = -1
seed = 0

d_model = 128
d_mlp = 4 * d_model
num_heads = 4
d_head = d_model // num_heads
d_vocab = p + 1
n_ctx = 3
num_layers = 1

random_answers = np.random.randint(low=0, high=p, size=(p, p))
fns_dict = {
    "add": lambda x, y: (x + y) % p,
    "subtract": lambda x, y: (x - y) % p,
    "x2xyy2": lambda x, y: (x**2 + x * y + y**2) % p,
    "rand": lambda x, y: random_answers[x][y],
}
target_fn = fns_dict[fn_name]

all_data = t.tensor([(i, j, p) for i in range(p) for j in range(p)]).to(device)
labels = t.tensor([target_fn(i, j) for i, j, _ in all_data]).to(device)


def unflatten_first(tensor):
    if tensor.shape[0] == p * p:
        return einops.rearrange(tensor, "(x y) ... -> x y ...", x=p, y=p)
    else:
        return tensor


def cross_entropy_high_precision(logits, labels):
    logprobs = F.log_softmax(logits.cpu().to(t.float64), dim=-1)
    prediction_logprobs = t.gather(logprobs, index=labels.cpu()[:, None], dim=-1)
    loss = -t.mean(prediction_logprobs)
    return loss


is_train = None
is_test = None


def test_logits(logits, bias_correction=False, original_logits=None, mode="all"):
    if logits.shape[1] == p * p:
        logits = logits.T
    if logits.shape == t.Size([p * p, p + 1]):
        logits = logits[:, :-1]
    logits = logits.reshape(p * p, p)
    if bias_correction:
        original_logits = original_logits.reshape(p * p, p)
        logits = einops.reduce(original_logits - logits, "batch ... -> ...", "mean") + logits
    if mode == "train":
        return cross_entropy_high_precision(logits[is_train], labels[is_train])
    elif mode == "test":
        return cross_entropy_high_precision(logits[is_test], labels[is_test])
    elif mode == "all":
        return cross_entropy_high_precision(logits, labels)


def gen_train_test(frac_train, num, seed=0):
    pairs = [(i, j, num) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train * len(pairs))
    return pairs[:div], pairs[div:]


train, test = gen_train_test(frac_train, p, seed)

is_train = []
is_test = []
for x in range(p):
    for y in range(p):
        if (x, y, 113) in train:
            is_train.append(True)
            is_test.append(False)
        else:
            is_train.append(False)
            is_test.append(True)
is_train = np.array(is_train)
is_test = np.array(is_test)


# ============================================================
# Plotting functions
# ============================================================

def imshow(
    tensor: t.Tensor,
    xaxis=None,
    yaxis=None,
    animation_name="Snapshot",
    vline_positions=[],
    vline_labels=[],
    hline_positions=[],
    hline_labels=[],
    animation_labels=[],
    filename: str | None = None,
    **kwargs,
):
    if tensor.shape[0] == p * p:
        tensor = unflatten_first(tensor)
    tensor = t.squeeze(tensor)
    fig = px.imshow(
        to_numpy(tensor),
        labels={"x": xaxis, "y": yaxis, "animation_frame": animation_name},
        **kwargs,
    )
    if animation_labels:
        for i, label in enumerate(animation_labels):
            fig.layout.sliders[0].steps[i]["label"] = label
    for x, text in zip(vline_positions, vline_labels):
        fig.add_vline(x=x - 0.5, line_width=1, annotation_text=text, annotation_position="top left")
    for y, text in zip(hline_positions, hline_labels):
        fig.add_hline(y=y - 0.5, line_width=1, annotation_text=text, annotation_position="top left")
    y_axis, x_axis = [s for i, s in enumerate(tensor.shape) if i != kwargs.get("animation_frame", None)]
    fig.update_yaxes(range=[y_axis - 0.5, 0 - 0.5], autorange=False)
    fig.update_xaxes(range=[0 - 0.5, x_axis - 0.5], autorange=False)
    if filename is not None:
        fig.write_html(filename)
    fig.show()


imshow = partial(imshow, color_continuous_scale="Blues")
imshow_div = partial(imshow, color_continuous_scale="RdBu", color_continuous_midpoint=0.0)

inputs_heatmap: Callable = partial(
    imshow,
    xaxis="Input 1",
    yaxis="Input 2",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
)


def line(x, y=None, hover=None, xaxis="", yaxis="", filename: str | None = None, **kwargs):
    if isinstance(y, t.Tensor):
        y = to_numpy(y.flatten())
    if isinstance(x, t.Tensor):
        x = to_numpy(x.flatten())
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if x.ndim == 1:
        fig.update_layout(showlegend=False)
    if filename is not None:
        fig.write_html(filename)
    fig.show()


def scatter(x, y, title="", xaxis="", yaxis="", colorbar_title="", filename: str | None = None, **kwargs):
    fig = px.scatter(
        x=to_numpy(x.flatten()),
        y=to_numpy(y.flatten()),
        title=title,
        labels={"color": colorbar_title},
        color_continuous_scale="sunsetdark_r",
        **kwargs,
    )
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if "xaxis_range" in kwargs:
        fig.update_xaxes(range=kwargs["xaxis_range"])
    if "yaxis_range" in kwargs:
        fig.update_yaxes(range=kwargs["yaxis_range"])
    if filename is not None:
        fig.write_html(filename)
    fig.show()


def lines(
    lines_list,
    x=None,
    mode="lines",
    labels=None,
    xaxis="",
    yaxis="",
    title="",
    log_y=False,
    hover=None,
    filename: str | None = None,
    **kwargs,
):
    if isinstance(lines_list, t.Tensor):
        lines_list = [to_numpy(lines_list[i]) for i in range(lines_list.shape[0])]
    if x is None:
        x = np.arange(len(lines_list[0]))
    fig = go.Figure(layout={"title": title})
    fig.update_xaxes(title=xaxis).update_yaxes(title=yaxis).update_layout(
        width=kwargs.pop("width", None), height=kwargs.pop("height", None)
    )
    for c, line_data in enumerate(lines_list):
        if isinstance(line_data, t.Tensor):
            line_data = to_numpy(line_data)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line_data, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    if filename is not None:
        fig.write_html(filename)
    fig.show()


def animate_lines(
    lines_list,
    snapshot_index=None,
    snapshot="snapshot",
    hover=None,
    xaxis="x",
    yaxis="y",
    title="",
    filename: str | None = None,
    **kwargs,
):
    if isinstance(lines_list, list):
        lines_list = t.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    rows = []
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[1]):
            rows.append([lines_list[i][j], snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=[yaxis, snapshot, xaxis])
    fig = px.line(
        df, x=xaxis, y=yaxis, title=title, animation_frame=snapshot,
        range_y=[lines_list.min(), lines_list.max()], hover_name=hover, **kwargs,
    )
    if filename is not None:
        fig.write_html(filename)
    fig.show()


def imshow_fourier(
    tensor,
    title="",
    animation_name="snapshot",
    facet_labels=[],
    animation_labels=[],
    filename: str | None = None,
    **kwargs,
):
    if tensor.shape[0] == p * p:
        tensor = unflatten_first(tensor)
    if tuple(tensor.shape[:2]) == (p, p):
        tensor = tensor.transpose(0, 1)
    tensor = t.squeeze(tensor)
    fig = px.imshow(
        to_numpy(tensor),
        x=fourier_basis_names,
        y=fourier_basis_names,
        labels={"x": "x Component", "y": "y Component", "animation_frame": animation_name},
        title=title,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    )
    fig.update(data=[{"hovertemplate": "%{x}x * %{y}y<br>Value:%{z:.4f}"}])
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]["text"] = label
    if animation_labels:
        for i, label in enumerate(animation_labels):
            fig.layout.sliders[0].steps[i]["label"] = label
    if filename is not None:
        fig.write_html(filename)
    fig.show()


def animate_multi_lines(
    lines_list,
    y_index=None,
    snapshot_index=None,
    snapshot="snapshot",
    hover=None,
    swap_y_animate=False,
    filename: str | None = None,
    **kwargs,
):
    if isinstance(lines_list, list):
        lines_list = t.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list)
    lines_list = lines_list.transpose(2, 0, 1)
    if swap_y_animate:
        lines_list = lines_list.transpose(1, 0, 2)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if y_index is None:
        y_index = [str(i) for i in range(lines_list.shape[1])]
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    rows = []
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append(list(lines_list[i, :, j]) + [snapshot_index[i], j])
    df = pd.DataFrame(rows, columns=y_index + [snapshot, "x"])
    fig = px.line(
        df, x="x", y=y_index, animation_frame=snapshot,
        range_y=[lines_list.min(), lines_list.max()], hover_name=hover, **kwargs,
    )
    if filename is not None:
        fig.write_html(filename)
    fig.show()


def animate_scatter(
    lines_list,
    snapshot_index=None,
    snapshot="snapshot",
    hover=None,
    yaxis="y",
    xaxis="x",
    color=None,
    color_name="color",
    filename: str | None = None,
    **kwargs,
):
    if isinstance(lines_list, list):
        lines_list = t.stack(lines_list, axis=0)
    lines_list = to_numpy(lines_list)
    if snapshot_index is None:
        snapshot_index = np.arange(lines_list.shape[0])
    if hover is not None:
        hover = [i for j in range(len(snapshot_index)) for i in hover]
    if color is None:
        color = np.ones(lines_list.shape[-1])
    if isinstance(color, t.Tensor):
        color = to_numpy(color)
    if len(color.shape) == 1:
        color = einops.repeat(color, "x -> snapshot x", snapshot=lines_list.shape[0])
    rows = []
    for i in range(lines_list.shape[0]):
        for j in range(lines_list.shape[2]):
            rows.append([
                lines_list[i, 0, j].item(),
                lines_list[i, 1, j].item(),
                snapshot_index[i],
                color[i, j],
            ])
    df = pd.DataFrame(rows, columns=[xaxis, yaxis, snapshot, color_name])
    fig = px.scatter(
        df, x=xaxis, y=yaxis, animation_frame=snapshot,
        range_x=[lines_list[:, 0].min(), lines_list[:, 0].max()],
        range_y=[lines_list[:, 1].min(), lines_list[:, 1].max()],
        hover_name=hover, color=color_name, color_continuous_scale="sunsetdark_r", **kwargs,
    )
    if filename is not None:
        fig.write_html(filename)
    fig.show()


# ============================================================
# Helper function to load converted state dict into model
# ============================================================

def load_in_state_dict(model, state_dict):
    """Load a converted state dict into our GrokkingTransformer model."""
    model.load_state_dict(state_dict)
    return model
