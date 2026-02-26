# %%

import copy
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import torch as t
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenameConfig, AttnProbFunction
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm import tqdm

# Add parent directory for plotly_utils, and conversion directory for OthelloGPT model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_conversion"))

from plotly_utils import to_numpy
import part53_othellogpt.utils as utils
import part53_othellogpt.tests as tests
from convert_othello_model import OthelloGPT, load_model

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"


class FakeConfig:
    """nnterp calls model.config with __contains__ checks."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)
    def __contains__(self, item): return hasattr(self, item)
    def __getitem__(self, item): return getattr(self, item)


class SourceSoftmaxAttnProbFunction(AttnProbFunction):
    """Universal attention prob accessor using NNsight's source tracing."""
    def get_attention_prob_source(self, attention_module, return_module_source=False):
        if return_module_source: return attention_module.source
        return attention_module.source.F_softmax_0


def wrap_othello_model(model):
    """Wrap an OthelloGPT with StandardizedTransformer."""
    model.config = FakeConfig(num_attention_heads=8, hidden_size=512, vocab_size=61)
    rename_config = RenameConfig(
        layers_name="blocks", attn_name="attn", mlp_name="mlp",
        ln_final_name="ln_final", lm_head_name="unembed",
        attn_prob_source=SourceSoftmaxAttnProbFunction(),
    )
    wrapped = StandardizedTransformer(
        model, rename_config=rename_config,
        device_map=None, check_renaming=False,
    )
    wrapped.attention_probabilities.enabled = True
    return wrapped


# %%

if MAIN:
    section_dir = Path(__file__).resolve().parent

    # Load model
    model = load_model(
        str(section_dir.parent / "_conversion" / "othello_gpt_converted.pt"),
        device=str(device),
    )
    model.eval()

    # Wrap with nnterp for activation access
    nnsight_model = wrap_othello_model(model)

# %%

if MAIN:
    # An example input: 10 moves in a game
    sample_input = t.tensor([[20, 19, 18, 10, 2, 1, 27, 3, 41, 42]]).to(device)

    with t.no_grad():
        logits = model(sample_input)
    logprobs = logits.log_softmax(-1)

    assert logprobs.shape == (1, 10, 61)
    assert logprobs[0, 0].topk(3).indices.tolist() == [21, 33, 19]

# %%

if MAIN:
    MIDDLE_SQUARES = [27, 28, 35, 36]
    ALL_SQUARES = [i for i in range(64) if i not in MIDDLE_SQUARES]

    logprobs_board = t.full(size=(8, 8), fill_value=-13.0, device=device)
    logprobs_board.flatten()[ALL_SQUARES] = logprobs[0, 0, 1:]

    utils.plot_board_values(logprobs_board, title="Example Log Probs", width=500)

# %%

if MAIN:
    TOKEN_IDS_2D = np.array(
        [str(i) if i in ALL_SQUARES else "" for i in range(64)]
    ).reshape(8, 8)
    BOARD_LABELS_2D = np.array(
        ["ABCDEFGH"[i // 8] + f"{i % 8}" for i in range(64)]
    ).reshape(8, 8)

    print(TOKEN_IDS_2D)
    print(BOARD_LABELS_2D)

    utils.plot_board_values(
        t.stack([logprobs_board, logprobs_board]),
        title="Example Log Probs (with annotated token IDs)",
        width=800,
        text=np.stack([TOKEN_IDS_2D, BOARD_LABELS_2D]),
        board_titles=["Labelled by token ID", "Labelled by board label"],
    )

# %%

if MAIN:
    logprobs_multi_board = t.full(size=(10, 8, 8), fill_value=-13.0, device=device)
    logprobs_multi_board.flatten(1, -1)[:, ALL_SQUARES] = logprobs[0, :, 1:]

    utils.plot_board_values(
        logprobs_multi_board,
        title="Example Log Probs",
        width=1000,
        boards_per_row=5,
        board_titles=[f"Logprobs after move {i}" for i in range(1, 11)],
    )

# %%

if MAIN:
    board_states = t.zeros((10, 8, 8), dtype=t.int32)
    legal_moves = t.zeros((10, 8, 8), dtype=t.int32)

    board = utils.OthelloBoardState()
    for i, token_id in enumerate(sample_input.squeeze()):
        board.umpire(utils.id_to_square(token_id))
        board_states[i] = t.from_numpy(board.state)
        legal_moves[i].flatten()[board.get_valid_moves()] = 1

    legal_moves_annotation = np.where(to_numpy(legal_moves), "o", "").tolist()

    utils.plot_board_values(
        board_states,
        title="Board states",
        width=1000,
        boards_per_row=5,
        board_titles=[f"State after move {i}" for i in range(1, 11)],
        text=legal_moves_annotation,
    )

# %%

if MAIN:
    board_seqs_id = t.from_numpy(
        np.load(section_dir / "board_seqs_id_small.npy")
    ).long()
    board_seqs_square = t.from_numpy(
        np.load(section_dir / "board_seqs_square_small.npy")
    ).long()

    print(
        f"board_seqs_id: shape {tuple(board_seqs_id.shape)}, "
        f"range: {board_seqs_id.min()} to {board_seqs_id.max()}"
    )
    print(
        f"board_seqs_square: shape {tuple(board_seqs_square.shape)}, "
        f"range: {board_seqs_square.min()} to {board_seqs_square.max()}"
    )

# %%


def get_board_states_and_legal_moves(
    games_square: Int[Tensor, "n_games n_moves"],
) -> tuple[
    Int[Tensor, "n_games n_moves rows cols"],
    Int[Tensor, "n_games n_moves rows cols"],
    list,
]:
    """
    Returns:
        states:                 (n_games, n_moves, 8, 8)
        legal_moves:            (n_games, n_moves, 8, 8)
        legal_moves_annotation: list of "o" strings for legal moves
    """
    n_games, n_moves = games_square.shape
    states = t.zeros((n_games, 60, 8, 8), dtype=t.int32)
    legal_moves = t.zeros((n_games, 60, 8, 8), dtype=t.int32)

    for n in range(n_games):
        board = utils.OthelloBoardState()
        for i in range(n_moves):
            board.umpire(games_square[n, i].item())
            states[n, i] = t.from_numpy(board.state)
            legal_moves[n, i].flatten()[board.get_valid_moves()] = 1

    legal_moves_annotation = np.where(to_numpy(legal_moves), "o", "").tolist()

    return states, legal_moves, legal_moves_annotation


if MAIN:
    num_games = 50

    focus_games_id = board_seqs_id[:num_games]
    focus_games_square = board_seqs_square[:num_games]
    focus_states, focus_legal_moves, focus_legal_moves_annotation = (
        get_board_states_and_legal_moves(focus_games_square)
    )

    print("focus states:", focus_states.shape)
    print("focus_legal_moves", tuple(focus_legal_moves.shape))

    utils.plot_board_values(
        focus_states[0, :10],
        title="Board states",
        width=1000,
        boards_per_row=5,
        board_titles=[
            f"Move {i}, {'white' if i % 2 == 1 else 'black'} to play"
            for i in range(1, 11)
        ],
        text=np.where(to_numpy(focus_legal_moves[0, :10]), "o", "").tolist(),
    )

# %%


def run_with_cache(
    nnsight_model: StandardizedTransformer,
    input_ids: t.Tensor,
    names_filter: Callable | None = None,
) -> tuple[t.Tensor, dict]:
    """
    Run the model on input_ids and return (logits, cache_dict).

    cache_dict keys include:
      - ("resid_post", layer): block output [batch, seq, d_model]
      - ("attn_out", layer): attention output [batch, seq, d_model]
      - ("mlp_out", layer): MLP output [batch, seq, d_model]
      - ("post", layer, "mlp"): MLP post-activation [batch, seq, d_mlp]

    The names_filter is kept for API compatibility but ignored (we always cache
    whatever we need).
    """
    n_layers = nnsight_model._model.n_layers

    resid_posts = []
    attn_outs = []
    mlp_outs = []

    with nnsight_model.trace(input_ids):
        for i in range(n_layers):
            attn_o = nnsight_model.blocks[i].attn.output.save()
            mlp_o = nnsight_model.blocks[i].mlp.output.save()
            resid_o = nnsight_model.blocks[i].output.save()
            attn_outs.append(attn_o)
            mlp_outs.append(mlp_o)
            resid_posts.append(resid_o)
        logits_out = nnsight_model.lm_head.output.save()

    cache = {}
    for i in range(n_layers):
        cache[("resid_post", i)] = resid_posts[i]
        cache[("attn_out", i)] = attn_outs[i]
        cache[("mlp_out", i)] = mlp_outs[i]

    return logits_out, cache


def run_with_full_cache(
    nnsight_model: StandardizedTransformer,
    input_ids: t.Tensor,
) -> tuple[t.Tensor, dict]:
    """
    Like run_with_cache but also caches MLP post-activation (after GELU).
    """
    n_layers = nnsight_model._model.n_layers

    resid_posts = []
    attn_outs = []
    mlp_outs = []
    mlp_posts = []

    with nnsight_model.trace(input_ids):
        for i in range(n_layers):
            attn_o = nnsight_model.blocks[i].attn.output.save()
            mlp_o = nnsight_model.blocks[i].mlp.output.save()
            resid_o = nnsight_model.blocks[i].output.save()
            attn_outs.append(attn_o)
            mlp_outs.append(mlp_o)
            resid_posts.append(resid_o)
        logits_out = nnsight_model.lm_head.output.save()

    cache = {}
    for i in range(n_layers):
        cache[("resid_post", i)] = resid_posts[i]
        cache[("attn_out", i)] = attn_outs[i]
        cache[("mlp_out", i)] = mlp_outs[i]

    return logits_out, cache


if MAIN:
    focus_logits, focus_cache = run_with_cache(
        nnsight_model, focus_games_id[:, :-1].to(device)
    )
    print(focus_logits.shape)

# %%

# We also need MLP post-activations for neuron analysis. To get these we need
# to hook into the MLP internals. We'll compute them directly from the
# model weights and cached residual activations.


def get_mlp_post_activations(
    model: OthelloGPT,
    cache: dict,
    layer: int,
) -> t.Tensor:
    """
    Compute MLP post-GELU activations from cached residual stream.

    This replicates TL's cache["post", layer, "mlp"] which contains the
    activations after the GELU nonlinearity but before the output projection.
    """
    from convert_othello_model import LayerNormPre

    # Get the input to the MLP: apply ln2 to resid_mid
    # resid_mid = resid_post of previous layer + attn_out of this layer
    # But we don't cache resid_mid, so we compute:
    # For the MLP input, we need the residual before MLP = block input + attn_out
    # which is what ln2 receives. We can get this from block output - mlp_out.
    resid_post = cache[("resid_post", layer)]
    mlp_out = cache[("mlp_out", layer)]
    resid_mid = resid_post - mlp_out  # residual before MLP

    # Apply ln2
    ln2 = model.blocks[layer].ln2
    ln2_out = ln2(resid_mid)

    # Apply first half of MLP: W_in @ x + b_in, then GELU
    mlp = model.blocks[layer].mlp
    pre_act = t.addmm(
        mlp.b_in, ln2_out.reshape(-1, ln2_out.shape[-1]), mlp.W_in
    ).reshape(*ln2_out.shape[:-1], -1)
    import torch.nn.functional as F
    post_act = F.gelu(pre_act)
    return post_act


# %%

if MAIN:
    full_linear_probe = t.load(
        section_dir / "main_linear_probe.pth",
        map_location=str(device),
        weights_only=True,
    )

    print(full_linear_probe.shape)

    black_to_play, white_to_play, _ = (0, 1, 2)
    empty, white, black = (0, 1, 2)

# %%

if MAIN:
    # Get the "black vs white" probe directions for odd & even moves
    black_vs_white_dir_odd_moves = (
        full_linear_probe[black_to_play, :, :, :, black]
        - full_linear_probe[black_to_play, :, :, :, white]
    )
    black_vs_white_dir_even_moves = (
        full_linear_probe[white_to_play, :, :, :, black]
        - full_linear_probe[white_to_play, :, :, :, white]
    )

    all_dirs = t.stack(
        [black_vs_white_dir_odd_moves, black_vs_white_dir_even_moves]
    )
    all_dirs = einops.rearrange(
        all_dirs, "parity d_model rows cols -> d_model (parity rows cols)"
    )

    all_dirs_normed = all_dirs / all_dirs.norm(dim=0, keepdim=True)
    cosine_similarities = einops.einsum(
        all_dirs_normed,
        all_dirs_normed,
        "d_model mode_row_col_1, d_model mode_row_col_2 -> mode_row_col_1 mode_row_col_2",
    )

    fig = px.imshow(
        to_numpy(cosine_similarities),
        title="Cosine Sim of B-W Linear Probe Directions by Cell",
        x=[f"{label} (O)" for label in BOARD_LABELS_2D.flatten()]
        + [f"{label} (E)" for label in BOARD_LABELS_2D.flatten()],
        y=[f"{label} (O)" for label in BOARD_LABELS_2D.flatten()]
        + [f"{label} (E)" for label in BOARD_LABELS_2D.flatten()],
        width=900,
        height=800,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
    )
    fig.show()

# %%

if MAIN:
    linear_probe = t.stack(
        [
            full_linear_probe[
                [black_to_play, white_to_play], ..., [empty, empty]
            ].mean(0),
            full_linear_probe[
                [black_to_play, white_to_play], ..., [white, black]
            ].mean(0),
            full_linear_probe[
                [black_to_play, white_to_play], ..., [black, white]
            ].mean(0),
        ],
        dim=-1,
    )

# %%


def plot_probe_outputs(
    cache: dict,
    linear_probe: Tensor,
    layer: int,
    game_index: int,
    move: int,
    title: str = "Probe outputs",
):
    """Plot probe outputs using cached residual stream activations."""
    residual_stream = cache[("resid_post", layer)][game_index, move]
    probe_out = einops.einsum(
        residual_stream,
        linear_probe,
        "d_model, d_model row col options -> options row col",
    )

    utils.plot_board_values(
        probe_out.softmax(dim=0),
        title=title,
        width=900,
        height=400,
        board_titles=["P(Empty)", "P(Their's)", "P(Mine)"],
    )


if MAIN:
    layer = 6
    game_index = 0
    move = 29

    utils.plot_board_values(
        focus_states[game_index, move],
        title="Focus game states",
        width=400,
        height=400,
        text=focus_legal_moves_annotation[game_index][move],
    )

    plot_probe_outputs(
        focus_cache,
        linear_probe,
        layer,
        game_index,
        move,
        title="Probe outputs after move 29 (black to play)",
    )

# %%

if MAIN:
    layer = 3
    game_index = 0
    move = 29

    plot_probe_outputs(
        focus_cache,
        linear_probe,
        layer,
        game_index,
        move,
        title="Probe outputs (layer 4) after move 29 (black to play)",
    )

# %%

if MAIN:
    layer = 4
    game_index = 0
    move = 30

    utils.plot_board_values(
        focus_states[game_index, move],
        text=focus_legal_moves_annotation[game_index][move],
        title="Focus game states",
        width=400,
        height=400,
    )
    plot_probe_outputs(
        focus_cache,
        linear_probe,
        layer,
        game_index,
        move,
        title="Probe outputs (layer 4) after move 30 (white to play)",
    )

# %%

if MAIN:
    # Convert board states to "theirs vs mine" basis
    focus_states_theirs_vs_mine = (
        focus_states
        * (-1 + 2 * (t.arange(focus_states.shape[1]) % 2))[None, :, None, None]
    )
    focus_states_theirs_vs_mine[focus_states_theirs_vs_mine == 1] = 2
    focus_states_theirs_vs_mine[focus_states_theirs_vs_mine == -1] = 1

    # Get probe values at layer 6
    probe_out = einops.einsum(
        focus_cache[("resid_post", 6)],
        linear_probe,
        "game move d_model, d_model row col options -> game move row col options",
    )
    probe_predictions = probe_out.argmax(dim=-1)

    # Get accuracy
    correct_middle_odd_answers = (
        probe_predictions.cpu() == focus_states_theirs_vs_mine[:, :-1]
    )[:, 5:-5:2]
    accuracies_odd = einops.reduce(
        correct_middle_odd_answers.float(), "game move row col -> row col", "mean"
    )

    correct_middle_even_answers = (
        probe_predictions.cpu() == focus_states_theirs_vs_mine[:, :-1]
    )[:, 6:-5:2]
    accuracies_even = einops.reduce(
        correct_middle_even_answers.float(), "game move row col -> row col", "mean"
    )

    correct_middle_answers = (
        probe_predictions.cpu() == focus_states_theirs_vs_mine[:, :-1]
    )[:, 5:-5]
    accuracies = einops.reduce(
        correct_middle_answers.float(), "game move row col -> row col", "mean"
    )

    utils.plot_board_values(
        1 - t.stack([accuracies_odd, accuracies_even, accuracies], dim=0),
        title="Average Error Rate of Linear Probe",
        width=1000,
        height=400,
        board_titles=["Black to play", "White to play", "All moves"],
        zmax=0.25,
        zmin=-0.25,
    )

# %%

if MAIN:
    blank_probe = (
        linear_probe[..., 0]
        - linear_probe[..., 1] * 0.5
        - linear_probe[..., 2] * 0.5
    )
    my_probe = linear_probe[..., 2] - linear_probe[..., 1]

    tests.test_my_probes(blank_probe, my_probe, linear_probe)

# %%

if MAIN:
    game_index = 0
    move = 20

    utils.plot_board_values(
        focus_states[game_index, move],
        title="Focus game states",
        width=400,
        height=400,
        text=focus_legal_moves_annotation[game_index][move],
    )

    logprobs = t.full(size=(8, 8), fill_value=-13.0, device=device)
    logprobs.flatten()[ALL_SQUARES] = (
        focus_logits[game_index, move].log_softmax(dim=-1)[1:]
    )
    utils.plot_board_values(
        logprobs, title=f"Logprobs after move {move}", width=450, height=400
    )

# %%

if MAIN:
    cell_r = 5
    cell_c = 4
    print(f"Flipping the color of cell {'ABCDEFGH'[cell_r]}{cell_c}")

    board = utils.OthelloBoardState()
    board.update(focus_games_square[game_index, : move + 1].tolist())
    valid_moves = board.get_valid_moves()
    flipped_board = copy.deepcopy(board)
    flipped_board.state[cell_r, cell_c] *= -1
    flipped_legal_moves = flipped_board.get_valid_moves()

    newly_legal = [
        utils.square_to_label(move)
        for move in flipped_legal_moves
        if move not in valid_moves
    ]
    newly_illegal = [
        utils.square_to_label(move)
        for move in valid_moves
        if move not in flipped_legal_moves
    ]
    print("newly_legal", newly_legal)
    print("newly_illegal", newly_illegal)

# %%


def apply_scale(
    resid: Float[Tensor, "batch seq d_model"],
    flip_dir: Float[Tensor, "d_model"],
    scale: int,
    pos: int,
) -> Float[Tensor, "batch seq d_model"]:
    """
    Returns a version of the residual stream, modified by the amount `scale` in the
    direction `flip_dir` at the sequence position `pos`.
    """
    flip_dir_normed = flip_dir / flip_dir.norm()

    alpha = resid[0, pos] @ flip_dir_normed
    resid[0, pos] -= (scale + 1) * alpha * flip_dir_normed

    return resid


if MAIN:
    tests.test_apply_scale(apply_scale)

# %%

if MAIN:
    flip_dir = my_probe[:, cell_r, cell_c]

    logprobs_flipped = []
    layer = 4
    scales = [0, 1, 2, 4, 8, 16]

    for scale in scales:
        # Use NNsight intervention to apply the flip
        input_ids = focus_games_id[game_index : game_index + 1, : move + 1].to(
            device
        )

        with nnsight_model.trace(input_ids):
            # Get the residual after the target layer and modify it
            resid = nnsight_model.blocks[layer].output
            flip_dir_normed = flip_dir / flip_dir.norm()
            alpha_val = resid[0, move] @ flip_dir_normed
            nnsight_model.blocks[layer].output[0, move] = (
                resid[0, move] - (scale + 1) * alpha_val * flip_dir_normed
            )
            # Read logits
            flipped_logits_saved = nnsight_model.lm_head.output.save()

        flipped_logits = flipped_logits_saved[0, move]

        logprobs_flipped_single = (
            t.zeros((64,), dtype=t.float32, device=device) - 10.0
        )
        logprobs_flipped_single[ALL_SQUARES] = flipped_logits.log_softmax(
            dim=-1
        )[1:]
        logprobs_flipped.append(logprobs_flipped_single)

    flip_state_big = t.stack(logprobs_flipped)
    logprobs_repeated = einops.repeat(
        logprobs.flatten(), "d -> b d", b=6
    )
    color = t.zeros((len(scales), 64)) + 0.2
    color[:, utils.to_square(newly_legal)] = 1
    color[:, utils.to_square(newly_illegal)] = -1

    # Use plotly scatter instead of neel_plotly scatter
    fig = px.scatter(
        x=to_numpy(flip_state_big.flatten()),
        y=to_numpy(logprobs_repeated.flatten()),
        color=to_numpy(color.flatten()),
        hover_name=[f"{r}{c}" for r in "ABCDEFGH" for c in range(8)] * 6,
        facet_col=[i for i in range(6) for _ in range(64)],
        labels={"x": "Flipped", "y": "Original", "color": "Newly Legal"},
        title=f"Original vs Flipped {utils.square_to_label(8 * cell_r + cell_c)} at Layer {layer}",
        color_continuous_scale="Geyser",
        width=1400,
    )
    fig.show()

# %%

if MAIN:
    layer = 6
    game_index = 1
    move = 20

    utils.plot_board_values(
        focus_states[game_index, move],
        text=focus_legal_moves_annotation[game_index][move],
        title=f"Focus game #{game_index}, board after move {move}",
        width=400,
        height=400,
    )

    plot_probe_outputs(
        focus_cache,
        linear_probe,
        layer,
        game_index,
        move,
        title=f"Probe outputs (layer {layer})",
    )

# %%


def calculate_attn_and_mlp_probe_score_contributions(
    cache: dict,
    probe: Float[Tensor, "d_model rows cols"],
    layer: int,
    game_index: int,
    move: int,
) -> tuple[Float[Tensor, "layers rows cols"], Float[Tensor, "layers rows cols"]]:
    """Calculate attention and MLP contributions to probe scores."""
    attn_contributions = einops.einsum(
        t.stack(
            [cache[("attn_out", l)][game_index, move] for l in range(layer + 1)]
        ),
        probe,
        "layers d_model, d_model rows cols -> layers rows cols",
    )
    mlp_contributions = einops.einsum(
        t.stack(
            [cache[("mlp_out", l)][game_index, move] for l in range(layer + 1)]
        ),
        probe,
        "layers d_model, d_model rows cols -> layers rows cols",
    )

    return (attn_contributions, mlp_contributions)


if MAIN:
    layer = 6
    attn_contributions, mlp_contributions = (
        calculate_attn_and_mlp_probe_score_contributions(
            focus_cache, my_probe, layer, game_index, move
        )
    )

    utils.plot_board_values(
        mlp_contributions,
        title=f"MLP Contributions to my vs their (game #{game_index}, move {move})",
        board_titles=[f"Layer {i}" for i in range(layer + 1)],
        width=1400,
        height=340,
    )
    utils.plot_board_values(
        attn_contributions,
        title=f"Attn Contributions to my vs their (game #{game_index}, move {move})",
        board_titles=[f"Layer {i}" for i in range(layer + 1)],
        width=1400,
        height=340,
    )

# %%


def calculate_accumulated_probe_score(
    cache: dict,
    probe: Float[Tensor, "d_model rows cols"],
    layer: int,
    game_index: int,
    move: int,
) -> Float[Tensor, "layers rows cols"]:
    """Calculate accumulated residual stream probe scores."""
    residual_stream_score = einops.einsum(
        t.stack(
            [
                cache[("resid_post", l)][game_index, move]
                for l in range(layer + 1)
            ]
        ),
        probe,
        "layer d_model, d_model rows cols -> layer rows cols",
    )

    return residual_stream_score


if MAIN:
    residual_stream_score = calculate_accumulated_probe_score(
        focus_cache, my_probe, layer, game_index, move
    )

    utils.plot_board_values(
        residual_stream_score,
        title=f"Residual stream probe values for 'my vs their' (game #{game_index}, move {move})",
        board_titles=[f"Layer {i}" for i in range(layer + 1)],
        width=1400,
        height=340,
    )

# %%

if MAIN:
    attn_contributions, mlp_contributions = (
        calculate_attn_and_mlp_probe_score_contributions(
            focus_cache, blank_probe, layer, game_index, move
        )
    )
    utils.plot_board_values(
        mlp_contributions,
        title=f"MLP Contributions to blank probe (game #{game_index}, move {move})",
        board_titles=[f"Layer {i}" for i in range(layer + 1)],
        width=1400,
        height=340,
    )
    utils.plot_board_values(
        attn_contributions,
        title=f"Attn Contributions to blank probe (game #{game_index}, move {move})",
        board_titles=[f"Layer {i}" for i in range(layer + 1)],
        width=1400,
        height=340,
    )

    residual_stream_score = calculate_accumulated_probe_score(
        focus_cache, blank_probe, layer, game_index, move
    )
    utils.plot_board_values(
        residual_stream_score,
        title=f"Residual stream probe values for 'blank' (game #{game_index}, move {move})",
        board_titles=[f"Layer {i}" for i in range(layer + 1)],
        width=1400,
        height=340,
    )

# %%

if MAIN:
    # Scale the probes down to be unit norm per cell
    blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
    my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)

    # Set the center blank probes to 0
    blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

# %%


def get_w_in(
    model: OthelloGPT,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    """Returns the input weights for the given neuron."""
    w_in = model.blocks[layer].mlp.W_in[:, neuron].detach().clone()
    if normalize:
        w_in /= w_in.norm(dim=0, keepdim=True)
    return w_in


def get_w_out(
    model: OthelloGPT,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    """Returns the output weights for the given neuron."""
    w_out = model.blocks[layer].mlp.W_out[neuron, :].detach().clone()
    if normalize:
        w_out /= w_out.norm(dim=0, keepdim=True)
    return w_out


def calculate_neuron_input_weights(
    model: OthelloGPT,
    probe: Float[Tensor, "d_model row col"],
    layer: int,
    neuron: int,
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the input weights for the given neuron, at each square
    on the board, projected along the corresponding probe directions.
    """
    w_in = get_w_in(model, layer, neuron, normalize=True)
    return einops.einsum(w_in, probe, "d_model, d_model row col -> row col")


def calculate_neuron_output_weights(
    model: OthelloGPT,
    probe: Float[Tensor, "d_model row col"],
    layer: int,
    neuron: int,
) -> Float[Tensor, "rows cols"]:
    """
    Returns tensor of the output weights for the given neuron, at each square
    on the board, projected along the corresponding probe directions.
    """
    w_out = get_w_out(model, layer, neuron, normalize=True)
    return einops.einsum(w_out, probe, "d_model, d_model row col -> row col")


if MAIN:
    tests.test_calculate_neuron_input_weights(calculate_neuron_input_weights, model)
    tests.test_calculate_neuron_output_weights(calculate_neuron_output_weights, model)

# %%

if MAIN:
    layer = 5
    neuron = 1393

    w_in_L5N1393_blank = calculate_neuron_input_weights(
        model, blank_probe_normalised, layer, neuron
    )
    w_in_L5N1393_my = calculate_neuron_input_weights(
        model, my_probe_normalised, layer, neuron
    )

    utils.plot_board_values(
        t.stack([w_in_L5N1393_blank, w_in_L5N1393_my]),
        title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
        board_titles=["Blank In", "My In"],
        width=650,
        height=380,
    )

# %%

if MAIN:
    # Get W_U from the unembed layer (nn.Linear stores [d_vocab, d_model])
    W_U = model.unembed.weight.T  # [d_model, d_vocab]

    # Neuron output weights' cos sim with unembedding
    w_out_L5N1393 = get_w_out(model, layer, neuron, normalize=True)
    W_U_normalized = W_U[:, 1:] / W_U[:, 1:].norm(dim=0, keepdim=True)
    cos_sim = w_out_L5N1393 @ W_U_normalized

    cos_sim_rearranged = t.zeros((8, 8), device=device)
    cos_sim_rearranged.flatten()[ALL_SQUARES] = cos_sim

    utils.plot_board_values(
        cos_sim_rearranged,
        title=f"Cosine sim of neuron L{layer}N{neuron} with W<sub>U</sub> directions",
        width=450,
        height=380,
    )

# %%

if MAIN:
    w_in_L5N1393 = get_w_in(model, layer, neuron, normalize=True)
    w_out_L5N1393 = get_w_out(model, layer, neuron, normalize=True)

    U, S, Vh = t.svd(
        t.cat(
            [
                my_probe.reshape(model.d_model, 64),
                blank_probe.reshape(model.d_model, 64),
            ],
            dim=1,
        )
    )

    probe_space_basis = U[:, :-4]

    print(
        f"Fraction of input weights in probe basis: "
        f"{((w_in_L5N1393 @ probe_space_basis).pow(2).sum()):.4f}"
    )
    print(
        f"Fraction of output weights in probe basis: "
        f"{((w_out_L5N1393 @ probe_space_basis).pow(2).sum()):.4f}"
    )

# %%

if MAIN:
    # For neuron analysis, we need MLP post-activations
    # Compute them for all layers for focus games
    focus_mlp_post = {}
    for l in range(model.n_layers):
        focus_mlp_post[l] = get_mlp_post_activations(model, focus_cache, l)

# %%

if MAIN:
    layer = 3
    top_neurons = (
        focus_mlp_post[layer][:, 3:-3]
        .std(dim=[0, 1])
        .argsort(descending=True)[:10]
    )

    utils.plot_board_values(
        t.stack(
            [
                calculate_neuron_output_weights(
                    model, blank_probe_normalised, layer, n
                )
                for n in top_neurons
            ]
        ),
        title=f"Cosine sim of output weights and the 'blank color' probe for top layer {layer} neurons (by std dev)",
        board_titles=[f"L{layer}N{n.item()}" for n in top_neurons],
        width=1600,
        height=360,
    )

    utils.plot_board_values(
        t.stack(
            [
                calculate_neuron_output_weights(
                    model, my_probe_normalised, layer, n
                )
                for n in top_neurons
            ]
        ),
        title=f"Cosine sim of output weights and the 'my color' probe for top layer {layer} neurons (by std dev)",
        board_titles=[f"L{layer}N{n.item()}" for n in top_neurons],
        width=1600,
        height=360,
    )

# %%

if MAIN:
    layer = 4
    top_neurons = (
        focus_mlp_post[layer][:, 3:-3]
        .std(dim=[0, 1])
        .argsort(descending=True)[:10]
    )

    utils.plot_board_values(
        t.stack(
            [
                calculate_neuron_output_weights(
                    model, blank_probe_normalised, layer, n
                )
                for n in top_neurons
            ]
        ),
        title=f"Cosine sim of output weights and the 'blank color' probe for top layer {layer} neurons (by std dev)",
        board_titles=[f"L{layer}N{n.item()}" for n in top_neurons],
        width=1600,
        height=360,
    )

    utils.plot_board_values(
        t.stack(
            [
                calculate_neuron_output_weights(
                    model, my_probe_normalised, layer, n
                )
                for n in top_neurons
            ]
        ),
        title=f"Cosine sim of output weights and the 'my color' probe for top layer {layer} neurons (by std dev)",
        board_titles=[f"L{layer}N{n.item()}" for n in top_neurons],
        width=1600,
        height=360,
    )

# %%

if MAIN:
    layer = 4
    top_neurons = (
        focus_mlp_post[layer][:, 3:-3]
        .std(dim=[0, 1])
        .argsort(descending=True)[:10]
    )
    w_out_stack = t.stack(
        [get_w_out(model, layer, neuron, normalize=True) for neuron in top_neurons]
    )

    W_U_normalized = W_U[:, 1:] / W_U[:, 1:].norm(dim=0, keepdim=True)
    cos_sim = w_out_stack @ W_U_normalized

    cos_sim_rearranged = t.zeros((10, 8, 8), device=device)
    cos_sim_rearranged.flatten(1, -1)[:, ALL_SQUARES] = cos_sim

    utils.plot_board_values(
        cos_sim_rearranged,
        title=f"Cosine sim of top neurons with W<sub>U</sub> directions (layer {layer})",
        board_titles=[f"L{layer}N{n.item()}" for n in top_neurons],
        width=1500,
        height=320,
    )

# %%

if MAIN:
    cell_r = 5
    cell_c = 4
    print(f"Flipping the color of cell {'ABCDEFGH'[cell_r]}{cell_c}")

    board = utils.OthelloBoardState()
    board.update(focus_games_square[game_index, : move + 1].tolist())
    valid_moves = board.get_valid_moves()
    flipped_board = copy.deepcopy(board)
    flipped_board.state[cell_r, cell_c] *= -1
    flipped_legal_moves = flipped_board.get_valid_moves()

    newly_legal = [
        utils.square_to_label(move)
        for move in flipped_legal_moves
        if move not in valid_moves
    ]
    newly_illegal = [
        utils.square_to_label(move)
        for move in valid_moves
        if move not in flipped_legal_moves
    ]
    print("newly_legal", newly_legal)
    print("newly_illegal", newly_illegal)

# %%

if MAIN:
    game_index = 4
    move = 20

    original_game_id = focus_games_id[game_index, : move + 1]
    corrupted_game_id = original_game_id.clone()
    corrupted_game_id[-1] = utils.label_to_id("C0")
    original_game_square = t.tensor([utils.id_to_square(original_game_id)])
    corrupted_game_square = t.tensor([utils.id_to_square(corrupted_game_id)])

    original_state, original_legal_moves, original_legal_moves_annotation = (
        get_board_states_and_legal_moves(original_game_square)
    )
    corrupted_state, corrupted_legal_moves, corrupted_legal_moves_annotation = (
        get_board_states_and_legal_moves(corrupted_game_square)
    )
    utils.plot_board_values(
        t.stack([original_state[0, move], corrupted_state[0, move]]),
        text=[
            original_legal_moves_annotation[0][move],
            corrupted_legal_moves_annotation[0][move],
        ],
        title="Focus game states",
        board_titles=[
            "Original game (black plays E0)",
            "Corrupted game (black plays C0)",
        ],
        width=650,
        height=380,
    )

# %%

if MAIN:
    original_logits, original_cache = run_with_cache(
        nnsight_model, original_game_id.unsqueeze(0).to(device)
    )
    corrupted_logits, corrupted_cache = run_with_cache(
        nnsight_model, corrupted_game_id.unsqueeze(0).to(device)
    )

    original_log_probs = original_logits.log_softmax(dim=-1)
    corrupted_log_probs = corrupted_logits.log_softmax(dim=-1)

# %%

if MAIN:
    F0_index = utils.label_to_id("F0")
    original_F0_log_prob = original_log_probs[0, -1, F0_index]
    corrupted_F0_log_prob = corrupted_log_probs[0, -1, F0_index]

    print(
        "Check that the model predicts F0 is legal in original game & illegal in corrupted game:"
    )
    print(f"Clean log prob: {original_F0_log_prob.item():.2f}")
    print(f"Corrupted log prob: {corrupted_F0_log_prob.item():.2f}\n")


def patching_metric(
    patched_logits: Float[Tensor, "batch seq d_vocab"],
) -> Float[Tensor, ""]:
    """
    Calibrated so it equals 0 on corrupted input and 1 on original input.
    """
    patched_log_probs = patched_logits.log_softmax(dim=-1)
    return (patched_log_probs[0, -1, F0_index] - corrupted_F0_log_prob) / (
        original_F0_log_prob - corrupted_F0_log_prob
    )


if MAIN:
    tests.test_patching_metric(
        patching_metric, original_log_probs, corrupted_log_probs
    )

# %%


def get_act_patch_resid_pre(
    nnsight_model: StandardizedTransformer,
    corrupted_input: Tensor,
    clean_cache: dict,
    patching_metric: Callable,
) -> Float[Tensor, "2 n_layers"]:
    """
    Returns an array of patching results for (attn_out, mlp_out) at each layer.

    Uses NNsight to patch clean activations into the corrupted run at the final
    sequence position.
    """
    n_layers = nnsight_model._model.n_layers
    results = t.zeros(2, n_layers, device=device, dtype=t.float32)

    for layer in tqdm(range(n_layers)):
        # Patch attention output
        with nnsight_model.trace(corrupted_input):
            clean_attn = clean_cache[("attn_out", layer)][0, -1, :]
            nnsight_model.blocks[layer].attn.output[0, -1, :] = clean_attn
            patched_logits_attn = nnsight_model.lm_head.output.save()
        results[0, layer] = patching_metric(patched_logits_attn)

        # Patch MLP output
        with nnsight_model.trace(corrupted_input):
            clean_mlp = clean_cache[("mlp_out", layer)][0, -1, :]
            nnsight_model.blocks[layer].mlp.output[0, -1, :] = clean_mlp
            patched_logits_mlp = nnsight_model.lm_head.output.save()
        results[1, layer] = patching_metric(patched_logits_mlp)

    return results


if MAIN:
    patching_results = get_act_patch_resid_pre(
        nnsight_model,
        corrupted_game_id.unsqueeze(0).to(device),
        original_cache,
        patching_metric,
    )

    pd.options.plotting.backend = "plotly"
    pd.DataFrame(
        to_numpy(patching_results.T), columns=["attn", "mlp"]
    ).plot.line(
        title="Layer Output Patching Effect on F0 Log Prob",
        width=700,
        labels={"value": "Patching Effect", "index": "Layer"},
    ).show()

# %%

if MAIN:
    layer = 5
    neuron = 1393

    w_out_val = get_w_out(model, layer, neuron, normalize=False)
    w_out_W_U_basis = w_out_val @ W_U[:, 1:]

    w_out_W_U_basis_rearranged = t.zeros((8, 8), device=device)
    w_out_W_U_basis_rearranged.flatten()[ALL_SQUARES] = w_out_W_U_basis

    utils.plot_board_values(
        w_out_W_U_basis_rearranged,
        title=f"Cosine sim of neuron L{layer}N{neuron} with W<sub>U</sub> directions",
        width=450,
        height=380,
    )

# %%

if MAIN:
    c0_U = W_U[:, utils.label_to_id("C0")].detach()
    c0_U /= c0_U.norm()

    d1_U = W_U[:, utils.label_to_id("D1")].detach()
    d1_U /= d1_U.norm()

    print(f"Cosine sim of C0 and D1 unembeds: {c0_U @ d1_U:.3f}")

# %%

if MAIN:
    w_out_norm = get_w_out(model, layer, neuron, normalize=True)
    U_decomp, S_decomp, Vh_decomp = t.svd(W_U[:, 1:])
    print(
        f"Fraction of variance captured by W_U: "
        f"{((w_out_norm @ U_decomp).norm().item() ** 2):.4f}"
    )

# %%

if MAIN:
    neuron_acts = focus_mlp_post[5][:, :, neuron]

    fig = px.imshow(
        to_numpy(neuron_acts),
        title=f"L{layer}N{neuron} Activations over 50 games",
        labels={"x": "Move", "y": "Game"},
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        aspect="auto",
        width=900,
        height=400,
    )
    fig.show()

# %%

if MAIN:
    top_moves = neuron_acts > neuron_acts.quantile(0.99)
    top_focus_states = focus_states[:, :-1][top_moves.cpu()]
    top_focus_states_flip = focus_states_theirs_vs_mine[:, :-1][top_moves.cpu()]
    utils.plot_board_values(
        top_focus_states,
        boards_per_row=10,
        board_titles=[f"{act=:.2f}" for act in neuron_acts[top_moves]],
        title=f"Top 30 moves for neuron L{layer}N{neuron}",
        width=1600,
        height=500,
    )

    utils.plot_board_values(
        t.stack(
            [
                top_focus_states_flip == 0,
                top_focus_states_flip == 1,
                top_focus_states_flip == 2,
            ]
        )
        .float()
        .mean(1),
        board_titles=["Blank", "Theirs", "Mine"],
        title=f"Aggregated top 30 moves for neuron L{layer}N{neuron}, in 'blank/mine/theirs' basis",
        width=800,
        height=380,
    )

# %%

if MAIN:
    focus_states_theirs_vs_mine_pm1 = t.zeros_like(
        focus_states_theirs_vs_mine, device=device
    )
    focus_states_theirs_vs_mine_pm1[focus_states_theirs_vs_mine == 2] = 1
    focus_states_theirs_vs_mine_pm1[focus_states_theirs_vs_mine == 1] = -1

    board_state_at_top_moves = (
        focus_states_theirs_vs_mine_pm1[:, :-1][top_moves].float().mean(0)
    )

    utils.plot_board_values(
        board_state_at_top_moves,
        title=f"Aggregated top 30 moves for neuron L{layer}N{neuron}<br>(1 = theirs, -1 = mine)",
        height=380,
        width=450,
    )

# %%

if MAIN:
    layer = 5
    top_neurons = (
        focus_mlp_post[layer].std(dim=[0, 1]).argsort(descending=True)[:10]
    )
    board_states_list = []
    output_weights_in_logit_basis = []

    for neuron_idx in top_neurons:
        w_out_val = get_w_out(model, layer, neuron_idx, normalize=False)
        state = t.zeros(8, 8, device=device)
        state.flatten()[ALL_SQUARES] = w_out_val @ W_U[:, 1:]
        output_weights_in_logit_basis.append(state)

        neuron_acts_local = focus_mlp_post[5][:, :, neuron_idx]
        top_moves_local = neuron_acts_local > neuron_acts_local.quantile(0.99)
        board_state_at_top = (
            focus_states_theirs_vs_mine_pm1[:, :-1][top_moves_local]
            .float()
            .mean(0)
        )
        board_states_list.append(board_state_at_top)

    output_weights_in_logit_basis = t.stack(output_weights_in_logit_basis)
    board_states_stacked = t.stack(board_states_list)

    utils.plot_board_values(
        output_weights_in_logit_basis,
        title=f"Output weights of top 10 neurons in layer {layer}, in the output logit basis",
        board_titles=[f"L{layer}N{n.item()}" for n in top_neurons],
        width=1600,
        height=360,
    )
    utils.plot_board_values(
        board_states_stacked,
        title=f"Aggregated top 30 moves for each top 10 neuron in layer {layer}",
        board_titles=[f"L{layer}N{n.item()}" for n in top_neurons],
        width=1600,
        height=360,
    )

# %%

if MAIN:
    c0 = focus_states_theirs_vs_mine_pm1[:, :, 2, 0]
    d1 = focus_states_theirs_vs_mine_pm1[:, :, 3, 1]
    e2 = focus_states_theirs_vs_mine_pm1[:, :, 4, 2]

    label = (c0 == 0) & (d1 == -1) & (e2 == 1)

    neuron_acts = focus_mlp_post[5][:, :, 1393]

    def make_spectrum_plot(
        neuron_acts: Float[Tensor, "batch"],
        label: Bool[Tensor, "batch"],
        **kwargs,
    ) -> None:
        px.histogram(
            pd.DataFrame(
                {"acts": neuron_acts.tolist(), "label": label.tolist()}
            ),
            x="acts",
            color="label",
            histnorm="percent",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Bold,
            nbins=100,
            **kwargs,
        ).show()

    make_spectrum_plot(
        neuron_acts.flatten(),
        label[:, :-1].flatten(),
        title="Spectrum plot for neuron L5N1393 testing C0==BLANK & D1==THEIRS & E2==MINE",
        width=1200,
        height=400,
    )

# %%

if MAIN:
    utils.plot_board_values(
        focus_states[0, :16],
        boards_per_row=8,
        board_titles=[f"Move {i}" for i in range(1, 17)],
        title="First 16 moves of first game",
        width=1400,
        height=440,
    )

# %%

# Note: The probe training section uses gradient-based optimization. We keep
# model inference with NNsight for activation extraction but use standard
# PyTorch for the probe training loop.


@dataclass
class ProbeTrainingArgs:
    layer: int = 6
    pos_start: int = 5
    pos_end: int = -5

    options: int = 3
    rows: int = 8
    cols: int = 8

    epochs: int = 3
    num_games: int = 10_000

    batch_size: int = 32
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.01

    use_wandb: bool = False
    wandb_project: str | None = "othellogpt-probe"
    wandb_name: str | None = None

    def setup_linear_probe(self, model: OthelloGPT):
        linear_probe = t.randn(
            model.d_model, self.rows, self.cols, self.options, device=device
        ) / np.sqrt(model.d_model)
        linear_probe.requires_grad = True
        return linear_probe


# %%


class LinearProbeTrainer:
    def __init__(self, model: OthelloGPT, nnsight_model: StandardizedTransformer, args: ProbeTrainingArgs):
        self.model = model
        self.nnsight_model = nnsight_model
        self.args = args
        self.linear_probe = args.setup_linear_probe(model)

    def training_step(self, indices: Int[Tensor, "n_games"]) -> Float[Tensor, ""]:
        indices_cpu = indices.cpu()
        games_id = board_seqs_id[indices_cpu]
        games_square = board_seqs_square[indices_cpu]

        pos_start = self.args.pos_start
        pos_end = self.args.pos_end + self.model.n_ctx

        # Get residual stream activations using NNsight
        with t.inference_mode():
            with self.nnsight_model.trace(games_id[:, :-1].to(device)):
                resid_post = self.nnsight_model.blocks[self.args.layer].output.save()

        # Slice for even moves only
        pos_start_even = pos_start + (pos_start % 2)
        seqpos_indices = np.arange(pos_start_even, pos_end, 2)
        resid_post_sliced = resid_post[:, seqpos_indices]

        probe_logits = einops.einsum(
            resid_post_sliced,
            self.linear_probe,
            "batch pos d_model, d_model rows cols options -> batch pos rows cols options",
        )
        probe_logprobs = probe_logits.log_softmax(-1)

        state = get_board_states_and_legal_moves(games_square)[0]
        state = state[:, seqpos_indices]
        state[state == -1] = 2

        from eindex import eindex

        correct_probe_logprobs = eindex(
            probe_logprobs, state, "game pos row col [game pos row col]"
        )
        loss = -einops.reduce(
            correct_probe_logprobs, "game pos row col -> row col", "mean"
        ).sum()

        self.step += 1
        return loss

    def shuffle_training_indices(self):
        n_indices = self.args.num_games - (
            self.args.num_games % self.args.batch_size
        )
        full_train_indices = t.randperm(self.args.num_games)[:n_indices]
        full_train_indices = einops.rearrange(
            full_train_indices,
            "(batch_idx game_idx) -> batch_idx game_idx",
            game_idx=self.args.batch_size,
        )
        return full_train_indices

    def train(self):
        self.step = 0

        optimizer = t.optim.AdamW(
            [self.linear_probe],
            lr=self.args.lr,
            betas=self.args.betas,
            weight_decay=self.args.weight_decay,
        )

        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch + 1}/{self.args.epochs}")
            full_train_indices = self.shuffle_training_indices()
            progress_bar = tqdm(full_train_indices)
            for indices in progress_bar:
                loss = self.training_step(indices)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.set_description(f"Loss = {loss:.4f}")


if MAIN:
    t.set_grad_enabled(True)

    args = ProbeTrainingArgs()
    trainer = LinearProbeTrainer(model, nnsight_model, args)
    trainer.train()

# %%

if MAIN:
    # Evaluate probe on focus games
    probe_out = einops.einsum(
        focus_cache[("resid_post", args.layer)],
        trainer.linear_probe,
        "game move d_model, d_model row col options -> game move row col options",
    )
    probe_out_value = probe_out.argmax(dim=-1).cpu()

    is_correct = probe_out_value == focus_states_theirs_vs_mine[:, :-1]
    accuracies_odd = einops.reduce(
        is_correct[:, 5:-5:2].float(), "game move row col -> row col", "mean"
    )
    accuracies_even = einops.reduce(
        is_correct[:, 6:-6:2].float(), "game move row col -> row col", "mean"
    )
    accuracies_all = einops.reduce(
        is_correct[:, 5:-5].float(), "game move row col -> row col", "mean"
    )

    utils.plot_board_values(
        1 - t.stack([accuracies_odd, accuracies_even, accuracies_all], dim=0),
        title="Average Error Rate of Linear Probe",
        board_titles=["Black to play", "White to play", "All Moves"],
        zmax=0.25,
        zmin=-0.25,
        height=400,
        width=900,
    )

# %%
