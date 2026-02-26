import numpy as np
import plotly.express as px
import torch


torch.set_grad_enabled(True)

"""
File structure:

    (1) Classes to calculate othello board states & related utilities
    (2) Map between different representations of board positions / moves
    (3) Plotting & animation functions
"""


# ! (1) Classes to calculate othello board states & related utilities

rows = list("abcdefgh")
columns = [str(_) for _ in range(1, 9)]


def permit(s):
    s = s.lower()
    if len(s) != 2:
        return -1
    if s[0] not in rows or s[1] not in columns:
        return -1
    return rows.index(s[0]) * 8 + columns.index(s[1])


def permit_reverse(integer):
    r, c = integer // 8, integer % 8
    return "".join([rows[r], columns[c]])


start_hands = [permit(_) for _ in ["d5", "d4", "e4", "e5"]]
eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]


class OthelloBoardState:
    # 1 is black, -1 is white
    def __init__(self, board_size=8):
        self.board_size = board_size * board_size
        board = np.zeros((8, 8))
        board[3, 4] = 1
        board[3, 3] = -1
        board[4, 3] = 1
        board[4, 4] = -1
        self.initial_state = board
        self.state = self.initial_state
        self.age = np.zeros((8, 8))
        self.next_hand_color = 1
        self.history = []

    def get_occupied(self):
        board = self.state
        tbr = board.flatten() != 0
        return tbr.tolist()

    def get_state(self):
        board = self.state + 1  # white 0, blank 1, black 2
        tbr = board.flatten()
        return tbr.tolist()

    def get_age(self):
        return self.age.flatten().tolist()

    def get_next_hand_color(self):
        return (self.next_hand_color + 1) // 2

    def update(self, moves, prt=False):
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            if prt:
                self.__print__()

    def umpire(self, move):
        r, c = move // 8, move % 8
        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"
        color = self.next_hand_color
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) == 0:
            color *= -1
            self.next_hand_color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
        if len(tbf) == 0:
            valids = self.get_valid_moves()
            if len(valids) == 0:
                assert 0, "Both color cannot put piece, game should have ended!"
            else:
                assert 0, "Illegal move!"

        self.age += 1
        for ff in tbf:
            self.state[ff[0], ff[1]] *= -1
            self.age[ff[0], ff[1]] = 0
        self.state[r, c] = color
        self.age[r, c] = 0
        self.next_hand_color *= -1
        self.history.append(move)

    def __print__(self):
        print("-" * 20)
        print([permit_reverse(_) for _ in self.history])
        a = "abcdefgh"
        for k, row in enumerate(self.state.tolist()):
            tbp = []
            for ele in row:
                if ele == -1:
                    tbp.append("O")
                elif ele == 0:
                    tbp.append(" ")
                else:
                    tbp.append("X")
            print(" ".join([a[k]] + tbp))
        tbp = [str(k) for k in range(1, 9)]
        print(" ".join([" "] + tbp))
        print("-" * 20)

    def tentative_move(self, move):
        r, c = move // 8, move % 8
        if not self.state[r, c] == 0:
            return 0
        color = self.next_hand_color
        tbf = []
        for direction in eights:
            buffer = []
            cur_r, cur_c = r, c
            while 1:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) != 0:
            return 1
        else:
            color *= -1
            for direction in eights:
                buffer = []
                cur_r, cur_c = r, c
                while 1:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
            if len(tbf) == 0:
                return 0
            else:
                return 2

    def get_valid_moves(self):
        regular_moves = []
        forfeit_moves = []
        for move in range(64):
            x = self.tentative_move(move)
            if x == 1:
                regular_moves.append(move)
            elif x == 2:
                forfeit_moves.append(move)
        if len(regular_moves):
            return regular_moves
        elif len(forfeit_moves):
            return forfeit_moves
        else:
            return []

    def get_gt(self, moves, func, prt=False):
        container = []
        if prt:
            self.__print__()
        for _, move in enumerate(moves):
            self.umpire(move)
            container.append(getattr(self, func)())
            if prt:
                self.__print__()
        return container


# ! (2) Map between different representations of board positions / moves

MIDDLE_SQUARES = [27, 28, 35, 36]
ALL_SQUARES = [i for i in range(64) if i not in MIDDLE_SQUARES]

VOCAB = list(range(61))

ID_TO_SQUARE = {0: -100, **{id: square for id, square in enumerate(ALL_SQUARES, start=1)}}
SQUARE_TO_ID = {square: id for id, square in ID_TO_SQUARE.items()}

alpha = "ABCDEFGH"


def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"


board_labels = list(map(to_board_label, ALL_SQUARES))


def str_to_id(s):
    return SQUARE_TO_ID[s] - 1


def to_id(x):
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_id(x.item())
    elif isinstance(x, list) or isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return [to_id(i) for i in x]
    elif isinstance(x, int):
        return SQUARE_TO_ID[x]
    elif isinstance(x, str):
        x = x.upper()
        return to_id(to_square(x))


def to_square(x):
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_square(x.item())
    elif isinstance(x, list) or isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return [to_square(i) for i in x]
    elif isinstance(x, int):
        return ID_TO_SQUARE[x]
    elif isinstance(x, str):
        x = x.upper()
        return 8 * alpha.index(x[0]) + int(x[1])


def to_label(x, from_square=True):
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_label(x.item(), from_square=from_square)
    elif isinstance(x, list) or isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return [to_label(i, from_square=from_square) for i in x]
    elif isinstance(x, int):
        if from_square:
            return to_board_label(to_square(x))
        else:
            return to_board_label(x)
    elif isinstance(x, str):
        return x


def square_to_label(x):
    return to_label(x, from_square=False)


def id_to_label(x):
    return to_label(x, from_square=True)


def id_to_square(x):
    return to_square(x)


def label_to_square(x):
    return to_square(x)


def square_to_id(x):
    return to_id(x)


def label_to_id(x):
    return to_id(x)


def moves_to_state(moves):
    state = np.zeros((8, 8), dtype=bool)
    for move in moves:
        state[move // 8, move % 8] = 1.0
    return state


# ! (3) Plotting & animation functions


def to_numpy(tensor):
    """Helper to convert tensor to numpy."""
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def reorder_list_in_plotly_way(L: list, col_wrap: int):
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    return L_new


def plot_board_values(
    state: torch.Tensor,
    board_titles: list[str] | None = None,
    boards_per_row: int | None = None,
    text: list[str] | list[list[str]] | None = None,
    filename: str | None = None,
    **kwargs,
):
    state = to_numpy(state)

    if state.ndim == 3:
        boards_per_row = boards_per_row or state.shape[0]
        kwargs |= dict(facet_col=0, facet_col_wrap=boards_per_row)

    if state.dtype in [np.int64, np.int32]:
        kwargs |= dict(color_continuous_scale="Greys")
    elif state.max().item() > 0:
        kwargs |= dict(color_continuous_scale="RdBu", color_continuous_midpoint=0.0)
    else:
        kwargs |= dict(color_continuous_scale="Blues")

    fig = px.imshow(
        to_numpy(state),
        y=list("ABCDEFGH"),
        x=[str(i) for i in range(8)],
        aspect="equal",
        **kwargs,
    )

    if board_titles is not None:
        board_titles = reorder_list_in_plotly_way(board_titles, boards_per_row)
        for i, title in enumerate(board_titles):
            fig.layout.annotations[i]["text"] = title

    if text is not None:
        text_arr: np.ndarray = np.array(text)
        try:
            text_arr = np.broadcast_to(
                text_arr, state.shape if state.ndim == 3 else (1, *state.shape)
            )
        except ValueError:
            raise ValueError(
                f"Shape mismatch: {text_arr.shape=} should be broadcastable to {state.shape=}"
            )
        for i, _text in enumerate(text_arr.tolist()):
            fig.data[i].update(text=_text, texttemplate="%{text}", textfont={"size": 12})

    if state.dtype in [np.int64, np.int32]:
        fig.update_coloraxes(showscale=False)
    fig.show()
    if filename is not None:
        fig.write_html(filename)
