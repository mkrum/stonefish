import chess
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

import stonefish.utils as ut
from stonefish.rep import MoveRep
import matplotlib.patches as patches

symbols = {
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
    "P": "♙",
    "k": "♚",
    "q": "♛",
    "r": "♜",
    "b": "♝",
    "n": "♞",
    "p": "♟︎",
}


def make_board(ax):
    """
    Adds checkers to the board to make it look more natural
    """
    X, Y = np.meshgrid(np.arange(8), np.arange(8))
    checker = (((X + Y) % 2) + 0.3) / 2
    ax.imshow(checker, cmap="Greys", vmax=1.0, vmin=0.0)


def add_piece(ax, x, y, piece, alpha=1.0, color="black"):
    """
    Adds a pieces to the board, x and y should be the war x,y coords
    """
    ax.text(
        x,
        y + 0.05,
        symbols[piece],
        fontsize=32,
        ha="center",
        va="center",
        alpha=alpha,
        color=color,
    )


def add_arrow(ax, from_x, from_y, to_x, to_y, alpha=1.0, color="black"):
    """
    Adds a pieces to the board, x and y should be the war x,y coords
    """
    ax.arrow(
        from_x,
        from_y + 0.05,
        to_x - from_x,
        to_y - from_y,
        alpha=alpha,
        color=color,
        head_width=0.1,
    )


def plot_board(ax, board, checkers=True):
    """
    Creates a board image on the specified axis
    """
    ax.set_xlim([-0.5, 7.5])
    ax.set_ylim([7.5, -0.5])
    for i in range(8):
        ax.axhline(i - 0.5, 0, 8, color="black")
        ax.axvline(i - 0.5, 0, 8, color="black")

    ax.tick_params(labeltop=True, labelright=True, length=0)

    ax.set_yticks(list(range(8)))
    ax.set_xticks(list(range(8)))
    ax.set_yticklabels(("8", "7", "6", "5", "4", "3", "2", "1"))
    ax.set_xticklabels(("a", "b", "c", "d", "e", "f", "g", "h"))

    if checkers:
        make_board(ax)

    X, Y = np.meshgrid(np.arange(8), np.arange(8))
    locs = np.stack([X.flatten(), Y.flatten()], axis=1)

    for (i, square) in enumerate(chess.SQUARES_180):
        piece = board.piece_at(square)
        if piece:
            add_piece(ax, locs[i][0], locs[i][1], piece.symbol())


def plot_move(ax, board: chess.Board, move: chess.Move, alpha=1.0, color="black"):
    move = MoveRep.from_uci(move)

    from_square = move.to_str_list()[0]
    to_square = move.to_str_list()[1]

    square = chess.SQUARE_NAMES.index(from_square)
    piece = board.piece_at(square)

    from_col, from_row = square_to_grid(from_square)
    to_col, to_row = square_to_grid(to_square)

    add_arrow(ax, from_row, from_col, to_row, to_col, alpha=alpha, color=color)
    add_piece(ax, to_row, to_col, piece.symbol(), alpha=alpha, color=color)


def square_to_grid(square: str) -> Tuple[int, int]:
    row_val = square[0]
    col_val = square[1]
    col = ["8", "7", "6", "5", "4", "3", "2", "1"].index(col_val)
    row = ["a", "b", "c", "d", "e", "f", "g", "h"].index(row_val)
    return col, row


def mark_square(ax, square: str):

    row, col = square_to_grid(square)

    rect = patches.Rectangle(
        (col - 0.5, row - 0.5),
        1.0,
        1.0,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
        zorder=3,
    )

    # Add the patch to the Axes
    ax.add_patch(rect)


def mark_move(ax, move):
    move = MoveRep.from_uci(move)

    from_square = move.to_str_list()[0]
    to_square = move.to_str_list()[1]
    mark_square(ax, from_square)
    mark_square(ax, to_square)


if __name__ == "__main__":
    board = ut.randomize_board()
    move = np.random.choice(list(board.legal_moves))
    fig, ax = plt.subplots(1, 1)

    plot_board(ax, board, checkers=True)
    plot_move(ax, board, move)
    mark_square(ax, "e1")
    plt.show()
