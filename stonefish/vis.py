
import chess
import matplotlib.pyplot as plt
import numpy as np

import stonefish.utils as ut

symbols = {
    'K': '♔',
    'Q': '♕',
    'R': '♖',
    'B': '♗',
    'N': '♘',
    'P': '♙',
    'k': '♚',
    'q': '♛',
    'r': '♜',
    'b': '♝',
    'n': '♞',
    'p': '♟︎',
}

def make_board(ax):
    """ 
    Adds checkers to the board to make it look more natural
    """
    X, Y = np.meshgrid(np.arange(8), np.arange(8))
    checker = (((X + Y) % 2) + 0.3) / 2
    ax.imshow(checker, cmap="Greys", vmax=1.0, vmin=0.0)

def add_piece(ax, x, y, piece):
    """
    Adds a pieces to the board, x and y should be the war x,y coords
    """
    ax.text(x, y + 0.05, symbols[piece], fontsize=32, ha='center', va='center')

def plot_board(ax, board, checkers=True):
    """
    Creates a board image on the specified axis
    """
    ax.set_xlim([-0.5, 7.5])
    ax.set_ylim([7.5, -0.5])

    ax.tick_params(labeltop=True, labelright=True, length=0)
    
    ax.set_yticks(list(range(8)))
    ax.set_xticks(list(range(8)))
    ax.set_yticklabels(('8', '7', '6', '5', '4', '3', '2', '1'))
    ax.set_xticklabels(('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'))

    if checkers:
        make_board(ax)

    X, Y = np.meshgrid(np.arange(8), np.arange(8))
    locs = np.stack([X.flatten(), Y.flatten()], axis=1)

    for (i, square) in enumerate(chess.SQUARES_180):
        piece = board.piece_at(square)
        if piece:
            add_piece(ax, locs[i][0], locs[i][1], piece.symbol())

if __name__ == '__main__':
    board = ut.randomize_board()
    fig, ax = plt.subplots(1,1)
    plot_board(ax, board, checkers=False)
    ax.imshow(np.random.rand(8, 8), alpha=0.2, cmap='hot_r')
    plt.show()
