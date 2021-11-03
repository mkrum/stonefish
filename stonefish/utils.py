import chess
import numpy as np


def randomize_board(steps=range(10, 20)):
    steps = list(steps)
    board = chess.Board()
    for _ in range(np.random.choice(steps)):
        moves = list(board.legal_moves)
        board.push(np.random.choice(moves))

        # If we hit the end, just try again
        if board.is_game_over():
            return randomize_board()

    return board
