"""
Conversion utilities for different chess board representations.

This module provides functions to convert between standard chess.Board objects
and Leela Chess Zero tensor representations.
"""

import chess
import numpy as np


def board_to_lczero_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board to Leela Chess Zero tensor representation.

    Returns a tensor of shape (8, 8, 20) where:
    - Planes 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
    - Planes 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
    - Plane 12: Side to move (1 if black, 0 if white)
    - Planes 13-16: Castling rights (WK, WQ, BK, BQ)
    - Plane 17: En passant square
    - Plane 18: Halfmove clock
    - Plane 19: Fullmove number
    """
    tensor = np.zeros((8, 8, 20), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            idx = piece.piece_type - 1
            plane = idx if piece.color == chess.WHITE else idx + 6
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[7 - rank, file, plane] = 1.0

    tensor[:, :, 12] = float(board.turn == chess.BLACK)

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 16] = 1.0

    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor[7 - rank, file, 17] = 1.0

    tensor[:, :, 18] = board.halfmove_clock
    tensor[:, :, 19] = board.fullmove_number

    return tensor


def lczero_tensor_to_board(tensor: np.ndarray) -> chess.Board:
    """
    Convert a Leela Chess Zero tensor representation back to a chess.Board.

    Args:
        tensor: Tensor of shape (8, 8, 20) in Lczero format

    Returns:
        chess.Board object representing the position
    """
    board = chess.Board()
    board.clear()

    white_pieces = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    # White pieces (planes 0-5)
    for plane in range(6):
        piece_type = white_pieces[plane]
        for row in range(8):
            for col in range(8):
                if tensor[row, col, plane] == 1:
                    square = chess.square(col, 7 - row)
                    board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))

    # Black pieces (planes 6-11)
    for plane in range(6, 12):
        piece_type = white_pieces[plane - 6]
        for row in range(8):
            for col in range(8):
                if tensor[row, col, plane] == 1:
                    square = chess.square(col, 7 - row)
                    board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))

    # Side to move
    board.turn = chess.BLACK if np.any(tensor[:, :, 12]) else chess.WHITE

    # Castling rights
    if np.any(tensor[:, :, 13]):
        board.castling_rights |= chess.BB_H1
    if np.any(tensor[:, :, 14]):
        board.castling_rights |= chess.BB_A1
    if np.any(tensor[:, :, 15]):
        board.castling_rights |= chess.BB_H8
    if np.any(tensor[:, :, 16]):
        board.castling_rights |= chess.BB_A8

    # En passant
    ep_indices = np.argwhere(tensor[:, :, 17] == 1)
    if len(ep_indices) == 1:
        row, col = ep_indices[0]
        board.ep_square = chess.square(col, 7 - row)
    else:
        board.ep_square = None

    # Halfmove clock and fullmove number
    board.halfmove_clock = int(tensor[0, 0, 18])
    board.fullmove_number = int(tensor[0, 0, 19])

    return board
