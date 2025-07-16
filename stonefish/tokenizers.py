"""
Concrete implementations of BoardTokenizer and MoveTokenizer protocols.
"""

from typing import List

import chess
import torch
from fastchessenv import CBoard, CMove

from stonefish.convert import board_to_lczero_tensor, lczero_tensor_to_board
from stonefish.types import BoardTokenizer, MoveTokenizer


class FlatBoardTokenizer(BoardTokenizer):
    """BoardTokenizer for flat board representations using fastchessenv 69-dim representation"""

    def from_board(self, board: chess.Board) -> torch.Tensor:
        """Convert a single board into a tensor"""
        cboard = CBoard.from_board(board)
        array = cboard.to_array()
        return torch.tensor(array, dtype=torch.float32)

    def to_board(self, board_array: torch.Tensor) -> chess.Board:
        """Convert a tensor into a single board"""
        array = board_array.cpu().numpy()
        cboard = CBoard.from_array(array)
        return cboard.to_board()

    def from_fen(self, fen: str) -> torch.Tensor:
        """Convert FEN string into a tensor"""
        cboard = CBoard.from_fen(fen)
        array = cboard.to_array()
        return torch.tensor(array, dtype=torch.float32)

    def to_fen(self, board_array: torch.Tensor) -> str:
        """Convert tensor into FEN string"""
        array = board_array.cpu().numpy()
        cboard = CBoard.from_array(array)
        return str(cboard.to_fen())

    def from_board_batch(self, boards: List[chess.Board]) -> torch.Tensor:
        """Convert multiple boards into a batched tensor"""
        tensors = [self.from_board(board) for board in boards]
        return torch.stack(tensors)

    def to_board_batch(self, board_arrays: torch.Tensor) -> List[chess.Board]:
        """Convert batched tensors into multiple boards"""
        return [self.to_board(board_array) for board_array in board_arrays]

    @property
    def shape(self):
        """Shape of a single encoded board tensor"""
        return (69,)


class FlatMoveTokenizer(MoveTokenizer):
    """MoveTokenizer for flat move representations using fastchessenv 5700-dim space"""

    def from_move(self, move: chess.Move) -> torch.Tensor:
        """Convert a single move into a tensor"""
        cmove = CMove.from_move(move)
        move_int = cmove.to_int()
        return torch.tensor(move_int, dtype=torch.long)

    def to_move(self, move_array: torch.Tensor) -> chess.Move:
        """Convert a tensor into a single move"""
        move_int = move_array.item()
        cmove = CMove.from_int(move_int)
        return cmove.to_move()

    def from_uci(self, uci: str) -> torch.Tensor:
        """Convert UCI string into a tensor"""
        cmove = CMove.from_str(uci)
        move_int = cmove.to_int()
        return torch.tensor(move_int, dtype=torch.long)

    def to_uci(self, move_array: torch.Tensor) -> str:
        """Convert tensor into UCI string"""
        move_int = move_array.item()
        cmove = CMove.from_int(move_int)
        return str(cmove.to_str())

    def from_move_batch(self, moves: List[chess.Move]) -> torch.Tensor:
        """Convert multiple moves into a batched tensor"""
        tensors = [self.from_move(move) for move in moves]
        return torch.stack(tensors)

    def to_move_batch(self, move_arrays: torch.Tensor) -> List[chess.Move]:
        """Convert batched tensors into multiple moves"""
        return [self.to_move(move_array) for move_array in move_arrays]

    @property
    def shape(self):
        """Shape of a single encoded move tensor"""
        return ()  # Scalar tensor

    @property
    def vocab_size(self) -> int:
        """Size of move vocabulary (5700 for chess)"""
        return 5700


class LCZeroBoardTokenizer(BoardTokenizer):
    """BoardTokenizer for spatial board representations using Leela Chess Zero 8x8x20 format"""

    def from_board(self, board: chess.Board) -> torch.Tensor:
        """Convert a single board into a tensor"""
        array = board_to_lczero_tensor(board)
        return torch.tensor(array, dtype=torch.float32)

    def to_board(self, board_array: torch.Tensor) -> chess.Board:
        """Convert a tensor into a single board"""
        array = board_array.cpu().numpy()
        return lczero_tensor_to_board(array)

    def from_fen(self, fen: str) -> torch.Tensor:
        """Convert FEN string into a tensor"""
        board = chess.Board(fen)
        return self.from_board(board)

    def to_fen(self, board_array: torch.Tensor) -> str:
        """Convert tensor into FEN string"""
        board = self.to_board(board_array)
        return str(board.fen())

    def from_board_batch(self, boards: List[chess.Board]) -> torch.Tensor:
        """Convert multiple boards into a batched tensor"""
        tensors = [self.from_board(board) for board in boards]
        return torch.stack(tensors)

    def to_board_batch(self, board_arrays: torch.Tensor) -> List[chess.Board]:
        """Convert batched tensors into multiple boards"""
        return [self.to_board(board_array) for board_array in board_arrays]

    @property
    def shape(self):
        """Shape of a single encoded board tensor"""
        return (8, 8, 20)
