from typing import List, Mapping, Protocol

import chess
import torch


class ChessLogits(Protocol):
    """Raw model logits for a single position (1x5700 fastchessenv space)"""

    _move_map: List[tuple]

    def filter_legal_moves(self, board: chess.Board) -> torch.Tensor:
        """Extract logits for only legal moves"""
        ...

    def apply_temperature(self, temperature: float) -> torch.Tensor:
        """Apply temperature scaling"""
        ...


class ChessPolicy(Protocol):
    """Probability distribution over legal chess moves for a given position"""

    def __call__(self, board: chess.Board) -> Mapping[chess.Move, float]:
        """Return probability distribution over legal moves"""
        ...

    def sample(self, board: chess.Board, temperature: float = 1.0) -> chess.Move:
        """Sample a move from the distribution"""
        ...

    def best_move(self, board: chess.Board) -> chess.Move:
        """Return highest probability legal move"""
        ...


class ChessAgent(Protocol):
    """Move selection strategy"""

    def __call__(self, board: chess.Board) -> chess.Move:
        """Select a move for the given board"""
        ...


class StackedChessAgent(Protocol):
    """Batched move selection"""

    def __call__(self, boards: List[chess.Board]) -> List[chess.Move]:
        """Select moves for multiple boards"""
        ...


class ChessModel(Protocol):
    """Training interface for chess models"""

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass for training with teacher forcing support"""
        ...

    def inference(self, state: torch.Tensor) -> torch.Tensor:
        """Inference pass for move selection"""
        ...


class BoardTokenizer(Protocol):

    def from_board(self, board: chess.Board) -> torch.Tensor:
        """Convert a single board into a tensor"""
        ...

    def to_board(self, board_array: torch.Tensor) -> chess.Board:
        """Convert a tensor into a single board"""
        ...

    def from_fen(self, fen: str) -> torch.Tensor:
        """Convert FEN string into a tensor"""
        ...

    def to_fen(self, board_array: torch.Tensor) -> str:
        """Convert tensor into FEN string"""
        ...

    def from_board_batch(self, boards: List[chess.Board]) -> torch.Tensor:
        """Convert multiple boards into a batched tensor"""
        ...

    def to_board_batch(self, board_arrays: torch.Tensor) -> List[chess.Board]:
        """Convert batched tensors into multiple boards"""
        ...

    @property
    def shape(self):
        """Shape of a single encoded board tensor"""
        ...


class MoveTokenizer(Protocol):

    def from_move(self, move: chess.Move) -> torch.Tensor:
        """Convert a single move into a tensor"""
        ...

    def to_move(self, move_array: torch.Tensor) -> chess.Move:
        """Convert a tensor into a single move"""
        ...

    def from_uci(self, uci: str) -> torch.Tensor:
        """Convert UCI string into a tensor"""
        ...

    def to_uci(self, move_array: torch.Tensor) -> str:
        """Convert tensor into UCI string"""
        ...

    def from_move_batch(self, moves: List[chess.Move]) -> torch.Tensor:
        """Convert multiple moves into a batched tensor"""
        ...

    def to_move_batch(self, move_arrays: torch.Tensor) -> List[chess.Move]:
        """Convert batched tensors into multiple moves"""
        ...

    @property
    def shape(self):
        """Shape of a single encoded move tensor"""
        ...

    @property
    def vocab_size(self) -> int:
        """Size of move vocabulary (e.g., 5700 for chess)"""
        ...
