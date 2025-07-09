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
