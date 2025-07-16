"""
Chess policy implementations for unified model interface.

This module implements the probability space abstraction for chess models,
allowing ResNet and Transformer models to work with the same evaluation
and agent interfaces.
"""

from dataclasses import dataclass, field
from typing import List, Mapping

import chess
import torch
import torch.nn.functional
from fastchessenv import CMove
from torch.distributions import Categorical

from stonefish.types import ChessAgent, ChessLogits, ChessPolicy


@dataclass
class FastChessLogits:
    """ChessLogits implementation for fastchessenv 5700-dim space"""

    logits: torch.Tensor
    _move_map: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        if self.logits.dim() != 1 or self.logits.size(0) != 5700:
            raise ValueError(f"Expected (5700,) tensor, got {self.logits.shape}")

    def filter_legal_moves(self, board: chess.Board) -> torch.Tensor:
        """Extract logits for only legal moves"""
        legal_moves = list(board.legal_moves)

        # Map legal moves to fastchessenv indices
        legal_indices = []
        self._move_map = []  # Reset for this board

        for i, move in enumerate(legal_moves):
            try:
                move_idx = CMove.from_str(move.uci()).to_int()
                if move_idx < len(self.logits):  # Valid index
                    legal_indices.append(move_idx)
                    self._move_map.append((i, move_idx))
            except Exception:
                # Skip moves that can't be mapped
                continue

        if not legal_indices:
            # Fallback: return logits for first valid index
            return torch.tensor([self.logits[0]])

        return self.logits[legal_indices]

    def apply_temperature(self, temperature: float) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        return self.logits / temperature


@dataclass
class ModelChessPolicy:
    """ChessPolicy implementation using ChessLogits"""

    logits: ChessLogits

    def __call__(self, board: chess.Board) -> Mapping[chess.Move, float]:
        """Return probability distribution over legal moves"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        # Get filtered logits
        legal_logits = self.logits.filter_legal_moves(board)
        legal_probs = torch.nn.functional.softmax(legal_logits, dim=0)

        # Map back to moves using stored move map
        move_probs = {}
        for i, (legal_move_idx, _) in enumerate(self.logits._move_map):
            if i < len(legal_probs):
                move_probs[legal_moves[legal_move_idx]] = legal_probs[i].item()
        return move_probs

    def sample(self, board: chess.Board, temperature: float = 1.0) -> chess.Move:
        """Sample a move from the distribution"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Apply temperature and get probabilities
        legal_logits = self.logits.filter_legal_moves(
            board
        )  # Use original logits object
        legal_probs = torch.nn.functional.softmax(legal_logits / temperature, dim=0)

        # Sample from distribution
        idx = Categorical(legal_probs).sample().item()
        legal_move_idx, _ = self.logits._move_map[idx]
        return legal_moves[legal_move_idx]

    def best_move(self, board: chess.Board) -> chess.Move:
        """Return highest probability legal move"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Get probabilities and find best
        legal_logits = self.logits.filter_legal_moves(board)
        best_idx = legal_logits.argmax().item()

        legal_move_idx, _ = self.logits._move_map[best_idx]
        return legal_moves[legal_move_idx]


@dataclass
class PolicyAgent:
    """ChessAgent implementation using ChessPolicy"""

    policy: ChessPolicy
    temperature: float = 1.0
    sample: bool = True

    def __call__(self, board: chess.Board) -> chess.Move:
        """Select a move using the policy"""
        if self.sample:
            return self.policy.sample(board, self.temperature)
        else:
            return self.policy.best_move(board)


@dataclass
class ModelChessAgent:
    """Complete ChessAgent that wraps a model and handles the full pipeline"""

    model: torch.nn.Module
    temperature: float = 1.0
    sample: bool = True

    def __post_init__(self):
        self.device = next(self.model.parameters()).device

    def __call__(self, board: chess.Board) -> chess.Move:
        """Select a move for the given board"""
        # Convert board to tensor using model's tokenizer
        board_tensor = (
            self.model.board_tokenizer.from_board(board).unsqueeze(0).to(self.device)
        )

        # Get model prediction
        with torch.no_grad():
            logits = self.model.inference(board_tensor)[0]  # Remove batch dimension

        # Create policy pipeline
        chess_logits = FastChessLogits(logits)
        policy = ModelChessPolicy(chess_logits)
        agent = PolicyAgent(policy, self.temperature, self.sample)

        return agent(board)


def eval_policy_cross_entropy(policy: ChessPolicy, positions: List[tuple]) -> float:
    """
    Evaluate policy using cross-entropy loss against ground truth moves.

    Args:
        policy: ChessPolicy to evaluate
        positions: List of (board, true_move) tuples

    Returns:
        Average cross-entropy loss
    """
    total_loss = 0.0
    count = 0

    for board, true_move in positions:
        move_probs = policy(board)
        prob = move_probs[true_move]  # Assume move always has probability mass
        total_loss += -torch.log(torch.tensor(prob)).item()
        count += 1

    return total_loss / count if count > 0 else float("inf")


def eval_agent_accuracy(agent: ChessAgent, positions: List[tuple]) -> float:
    """
    Evaluate agent accuracy against ground truth moves.

    Args:
        agent: ChessAgent to evaluate
        positions: List of (board, true_move) tuples

    Returns:
        Accuracy as fraction of correct moves
    """
    correct = 0
    total = len(positions)

    for board, true_move in positions:
        predicted_move = agent(board)
        if predicted_move == true_move:
            correct += 1

    return correct / total if total > 0 else 0.0
