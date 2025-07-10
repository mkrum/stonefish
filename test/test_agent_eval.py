"""
Tests for agent evaluation system
"""

import chess

from stonefish.env import RandomAgent
from stonefish.eval.agent import (
    eval_agent_vs_random,
    evaluate_agent_vs_random,
    play_game,
)
from stonefish.policy import ModelChessAgent
from stonefish.resnet import ChessResNet


def test_random_agent():
    """Test RandomAgent basic functionality"""
    agent = RandomAgent()
    board = chess.Board()

    move = agent(board)
    assert move in board.legal_moves
    assert agent.name == "RandomAgent"


def test_play_game():
    """Test playing a game between two random agents"""
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    result, pgn = play_game(agent1, agent2, max_moves=50)

    # Should get a valid chess result
    assert result in ["1-0", "0-1", "1/2-1/2"]
    # PGN should contain headers and result
    assert "RandomAgent" in pgn
    assert result in pgn


def test_evaluate_agent_vs_random():
    """Test agent evaluation over multiple games"""
    agent = RandomAgent()

    results = evaluate_agent_vs_random(agent, num_games=10)

    # Should return TestInfo objects
    assert len(results) == 4

    # Check required metrics present
    metric_names = [info.loss_type for info in results]
    assert "agent_win_rate" in metric_names
    assert "agent_wins" in metric_names
    assert "agent_losses" in metric_names
    assert "agent_draws" in metric_names


def test_eval_agent_vs_random_with_model():
    """Test training-compatible evaluation function"""
    # Create a small model
    model = ChessResNet(input_dim=69, hidden_dim=64, num_blocks=1, output_dim=5700)

    # Run evaluation
    test_infos = eval_agent_vs_random(model, None, None, max_batch=5)

    # Should return TestInfo objects
    assert len(test_infos) == 4

    # Check required metrics present
    metric_names = [info.loss_type for info in test_infos]
    assert "agent_win_rate" in metric_names
    assert "agent_wins" in metric_names
    assert "agent_losses" in metric_names
    assert "agent_draws" in metric_names


def test_model_agent_game():
    """Test that model agent can actually play a game"""
    model = ChessResNet(input_dim=69, hidden_dim=64, num_blocks=1, output_dim=5700)
    agent = ModelChessAgent(model, model_type="standard")
    random_agent = RandomAgent()

    result, pgn = play_game(agent, random_agent, max_moves=20)

    # Should complete without error
    assert result in ["1-0", "0-1", "1/2-1/2"]
    assert len(pgn) > 0


def test_evaluate_agent_vs_stockfish():
    """Test agent evaluation against Stockfish"""
    from stonefish.eval.agent import evaluate_agent_vs_stockfish

    agent = RandomAgent()

    # Use depth 1 for fast testing
    results = evaluate_agent_vs_stockfish(agent, num_games=4, stockfish_depth=1)

    # Should return TestInfo objects
    assert len(results) == 5

    # Check required metrics present
    metric_names = [info.loss_type for info in results]
    assert "stockfish_win_rate" in metric_names
    assert "stockfish_wins" in metric_names
    assert "stockfish_losses" in metric_names
    assert "stockfish_draws" in metric_names
    assert "stockfish_depth" in metric_names


def test_eval_agent_vs_stockfish_with_model():
    """Test training-compatible Stockfish evaluation function"""
    from stonefish.eval.agent import eval_agent_vs_stockfish

    model = ChessResNet(input_dim=69, hidden_dim=64, num_blocks=1, output_dim=5700)

    # Run evaluation with depth 1 for speed
    test_infos = eval_agent_vs_stockfish(
        model, None, None, max_batch=3, stockfish_depth=1
    )

    # Should return TestInfo objects
    assert len(test_infos) == 5

    # Check required metrics present
    metric_names = [info.loss_type for info in test_infos]
    assert "stockfish_win_rate" in metric_names
    assert "stockfish_wins" in metric_names
    assert "stockfish_losses" in metric_names
    assert "stockfish_draws" in metric_names
    assert "stockfish_depth" in metric_names
