"""
Agent evaluation by playing games against random opponent.
"""

import chess
import chess.pgn
from mllg import TestInfo

from stonefish.env import RandomAgent, StockfishAgent
from stonefish.types import ChessAgent


def play_game(
    white_agent: ChessAgent, black_agent: ChessAgent, max_moves: int = 500
) -> tuple[str, str]:
    """Play game and return (result, pgn)"""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = getattr(white_agent, "name", "Agent")
    game.headers["Black"] = getattr(black_agent, "name", "Agent")

    node = game
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        agent = white_agent if board.turn == chess.WHITE else black_agent

        try:
            move = agent(board)
            if move and move in board.legal_moves:
                board.push(move)
                node = node.add_variation(move)
                move_count += 1
            else:
                # Invalid move = failing agent loses
                agent_name = (
                    getattr(white_agent, "name", "white")
                    if board.turn == chess.WHITE
                    else getattr(black_agent, "name", "black")
                )
                print(
                    f"DEBUG: Agent {agent_name} made invalid move: {move} (legal moves: {len(list(board.legal_moves))})"
                )
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                break
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
        ) as e:
            # Agent crash = failing agent loses
            print(
                f"DEBUG: Agent {getattr(white_agent, 'name', 'white') if board.turn == chess.WHITE else getattr(black_agent, 'name', 'black')} crashed: {type(e).__name__}: {e}"
            )
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            break
    else:
        # Normal game end or max moves reached
        result = board.result() if board.is_game_over() else "1/2-1/2"
    game.headers["Result"] = result

    return result, str(game)


def evaluate_agent_vs_random(agent: ChessAgent, num_games: int = 100) -> list[TestInfo]:
    """Evaluate agent against random opponent"""
    random_agent = RandomAgent()
    wins = losses = draws = 0

    for i in range(num_games):
        if i < num_games // 2:
            # Agent as white
            result, _ = play_game(agent, random_agent)
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            # Agent as black
            result, _ = play_game(random_agent, agent)
            if result == "0-1":
                wins += 1
            elif result == "1-0":
                losses += 1
            else:
                draws += 1

    return [
        TestInfo("agent_win_rate", wins / num_games),
        TestInfo("agent_wins", wins),
        TestInfo("agent_losses", losses),
        TestInfo("agent_draws", draws),
    ]


def evaluate_agent_vs_stockfish(
    agent: ChessAgent, num_games: int = 100, stockfish_depth: int = 1
) -> list[TestInfo]:
    """Evaluate agent against Stockfish at configurable difficulty"""
    stockfish_agent = StockfishAgent(depth=stockfish_depth)
    wins = losses = draws = 0

    for i in range(num_games):
        if i < num_games // 2:
            # Agent as white
            result, _ = play_game(agent, stockfish_agent)
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            # Agent as black
            result, _ = play_game(stockfish_agent, agent)
            if result == "0-1":
                wins += 1
            elif result == "1-0":
                losses += 1
            else:
                draws += 1

    return [
        TestInfo("stockfish_win_rate", wins / num_games),
        TestInfo("stockfish_wins", wins),
        TestInfo("stockfish_losses", losses),
        TestInfo("stockfish_draws", draws),
        TestInfo("stockfish_depth", stockfish_depth),
    ]


def eval_agent_vs_random(model, dataloader, train_fn, max_batch=20) -> list[TestInfo]:
    """Training-compatible evaluation function against random opponent"""
    from stonefish.policy import ModelChessAgent

    agent = ModelChessAgent(model, model_type="standard")
    return evaluate_agent_vs_random(agent, num_games=max_batch)


def eval_agent_vs_stockfish(
    model, dataloader, train_fn, max_batch=20, stockfish_depth=1
) -> list[TestInfo]:
    """Training-compatible evaluation function against Stockfish"""
    from stonefish.policy import ModelChessAgent

    agent = ModelChessAgent(model, model_type="standard")
    return evaluate_agent_vs_stockfish(
        agent, num_games=max_batch, stockfish_depth=stockfish_depth
    )
