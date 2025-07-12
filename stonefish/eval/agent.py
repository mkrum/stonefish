"""
Agent evaluation by playing games against random opponent.
"""

import io

import chess
import chess.pgn
from mllg import TestInfo

import wandb
from stonefish.env import RandomAgent, StockfishAgent
from stonefish.eval.base import _create_pgn_html
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
    return evaluate_agents(
        agent, random_agent, num_games=num_games, opponent_name="random"
    )


def evaluate_agent_vs_stockfish(
    agent: ChessAgent, num_games: int = 100, stockfish_depth: int = 1
) -> list[TestInfo]:
    """Evaluate agent against Stockfish at configurable difficulty"""
    stockfish_agent = StockfishAgent(depth=stockfish_depth)
    return evaluate_agents(
        agent, stockfish_agent, num_games=num_games, opponent_name="stockfish"
    )


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


def get_pgns_between_agents(
    white_agent: ChessAgent, black_agent: ChessAgent, num_games: int = 2
) -> list[str]:
    """Get PGN strings from games between two agents"""
    pgns = []

    for _ in range(num_games):
        _, pgn = play_game(white_agent, black_agent)
        pgns.append(pgn)

    return pgns


def evaluate_agents(
    agent1: ChessAgent,
    agent2: ChessAgent,
    num_games: int = 10,
    opponent_name: str = "opponent",
) -> list[TestInfo]:
    """Evaluate agent1 vs agent2, with agent1 playing both colors"""
    wins = losses = draws = 0

    for i in range(num_games):
        if i < num_games // 2:
            # Agent1 as white
            result, _ = play_game(agent1, agent2)
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            # Agent1 as black
            result, _ = play_game(agent2, agent1)
            if result == "0-1":
                wins += 1
            elif result == "1-0":
                losses += 1
            else:
                draws += 1

    return [
        TestInfo(f"against_{opponent_name}_win_rate", wins / num_games),
        TestInfo(f"against_{opponent_name}_loss_rate", losses / num_games),
    ]


def training_agent_eval(
    agent_config,
    games_vs_random: int = 10,
    games_vs_stockfish: int = 10,
    stockfish_depth: int = 1,
    pgn_games: int = 2,
):
    """Creates a training-compatible agent evaluation function"""

    def eval_fn(model, epoch):
        # Create agent from model using config
        agent = agent_config(model=model)

        # Collect metrics for wandb
        wandb_metrics = {}

        # Evaluate vs random
        if games_vs_random > 0:
            random_agent = RandomAgent()
            random_results = evaluate_agents(
                agent, random_agent, num_games=games_vs_random, opponent_name="random"
            )
            for result in random_results:
                wandb_metrics[result.loss_type] = result.loss

        # Evaluate vs stockfish
        if games_vs_stockfish > 0:
            stockfish_agent = StockfishAgent(depth=stockfish_depth)
            stockfish_results = evaluate_agents(
                agent,
                stockfish_agent,
                num_games=games_vs_stockfish,
                opponent_name="stockfish",
            )
            for result in stockfish_results:
                wandb_metrics[result.loss_type] = result.loss

        # Generate PGNs for visualization
        if pgn_games > 0:
            # Get sample games vs random
            random_agent = RandomAgent()
            random_pgns = get_pgns_between_agents(
                agent, random_agent, num_games=pgn_games
            )

            # Convert random PGN strings to game objects
            random_games = []
            for pgn_str in random_pgns:
                game = chess.pgn.read_game(io.StringIO(pgn_str))
                if game:
                    random_games.append(game)

            # Create wandb HTML viewer for random games (one game only)
            if random_games:
                html_content = _create_pgn_html(str(random_games[0]))
                wandb_metrics["eval/PGNAgainstRandom"] = wandb.Html(html_content)

            # Get sample games vs stockfish
            stockfish_agent = StockfishAgent(depth=stockfish_depth)
            stockfish_pgns = get_pgns_between_agents(
                agent, stockfish_agent, num_games=pgn_games
            )

            # Convert stockfish PGN strings to game objects
            stockfish_games = []
            for pgn_str in stockfish_pgns:
                game = chess.pgn.read_game(io.StringIO(pgn_str))
                if game:
                    stockfish_games.append(game)

            # Create wandb HTML viewer for stockfish games (one game only)
            if stockfish_games:
                html_content = _create_pgn_html(str(stockfish_games[0]))
                wandb_metrics["eval/PGNAgainstStockfishOne"] = wandb.Html(html_content)

        return wandb_metrics

    return eval_fn
