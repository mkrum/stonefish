"""
Agent Evaluations

Evaluations that focus on the entire "system", building on top things that map
from a chess.Board to a chess.Move (ChessAgent).
"""

import io

import chess
import chess.pgn
import numpy as np
import torch
from fastchessenv import CBoard, CChessEnv
from fastchessenv.env import SFCChessEnv
from mllg import TestInfo
from tqdm import tqdm

import wandb
from stonefish.convert import board_to_lczero_tensor
from stonefish.env import RandomAgent, StockfishAgent
from stonefish.eval.base import _create_pgn_html
from stonefish.tokenizers import LCZeroBoardTokenizer
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
                # Move is valid
                board.push(move)
                node = node.add_variation(move)
                move_count += 1
            else:
                # Catch for invalid moves, failing agent loses
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
            # If the agent crashes or throws an error, we just say it lost
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

    # Create progress bar with descriptive text
    pbar = tqdm(range(num_games), desc="Playing games", unit="game")

    for i in pbar:
        if i < num_games // 2:
            # Agent1 as white
            pbar.set_description(f"Game {i+1}/{num_games} (Agent1 as white)")
            result, _ = play_game(agent1, agent2)
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            # Agent1 as black
            pbar.set_description(f"Game {i+1}/{num_games} (Agent1 as black)")
            result, _ = play_game(agent2, agent1)
            if result == "0-1":
                wins += 1
            elif result == "1-0":
                losses += 1
            else:
                draws += 1

        # Update progress bar postfix with current stats
        pbar.set_postfix({"wins": wins, "losses": losses, "draws": draws})

    return [
        TestInfo(f"against_{opponent_name}_win_rate", wins / num_games),
        TestInfo(f"against_{opponent_name}_loss_rate", losses / num_games),
    ]


def training_agent_eval(
    agent_config,
    games_vs_random: int = 10,
    games_vs_stockfish: int = 10,
    stockfish_depths: list[int] | int = 1,
    pgn_games: int = 2,
):
    """Creates a training-compatible agent evaluation function"""
    if isinstance(stockfish_depths, int):
        stockfish_depths = [stockfish_depths]

    def eval_fn(model, epoch):
        agent = agent_config(model=model)
        wandb_metrics = {}

        # Evaluate vs random
        if games_vs_random > 0:
            random_agent = RandomAgent()
            random_results = evaluate_agents(
                agent, random_agent, num_games=games_vs_random, opponent_name="random"
            )
            for result in random_results:
                wandb_metrics[result.loss_type] = result.loss

        # Evaluate vs stockfish at each depth
        for depth in stockfish_depths:
            if games_vs_stockfish > 0:
                sf_name = f"stockfish_d{depth}"
                stockfish_agent = StockfishAgent(depth=depth)
                stockfish_results = evaluate_agents(
                    agent,
                    stockfish_agent,
                    num_games=games_vs_stockfish,
                    opponent_name=sf_name,
                )
                for result in stockfish_results:
                    wandb_metrics[result.loss_type] = result.loss

        # Generate PGNs for visualization
        if pgn_games > 0:
            if games_vs_random > 0:
                random_agent = RandomAgent()
                random_pgns = get_pgns_between_agents(
                    agent, random_agent, num_games=pgn_games
                )
                random_games = []
                for pgn_str in random_pgns:
                    game = chess.pgn.read_game(io.StringIO(pgn_str))
                    random_games.append(game)
                html_content = _create_pgn_html(str(random_games[0]))
                wandb_metrics["eval/PGNAgainstRandom"] = wandb.Html(html_content)

            if games_vs_stockfish > 0:
                # PGN against the highest depth
                depth = stockfish_depths[-1]
                stockfish_agent = StockfishAgent(depth=depth)
                stockfish_pgns = get_pgns_between_agents(
                    agent, stockfish_agent, num_games=pgn_games
                )
                stockfish_games = []
                for pgn_str in stockfish_pgns:
                    game = chess.pgn.read_game(io.StringIO(pgn_str))
                    stockfish_games.append(game)
                html_content = _create_pgn_html(str(stockfish_games[0]))
                wandb_metrics[f"eval/PGNAgainstStockfishD{depth}"] = wandb.Html(
                    html_content
                )

        return wandb_metrics

    return eval_fn


def _state_to_tensor(state, model):
    """Convert CChessEnv flat state (N, 69) to model input tensor."""
    if isinstance(model.board_tokenizer, LCZeroBoardTokenizer):
        boards = []
        for i in range(state.shape[0]):
            cboard = CBoard.from_array(state[i])
            py_board = cboard.to_board()
            boards.append(board_to_lczero_tensor(py_board))
        return torch.from_numpy(np.stack(boards))
    else:
        return torch.from_numpy(state).float()


def _sample_moves_from_logits(logits, mask, temperature, sample):
    """Sample moves from model logits masked by legal moves.

    Args:
        logits: (N, output_dim) model output tensor
        mask: (N, 5632) legal move mask from env
        temperature: temperature for sampling
        sample: if False, take argmax instead of sampling
    Returns:
        (N,) int32 array of move indices for env.step()
    """
    n = logits.shape[0]
    mask_dim = mask.shape[1]
    # Slice logits to match env mask dimension
    logits_masked = logits[:, :mask_dim].cpu()
    mask_t = torch.from_numpy(mask).bool()

    # Set illegal moves to -inf
    logits_masked[~mask_t] = float("-inf")
    logits_masked = logits_masked / temperature

    moves = np.zeros(n, dtype=np.int32)
    probs = torch.softmax(logits_masked, dim=-1)
    for i in range(n):
        if sample:
            moves[i] = torch.multinomial(probs[i], 1).item()
        else:
            moves[i] = probs[i].argmax().item()
    return moves


def _play_parallel_games(model, env, num_games, temperature, sample, max_moves=500):
    """Play num_games in parallel using CChessEnv, return (wins, losses, draws)."""
    device = next(model.parameters()).device
    state, mask = env.reset()

    wins = losses = draws = 0
    games_completed = 0
    steps = 0

    pbar = tqdm(total=num_games, desc="Parallel games", unit="game")

    while games_completed < num_games:
        state_tensor = _state_to_tensor(state, model).to(device)
        with torch.no_grad():
            logits = model.inference(state_tensor)

        moves = _sample_moves_from_logits(logits, mask, temperature, sample)
        state, mask, reward, done = env.step(moves)
        steps += 1

        for i in np.where(done)[0]:
            if games_completed >= num_games:
                break
            if reward[i] > 0:
                wins += 1
            elif reward[i] < 0:
                losses += 1
            else:
                draws += 1
            games_completed += 1
            pbar.update(1)
            pbar.set_postfix({"wins": wins, "losses": losses, "draws": draws})

        # Safety: if games are too long, count remaining as draws
        if steps > max_moves:
            remaining = num_games - games_completed
            draws += remaining
            games_completed = num_games
            pbar.update(remaining)
            break

    pbar.close()
    return wins, losses, draws


def parallel_training_agent_eval(
    games_vs_random: int = 10,
    games_vs_stockfish: int = 10,
    stockfish_depths: list[int] | int = 1,
    temperature: float = 1.0,
    sample: bool = True,
    max_moves: int = 500,
):
    """Creates a parallel agent evaluation function using CChessEnv."""
    if isinstance(stockfish_depths, int):
        stockfish_depths = [stockfish_depths]

    def eval_fn(model, epoch):
        wandb_metrics = {}

        if games_vs_random > 0:
            env = CChessEnv(games_vs_random, max_step=max_moves)
            w, loss, d = _play_parallel_games(
                model, env, games_vs_random, temperature, sample, max_moves
            )
            wandb_metrics["against_random_win_rate"] = w / games_vs_random
            wandb_metrics["against_random_loss_rate"] = loss / games_vs_random

        for depth in stockfish_depths:
            if games_vs_stockfish > 0:
                sf_name = f"stockfish_d{depth}"
                env = SFCChessEnv(games_vs_stockfish, depth=depth, max_step=max_moves)
                w, loss, d = _play_parallel_games(
                    model, env, games_vs_stockfish, temperature, sample, max_moves
                )
                wandb_metrics[f"against_{sf_name}_win_rate"] = w / games_vs_stockfish
                wandb_metrics[f"against_{sf_name}_loss_rate"] = (
                    loss / games_vs_stockfish
                )
                del env

        return wandb_metrics

    return eval_fn
