"""
Evaluate chess models on Lichess puzzles dataset
Adapted from LichessEval for stonefish models
"""

import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess
import datasets
import glicko2
import numpy as np
import pandas as pd

import stonefish.config
from stonefish.types import ChessAgent


@dataclass
class PuzzleResult:
    """Result of a single puzzle attempt"""

    puzzle_id: str
    rating: int
    rating_deviation: int
    fen: str
    predicted_move: str
    correct_move: str
    is_correct: bool
    is_legal_move: bool
    move_index: int  # Which move in the puzzle sequence
    player_rating_estimate: float = 0.0
    player_rd_estimate: float = 0.0


class LichessPuzzleSampler:
    """Efficiently sample puzzles near target ratings without replacement"""

    def __init__(self, dataset):
        self.data = dataset
        self.data = self.data.sort("Rating")
        self.ratings = self.data["Rating"]
        self.seen = set()

    def sample(self, target: int, size: int = 1):
        idx = bisect.bisect_left(self.ratings, target)

        window = len(self.seen) + 10 * size
        start = max(0, idx - window)
        end = min(len(self.data), idx + window)

        sample_range = list(range(start, end))
        sample_range = list(filter(lambda x: int(x) not in self.seen, sample_range))

        sample_indices = np.random.choice(
            sample_range, size=min(size, len(sample_range)), replace=False
        )
        for idx in sample_indices:
            self.seen.add(int(idx))
        return [self.data[int(idx)] for idx in sample_indices]

    def random_sample(self, size):
        idxes = np.random.choice(len(self.data), size, replace=False)
        for idx in idxes:
            self.seen.add(int(idx))

        return [self.data[int(idx)] for idx in idxes]


def evaluate_puzzle(
    agent: ChessAgent,
    puzzle_sample: Dict,
) -> List[PuzzleResult]:
    """
    Evaluate a single puzzle with potentially multiple moves

    Args:
        agent: Chess agent to evaluate
        puzzle_sample: Dictionary containing puzzle data from dataset

    Returns:
        List of PuzzleResult for each move in the puzzle
    """
    results = []

    # Extract puzzle data
    puzzle_id = puzzle_sample["PuzzleId"]
    rating = puzzle_sample["Rating"]
    rating_deviation = puzzle_sample["RatingDeviation"]
    fen = puzzle_sample["FEN"]

    # Initialize board
    board = chess.Board(fen)

    # Parse moves - first move is the setup, rest are the puzzle
    moves = puzzle_sample["Moves"].split()

    # The first move is the opponent's move that sets up the puzzle
    # We need to respond to this, then the opponent responds, etc.
    for idx, (setup_move, solution_move) in enumerate(
        zip(moves[::2], moves[1::2], strict=False)
    ):
        # Apply the setup move (opponent's move)
        board.push_uci(setup_move)

        # Get the agent's prediction
        try:
            predicted_move = agent(board)
            is_legal_move = (
                predicted_move in board.legal_moves if predicted_move else False
            )
            predicted_uci = predicted_move.uci() if predicted_move else "none"
        except Exception as e:
            print(f"Error getting move from agent: {e}")
            predicted_move = None
            predicted_uci = "error"
            is_legal_move = False

        # Check if prediction is correct
        is_correct = predicted_uci == solution_move

        # Apply the correct move to continue the puzzle
        board.push_uci(solution_move)

        results.append(
            PuzzleResult(
                puzzle_id=puzzle_id,
                rating=rating,
                rating_deviation=rating_deviation,
                fen=fen,
                predicted_move=predicted_uci,
                correct_move=solution_move,
                is_correct=is_correct,
                is_legal_move=is_legal_move,
                move_index=idx,
            )
        )

        # Check if puzzle is complete (checkmate or no more moves)
        if board.is_game_over() or (idx * 2 + 2) >= len(moves):
            break

    return results


def evaluate_puzzles(
    agent: ChessAgent,
    dataset: datasets.Dataset,
    num_puzzles: Optional[int] = None,
    batch_size: int = 10,
    warmup_puzzles: int = 2000,
) -> pd.DataFrame:
    """
    Evaluate an agent on puzzles using adaptive selection

    Args:
        agent: Chess agent to evaluate
        dataset: Lichess puzzles dataset
        num_puzzles: Number of puzzles to evaluate (None for 100)
        warmup_puzzles: Number of random puzzles for initial rating estimate

    Returns:
        DataFrame with evaluation results
    """
    if num_puzzles is None:
        num_puzzles = 100

    # Initialize sampler and Glicko2 player
    sampler = LichessPuzzleSampler(dataset)
    player = glicko2.Player(1500, 200, 0.04)

    batch: List[Tuple[int, int, int]] = []
    batch_size = 100

    all_results = []

    # Warmup phase with random puzzles
    if warmup_puzzles > 0:
        print(f"Warmup phase: evaluating {warmup_puzzles} random puzzles...")
        warmup_batch = sampler.random_sample(warmup_puzzles)

        warmup_results = []
        for puzzle in warmup_batch:
            puzzle_results = evaluate_puzzle(agent, puzzle)
            puzzle_solved = all(r.is_correct for r in puzzle_results)
            warmup_results.append((puzzle["Rating"], 100, 1 if puzzle_solved else 0))

            # Store results
            for result in puzzle_results:
                result.player_rating_estimate = player.rating
                result.player_rd_estimate = player.rd
                all_results.append(result)

        ratings, rds, results = zip(*warmup_results, strict=False)
        player.update_player(ratings, rds, results)
        print(f"After warmup: Rating = {player.rating:.0f} ± {player.rd:.0f}")

    print(
        f"\nStarting adaptive puzzle evaluation (initial rating: {player.rating:.0f})"
    )

    remaining_puzzles = num_puzzles - warmup_puzzles
    puzzles_done = 0

    while puzzles_done < remaining_puzzles:
        # Sample a batch of puzzles near current rating
        batch_to_sample = min(batch_size, remaining_puzzles - puzzles_done)
        puzzle_batch = sampler.sample(int(player.rating), size=batch_to_sample)

        batch = []
        for puzzle in puzzle_batch:
            # Evaluate puzzle
            puzzle_results = evaluate_puzzle(agent, puzzle)

            # Check if fully solved
            puzzle_solved = all(r.is_correct for r in puzzle_results)

            # Add to batch
            puzzle_rating = puzzle["Rating"]
            batch.append((puzzle_rating, 50, 1 if puzzle_solved else 0))

            # Store results with rating estimate
            for result in puzzle_results:
                result.player_rating_estimate = player.rating
                result.player_rd_estimate = player.rd
                all_results.append(result)

            puzzles_done += 1

            # Progress update
            if puzzles_done % batch_size == 0:
                print(
                    f"Progress: {warmup_puzzles + puzzles_done}/{num_puzzles}, "
                    f"Rating: {player.rating:.0f} ± {player.rd:.0f}"
                )

        # Update rating after batch
        if batch:
            player_ratings, stds, wins = zip(*batch, strict=False)
            player.update_player(player_ratings, stds, wins)

    return pd.DataFrame([r.__dict__ for r in all_results])


def main():
    """Example usage"""
    import argparse

    from stonefish.eval.agent_loader import load_agent

    parser = argparse.ArgumentParser(description="Evaluate chess model on puzzles")
    parser.add_argument("--agent", required=True, help="Agent specification")
    parser.add_argument(
        "--num_puzzles", type=int, default=100, help="Number of puzzles"
    )
    parser.add_argument("--device", default="cpu", help="Device to use")

    args = parser.parse_args()

    # Load agent
    print(f"Loading agent: {args.agent}")
    agent = load_agent(args.agent, device=args.device)

    # Load puzzles dataset
    print("Loading Lichess puzzles dataset...")
    dataset = datasets.load_dataset("Lichess/chess-puzzles")["train"]

    # Filter by popularity to get good quality puzzles
    dataset = dataset.filter(lambda x: x["Popularity"] > 0)

    # Evaluate
    print(f"Evaluating on {args.num_puzzles} puzzles...")
    results_df = evaluate_puzzles(
        agent=agent,
        dataset=dataset,
        num_puzzles=args.num_puzzles,
    )

    # Print statistics
    print("\n" + "=" * 50)
    print("PUZZLE EVALUATION RESULTS")
    print("=" * 50)
    print(results_df)


if __name__ == "__main__":
    stonefish.config.expose_modules()
    main()
