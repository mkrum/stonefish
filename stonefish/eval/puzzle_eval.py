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
import wandb
from mllg import TestInfo

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


def evaluate_puzzle_by_rating_brackets(
    agent: ChessAgent,
    dataset: datasets.Dataset,
    puzzles_per_bracket: int = 100,
) -> Dict[str, float]:
    """
    Evaluate agent on puzzles from fixed rating brackets

    Args:
        agent: Chess agent to evaluate
        dataset: Lichess puzzles dataset
        puzzles_per_bracket: Number of puzzles to sample from each bracket

    Returns:
        Dictionary with accuracy for each rating bracket
    """
    # Define rating brackets
    rating_brackets = [
        (0, 500),
        (500, 1000),
        (1000, 1500),
        (1500, 2000),
        (2000, 2500),
        (2500, 3000),
    ]

    results = {}

    for min_rating, max_rating in rating_brackets:
        # Filter dataset for this bracket
        bracket_data = dataset.filter(
            lambda x, min_r=min_rating, max_r=max_rating: min_r <= x["Rating"] < max_r
        )

        if len(bracket_data) == 0:
            continue

        # Sample puzzles from this bracket
        num_to_sample = min(puzzles_per_bracket, len(bracket_data))
        indices = np.random.choice(len(bracket_data), num_to_sample, replace=False)
        sampled_puzzles = [bracket_data[int(i)] for i in indices]

        # Evaluate on these puzzles
        bracket_results = []
        for puzzle in sampled_puzzles:
            puzzle_results = evaluate_puzzle(agent, puzzle)
            puzzle_solved = all(r.is_correct for r in puzzle_results)
            bracket_results.append(puzzle_solved)

        # Calculate accuracy for this bracket
        bracket_accuracy = np.mean(bracket_results)
        bracket_name = f"{min_rating}-{max_rating}"
        results[bracket_name] = bracket_accuracy

    return results


def _puzzle_eval_fn(
    model, epoch, agent_config, dataset, num_puzzles, warmup_puzzles, batch_size
):
    """Helper function for training_puzzle_eval"""
    # Create agent from model using config
    agent = agent_config(model=model)

    # Evaluate on puzzles
    results_df = evaluate_puzzles(
        agent=agent,
        dataset=dataset,
        num_puzzles=num_puzzles,
        warmup_puzzles=warmup_puzzles,
        batch_size=batch_size,
    )

    # Calculate final rating
    final_rating = results_df["player_rating_estimate"].iloc[-1]
    final_rd = results_df["player_rd_estimate"].iloc[-1]

    # Calculate accuracy metrics
    move_accuracy = results_df["is_correct"].mean()
    puzzle_completion = results_df.groupby("puzzle_id")["is_correct"].all().mean()

    # Create metrics for wandb
    wandb_metrics = {
        "eval/puzzle_rating": final_rating,
        "eval/puzzle_rating_deviation": final_rd,
        "eval/puzzle_move_accuracy": move_accuracy,
        "eval/puzzle_completion_rate": puzzle_completion,
    }

    # Log to wandb
    wandb.log(wandb_metrics)

    # Also return as TestInfo objects for compatibility
    return [
        TestInfo(loss_type="puzzle_rating", loss=final_rating),
        TestInfo(loss_type="puzzle_move_accuracy", loss=move_accuracy),
    ]


def _puzzle_bracket_eval_fn(model, epoch, agent_config, dataset, puzzles_per_bracket):
    """Helper function for training_puzzle_eval_by_rating"""
    # Create agent from model using config
    agent = agent_config(model=model)

    # Evaluate on rating brackets
    bracket_results = evaluate_puzzle_by_rating_brackets(
        agent=agent,
        dataset=dataset,
        puzzles_per_bracket=puzzles_per_bracket,
    )

    # Create metrics for wandb
    wandb_metrics = {}
    test_infos = []

    for bracket_name, accuracy in bracket_results.items():
        wandb_metrics[f"eval/puzzle_acc_{bracket_name}"] = accuracy
        test_infos.append(
            TestInfo(loss_type=f"puzzle_acc_{bracket_name}", loss=accuracy)
        )

    # Calculate overall average
    if test_infos:
        overall_avg = np.mean([info.loss for info in test_infos])
        wandb_metrics["eval/puzzle_acc_average"] = overall_avg

    # Log to wandb
    wandb.log(wandb_metrics)

    return test_infos


def training_puzzle_eval(
    agent_config,
    num_puzzles: int = 500,
    warmup_puzzles: int = 100,
    batch_size: int = 50,
):
    """Creates a training-compatible puzzle evaluation function"""

    # Load dataset once
    dataset = datasets.load_dataset("Lichess/chess-puzzles")["train"]
    dataset = dataset.filter(lambda x: x["Popularity"] > 0)

    # Return a lambda that calls the helper function with the dataset
    return lambda model, epoch: _puzzle_eval_fn(
        model, epoch, agent_config, dataset, num_puzzles, warmup_puzzles, batch_size
    )


def training_puzzle_eval_by_rating(
    agent_config,
    puzzles_per_bracket: int = 100,
):
    """Creates a training-compatible puzzle evaluation function for rating brackets"""

    # Load dataset once
    dataset = datasets.load_dataset("Lichess/chess-puzzles")["train"]
    dataset = dataset.filter(lambda x: x["Popularity"] > 0)

    # Return a lambda that calls the helper function with the dataset
    return lambda model, epoch: _puzzle_bracket_eval_fn(
        model, epoch, agent_config, dataset, puzzles_per_bracket
    )


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
