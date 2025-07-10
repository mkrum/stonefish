"""
Evaluate chess agents with various evaluation methods.

Usage:
    python -m stonefish.eval --agent model:config.yml:checkpoint.pth --eval-games-random 100
"""

import argparse
from typing import List

from mllg import TestInfo

from stonefish.eval.agent import evaluate_agent_vs_random, evaluate_agent_vs_stockfish
from stonefish.eval.agent_loader import load_agent


def print_results_table(results: List[TestInfo], title: str):
    """Print evaluation results in a table format."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for info in results:
        if isinstance(info.loss, float):
            print(f"{info.loss_type:<30} {info.loss:>10.4f}")
        else:
            print(f"{info.loss_type:<30} {info.loss:>10}")

    print("=" * 60)


def run_game_evaluations(agent, args) -> List[TestInfo]:
    """Run all requested game-based evaluations."""
    all_results = []

    if args.eval_games_random:
        print(f"\nEvaluating vs random opponent ({args.eval_games_random} games)...")
        results = evaluate_agent_vs_random(agent, num_games=args.eval_games_random)
        all_results.extend(results)
        print_results_table(results, "VS RANDOM OPPONENT")

    if args.eval_games_stockfish:
        print(
            f"\nEvaluating vs Stockfish depth {args.stockfish_depth} ({args.eval_games_stockfish} games)..."
        )
        results = evaluate_agent_vs_stockfish(
            agent,
            num_games=args.eval_games_stockfish,
            stockfish_depth=args.stockfish_depth,
        )
        all_results.extend(results)
        print_results_table(results, f"VS STOCKFISH (depth {args.stockfish_depth})")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate chess agents with various evaluation methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Agent specification
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent specification: model:config.yml:checkpoint.pth, random, or stockfish:depth=N",
    )
    parser.add_argument(
        "--agent2", help="Second agent for comparison evaluations (optional)"
    )

    # Device
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on"
    )

    # Game-based evaluations
    parser.add_argument(
        "--eval-games-random",
        type=int,
        help="Number of games to play vs random opponent",
    )
    parser.add_argument(
        "--eval-games-stockfish", type=int, help="Number of games to play vs Stockfish"
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=1,
        help="Stockfish search depth for game evaluation",
    )

    args = parser.parse_args()

    # Load agent
    print(f"Loading agent: {args.agent}")
    agent = load_agent(args.agent, device=args.device)

    # Run evaluations
    all_results = []

    # Game-based evaluations
    game_results = run_game_evaluations(agent, args)
    all_results.extend(game_results)

    # Summary
    if all_results:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Agent: {args.agent}")
        print(f"Total metrics: {len(all_results)}")
        print("=" * 60)


if __name__ == "__main__":
    main()
