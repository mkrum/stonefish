"""
Evaluate chess agents against each other.

Usage:
    python -m stonefish.eval --agent1 model:config.yml:checkpoint.pth --agent2 random --games 100
"""

import argparse
from typing import List

from mllg import TestInfo

from stonefish.config import expose_modules
from stonefish.eval.agent import evaluate_agents
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate two chess agents against each other",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Agent specifications
    parser.add_argument(
        "--agent1",
        required=True,
        help="First agent: model:config.yml:checkpoint.pth, random, or stockfish:depth=N",
    )
    parser.add_argument(
        "--agent2",
        required=True,
        help="Second agent: model:config.yml:checkpoint.pth, random, or stockfish:depth=N",
    )

    # Number of games
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play (default: 100)",
    )

    # Device
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on"
    )

    args = parser.parse_args()

    # Load agents
    print(f"Loading agent 1: {args.agent1}")
    agent1 = load_agent(args.agent1, device=args.device)

    print(f"Loading agent 2: {args.agent2}")
    agent2 = load_agent(args.agent2, device=args.device)

    # Run evaluation
    print(f"\nPlaying {args.games} games between agents...")
    results = evaluate_agents(agent1, agent2, num_games=args.games)

    # Display results
    print_results_table(results, f"{args.agent1} vs {args.agent2}")

    # Extract wins/losses from results
    wins = int(results[0].loss * args.games)  # win_rate * num_games
    losses = int(results[1].loss * args.games)  # loss_rate * num_games
    draws = args.games - wins - losses

    # Clear summary
    print(f"\n{args.agent1}: {wins} wins")
    print(f"{args.agent2}: {losses} wins")
    print(f"Draws: {draws}")


if __name__ == "__main__":
    expose_modules()
    main()
