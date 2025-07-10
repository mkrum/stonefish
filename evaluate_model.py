#!/usr/bin/env python
"""
Evaluate a trained ResNet chess model against random and Stockfish opponents.
"""

import argparse

import torch

from stonefish.eval.agent import evaluate_agent_vs_random, evaluate_agent_vs_stockfish
from stonefish.policy import ModelChessAgent
from stonefish.resnet import ChessResNet


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load a trained chess model from checkpoint (auto-detects type)"""
    from stonefish.resnet import ChessConvNet

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model state dict (handle different checkpoint formats)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # Assume the checkpoint is the state dict itself
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Detect model type and parameters from state dict
    if any("conv" in key for key in state_dict.keys()):
        print("Detected convolutional model (ChessConvNet)")

        # Detect num_filters from first conv layer
        conv_weight_key = "conv_input.0.weight"
        if conv_weight_key in state_dict:
            num_filters = state_dict[conv_weight_key].shape[0]
        else:
            num_filters = 256  # fallback

        # Count number of ResBlocks
        num_blocks = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("res_blocks.") and k.endswith(".conv1.weight")
            ]
        )

        # Detect input channels
        input_channels = (
            state_dict[conv_weight_key].shape[1]
            if conv_weight_key in state_dict
            else 20
        )

        print(
            f"Architecture: {input_channels} input channels, {num_filters} filters, {num_blocks} blocks"
        )

        model = ChessConvNet(
            input_channels=input_channels,
            num_filters=num_filters,
            num_blocks=num_blocks,
            output_dim=5700,
        )
        model_type = "conv"
    else:
        print("Detected linear model (ChessResNet)")

        # Try to detect hidden_dim from first linear layer
        if "input_proj.0.weight" in state_dict:
            hidden_dim = state_dict["input_proj.0.weight"].shape[0]
            input_dim = state_dict["input_proj.0.weight"].shape[1]
        else:
            hidden_dim = 4096
            input_dim = 69

        # Count ResBlocks
        num_blocks = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("res_blocks.") and k.endswith(".block.0.weight")
            ]
        )

        print(
            f"Architecture: {input_dim} input dim, {hidden_dim} hidden dim, {num_blocks} blocks"
        )

        model = ChessResNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            output_dim=5700,
        )
        model_type = "standard"

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model, model_type


def print_results_table(random_results, stockfish_results, args):
    """Print evaluation results in a nice table format"""

    # Extract values
    random_metrics = {r.loss_type: r.loss for r in random_results}
    stockfish_metrics = {r.loss_type: r.loss for r in stockfish_results}

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print("")

    # Random opponent results
    print("VS RANDOM OPPONENT")
    print("-" * 30)
    print(f"Games played:  {args.games_random}")
    print(f"Wins:          {random_metrics['agent_wins']}")
    print(f"Losses:        {random_metrics['agent_losses']}")
    print(f"Draws:         {random_metrics['agent_draws']}")
    print(f"Win rate:      {random_metrics['agent_win_rate']:.1%}")
    print("")

    # Stockfish results
    print(f"VS STOCKFISH (depth {args.stockfish_depth})")
    print("-" * 30)
    print(f"Games played:  {args.games_stockfish}")
    print(f"Wins:          {stockfish_metrics['stockfish_wins']}")
    print(f"Losses:        {stockfish_metrics['stockfish_losses']}")
    print(f"Draws:         {stockfish_metrics['stockfish_draws']}")
    print(f"Win rate:      {stockfish_metrics['stockfish_win_rate']:.1%}")
    print("")

    # Summary table
    print("SUMMARY")
    print("-" * 30)
    print(f"{'Opponent':<15} {'Win Rate':<10} {'Games':<6}")
    print(
        f"{'Random':<15} {random_metrics['agent_win_rate']:<10.1%} {args.games_random:<6}"
    )
    print(
        f"{'Stockfish-{args.stockfish_depth}':<15} {stockfish_metrics['stockfish_win_rate']:<10.1%} {args.games_stockfish:<6}"
    )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained chess model")
    parser.add_argument(
        "--checkpoint", default="latest_checkpoint.pth", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on"
    )
    parser.add_argument(
        "--games-random",
        type=int,
        default=100,
        help="Number of games vs random opponent",
    )
    parser.add_argument(
        "--games-stockfish", type=int, default=50, help="Number of games vs Stockfish"
    )
    parser.add_argument(
        "--stockfish-depth", type=int, default=1, help="Stockfish search depth"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, model_type = load_model(args.checkpoint, args.device)

    # Create agent
    agent = ModelChessAgent(model, model_type=model_type, temperature=1.0, sample=True)

    print(f"Evaluating vs random opponent ({args.games_random} games)...")
    random_results = evaluate_agent_vs_random(agent, num_games=args.games_random)

    print(
        f"Evaluating vs Stockfish depth {args.stockfish_depth} ({args.games_stockfish} games)..."
    )
    stockfish_results = evaluate_agent_vs_stockfish(
        agent, num_games=args.games_stockfish, stockfish_depth=args.stockfish_depth
    )

    print_results_table(random_results, stockfish_results, args)


if __name__ == "__main__":
    main()
