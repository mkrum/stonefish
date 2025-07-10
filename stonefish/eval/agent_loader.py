"""
Agent loading utilities for evaluation system.

Supports loading agents from various sources:
- model:config.yml:checkpoint.pth - load architecture from config, weights from checkpoint
- random - random agent
- stockfish:depth=N - stockfish with parameters
"""

import torch
from yamlargs.config import YAMLConfig

from stonefish.env import RandomAgent, StockfishAgent
from stonefish.types import ChessAgent


def load_agent(agent_spec: str, device: str = "cpu") -> ChessAgent:
    """
    Load an agent from specification string.

    Args:
        agent_spec: Agent specification in format:
            - "model:config.yml:checkpoint.pth"
            - "random"
            - "stockfish:depth=N"
        device: Device to load model on

    Returns:
        ChessAgent instance
    """
    if agent_spec == "random":
        return RandomAgent()

    if agent_spec.startswith("stockfish:"):
        # Parse stockfish parameters - StockfishAgent only accepts depth
        depth = 1  # default
        param_str = agent_spec[10:]  # Remove "stockfish:"
        if param_str:
            for param in param_str.split(","):
                key, value = param.split("=")
                if key == "depth":
                    depth = int(value)
                # Ignore other parameters like time for now
        return StockfishAgent(depth=depth)

    if agent_spec.startswith("model:"):
        parts = agent_spec[6:].split(":")  # Remove "model:"

        if len(parts) != 2:
            raise ValueError(
                f"Model spec must be 'model:config.yml:checkpoint.pth', got: {agent_spec}"
            )

        config_path = parts[0]
        checkpoint_path = parts[1]
        return load_model_from_config(config_path, checkpoint_path, device)

    raise ValueError(f"Unknown agent type: {agent_spec}")


def load_model_from_config(
    config_path: str, checkpoint_path: str, device: str = "cpu"
) -> ChessAgent:
    """Load model architecture from config and weights from checkpoint."""
    # Load config using yamlargs
    config = YAMLConfig.load(config_path)

    # Build model - call () to instantiate from LazyConstructor
    model = config["model"]().to(device)

    # Load checkpoint weights ONLY
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # Build agent from config - pass model to LazyConstructor then instantiate
    agent: ChessAgent = config["agent"](model=model)
    return agent
