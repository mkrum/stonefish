import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import chess
import chess.engine
import chess.pgn
import torch
import torch.nn as nn
from yamlargs.config import YAMLConfig

from stonefish.config import expose_modules


@dataclass
class BenchmarkResults:
    """Container for benchmark timing results"""

    model_name: str
    batch_size: int
    forward_times: List[float]
    backward_times: List[float]
    memory_mb: float
    num_parameters: int

    @property
    def forward_mean(self) -> float:
        return statistics.mean(self.forward_times)

    @property
    def forward_std(self) -> float:
        return (
            statistics.stdev(self.forward_times) if len(self.forward_times) > 1 else 0.0
        )

    @property
    def backward_mean(self) -> float:
        return statistics.mean(self.backward_times)

    @property
    def backward_std(self) -> float:
        return (
            statistics.stdev(self.backward_times)
            if len(self.backward_times) > 1
            else 0.0
        )

    @property
    def throughput_fps(self) -> float:
        """Forward passes per second"""
        return self.batch_size / self.forward_mean

    def __str__(self) -> str:
        return (
            f"{self.model_name} (batch={self.batch_size}, {self.num_parameters:,} params): "
            f"Forward {self.forward_mean*1000:.2f}±{self.forward_std*1000:.2f}ms, "
            f"Backward {self.backward_mean*1000:.2f}±{self.backward_std*1000:.2f}ms, "
            f"Throughput {self.throughput_fps:.1f} FPS, "
            f"Memory {self.memory_mb:.1f}MB"
        )


def benchmark_model(
    model: torch.nn.Module,
    batch_sizes: Optional[List[int]] = None,
    num_trials: int = 100,
    warmup_trials: int = 10,
    device: str = "cuda",
) -> List[BenchmarkResults]:
    """
    Benchmark forward and backward pass speeds for a chess model.

    Args:
        model: The model to benchmark
        batch_sizes: List of batch sizes to test
        num_trials: Number of timing trials per batch size
        warmup_trials: Number of warmup runs before timing
        device: Device to run on ("cuda" or "cpu")

    Returns:
        List of BenchmarkResults for each batch size
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.train()  # Enable gradients for backward pass

    # Count parameters
    num_parameters = sum(p.numel() for p in model.parameters())

    results = []

    for batch_size in batch_sizes:
        print(
            f"Benchmarking {model.__class__.__name__} with batch size {batch_size}..."
        )

        # Create sample input based on model type
        if hasattr(model, "board_tokenizer"):
            # Use the model's tokenizer to create proper input
            boards = [chess.Board() for _ in range(batch_size)]
            input_tensor = model.board_tokenizer.from_board_batch(boards).to(device_obj)
            print(f"  Input tensor shape: {input_tensor.shape}")
        else:
            # Fallback for older models - assume flat input
            input_tensor = torch.randn(batch_size, 69, device=device_obj)

        # Create dummy targets for loss calculation
        targets = torch.randint(0, 5700, (batch_size,), device=device_obj)

        # Warmup runs
        for _ in range(warmup_trials):
            with torch.no_grad():
                _ = model.inference(input_tensor)

        torch.cuda.synchronize() if device_obj.type == "cuda" else None

        forward_times = []
        backward_times = []

        for _ in range(num_trials):
            # Forward pass timing
            torch.cuda.synchronize() if device_obj.type == "cuda" else None
            start_time = time.perf_counter()

            outputs = model.inference(input_tensor)

            torch.cuda.synchronize() if device_obj.type == "cuda" else None
            forward_time = time.perf_counter() - start_time
            forward_times.append(forward_time)

            # Backward pass timing
            loss = nn.CrossEntropyLoss()(outputs, targets)

            torch.cuda.synchronize() if device_obj.type == "cuda" else None
            start_time = time.perf_counter()

            loss.backward()

            torch.cuda.synchronize() if device_obj.type == "cuda" else None
            backward_time = time.perf_counter() - start_time
            backward_times.append(backward_time)

            # Clear gradients for next iteration
            model.zero_grad()

        # Measure memory usage
        if device_obj.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated(device_obj) / 1024 / 1024
            torch.cuda.reset_peak_memory_stats(device_obj)
        else:
            memory_mb = 0.0

        result = BenchmarkResults(
            model_name=model.__class__.__name__,
            batch_size=batch_size,
            forward_times=forward_times,
            backward_times=backward_times,
            memory_mb=memory_mb,
            num_parameters=num_parameters,
        )

        results.append(result)
        print(f"  {result}")

    return results


def print_benchmark_summary(results: List[BenchmarkResults]):
    """Print a formatted summary of benchmark results"""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by model name
    models: Dict[str, List[BenchmarkResults]] = {}
    for result in results:
        if result.model_name not in models:
            models[result.model_name] = []
        models[result.model_name].append(result)

    for model_name, model_results in models.items():
        print(f"\n{model_name}:")
        print("-" * len(model_name))
        print(
            f"{'Batch':<6} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Throughput (FPS)':<18} {'Memory (MB)':<12}"
        )
        print(f"Parameters: {model_results[0].num_parameters:,}")
        print("-" * 76)

        for result in model_results:
            print(
                f"{result.batch_size:<6} "
                f"{result.forward_mean*1000:>7.2f}±{result.forward_std*1000:<5.2f} "
                f"{result.backward_mean*1000:>7.2f}±{result.backward_std*1000:<6.2f} "
                f"{result.throughput_fps:>14.1f} "
                f"{result.memory_mb:>8.1f}"
            )


def load_model_from_config(config_path: str, device: str = "cuda"):
    """Load a model from config file (no weights needed for benchmarking)"""

    # Load config using yamlargs
    config = YAMLConfig.load(config_path)

    # Build model - call () to instantiate from LazyConstructor
    model = config["model"]()
    print(f"Model class: {model.__class__.__name__}")
    print(
        f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}"
    )
    model = model.to(device)

    return model


if __name__ == "__main__":
    import argparse

    expose_modules()

    parser = argparse.ArgumentParser(description="Benchmark chess model performance")
    parser.add_argument("config", type=str, help="Path to model config file")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run benchmarks on (auto-detects if not specified)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--trials", type=int, default=100, help="Number of timing trials per batch size"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup trials"
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"
    else:
        device_str = args.device

    device = torch.device(device_str)
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
        device_str = "cpu"

    print(f"Running benchmarks on {device}")
    print(f"Config: {args.config}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Trials per batch: {args.trials}")

    # Load model from config
    try:
        model = load_model_from_config(args.config, device_str)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded model: {model.__class__.__name__} ({num_params:,} parameters)")
        print(f"Board tokenizer type: {model.board_tokenizer.__class__.__name__}")
        print(f"Board tokenizer shape: {model.board_tokenizer.shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # Run benchmark
    results = benchmark_model(
        model,
        batch_sizes=args.batch_sizes,
        num_trials=args.trials,
        warmup_trials=args.warmup,
        device=device_str,
    )

    print_benchmark_summary(results)
