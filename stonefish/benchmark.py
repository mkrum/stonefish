"""Benchmark a model's training step time.

Loads a model from a config file and measures the full training step
(forward + backward + optimizer) with proper device synchronization.

Usage:
    uv run python -m stonefish.benchmark configs/test_binary.yml
    uv run python -m stonefish.benchmark configs/test_binary.yml --device cpu
    uv run python -m stonefish.benchmark configs/test_binary.yml --batch-size 4096
"""

import argparse
import time

import torch
import torch.nn.functional as functional
from yamlargs.config import YAMLConfig

from stonefish.config import expose_modules


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_step(model, opt, batch_size, device, num_trials=50, warmup=5):
    """Benchmark a full training step: forward + loss + backward + clip + optimizer."""
    model.train()
    x = torch.randn(batch_size, 8, 8, 20, device=device)
    y = torch.randint(0, 5700, (batch_size,), device=device)

    # Warmup
    for _ in range(warmup):
        opt.zero_grad()
        probs = model(x, y)
        loss = functional.cross_entropy(probs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sync(device)

    # Timed trials
    step_times = []
    for _ in range(num_trials):
        sync(device)
        t0 = time.perf_counter()
        opt.zero_grad()
        probs = model(x, y)
        loss = functional.cross_entropy(probs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sync(device)
        step_times.append(time.perf_counter() - t0)

    return step_times


def main():
    expose_modules()

    parser = argparse.ArgumentParser(description="Benchmark model training step")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    config = YAMLConfig.load(args.config)
    model = config["model"]().to(device)
    opt = config["opt"](model.parameters())
    if args.batch_size:
        batch_size = args.batch_size
    else:
        # Try to extract from config, fall back to 2048
        try:
            ctx = config["pretrain_context"]()
            batch_size = ctx.train_dl.batch_size
        except Exception:
            batch_size = 2048

    num_params = sum(p.numel() for p in model.parameters())

    print(f"Model:      {model.__class__.__name__}")
    print(f"Parameters: {num_params:,}")
    print(f"Device:     {device}")
    print(f"Batch size: {batch_size}")
    print(f"Trials:     {args.trials}")
    print()

    step_times = bench_step(model, opt, batch_size, device, args.trials, args.warmup)

    mean_ms = sum(step_times) / len(step_times) * 1000
    min_ms = min(step_times) * 1000
    max_ms = max(step_times) * 1000
    p50_ms = sorted(step_times)[len(step_times) // 2] * 1000
    samples_per_sec = batch_size / (sum(step_times) / len(step_times))

    print("=" * 50)
    print("STEP TIME (forward + backward + optimizer)")
    print("=" * 50)
    print(f"  {'mean:':<8} {mean_ms:>8.1f} ms")
    print(f"  {'min:':<8} {min_ms:>8.1f} ms")
    print(f"  {'max:':<8} {max_ms:>8.1f} ms")
    print(f"  {'p50:':<8} {p50_ms:>8.1f} ms")
    print()
    print(f"  {'samples/sec:':<14} {samples_per_sec:>10,.0f}")
    print(f"  {'batches/sec:':<14} {1000/mean_ms:>10.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
