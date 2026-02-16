"""Benchmark harness for the binary chess data loader.

Usage:
    uv run python -m stonefish.benchmark_data --data-path chessdataloader/data/elite.bin
    uv run python -m stonefish.benchmark_data --data-path chessdataloader/data/elite.bin --simulate-ms 700
    uv run python -m stonefish.benchmark_data --data-path chessdataloader/data/elite.bin --num-workers 1
"""

import argparse
import json
import resource
import statistics
import sys
import time

import numpy as np

from stonefish.binary_dataset import BinaryChessDataLoader


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)  # bytes on macOS
    return usage.ru_maxrss / 1024  # KB on Linux


def main():
    parser = argparse.ArgumentParser(description="Benchmark binary chess data loader")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--board-format", choices=["lczero", "flat"], default="lczero")
    parser.add_argument(
        "--move-format", choices=["cmove", "from_to_promo"], default="cmove"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--simulate-ms",
        type=float,
        default=0,
        help="Simulate a model step by sleeping this many ms per batch (releases GIL)",
    )
    args = parser.parse_args()

    loader = BinaryChessDataLoader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        board_format=args.board_format,
        move_format=args.move_format,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    simulate_sec = args.simulate_ms / 1000
    total_batches = (
        min(len(loader), args.max_batches) if args.max_batches > 0 else len(loader)
    )

    print(f"Dataset:      {loader.num_samples:,} samples")
    print(f"Batch size:   {args.batch_size:,}")
    print(f"Workers:      {args.num_workers}")
    print(f"Prefetch:     {args.prefetch_factor}")
    print(f"Board format: {args.board_format}")
    print(f"Move format:  {args.move_format}")
    print(
        f"Batches:      {total_batches:,}"
        + (f" (of {len(loader):,})" if args.max_batches > 0 else "")
    )
    if simulate_sec > 0:
        print(f"Simulate:     {args.simulate_ms:.1f} ms/batch")
    print()

    batch_latencies: list[float] = []
    total_samples = 0
    num_batches = 0

    wall_start = time.perf_counter()
    t0 = wall_start

    for boards, _moves in loader:
        t1 = time.perf_counter()
        batch_latencies.append(t1 - t0)
        total_samples += boards.shape[0]
        num_batches += 1
        if args.max_batches > 0 and num_batches >= args.max_batches:
            break
        if simulate_sec > 0:
            time.sleep(simulate_sec)
        t0 = time.perf_counter()

    wall_time = time.perf_counter() - wall_start

    samples_per_sec = total_samples / wall_time
    batches_per_sec = num_batches / wall_time
    lat_min = min(batch_latencies)
    lat_max = max(batch_latencies)
    lat_mean = statistics.mean(batch_latencies)
    lat_p50 = np.percentile(batch_latencies, 50)
    lat_p95 = np.percentile(batch_latencies, 95)
    lat_p99 = np.percentile(batch_latencies, 99)
    peak_rss = get_peak_rss_mb()

    print("=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"{'Total wall time:':<30} {wall_time:.2f} s")
    print(f"{'Total samples:':<30} {total_samples:,}")
    print(f"{'Total batches:':<30} {num_batches:,}")
    print(f"{'Throughput (samples/sec):':<30} {samples_per_sec:,.0f}")
    print(f"{'Throughput (batches/sec):':<30} {batches_per_sec:,.1f}")
    print()
    print("Batch Latency (seconds):")
    print(f"  {'min:':<10} {lat_min:.6f}")
    print(f"  {'max:':<10} {lat_max:.6f}")
    print(f"  {'mean:':<10} {lat_mean:.6f}")
    print(f"  {'p50:':<10} {lat_p50:.6f}")
    print(f"  {'p95:':<10} {lat_p95:.6f}")
    print(f"  {'p99:':<10} {lat_p99:.6f}")
    print()
    print(f"{'Peak RSS:':<30} {peak_rss:.1f} MB")
    print("=" * 50)

    results = {
        "config": {
            "data_path": args.data_path,
            "batch_size": args.batch_size,
            "board_format": args.board_format,
            "move_format": args.move_format,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "num_samples": loader.num_samples,
            "simulate_ms": args.simulate_ms,
            "max_batches": args.max_batches,
        },
        "throughput": {
            "samples_per_sec": round(samples_per_sec, 1),
            "batches_per_sec": round(batches_per_sec, 2),
            "wall_time_sec": round(wall_time, 3),
        },
        "batch_latency_sec": {
            "min": round(lat_min, 6),
            "max": round(lat_max, 6),
            "mean": round(lat_mean, 6),
            "p50": round(float(lat_p50), 6),
            "p95": round(float(lat_p95), 6),
            "p99": round(float(lat_p99), 6),
        },
        "memory": {
            "peak_rss_mb": round(peak_rss, 1),
        },
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
