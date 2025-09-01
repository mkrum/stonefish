#!/usr/bin/env python
"""
Get timing statistics from W&B
"""

import argparse

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="W&B entity")
    parser.add_argument("--project", default="stonefish", help="W&B project")
    parser.add_argument("--run-id", default=None, help="Run ID (default: latest)")

    args = parser.parse_args()

    api = wandb.Api()

    if args.run_id:
        run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
    else:
        runs = api.runs(
            f"{args.entity}/{args.project}", order="-created_at", per_page=1
        )
        run = runs[0]

    print(f"Run: {run.name} ({run.id})")

    # Get timing metrics
    history = run.history()
    timing_cols = [col for col in history.columns if col.startswith("timing/")]

    df = history[timing_cols].dropna()
    print(df.mean())


if __name__ == "__main__":
    main()
