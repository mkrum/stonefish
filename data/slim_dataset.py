#!/usr/bin/env python
"""
Create a slimmed down dataset with only board and move columns
Process year/month by year/month to avoid disk space issues
"""

import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import datasets
from datasets import disable_caching

target_repo = "mkrum/OneBillionMoves"
source_repo = "mkrum/LichessParsedBlitz"


def process_year_month(year: int, month: int, dry_run: bool = False):
    """Process one year/month of data"""

    data_dir = f"data/year={year}/month={month:02}"
    print(f"\n{'='*60}")
    print(f"Processing {year}-{month:02}")
    print("=" * 60)

    # Check disk space before starting
    disk_usage = shutil.disk_usage("/")
    free_gb = disk_usage.free / (1024**3)
    print(f"Free disk space: {free_gb:.1f} GB")

    if free_gb < 50:  # Need at least 50GB free
        print(f"WARNING: Low disk space ({free_gb:.1f} GB). Skipping...")
        return False

    try:
        # Load dataset for this specific year/month
        print(f"Loading {source_repo}/{data_dir}...")
        start = time.time()

        # First try streaming to avoid the split info mismatch
        dataset = datasets.load_dataset(
            source_repo,
            data_dir=data_dir,
            streaming=True,  # Use streaming first to get the data
        )

        # Convert streaming dataset to regular dataset for this partition only
        print("Converting streaming dataset to regular dataset...")
        train_data_streaming = dataset["train"]

        # Collect the data (this loads it into memory)
        all_data: Dict[str, List[Any]] = {"board": [], "move": []}
        count = 0
        for example in train_data_streaming:
            all_data["board"].append(example["board"])
            all_data["move"].append(example["move"])
            count += 1
            if count % 100000 == 0:
                print(f"  Loaded {count:,} examples...")

        # Create a new dataset from the collected data
        from datasets import Dataset

        train_data = Dataset.from_dict(all_data)

        load_time = time.time() - start
        num_samples = len(train_data)
        print(f"Loaded {num_samples:,} samples in {load_time:.1f}s")

        # We already have only board and move columns
        slim_data = train_data

        # Verify the data
        print(f"Final columns: {slim_data.column_names}")
        print(f"Final size: {len(slim_data):,} samples")

        # Calculate size reduction by looking at actual data
        # Get a sample to compare
        sample = slim_data[0]

        # Original columns from LichessParsedBlitz
        original_columns = [
            "Event",
            "Site",
            "White",
            "Black",
            "Result",
            "UTCDate",
            "UTCTime",
            "WhiteElo",
            "BlackElo",
            "WhiteRatingDiff",
            "BlackRatingDiff",
            "ECO",
            "Opening",
            "TimeControl",
            "Termination",
            "movetext",
            "board",
            "move",
            "move_idx",
        ]

        # Estimate original size (board + move + all metadata)
        # Rough estimates based on typical values
        metadata_size_estimate = 500  # All the metadata fields
        board_move_size = len(sample["board"]) + len(sample["move"])
        original_size_estimate = board_move_size + metadata_size_estimate
        slim_size_estimate = board_move_size

        reduction_pct = (1 - slim_size_estimate / original_size_estimate) * 100

        print("\nSize analysis:")
        print(f"  Original columns: {len(original_columns)} fields")
        print("  Slim columns: 2 fields (board, move)")
        print(f"  Board size: {len(sample['board'])} chars")
        print(f"  Move size: {len(sample['move'])} chars")
        print(f"  Estimated metadata overhead: ~{metadata_size_estimate} chars")
        print(f"  Size reduction: ~{reduction_pct:.1f}%")

        # Show sample
        sample = slim_data[0]
        print(f"Sample: board='{sample['board'][:50]}...', move='{sample['move']}'")

        if not dry_run:
            # Push to hub
            print(f"Pushing to {target_repo}/{data_dir}...")
            start = time.time()
            slim_data.push_to_hub(target_repo, private=False, data_dir=data_dir)
            push_time = time.time() - start
            print(f"Pushed in {push_time:.1f}s")
        else:
            print("DRY RUN: Skipping push to hub")

        # Log success
        with open("process_log.csv", "a") as f:
            f.write(f"{year},{month},{num_samples},success\n")

        print(f"✓ Completed {year}-{month:02}")
        return True

    except Exception as e:
        print(f"✗ Error processing {year}-{month:02}: {e}")
        with open("process_log.csv", "a") as f:
            f.write(f"{year},{month},0,error: {e}\n")
        return False
    finally:
        # Always try to clean up
        try:
            datasets.disable_caching()
            import gc

            gc.collect()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Create OneBillionMoves dataset")
    parser.add_argument("--start-year", type=int, default=2013, help="Start year")
    parser.add_argument("--start-month", type=int, default=1, help="Start month")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument("--end-month", type=int, default=10, help="End month")
    parser.add_argument("--dry-run", action="store_true", help="Don't push to hub")
    parser.add_argument(
        "--single",
        nargs=2,
        type=int,
        metavar=("YEAR", "MONTH"),
        help="Process single year/month",
    )

    args = parser.parse_args()

    # Disable caching to save disk space
    disable_caching()

    # Create log file
    if not Path("process_log.csv").exists():
        with open("process_log.csv", "w") as f:
            f.write("year,month,samples,status\n")

    if args.single:
        # Process single year/month
        year, month = args.single
        process_year_month(year, month, dry_run=args.dry_run)
    else:
        # Process range
        current_year = args.start_year
        current_month = args.start_month

        while (current_year < args.end_year) or (
            current_year == args.end_year and current_month <= args.end_month
        ):
            success = process_year_month(
                current_year, current_month, dry_run=args.dry_run
            )

            if not success and not args.dry_run:
                print("\nStopping due to error or low disk space")
                break

            # Move to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

            # Small delay between uploads
            if not args.dry_run:
                time.sleep(2)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
