#!/usr/bin/env python
"""
Fast parallel version - process multiple months simultaneously
"""

import argparse
import csv
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import datasets
from datasets import disable_caching

target_repo = "mkrum/OneBillionMoves"
source_repo = "mkrum/LichessParsedBlitz"


def read_count_log(filepath="data/count_log.csv"):
    """Read year/month combinations from count_log.csv"""
    tasks = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        year = int(row[0])
                        month = int(row[1])
                        tasks.append((year, month))
                    except ValueError:
                        continue
    return list(set(tasks))  # Remove duplicates


def read_completed_months(filepath="parallel_process_log.csv"):
    """Read successfully completed months from log"""
    completed = set()
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4 and row[3] == "success":
                    try:
                        year = int(row[0])
                        month = int(row[1])
                        completed.add((year, month))
                    except ValueError:
                        continue
    return completed


def process_single_month(year_month_tuple):
    """Process one year/month of data - designed for parallel execution"""
    year, month = year_month_tuple
    data_dir = f"data/year={year}/month={month:02}"

    try:
        print(f"[{year}-{month:02}] Starting...")
        start = time.time()

        # Load with streaming
        dataset = datasets.load_dataset(source_repo, data_dir=data_dir, streaming=True)

        # Use map to process in batches (much faster than iterating)
        def extract_board_move(examples):
            """Extract only board and move columns"""
            return {"board": examples["board"], "move": examples["move"]}

        # Process in larger batches for efficiency
        print(f"[{year}-{month:02}] Processing data...")
        train_data = dataset["train"].map(
            extract_board_move,
            batched=True,
            batch_size=10000,  # Process 10k at a time
            remove_columns=[
                col
                for col in dataset["train"].column_names
                if col not in ["board", "move"]
            ],
        )

        # Convert to regular dataset for pushing
        print(f"[{year}-{month:02}] Collecting data...")
        boards = []
        moves = []
        count = 0

        for batch in train_data.iter(batch_size=10000):
            boards.extend(batch["board"])
            moves.extend(batch["move"])
            count += len(batch["board"])
            if count % 100000 == 0:
                print(f"[{year}-{month:02}] Collected {count:,} samples...")

        # Create dataset
        from datasets import Dataset

        slim_data = Dataset.from_dict({"board": boards, "move": moves})

        # Push to hub
        print(f"[{year}-{month:02}] Pushing {len(slim_data):,} samples to hub...")
        slim_data.push_to_hub(target_repo, private=False, data_dir=data_dir)

        elapsed = time.time() - start
        print(
            f"[{year}-{month:02}] ✓ Complete! {len(slim_data):,} samples in {elapsed:.1f}s"
        )

        return year, month, len(slim_data), "success", elapsed

    except Exception as e:
        print(f"[{year}-{month:02}] ✗ Error: {e}")
        return year, month, 0, f"error: {e}", 0


def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel OneBillionMoves creation"
    )
    parser.add_argument("--start-year", type=int, default=2013, help="Start year")
    parser.add_argument("--start-month", type=int, default=1, help="Start month")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument("--end-month", type=int, default=10, help="End month")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument("--test", action="store_true", help="Test with just 2 months")
    parser.add_argument(
        "--from-count-log",
        action="store_true",
        help="Read months from count_log.csv and skip already completed ones",
    )

    args = parser.parse_args()

    # Disable caching
    disable_caching()

    # Generate list of year/month tuples
    if args.from_count_log:
        # Read from count_log.csv
        print("Reading months from data/count_log.csv...")
        all_tasks = read_count_log()
        all_tasks.sort()  # Sort by year, month

        # Filter out already completed
        completed = read_completed_months()
        tasks = [task for task in all_tasks if task not in completed]

        print(f"Found {len(all_tasks)} total months")
        print(f"Already completed: {len(completed)}")
        print(f"Remaining to process: {len(tasks)}")
    else:
        # Use date range
        tasks = []
        current_year = args.start_year
        current_month = args.start_month

        while (current_year < args.end_year) or (
            current_year == args.end_year and current_month <= args.end_month
        ):
            tasks.append((current_year, current_month))
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

    if args.test:
        tasks = tasks[:2]  # Just test with first 2 months
        print(f"TEST MODE: Processing only {tasks}")

    print(f"Processing {len(tasks)} months with {args.workers} workers")
    print("=" * 60)

    # Process in parallel
    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_month, task): task for task in tasks}

        # Process as they complete
        for future in as_completed(futures):
            year, month, samples, status, elapsed = future.result()
            results.append((year, month, samples, status, elapsed))

            # Log to file
            with open("parallel_process_log.csv", "a") as f:
                f.write(f"{year},{month},{samples},{status},{elapsed:.1f}\n")

    # Summary
    total_time = time.time() - start_time
    total_samples = sum(r[2] for r in results if r[3] == "success")
    successful = sum(1 for r in results if r[3] == "success")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Processed {successful}/{len(tasks)} months successfully")
    print(f"Total samples: {total_samples:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per month: {total_time/len(tasks):.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    main()
