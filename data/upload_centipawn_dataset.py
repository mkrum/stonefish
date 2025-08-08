"""
Script to convert JSONL centipawn estimates to HuggingFace dataset and upload.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Union

import tqdm
from datasets import Dataset


def read_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Read JSONL file and return list of dictionaries."""
    with open(file_path, "r") as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            if line:
                yield json.loads(line)


def create_hf_dataset(jsonl_data: List[Dict[str, Any]]) -> Dataset:
    """Convert JSONL data to HuggingFace Dataset, separating mates from values."""
    boards = []
    moves = []
    values = []
    mates = []

    for item in jsonl_data:
        boards.append(item["board"])
        moves.append(item["moves"])

        # Separate mates from centipawn values
        item_values: List[Union[int, None]] = []
        item_mates: List[Union[int, None]] = []

        for value in item["values"]:
            # Check if this is a mate value (>= 1M or <= -1M)
            if abs(value) >= 1000000:
                mate_distance = int(value / 1000000)
                item_values.append(None)
                item_mates.append(mate_distance)
            else:
                item_values.append(value)
                item_mates.append(None)

        values.append(item_values)
        mates.append(item_mates)

    dataset_dict = {"board": boards, "moves": moves, "values": values, "mates": mates}

    return Dataset.from_dict(dataset_dict)


def upload_to_hf(dataset: Dataset, repo_name: str):
    """Upload dataset to HuggingFace Hub."""


def main():
    parser = argparse.ArgumentParser(
        description="Upload centipawn dataset to HuggingFace"
    )
    parser.add_argument("jsonl_file", help="Path to JSONL file")
    parser.add_argument(
        "repo_name", help="HuggingFace repo name (e.g., 'username/dataset-name')"
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.jsonl_file).exists():
        raise ValueError(f"Error: File {args.jsonl_file} not found")

    batch = []
    index = 0
    for d in read_jsonl(args.jsonl_file):
        batch.append(d)

        if len(batch) == 1000000:
            df = create_hf_dataset(batch)
            df.push_to_hub(args.repo_name, private=False, data_dir=f"data_{index}")
            index += 1
            batch = []

    df = create_hf_dataset(batch)
    df.push_to_hub(args.repo_name, private=False, append=True)


if __name__ == "__main__":
    main()
