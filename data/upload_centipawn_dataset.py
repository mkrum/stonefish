"""
Script to convert JSONL centipawn estimates to HuggingFace dataset and upload.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import tqdm
from datasets import Dataset


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def create_hf_dataset(jsonl_data: List[Dict[str, Any]]) -> Dataset:
    """Convert JSONL data to HuggingFace Dataset."""
    boards = []
    moves = []
    values = []

    for item in jsonl_data:
        boards.append(item["board"])
        moves.append(item["moves"])
        values.append(item["values"])

    dataset_dict = {"board": boards, "moves": moves, "values": values}

    return Dataset.from_dict(dataset_dict)


def upload_to_hf(dataset: Dataset, repo_name: str):
    """Upload dataset to HuggingFace Hub."""

    dataset.push_to_hub(repo_name, private=False)
    print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_name}")


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

    print(f"Reading JSONL file: {args.jsonl_file}")
    jsonl_data = read_jsonl(args.jsonl_file)
    print(f"Loaded {len(jsonl_data)} records")

    print("Converting to HuggingFace dataset...")
    dataset = create_hf_dataset(jsonl_data)
    print(f"Dataset created with {len(dataset)} rows")
    print(f"Features: {dataset.features}")

    print(f"Uploading to HuggingFace Hub: {args.repo_name}")
    upload_to_hf(dataset, args.repo_name)


if __name__ == "__main__":
    main()
