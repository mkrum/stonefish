import argparse

import datasets
from datasets import disable_caching
from pull_data import handle_movetext

repo_name = "mkrum/LichessParsed"


def process_single_game(game_data):
    """Process a single game and return list of board/move pairs"""

    white_elo = game_data.get("WhiteElo", 0)
    black_elo = game_data.get("BlackElo", 0)

    use_white = white_elo is not None and white_elo > 1999
    use_black = black_elo is not None and black_elo > 1999

    boards, moves = handle_movetext(game_data["movetext"])
    results = []

    for idx, (b, m) in enumerate(zip(boards, moves, strict=False)):

        move_data = {"board": b, "move": m, "move_idx": idx, **game_data}

        if idx % 2 == 0 and use_white:
            results.append(move_data)
        elif use_black:
            results.append(move_data)

    return results


def process_batch(batch):
    """Process a batch of games for use with dataset.map(batched=True)"""
    all_boards = []
    all_moves = []
    all_metadata = {k: [] for k in batch.keys()}

    # Process each game in the batch
    for i in range(len(batch["Event"])):
        game_data = {k: v[i] for k, v in batch.items()}
        results = process_single_game(game_data)

        for result in results:
            all_boards.append(result["board"])
            all_moves.append(result["move"])
            # Add all other metadata
            for k in batch.keys():
                all_metadata[k].append(result[k])

    # Return flattened batch
    return {"board": all_boards, "move": all_moves, **all_metadata}


def game_filter(game_data, elo_min: int = 1999):
    """Process a single game and return list of board/move pairs"""
    white_elo = game_data.get("WhiteElo", 0)
    black_elo = game_data.get("BlackElo", 0)

    if white_elo is None:
        white_elo = 0

    if black_elo is None:
        black_elo = 0

    one_side_good_enough = (white_elo > elo_min) or (black_elo > elo_min)

    is_classical = (
        "Classical" in game_data["Event"] or "Correspondence" in game_data["Event"]
    )
    ended_normally = game_data["Termination"] == "Normal"

    return one_side_good_enough and is_classical and ended_normally


def parse_year_month(
    target_repo_name: str,
    year: int,
    month: int,
    elo_min: int = 1999,
    batch_size: int = 1000,
    num_proc: int = 8,
):
    """Load dataset upfront and process with parallel map"""
    data_dir = f"data/year={year}/month={month:02}"

    print(f"Loading dataset for {year}-{month:02}...")
    # Load without streaming to enable num_proc
    data = datasets.load_dataset("Lichess/standard-chess-games", data_dir=data_dir)

    print(f"Processing {len(data['train'])} games with {num_proc} processes...")
    # Process with batched map and multiprocessing
    filtered = data["train"].filter(
        game_filter, num_proc=num_proc, desc="Filtering games"
    )
    print(f"{len(filtered)} valid games...")

    # Process with batched map and multiprocessing
    processed = filtered.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Processing games",
    )

    # Shuffle
    processed = processed.shuffle(seed=123)

    print(f"Pushing {len(processed)} positions to hub...")
    processed.push_to_hub(target_repo_name, private=False, data_dir=data_dir)
    print(f"Completed {year}-{month:02}")
    data.cleanup_cache_files()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, help="Year to process")
    parser.add_argument("month", type=int, help="Month to process")
    args = parser.parse_args()

    disable_caching()
    parse_year_month(repo_name, args.year, args.month, num_proc=8, batch_size=500)
