import datasets
import tqdm
from pull_data import handle_movetext

data = datasets.load_dataset("Lichess/standard-chess-games", streaming=True)
repo_name = "mkrum/BillionChessMoves"

elo_min = 1999

batch = []
index = 0

with tqdm.tqdm(total=10**9) as pbar:

    for d in data["train"]:
        white_elo = d.get("WhiteElo", 0)
        black_elo = d.get("BlackElo", 0)

        if white_elo is None:
            white_elo = 0

        if black_elo is None:
            black_elo = 0

        if white_elo > elo_min and black_elo > elo_min and d["Termination"] == "Normal":
            boards, moves = handle_movetext(d["movetext"])

            for b, m in zip(boards, moves, strict=False):
                batch.append({"board": b, "move": m, **d})
                pbar.update(1)

            if len(batch) >= 1000000:
                df = datasets.Dataset.from_dict(batch)
                df.push_to_hub(repo_name, private=False, data_dir=f"data_{index}")
                index += 1
                batch = []
