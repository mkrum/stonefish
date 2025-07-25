import re

import chess
import chess.pgn
import datasets


def clean_move(move):
    """Remove annotations like !, ?, ??, !?, etc."""
    return re.sub(r"[!?]+", "", move)


def handle_movetext(movetext):
    boards = []
    moves = []

    # Remove annotations inside braces
    cleaned = re.sub(r"\{[^}]*\}", "", movetext)
    # Remove move numbers (1., 1... etc.)
    cleaned = re.sub(r"\d+\.(\.\.)?", "", cleaned)
    # Split into tokens
    tokens = cleaned.strip().split()

    # Remove result markers
    tokens = [t for t in tokens if t not in ["1-0", "0-1", "1/2-1/2", "*"]]

    board = chess.Board()

    for token in tokens:
        move_str = clean_move(token)
        try:
            boards.append(board.fen())
            move = board.parse_san(move_str)
            board.push(move)
            moves.append(move.uci())
        except ValueError as e:
            print(f"Error parsing move {move_str} -> '{token}' on board:\n{board}\n{e}")
            break

    return boards, moves


def movetext_to_boards(data):
    movetext = data["movetext"]

    all_boards = []
    all_moves = []

    for movetext in data["movetext"]:
        try:
            boards, moves = handle_movetext(movetext)
        except Exception:
            boards, moves = [], []

        if len(boards) == len(moves):
            all_boards += boards
            all_moves += moves

    return {"board": all_boards, "move": all_moves}


if __name__ == "__main__":

    data = datasets.load_dataset("Lichess/tournament-chess-games", streaming=True)

    for d in data["train"]:
        boards, moves = handle_movetext(d["movetext"])
        breakpoint()

    # data = datasets.load_dataset("Lichess/tournament-chess-games")
    # data = data['train'].train_test_split(test_size=1000)
    # data = data.map(movetext_to_boards, batched=True, remove_columns=data['train'].column_names)
    # data.save_to_disk("parsed_games")

    # exit()
