"""
Simple pytorch dataset for the chess data
"""

from torch.utils.data import Dataset

from stonefish.rep import BoardRep, MoveRep
from stonefish.ttt import TTTBoardRep, TTTMoveRep


class ChessData(Dataset):
    def __init__(self, path):
        super().__init__()

        with open(path, "r") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        raw = self.data[i].rstrip().split(",")

        # Data is "fen,move"
        board_fen = raw[0]
        actions = raw[1]

        board_tokens = BoardRep.from_fen(board_fen)
        move = MoveRep.from_str(actions)
        return board_tokens.to_tensor(), move.to_tensor()


class TTTData(ChessData):
    def __getitem__(self, i):
        raw = self.data[i].rstrip().split(",")

        # Data is "board_str,move_int"
        board_str = raw[0]
        action = raw[1]

        board_tokens = TTTBoardRep.from_str(board_str)
        move = TTTMoveRep.from_int(int(action))
        return board_tokens.to_tensor(), move.to_tensor()
