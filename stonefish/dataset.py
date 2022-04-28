"""
Simple pytorch dataset for the chess data
"""

import torch
from torch.utils.data import Dataset

from stonefish.rep import BoardRep, MoveRep, MoveEnum
from stonefish.ttt import TTTBoardRep, TTTMoveRep
from torch.nn.utils.rnn import pad_sequence

from chessenv import CBoard


def default_collate_fn(batch):
    source, target = zip(*batch)

    source = pad_sequence(source, batch_first=True, padding_value=-100)
    target = pad_sequence(target, batch_first=True, padding_value=-100)

    return source, target


def single_default_collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=-100)


class ChessData(Dataset):
    def __init__(self, path, input_rep, output_rep):
        super().__init__()
        self.input_rep = input_rep
        self.output_rep = output_rep

        with open(path, "r") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        raw = self.data[i].rstrip().split(",")

        # Data is "fen,move"
        board_fen = raw[0]
        actions = raw[1]

        board_tokens = self.input_rep.from_fen(board_fen)
        move = self.output_rep.from_str(actions)

        try:
            move.to_str()
        except:
            import pdb

            pdb.set_trace()

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
