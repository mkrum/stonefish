"""
Simple pytorch dataset for the chess data
"""

import chess
import datasets
import torch
from fastchessenv import CBoard, CMove
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from stonefish.convert import board_to_lczero_tensor
from stonefish.ttt import TTTBoardRep, TTTMoveRep


def default_collate_fn(batch):
    source, target = zip(*batch, strict=False)

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

        return board_tokens.to_tensor(), move.to_tensor()


class FindKingData(Dataset):
    """
    Test dataset that trains the network to identify the location of the king
    of the side to move. This is testing its ability to understand the meta
    data and locality.
    """

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

        board_tokens = self.input_rep.from_fen(board_fen)
        board = board_tokens.to_board()

        my_king_loc = chess.square_name(board.king(board.turn))
        their_king_loc = chess.square_name(board.king(not board.turn))

        move = self.output_rep.from_str(my_king_loc + their_king_loc)
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


class HuggingFaceChessDataset(Dataset):
    """Chess dataset using HuggingFace datasets with fastchessenv board representation"""

    def __init__(self, split, sample_size=None, dataset_name="mkrum/ParsedChess"):
        # Load a larger dataset for the ResNet model
        if sample_size:
            self.data = datasets.load_dataset(dataset_name)[split].select(
                range(sample_size)
            )
        else:
            # Load the full dataset
            self.data = datasets.load_dataset(dataset_name)[split]

    def __getitem__(self, idx):
        row = self.data[idx]
        array = CBoard.from_fen(row["board"]).to_array()
        move = CMove.from_str(row["move"]).to_int()
        return torch.tensor(array).float(), torch.tensor(move).long()

    def __len__(self):
        return len(self.data)


class LczeroChessDataset(Dataset):
    """Chess dataset that uses the Leela Chess Zero board representation"""

    def __init__(self, split, sample_size=None, dataset_name="mkrum/ParsedChess"):
        # Load dataset
        if sample_size:
            self.data = datasets.load_dataset(dataset_name)[split].select(
                range(sample_size)
            )
        else:
            self.data = datasets.load_dataset(dataset_name)[split]

    def __getitem__(self, idx):
        row = self.data[idx]
        # Convert FEN to chess.Board
        board = chess.Board(row["board"])
        # Use the Lczero converter
        array = board_to_lczero_tensor(board)
        # Move handling remains the same
        move = CMove.from_str(row["move"]).to_int()
        return torch.tensor(array).float(), torch.tensor(move).long()

    def __len__(self):
        return len(self.data)
