"""
Simple pytorch dataset for the chess data
"""

import chess
import datasets
import torch
from fastchessenv import CMove
from huggingface_hub import HfApi
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset

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


class ChessDataset:
    """Unified chess dataset that can work in streaming or regular mode"""

    def __new__(cls, *args, streaming=False, **kwargs):
        if streaming:
            return StreamingChessDataset(*args, streaming=streaming, **kwargs)
        else:
            return StandardChessDataset(*args, streaming=streaming, **kwargs)


class StandardChessDataset(Dataset):
    """Standard chess dataset for non-streaming data"""

    def __init__(
        self,
        split,
        board_tokenizer=None,
        sample_size=None,
        dataset_name="mkrum/ParsedChess",
        streaming=False,
    ):
        self.streaming = False
        if sample_size:
            self.data = datasets.load_dataset(dataset_name)[split].select(
                range(sample_size)
            )
        else:
            self.data = datasets.load_dataset(dataset_name, streaming=False)[split]
        self.board_tokenizer = board_tokenizer

    def __getitem__(self, idx):
        row = self.data[idx]
        board = chess.Board(row["board"])
        board_tensor = self.board_tokenizer.from_board(board)
        move = CMove.from_str(row["move"]).to_int()
        return board_tensor, torch.tensor(move).long()

    def __len__(self):
        return len(self.data)


class StreamingChessDataset(IterableDataset):
    """Streaming chess dataset for large-scale data"""

    def __init__(
        self,
        split,
        board_tokenizer=None,
        sample_size=None,
        dataset_name="mkrum/ParsedChess",
        streaming=True,
    ):
        self.streaming = True
        if sample_size:
            raise ValueError("Can't use sample_size with streaming dataset")

        # Get dataset info for length
        api = HfApi()
        info = api.dataset_info(dataset_name)
        # This is basically hardcoded for mkrum/LichessParsedBlitz
        self.n_examples = info.cardData["dataset_info"]["splits"][0]["num_examples"]
        self.data = datasets.load_dataset(dataset_name, streaming=True)[split]
        self.board_tokenizer = board_tokenizer

    def __iter__(self):
        """Iterator for streaming datasets"""
        for row in self.data:
            board = chess.Board(row["board"])
            board_tensor = self.board_tokenizer.from_board(board)
            move = CMove.from_str(row["move"]).to_int()
            yield board_tensor, torch.tensor(move).long()

    def __len__(self):
        return self.n_examples
