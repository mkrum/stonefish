"""
Simple pytorch dataset for the chess data
"""

import os

import chess
import datasets
import torch
import torch.distributed as dist
import torch.utils.data
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

        # Load with larger buffer size to reduce chunk boundary delays
        # Set environment variable for larger streaming buffer
        os.environ["HF_DATASETS_STREAMING_BUFFER_SIZE"] = "100"  # MB

        self.data = datasets.load_dataset(
            dataset_name,
            streaming=True,
        )[split]
        self.board_tokenizer = board_tokenizer

    def __iter__(self):
        """Iterator for streaming datasets"""

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            worker_id = 0
            total_workers = 1
        else:
            worker_id = worker_info.id
            total_workers = worker_info.num_workers

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank_id = dist.get_rank()
        else:
            world_size = 1
            rank_id = 0

        total_workers *= world_size
        global_worker_id = worker_id * world_size + rank_id

        # Standard approach - each worker processes its assigned samples
        for i, row in enumerate(self.data):
            if i % total_workers == global_worker_id:
                board = chess.Board(row["board"])
                board_tensor = self.board_tokenizer.from_board(board)
                move = CMove.from_str(row["move"]).to_int()
                yield board_tensor, torch.tensor(move).long()

    def __len__(self):
        return self.n_examples
