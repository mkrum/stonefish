import torch
from torch.utils.data import DataLoader
from stonefish.dataset import ChessData, TTTData
from stonefish.rep import BoardRep, MoveRep
from stonefish.ttt import TTTBoardRep, TTTMoveRep


def test_dataset():
    dataset = ChessData("test/sample.csv", BoardRep, MoveRep)
    board_tensor, move_tensor = dataset[0]

    # Check the type
    assert isinstance(board_tensor, torch.LongTensor)
    assert isinstance(move_tensor, torch.LongTensor)

    # Check the shape
    assert board_tensor.shape == torch.Size([75])
    assert move_tensor.shape == torch.Size([3])


def test_in_dataloader():
    dataset = ChessData("test/sample.csv", BoardRep, MoveRep)
    dataloader = DataLoader(dataset, batch_size=8)
    board_tensor, move_tensor = next(iter(dataloader))

    # Check the type
    assert isinstance(board_tensor, torch.LongTensor)
    assert isinstance(move_tensor, torch.LongTensor)

    # Check the shape
    assert board_tensor.shape == torch.Size([8, 75])
    assert move_tensor.shape == torch.Size([8, 3])


def test_dataset():
    dataset = TTTData("data/ttt_test.csv", TTTBoardRep, TTTMoveRep)
    board_tensor, move_tensor = dataset[0]

    # Check the type
    assert isinstance(board_tensor, torch.LongTensor)
    assert isinstance(move_tensor, torch.LongTensor)

    # Check the shape
    assert board_tensor.shape == torch.Size([10])
    assert move_tensor.shape == torch.Size([3])
