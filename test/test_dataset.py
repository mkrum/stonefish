import chess
import torch
from torch.utils.data import DataLoader

from stonefish.convert import board_to_lczero_tensor, lczero_tensor_to_board
from stonefish.dataset import ChessData, default_collate_fn, single_default_collate_fn
from stonefish.rep import BoardRep, MoveRep


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


def test_board_to_lczero_conversion():
    """Test Lczero tensor conversion functions"""
    # Starting position
    board = chess.Board()
    tensor = board_to_lczero_tensor(board)

    # Check tensor shape
    assert tensor.shape == (8, 8, 20)

    # Check that pieces are placed correctly
    # White pawns should be on rank 2 (tensor row 6)
    assert tensor[6, :, 0].sum() == 8  # 8 white pawns

    # Black pawns should be on rank 7 (tensor row 1)
    assert tensor[1, :, 6].sum() == 8  # 8 black pawns

    # Test round-trip conversion
    reconstructed_board = lczero_tensor_to_board(tensor)
    assert reconstructed_board.fen() == board.fen()


def test_collate_functions():
    """Test collate functions for batching"""
    # Test default_collate_fn
    batch_data = [
        (torch.tensor([1, 2, 3]), torch.tensor([4, 5])),
        (torch.tensor([6, 7]), torch.tensor([8, 9, 10])),
    ]

    source, target = default_collate_fn(batch_data)

    # Should pad sequences
    assert source.shape[0] == 2  # Batch size
    assert target.shape[0] == 2  # Batch size

    # Test single_default_collate_fn
    single_batch = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
    ]

    result = single_default_collate_fn(single_batch)
    assert result.shape[0] == 2  # Batch size


def test_tensor_conversion_edge_cases():
    """Test edge cases in tensor conversion"""
    # Test en passant position
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
    tensor = board_to_lczero_tensor(board)
    reconstructed = lczero_tensor_to_board(tensor)

    assert board.ep_square == reconstructed.ep_square


def test_different_dataset_compatibility():
    """Test that different dataset types work with same models"""
    # Create mock data that all datasets should be able to handle
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Test that Lczero conversion works
    board = chess.Board(test_fen)
    lczero_tensor = board_to_lczero_tensor(board)

    # Should be compatible with ChessConvNet input expectations
    assert lczero_tensor.shape == (8, 8, 20)

    # Test tensor can be converted to torch and reshaped as needed
    torch_tensor = torch.tensor(lczero_tensor)

    # Test batch dimension handling
    batched = torch_tensor.unsqueeze(0)  # Add batch dimension
    assert batched.shape == (1, 8, 8, 20)

    # Test permutation for conv layers
    permuted = batched.permute(0, 3, 1, 2)  # BHWC -> BCHW
    assert permuted.shape == (1, 20, 8, 8)
