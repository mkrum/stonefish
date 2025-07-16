"""
Unit tests for tokenizer implementations.
"""

import chess
import torch

from stonefish.tokenizers import (
    FlatBoardTokenizer,
    FlatMoveTokenizer,
    LCZeroBoardTokenizer,
)


class TestFlatBoardTokenizer:
    """Test FlatBoardTokenizer functionality"""

    def setup_method(self):
        self.tokenizer = FlatBoardTokenizer()
        self.start_board = chess.Board()
        self.scholar_mate_board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )

    def test_shape_property(self):
        assert self.tokenizer.shape == (69,)

    def test_from_board_shape(self):
        tensor = self.tokenizer.from_board(self.start_board)
        assert tensor.shape == (69,)
        assert tensor.dtype == torch.float32

    def test_round_trip_conversion(self):
        # Start position
        tensor = self.tokenizer.from_board(self.start_board)
        reconstructed = self.tokenizer.to_board(tensor)
        # Compare board positions, not full FEN (clocks may differ)
        assert reconstructed.board_fen() == self.start_board.board_fen()
        assert reconstructed.turn == self.start_board.turn
        assert reconstructed.castling_rights == self.start_board.castling_rights
        assert reconstructed.ep_square == self.start_board.ep_square

        # Scholar's mate position
        tensor = self.tokenizer.from_board(self.scholar_mate_board)
        reconstructed = self.tokenizer.to_board(tensor)
        assert reconstructed.board_fen() == self.scholar_mate_board.board_fen()
        assert reconstructed.turn == self.scholar_mate_board.turn
        assert reconstructed.castling_rights == self.scholar_mate_board.castling_rights
        assert reconstructed.ep_square == self.scholar_mate_board.ep_square

    def test_fen_conversion(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
        board = chess.Board(fen + " 0 1")
        tensor = self.tokenizer.from_board(board)
        reconstructed = self.tokenizer.to_board(tensor)
        # Compare essential parts of FEN (excluding clocks)
        assert reconstructed.board_fen() == board.board_fen()
        assert reconstructed.turn == board.turn
        assert reconstructed.castling_rights == board.castling_rights
        assert reconstructed.ep_square == board.ep_square

    def test_batch_operations(self):
        boards = [self.start_board, self.scholar_mate_board]

        # Test batch conversion
        batch_tensor = self.tokenizer.from_board_batch(boards)
        assert batch_tensor.shape == (2, 69)

        # Test batch reconstruction
        reconstructed_boards = self.tokenizer.to_board_batch(batch_tensor)
        assert len(reconstructed_boards) == 2
        # Compare essential board state, not full FEN
        assert reconstructed_boards[0].board_fen() == boards[0].board_fen()
        assert reconstructed_boards[0].turn == boards[0].turn
        assert reconstructed_boards[1].board_fen() == boards[1].board_fen()
        assert reconstructed_boards[1].turn == boards[1].turn


class TestLCZeroBoardTokenizer:
    """Test LCZeroBoardTokenizer functionality"""

    def setup_method(self):
        self.tokenizer = LCZeroBoardTokenizer()
        self.start_board = chess.Board()
        self.scholar_mate_board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )

    def test_shape_property(self):
        assert self.tokenizer.shape == (8, 8, 20)

    def test_from_board_shape(self):
        tensor = self.tokenizer.from_board(self.start_board)
        assert tensor.shape == (8, 8, 20)
        assert tensor.dtype == torch.float32

    def test_round_trip_conversion(self):
        # Start position
        tensor = self.tokenizer.from_board(self.start_board)
        reconstructed = self.tokenizer.to_board(tensor)
        assert reconstructed.fen() == self.start_board.fen()

        # Scholar's mate position
        tensor = self.tokenizer.from_board(self.scholar_mate_board)
        reconstructed = self.tokenizer.to_board(tensor)
        assert reconstructed.fen() == self.scholar_mate_board.fen()

    def test_fen_conversion(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        tensor = self.tokenizer.from_fen(fen)
        reconstructed_fen = self.tokenizer.to_fen(tensor)
        assert reconstructed_fen == fen

    def test_batch_operations(self):
        boards = [self.start_board, self.scholar_mate_board]

        # Test batch conversion
        batch_tensor = self.tokenizer.from_board_batch(boards)
        assert batch_tensor.shape == (2, 8, 8, 20)

        # Test batch reconstruction
        reconstructed_boards = self.tokenizer.to_board_batch(batch_tensor)
        assert len(reconstructed_boards) == 2
        assert reconstructed_boards[0].fen() == boards[0].fen()
        assert reconstructed_boards[1].fen() == boards[1].fen()


class TestFlatMoveTokenizer:
    """Test FlatMoveTokenizer functionality"""

    def setup_method(self):
        self.tokenizer = FlatMoveTokenizer()
        self.moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("b8c6"),
        ]

    def test_shape_property(self):
        assert self.tokenizer.shape == ()

    def test_vocab_size_property(self):
        assert self.tokenizer.vocab_size == 5700

    def test_from_move_shape(self):
        tensor = self.tokenizer.from_move(self.moves[0])
        assert tensor.shape == ()
        assert tensor.dtype == torch.long

    def test_round_trip_conversion(self):
        for move in self.moves:
            tensor = self.tokenizer.from_move(move)
            reconstructed = self.tokenizer.to_move(tensor)
            assert reconstructed == move

    def test_uci_conversion(self):
        uci = "e2e4"
        tensor = self.tokenizer.from_uci(uci)
        reconstructed_uci = self.tokenizer.to_uci(tensor)
        assert reconstructed_uci == uci

    def test_batch_operations(self):
        # Test batch conversion
        batch_tensor = self.tokenizer.from_move_batch(self.moves)
        assert batch_tensor.shape == (4,)

        # Test batch reconstruction
        reconstructed_moves = self.tokenizer.to_move_batch(batch_tensor)
        assert len(reconstructed_moves) == 4
        for orig, recon in zip(self.moves, reconstructed_moves, strict=False):
            assert orig == recon

    def test_move_indices_in_range(self):
        """Test that move indices are within expected range"""
        for move in self.moves:
            tensor = self.tokenizer.from_move(move)
            move_idx = tensor.item()
            assert 0 <= move_idx < 5700


class TestTokenizerCompatibility:
    """Test compatibility between tokenizers and model expectations"""

    def setup_method(self):
        self.flat_board = FlatBoardTokenizer()
        self.lczero_board = LCZeroBoardTokenizer()
        self.move_tokenizer = FlatMoveTokenizer()
        self.board = chess.Board()
        self.move = chess.Move.from_uci("e2e4")

    def test_tensor_types(self):
        """Test that tokenizers produce correct tensor types"""
        flat_tensor = self.flat_board.from_board(self.board)
        lczero_tensor = self.lczero_board.from_board(self.board)
        move_tensor = self.move_tokenizer.from_move(self.move)

        assert flat_tensor.dtype == torch.float32
        assert lczero_tensor.dtype == torch.float32
        assert move_tensor.dtype == torch.long

    def test_batch_consistency(self):
        """Test that batch operations are consistent with single operations"""
        boards = [self.board] * 3
        moves = [self.move] * 3

        # Flat board tokenizer
        single_tensors = [self.flat_board.from_board(b) for b in boards]
        batch_tensor = self.flat_board.from_board_batch(boards)
        for i, single in enumerate(single_tensors):
            assert torch.allclose(single, batch_tensor[i])

        # LCZero board tokenizer
        single_tensors = [self.lczero_board.from_board(b) for b in boards]
        batch_tensor = self.lczero_board.from_board_batch(boards)
        for i, single in enumerate(single_tensors):
            assert torch.allclose(single, batch_tensor[i])

        # Move tokenizer
        single_tensors = [self.move_tokenizer.from_move(m) for m in moves]
        batch_tensor = self.move_tokenizer.from_move_batch(moves)
        for i, single in enumerate(single_tensors):
            assert torch.equal(single, batch_tensor[i])

    def test_device_agnostic(self):
        """Test that tokenizers work with different devices"""
        tensor = self.flat_board.from_board(self.board)

        # Test CPU
        cpu_board = self.flat_board.to_board(tensor)
        assert cpu_board.fen() == self.board.fen()

        # Test GPU if available
        if torch.cuda.is_available():
            gpu_tensor = tensor.cuda()
            gpu_board = self.flat_board.to_board(gpu_tensor)
            assert gpu_board.fen() == self.board.fen()
