import chess
import stonefish.utils as ut
from stonefish.rep import BoardToken, MoveToken, BoardRep, MoveRep


def dictrep_type_test(cls):
    for p in cls.valid_str():
        x = cls.from_str(p)

        assert x.from_str(x.to_str()) == x
        assert x.from_int(x.to_int()) == x
        assert x.from_int(x.to_int()) == x.from_str(x.to_str())

    for p in cls.valid_int():
        x = cls.from_int(p)

        assert x.from_str(x.to_str()) == x
        assert x.from_int(x.to_int()) == x
        assert x.from_int(x.to_int()) == x.from_str(x.to_str())

    x.sample()


def test_board_token():
    dictrep_type_test(BoardToken)

    assert BoardToken("p") != BoardToken("P")
    assert BoardToken("e") == BoardToken("e")

    BoardToken.sample()


def test_move_token():
    dictrep_type_test(MoveToken)


def test_board():

    for _ in range(100):
        real_board = ut.randomize_board(range(0, 50))

        board = BoardRep.from_board(real_board)
        other_board = chess.Board()
        other_board.set_fen(real_board.fen())

        assert board.to_fen() == real_board.fen()
        assert board == BoardRep.from_fen(real_board.fen())
        assert board.from_str_list(board.to_str_list()) == board
        assert board.from_int_list(board.to_int_list()) == board
        assert board.from_tensor(board.to_tensor()) == board


def test_move():
    for _ in range(10):
        move = MoveRep.sample()
        assert move.from_str_list(move.to_str_list()) == move
        assert move.from_int_list(move.to_int_list()) == move
        assert move.from_tensor(move.to_tensor()) == move
