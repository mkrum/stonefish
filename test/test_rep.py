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


def listrep_test(lr):
    assert lr.from_str_list(lr.to_str_list()) == lr
    assert lr.from_int_list(lr.to_int_list()) == lr
    assert lr.from_tensor(lr.to_tensor()) == lr
    assert lr.from_numpy(lr.to_numpy()) == lr


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
        listrep_test(board)


def test_move():
    move = MoveRep.from_str("e2d2")
    MoveRep.from_uci(MoveRep.to_uci(move)) == move
    for _ in range(10):
        move = MoveRep.sample()
        listrep_test(move)
