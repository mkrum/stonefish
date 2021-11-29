import chess
import stonefish.utils as ut
from stonefish.rep import BoardToken, MoveToken, BoardRep, MoveRep, create_tokenizer_rep
from stonefish.rep import *
import transformers


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


def test_language():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    BertBaseCased = create_tokenizer_rep("BertBaseCased", tokenizer)

    # Test with actual text
    test_str = "Outlined against a blue, gray October sky the Four Horsemen rode again."
    test = BertBaseCased.from_str(test_str)

    assert test.to_str() == test_str
    assert test.to_str(skip_special_tokens=False) != test_str
    listrep_test(test)

    # Test with gibberish
    gibberish_str = "adlkfjaldf. lakdsjflajfd"
    gibberish = BertBaseCased.from_str(gibberish_str)
    listrep_test(gibberish)
    assert gibberish.to_str() == gibberish_str
    assert gibberish.to_str(skip_special_tokens=False) != gibberish_str
