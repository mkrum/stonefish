import itertools
from typing import List, Dict
from dataclasses import dataclass, field

import chess
import torch
import numpy as np


class EnumRep:
    """
    Simple enumeration based representation.

    Manages a representation that maps a list of strings to unique integer
    values.

    >>> class MyEnumRep(EnumRep): _str_to_int = {"a": 0}; _int_to_str = {0: "a"}
    >>> x = MyEnumRep.from_str('a')
    >>> x.to_str()
    "a"
    >>> x.to_int()
    0
    >>> MyEnumRep.from_int(x.to_int())
    "a"
    """

    _str_to_int: Dict[str, int] = None
    _int_to_str: Dict[str, int] = None

    def __init__(self, str_value: str):
        """
        Initializer, use from_str or from_int instead.
        """
        assert isinstance(str_value, str)
        self._str_rep = str_value

    def __eq__(self, other):
        return self._str_rep == other._str_rep

    @classmethod
    def valid_str(cls) -> List[str]:
        """List of valid strings"""
        return list(cls._str_to_int.keys())

    @classmethod
    def valid_int(cls) -> List[int]:
        """List of valid integers"""
        return list(cls._str_to_int.values())

    @classmethod
    def sample(cls):
        """Randomly sample from the valid values"""
        str_value = np.random.choice(cls.valid_str())
        return cls(str_value)

    @classmethod
    def from_str(cls, str_value: str):
        """Create representation from a string value"""

        if str_value not in cls._str_to_int.keys():
            raise ValueError(f"{str_value} is not a valid piece string.")

        return cls(str_value)

    def to_str(self) -> str:
        """Convert representation into a string"""
        return self._str_rep

    def __str__(self) -> str:
        return self.to_str()

    @classmethod
    def from_int(cls, int_value: int):
        """Create representation from an integer value"""

        if int_value not in cls._int_to_str.keys():
            raise ValueError(f"{int_value} is not a valid piece int.")

        piece_str = cls._int_to_str[int_value]
        return cls(piece_str)

    def to_int(self) -> int:
        """Convert representation into a integer"""
        return self._str_to_int[self._str_rep]

    def __int__(self) -> int:
        return self.to_int()

    @classmethod
    def size(cls):
        return len(cls._int_to_str.keys())


class ListEnum:
    """
    Representation for a list of EnumReps

    >>> class MyEnum(EnumRep):
        _str_to_int = {"a": 0, "b": 1}
        _int_to_str = {0: "a", 1: "b"}
    >>> class MyListEnum(ListEnum):
        token_type = MyEnum
    >>> x = MyListEnum.from_str_list(["a", "b"])
    >>> x.to_int_list()
    [0, 1]
    >>> x.to_str_list()
    ["a", "b"]
    >>> x.to_tensor()
    tensor([0, 1])
    >>> x.from_tensor(x.to_tensor()).to_str_list()
    ["a", "b"]
    """

    token_type: EnumRep = None

    def __init__(self, list_of_values: List[EnumRep]):
        self._values = list_of_values

    def __eq__(self, other):
        return self._values == other._values

    def __len__(self):
        return len(self._values)

    @classmethod
    def sample(cls, n):
        return cls([cls.token_type.sample() for _ in range(n)])

    @classmethod
    def from_str_list(cls, list_vals: List[str]):
        return cls(list(map(lambda x: cls.token_type.from_str(x), list_vals)))

    def to_str_list(self) -> List[str]:
        return list(map(lambda x: x.to_str(), self._values))

    @classmethod
    def from_int_list(cls, list_vals):
        return cls(list(map(lambda x: cls.token_type.from_int(x), list_vals)))

    def to_int_list(self) -> List:
        return list(map(lambda x: x.to_int(), self._values))

    @classmethod
    def from_numpy(cls, arr):
        return cls(list(map(lambda x: cls.token_type.from_int(x), arr)))

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_int_list())

    @classmethod
    def from_tensor(cls, tensor_vals):
        list_vals = list(map(lambda x: x.item(), tensor_vals))
        return cls.from_int_list(list_vals)

    def to_tensor(self) -> torch.LongTensor:
        return torch.LongTensor(self.to_int_list())


class TupleEnum(ListEnum):
    """
    A tuple of EnumRep

    Implements the same behavior of a ListEnum, but requires the list to always
    be a specific size. Only supports single type tuples.
    """

    length = None
    token_type = None

    def __init__(self, list_of_values: List[EnumRep]):
        assert len(list_of_values) == self.length
        super().__init__(list_of_values)

    @classmethod
    def sample(cls):
        return cls([cls.token_type.sample() for _ in range(cls.length)])


class BoardToken(EnumRep):
    """
    Set of tokens that represent the board state

    This class is a representation for each of the individual tokens that will
    be used in the board state. It includes:
        1) Every piece as its standard symbol
        2) A symbol ('e') representing an empty square
        3) A symbol ('en') representing a valid en-passant location
        4) A symbol ('w' or 'b') representing the side to move
        5) The standard FEN symbols for castling rights ('K', 'Q', 'k', 'q')
        6) A set of symbols representing the lack of castling rights ('K!',
        'Q!', 'k!', 'q!')
        7) The numbers 0-9 for the halfmove and fullmove clocks

    These tokens allow us to make a fully invertible state representation.
    """

    _pieces = [
        "r",
        "n",
        "b",
        "q",
        "k",
        "p",
    ]
    _total_pieces = _pieces + [p.upper() for p in _pieces]
    _info_tokens = [
        "e",  # Empty square
        "w",  # White to move
        "b",  # Black to move
        "en",  # Represents a valid en-passant location
        "K",  # White can castle kingside
        "K!",  # White cannnot castle kingside
        "Q",  # White can castle queenside
        "Q!",  # White cannnot castle queenside
        "k",  # Black can castle kingside
        "k!",  # Black cannnot castle kingside
        "q",  # Black can castle queenside
        "q!",  # Black cannnot castle queenside
    ]

    _numbers = list(map(str, range(10)))  # numbers 0-9 for the move clocks

    _tokens = _numbers + _info_tokens + _pieces + [p.upper() for p in _pieces]

    _str_to_int: Dict[str, int] = {b: i for (i, b) in enumerate(_tokens)}
    _int_to_str: Dict[str, int] = {i: b for (i, b) in enumerate(_tokens)}


class MoveToken(EnumRep):
    """
    Set of tokens that represent UCI moves

    These tokens include:
        1) Every square, ('e2', d2')
        2) Every possible promotion move, ('h1q', a8r')
    """

    # Start with all the possible rows and columns on a chess board
    _columns = ["a", "b", "c", "d", "e", "f", "g", "h"]
    _rows = list(map(str, range(1, 9)))
    # Convert these into a list of all of the squares
    _squares = list(
        map(lambda x: x[0] + x[1], list(itertools.product(_columns, _rows)))
    )

    # Get all of the possible promotion pieces
    _promotion_pieces = ["r", "n", "b", "q"]
    # Get all of the possible promotion moves
    _top_promotions = list(
        map(
            lambda x: "".join(x),
            list(itertools.product(_columns, ["1"], _promotion_pieces)),
        )
    )
    _bottom_promotions = list(
        map(
            lambda x: "".join(x),
            list(itertools.product(_columns, ["8"], _promotion_pieces)),
        )
    )

    # All of the possible tokens in the move space.
    _tokens = _squares + _top_promotions + _bottom_promotions

    _str_to_int: Dict[str, int] = {m: i for (i, m) in enumerate(_tokens)}
    _int_to_str: Dict[str, int] = {i: m for (i, m) in enumerate(_tokens)}


class MoveRep(TupleEnum):
    """
    A representation for UCI moves

    A TupleEnum for our MoveTokens of size two. This can represent every
    possible UCI move. Allows for converting from UCI representation to a tensor
    and back.

    >>> move = MoveRep.from_str_list(['e2', 'e4'])
    >>> move.to_str_list()
    ['e2', 'e4']
    >>> move.to_int_list()
    [33, 35]
    >>> move.to_tensor()
    tensor([33, 35])
    >>> move.to_uci()
    Move.from_uci('e2e4')
    """

    length = 2
    token_type = MoveToken

    def __str__(self):
        return self._values[0].to_str() + self._values[1].to_str()

    @classmethod
    def from_str(cls, str_value: str):
        # I think this is valid? Might be missing some weird edge case
        from_str = cls.token_type.from_str(str_value[:2])
        to_str = cls.token_type.from_str(str_value[2:])
        return cls([from_str, to_str])

    @classmethod
    def from_uci(cls, uci_value):
        str_value = str(uci_value)
        return cls.from_str(str_value)

    def to_uci(self):
        """
        This class does not guarantee that the

        Invalid moves get the '0000' value to represent "no move"
        """
        str_value = str(self)
        try:
            move = chess.Move.from_uci(str_value)
        except ValueError:
            move = chess.Move.from_uci("0000")

        return move


class BoardRep(TupleEnum):
    """
    Representation of the board state

    Consists of a tuple of our BoardTokens, captures all of the information in a
    chess board. Allows for a conversion to and from python-chess boards, and FEN
    notation. This representation is positional, meaning each index has a unique
    interpretation:
        1) The first 0-63 items in the tuple represent all of the squares on
    the board
        2) 64 is the team to move
        3) 65,66,67,68,69 are the castling rights
        4) 70,71 is the half move clock, represented as individual digits (i.e.
        60 -> '6', '0', 1 -> '0', '1')
        5) 72,73,74 is the full move number, represented as individual digits (i.e.
        123 -> '1', '2', '3')
    >>> board = ut.randomize_board()
    >>> print(board)
    r . b q . b n r
    p . p . . p p .
    . . . k p . N p
    . Q . p . . . .
    . n P . . . . .
    . . . . . . . .
    P P . P P P P P
    R N B . K B . R
    >>> b = BoardRep.from_board(board)
    >>> print(b.to_board())
    r . b q . b n r
    p . p . . p p .
    . . . k p . N p
    . Q . p . . . .
    . n P . . . . .
    . . . . . . . .
    P P . P P P P P
    R N B . K B . R
    >>> b.to_fen()
    'r1bq1bnr/p1p2pp1/3kp1Np/1Q1p4/1nP5/8/PP1PPPPP/RNB1KB1R w KQ - 3 9'
    >>> b.to_str_list()
    ['r', 'e', 'b', 'q', 'e', 'b', 'n', 'r', 'p', 'e', 'p', 'e', 'e', 'p', 'p',
     'e', 'e', 'e', 'e', 'k', 'p', 'e', 'N', 'p', 'e', 'Q', 'e', 'p', 'e', 'e',
     'e', 'e', 'e', 'n', 'P', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e',
     'e', 'e', 'e', 'P', 'P', 'e', 'P', 'P', 'P', 'P', 'P', 'R', 'N', 'B', 'e',
     'K', 'B', 'e', 'R', 'w', 'K', 'Q', 'k!', 'q!', '0', '3', '0', '0', '9']
    >>> b.to_tensor()
    tensor([22, 10, 24, 25, 10, 24, 23, 22, 27, 10, 27, 10, 10, 27, 27, 10, 10,
        10, 10, 26, 27, 10, 29, 27, 10, 31, 10, 27, 10, 10, 10, 10, 10, 23, 33,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 33, 33, 10, 33, 33,
        33, 33, 33, 28, 29, 30, 10, 32, 30, 10, 28, 11, 32, 31, 19, 21,  0,  3,
         0,  0,  9])
    >>> BoardRep.from_tensor(b.to_tensor()).to_fen()
    'r1bq1bnr/p1p2pp1/3kp1Np/1Q1p4/1nP5/8/PP1PPPPP/RNB1KB1R w KQ - 3 9'
    """

    length = 74
    token_type = BoardToken

    @classmethod
    def from_board(cls, board):
        """Initializes from a chess.Board object"""
        builder = []

        # Get all the pieces/en passant
        for square in chess.SQUARES_180:
            piece = board.piece_at(square)

            if piece:
                builder.append(piece.symbol())
            elif square == board.ep_square and board.has_legal_en_passant():
                builder.append("en")
            else:
                builder.append("e")

        # Get team to move
        if board.turn == chess.WHITE:
            builder.append("w")
        else:
            builder.append("b")

        # Setup castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            builder.append("K")
        else:
            builder.append("K!")

        if board.has_queenside_castling_rights(chess.WHITE):
            builder.append("Q")
        else:
            builder.append("Q!")

        if board.has_kingside_castling_rights(chess.BLACK):
            builder.append("k")
        else:
            builder.append("k!")

        if board.has_queenside_castling_rights(chess.BLACK):
            builder.append("q")
        else:
            builder.append("q!")

        # Move clocks
        halfclock_str = f"{board.halfmove_clock:02}"
        fullmove_str = f"{board.fullmove_number:03}"

        builder += list(halfclock_str)
        builder += list(fullmove_str)

        # Make sure it is a fixed size
        assert len(builder) == cls.length
        return cls(list(map(cls.token_type.from_str, builder)))

    def to_board(self) -> chess.Board:
        """Converts a BoardRep to a chess.Board object"""

        # Initialize new board and remove the pieces
        board = chess.Board()
        board._clear_board()

        str_values = self.to_str_list()

        # Place the pieces, load en-passant
        piece_squares = str_values[:64]
        for (i, square) in enumerate(chess.SQUARES_180):

            piece_symbol = piece_squares[i]

            if piece_symbol == "e":
                continue
            elif piece_symbol == "en":
                board.ep_square = square
            else:
                piece = chess.Piece.from_symbol(piece_symbol)
                board._set_piece_at(square, piece.piece_type, piece.color)

        # Get the current turn
        to_move = str_values[64]
        if to_move == "w":
            board.turn = chess.WHITE
        else:
            board.turn = chess.BLACK

        # Get the castling rights
        castling_fen = ""
        for flag in str_values[65:69]:
            if "!" not in flag:
                castling_fen += flag

        board.set_castling_fen(castling_fen)

        # Get the clocks
        board.halfmove_clock = int("".join(str_values[69:71]))
        board.fullmove_number = int("".join(str_values[71:]))
        return board

    @classmethod
    def from_fen(cls, fen_str):
        """Loads from FEN"""
        as_board = chess.Board()
        as_board.set_fen(fen_str)
        ep_square = fen_str.split()[3]

        # En-passant loading seems to behave unexpectedly here, had to add this
        # to make sure it was properly set when loading from FEN. Not sure why.
        ep = None
        if ep_square != "-":
            ep = chess.SQUARE_NAMES.index(ep_square)
        as_board.ep_square = ep

        return cls.from_board(as_board)

    def to_fen(self) -> str:
        """Converts to FEN"""
        as_board = self.to_board()
        return as_board.fen()

    def __str__(self):
        return self.to_fen()
