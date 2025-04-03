from itertools import product
from typing import Dict

from stonefish.rep import EnumRep, TupleEnum


class TTTBoardToken(EnumRep):
    _values = ["<start>", "x", "o", "."]
    _str_to_int: Dict[str, int] = {b: i for (i, b) in enumerate(_values)}
    _int_to_str: Dict[int, str] = {i: b for (i, b) in enumerate(_values)}


class TTTMoveToken(EnumRep):
    _values = ["<start>", "r1", "r2", "r3", "c1", "c2", "c3"]
    _str_to_int: Dict[str, int] = {b: i for (i, b) in enumerate(_values)}
    _int_to_str: Dict[int, str] = {i: b for (i, b) in enumerate(_values)}


class TTTMoveRep(TupleEnum):

    length = 3
    token_type = TTTMoveToken

    _rows = ["r1", "r2", "r3"]
    _cols = ["c1", "c2", "c3"]
    _total = list(product(_rows, _cols))

    @classmethod
    def from_int(cls, int_val):
        str_list = cls._total[int_val]
        return cls.from_str_list(["<start>"] + list(str_list))

    def to_str(self):
        return str(self)

    def __str__(self):
        return self._values[0].to_str() + self._values[1].to_str()

    @classmethod
    def from_str(cls, str_value: str):
        from_str = cls.token_type.from_str(str_value[:2])
        to_str = cls.token_type.from_str(str_value[2:])
        start_token = cls.token_type.from_str("<start>")
        return cls([start_token, from_str, to_str])


class TTTBoardRep(TupleEnum):
    length = 10
    token_type = TTTBoardToken

    def to_str(self):
        print(self.to_str_list())

    @classmethod
    def from_str(cls, str_value: str):
        return cls.from_str_list(["<start>"] + list(str_value))
