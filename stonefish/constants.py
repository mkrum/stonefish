"""
Defines some useful constant values, including the maping from token to int for
both the boards and actions.
"""

import itertools

_pieces = [
    "r",
    "n",
    "b",
    "q",
    "k",
    "p",
]

# All of the possible tokens in the board space, each piece plus e for empty, w
# for white to move and b for black to move
BOARD_TOKENS = ["e", "w", "b"] + _pieces + [p.upper() for p in _pieces]

_columns = ["a", "b", "c", "d", "e", "f", "g", "h"]
_rows = list(map(str, range(1, 9)))

_promotion_pieces = ["r", "n", "b", "q"]
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

SQUARES = list(map(lambda x: x[0] + x[1], list(itertools.product(_columns, _rows))))

# All of the possible tokens in the move space.
MOVE_TOKENS = SQUARES + _top_promotions + _bottom_promotions

# A dictionary that converts each token to a unique id
BTOKEN_ID = {b: i for (i, b) in enumerate(BOARD_TOKENS)}
MTOKEN_ID = {m: i for (i, m) in enumerate(MOVE_TOKENS)}
