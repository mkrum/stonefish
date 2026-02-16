"""Storage format for encoding/decoding (board, move) pairs to fixed-size binary records.

Format (36 bytes per record):
  - Board (32 bytes): 64 squares × 4 bits each. Each nibble encodes a piece:
      0 = empty, 1-6 = white P/N/B/R/Q/K, 7-12 = black P/N/B/R/Q/K
  - Move (2 bytes): from_square (6 bits) | to_square (6 bits) | promotion (4 bits)
  - State (2 bytes):
      byte 0: side_to_move (1 bit) | castling KQkq (4 bits) | unused (3 bits)
      byte 1: en passant square (0-63) or 255 for none
"""

import struct

import chess

RECORD_SIZE = 36
RAW_RECORD_SIZE = 72

try:
    from stonefish._decode_fast import decode_to_bytes as _c_decode_to_bytes
    from stonefish._decode_fast import decode_to_chess as _c_decode_to_chess
    from stonefish._decode_fast import decode_to_lczero as _c_decode_to_lczero
    from stonefish._decode_fast import encode_board as _c_encode_board

    _HAS_C_EXT = True
except ImportError:
    _HAS_C_EXT = False


def _encode_python(board: chess.Board, move: chess.Move) -> bytes:
    # Board: 64 squares, 4 bits each, packed into 32 bytes.
    # Square i maps to nibble i. Nibbles are packed high-then-low per byte.
    board_bytes = bytearray(32)
    for byte_idx in range(32):
        sq_hi = 2 * byte_idx
        sq_lo = 2 * byte_idx + 1

        piece_hi = board.piece_at(sq_hi)
        if piece_hi is None:
            nib_hi = 0
        else:
            nib_hi = piece_hi.piece_type + (0 if piece_hi.color == chess.WHITE else 6)

        piece_lo = board.piece_at(sq_lo)
        if piece_lo is None:
            nib_lo = 0
        else:
            nib_lo = piece_lo.piece_type + (0 if piece_lo.color == chess.WHITE else 6)

        board_bytes[byte_idx] = (nib_hi << 4) | nib_lo

    # Move: from(6) | to(6) | promotion(4) = 16 bits
    promo = move.promotion if move.promotion else 0
    move_int = (move.from_square << 10) | (move.to_square << 4) | promo
    move_bytes = struct.pack(">H", move_int)

    # State byte 0: side_to_move(1) | castling(4) | unused(3)
    state0 = 0
    if board.turn == chess.BLACK:
        state0 |= 0x80
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 8
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 4
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 2
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 1
    state0 |= castling << 3

    # State byte 1: en passant square or 255
    state1 = board.ep_square if board.ep_square is not None else 255

    return bytes(board_bytes) + move_bytes + bytes([state0, state1])


encode = _c_encode_board if _HAS_C_EXT else _encode_python


# Pre-computed castling rights lookup table (4-bit castling field → bitboard).
# Bits: 3=White-K, 2=White-Q, 1=Black-k, 0=Black-q
# Castling rights are stored as rook-square bitboards.
_BB_H1 = 1 << 7  # White kingside rook
_BB_A1 = 1 << 0  # White queenside rook
_BB_H8 = 1 << 63  # Black kingside rook
_BB_A8 = 1 << 56  # Black queenside rook
_CASTLING_BITS = [(8, _BB_H1), (4, _BB_A1), (2, _BB_H8), (1, _BB_A8)]
_CASTLING_MASKS = [
    sum(mask for bit, mask in _CASTLING_BITS if i & bit) for i in range(16)
]

_MOVE_STRUCT = struct.Struct(">H")


def decode(data: bytes) -> tuple[chess.Board, chess.Move]:
    """Decode a single 36-byte record into a Board and Move."""
    board = chess.Board(fen=None)

    # Decode board: build bitboards from 4-bit nibble encoding.
    # Nibble values: 0=empty, 1-6=white PNBRQK, 7-12=black PNBRQK.
    piece_bbs = [0, 0, 0, 0, 0, 0]  # pawns, knights, bishops, rooks, queens, kings
    white = 0
    black = 0

    for byte_idx in range(32):
        b = data[byte_idx]
        sq = 2 * byte_idx

        for nib, s in ((b >> 4, sq), (b & 0x0F, sq + 1)):
            if not nib:
                continue
            bb = 1 << s
            if nib <= 6:
                white |= bb
                piece_bbs[nib - 1] |= bb
            else:
                black |= bb
                piece_bbs[nib - 7] |= bb

    board.pawns, board.knights, board.bishops = piece_bbs[0], piece_bbs[1], piece_bbs[2]
    board.rooks, board.queens, board.kings = piece_bbs[3], piece_bbs[4], piece_bbs[5]
    board.occupied_co[chess.WHITE] = white
    board.occupied_co[chess.BLACK] = black
    board.occupied = white | black

    # Decode move: from(6) | to(6) | promotion(4) = 16 bits
    move_int = _MOVE_STRUCT.unpack_from(data, 32)[0]
    from_sq = (move_int >> 10) & 0x3F
    to_sq = (move_int >> 4) & 0x3F
    promo = move_int & 0x0F
    move = chess.Move(from_sq, to_sq, promotion=promo if promo else None)

    # Decode state
    state0 = data[34]
    board.turn = chess.BLACK if (state0 & 0x80) else chess.WHITE
    board.castling_rights = _CASTLING_MASKS[(state0 >> 3) & 0x0F]
    board.ep_square = data[35] if data[35] != 255 else None

    return board, move


def decode_batch(data, count: int) -> list[tuple[chess.Board, chess.Move]]:
    """Decode count records from data using C extension (with pure-Python fallback)."""
    if _HAS_C_EXT:
        return _c_decode_to_chess(data, count)  # type: ignore[no-any-return]
    return [decode(data[i * RECORD_SIZE : (i + 1) * RECORD_SIZE]) for i in range(count)]


def decode_lczero_batch(data, count: int):
    """Decode count records directly into LCZero numpy tensors.

    Requires the C extension.

    Returns:
        Tuple of (boards, moves) where:
        - boards: float32 array of shape (count, 8, 8, 20)
        - moves: uint8 array of shape (count, 3)
    """
    if _HAS_C_EXT:
        return _c_decode_to_lczero(data, count)
    raise RuntimeError(
        "decode_lczero_batch requires the C extension (_decode_fast). "
        "Build it with: python setup.py build_ext --inplace"
    )


def decode_raw_batch(data, count: int) -> bytes:
    """Decode count records into a flat bytes object (72 bytes/record).

    Requires the C extension; raises RuntimeError if unavailable.
    """
    if not _HAS_C_EXT:
        raise RuntimeError(
            "decode_raw_batch requires the C extension (_decode_fast). "
            "Build it with: python setup.py build_ext --inplace"
        )
    return _c_decode_to_bytes(data, count)  # type: ignore[no-any-return]
