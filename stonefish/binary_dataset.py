"""Binary chess dataset loader for stonefish training.

Reads (board, move) pairs from compact binary files (36 bytes/record) with
C-accelerated decoding. Supports two board output formats:
- "lczero": (B, 8, 8, 20) float32 tensors via C extension (~2.7M samples/sec)
- "flat": (B, 69) float32 tensors via FlatBoardTokenizer (~772k samples/sec)

Two move encodings are supported via move_format:
- "cmove": Move bytes already contain a CMove index (uint16 big-endian).
           Used by elite.bin and other files produced by load_elite.py.
- "from_to_promo": Move bytes encode from_sq(6b)|to_sq(6b)|promo(4b).
                   Converted to CMove indices via a precomputed lookup table.
"""

import mmap
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

from stonefish.board_format import RECORD_SIZE, decode_batch, decode_lczero_batch

# Lazy-loaded at first use (fastchessenv requires compiled C libs, only available in Docker)
_MOVE_LOOKUP = None


def _ensure_move_lookup():
    """Build the (64, 64, 7) lookup table on first use (~20ms)."""
    global _MOVE_LOOKUP
    if _MOVE_LOOKUP is not None:
        return
    from fastchessenv import CMove

    lookup = torch.full((64, 64, 7), -1, dtype=torch.long)
    for i in range(5700):
        try:
            cmove = CMove.from_int(i)
            move = cmove.to_move()
            promo = move.promotion if move.promotion else 0
            lookup[move.from_square, move.to_square, promo] = i
        except Exception:
            continue
    _MOVE_LOOKUP = lookup


class _StubDataset:
    """Stub to satisfy training loop attribute access on .dataset."""

    streaming = False
    board_tokenizer = None


class BinaryChessDataLoader:
    """Reads binary chess data files for stonefish training.

    Combines mmap-based file reading, C-accelerated decoding, optional
    threaded prefetch, and move index conversion into a single class
    that plugs directly into PreTrainContext's training loop.

    For distributed training, each rank processes every world_size-th batch.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 2048,
        board_format: str = "lczero",
        move_format: str = "cmove",
        num_workers: int = 0,
        prefetch_factor: int = 2,
    ):
        if board_format not in ("lczero", "flat"):
            raise ValueError(
                f"board_format must be 'lczero' or 'flat', got {board_format!r}"
            )
        if move_format not in ("cmove", "from_to_promo"):
            raise ValueError(
                f"move_format must be 'cmove' or 'from_to_promo', got {move_format!r}"
            )

        if move_format == "from_to_promo":
            _ensure_move_lookup()

        self.board_format = board_format
        self.move_format = move_format
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.dataset = _StubDataset()

        self._decoder = (
            decode_lczero_batch if board_format == "lczero" else decode_batch
        )

        self.data_path = data_path
        self.file_size = os.path.getsize(data_path)
        self.num_samples = self.file_size // RECORD_SIZE
        self._specs = self._make_batch_specs()

        # Tokenizers for flat mode
        if board_format == "flat":
            from stonefish.tokenizers import FlatBoardTokenizer, FlatMoveTokenizer

            self._board_tokenizer = FlatBoardTokenizer()
            self._move_tokenizer = FlatMoveTokenizer()

    def _make_batch_specs(self):
        """Pre-compute (byte_offset, record_count) for each batch."""
        specs = []
        remaining = self.num_samples
        offset = 0
        while remaining > 0:
            count = min(self.batch_size, remaining)
            specs.append((offset, count))
            offset += count * RECORD_SIZE
            remaining -= count
        return specs

    def __len__(self) -> int:
        total = len(self._specs)
        if dist.is_available() and dist.is_initialized():
            return total // dist.get_world_size()  # type: ignore[no-any-return]
        return total

    # -- Move conversion --

    def _convert_moves(self, moves: torch.Tensor) -> torch.Tensor:
        """Convert (B, 3) decoded move fields to (B,) CMove indices.

        The C extension always decomposes the 2-byte move field into 3 values.
        For 'cmove' format, the original uint16 is reconstructed via bit ops.
        For 'from_to_promo' format, a precomputed lookup table is used.
        """
        if self.move_format == "cmove":
            # Reconstruct the uint16: (field0 << 10) | (field1 << 4) | field2
            return (moves[:, 0] << 10) | (moves[:, 1] << 4) | moves[:, 2]
        assert _MOVE_LOOKUP is not None
        return _MOVE_LOOKUP[moves[:, 0], moves[:, 1], moves[:, 2]]

    # -- Decoding helpers --

    def _decode_lczero(self, data, count):
        """Decode binary data → (boards tensor, move_indices tensor)."""
        boards_np, moves_np = self._decoder(data, count)
        boards = torch.from_numpy(boards_np)
        moves = torch.from_numpy(moves_np).long()
        move_indices = self._convert_moves(moves)
        return boards, move_indices

    def _decode_flat(self, data, count):
        """Decode binary data → (flat board tensor, move_indices tensor)."""
        batch = self._decoder(data, count)
        boards_list, moves_list = zip(*batch, strict=True)
        board_tensor = self._board_tokenizer.from_board_batch(list(boards_list))
        move_tensor = self._move_tokenizer.from_move_batch(list(moves_list))
        return board_tensor, move_tensor

    # -- Iteration --

    def _iter_sequential(self, specs):
        """Read and decode batches one at a time from an mmap'd file."""
        decode = (
            self._decode_lczero if self.board_format == "lczero" else self._decode_flat
        )
        with open(self.data_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mv = memoryview(mm)
            try:
                for offset, count in specs:
                    end = offset + count * RECORD_SIZE
                    yield decode(mv[offset:end], count)
            finally:
                mv.release()
                mm.close()

    def _iter_threaded(self, specs):
        """Prefetch-decode in a background thread."""
        decode = (
            self._decode_lczero if self.board_format == "lczero" else self._decode_flat
        )
        with open(self.data_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mv = memoryview(mm)
            pool = ThreadPoolExecutor(max_workers=1)
            pending = deque()
            spec_iter = iter(specs)
            max_pending = self.prefetch_factor
            try:

                def _do_decode(offset, count):
                    return decode(mv[offset : offset + count * RECORD_SIZE], count)

                for offset, count in spec_iter:
                    pending.append(pool.submit(_do_decode, offset, count))
                    if len(pending) >= max_pending:
                        break

                while pending:
                    batch = pending.popleft().result()
                    next_spec = next(spec_iter, None)
                    if next_spec is not None:
                        pending.append(pool.submit(_do_decode, *next_spec))
                    yield batch
            finally:
                for fut in pending:
                    fut.cancel()
                pool.shutdown(wait=False)
                mv.release()
                mm.close()

    def __iter__(self):
        # Distributed sharding: each rank gets every world_size-th batch
        specs = self._specs
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            if world_size > 1:
                specs = [s for i, s in enumerate(specs) if i % world_size == rank]

        if self.num_workers > 0:
            yield from self._iter_threaded(specs)
        else:
            yield from self._iter_sequential(specs)
