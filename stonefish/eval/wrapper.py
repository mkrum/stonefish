from dataclasses import dataclass
from typing import Any

import chess
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as functional
from fastchessenv import CMove
from torch.distributions.categorical import Categorical

from stonefish.mask import MoveMask
from stonefish.rep import CBoard, MoveRep


@dataclass(frozen=True)
class ModelEvalWrapper:
    model: Any
    params: Any
    max_sel: bool = True
    input_rep: Any = CBoard
    output_rep: Any = MoveRep

    def __call__(self, board: chess.Board):
        print("call")

        if board.turn == chess.BLACK:
            board = board.mirror()

        board_rep = CBoard.from_board(board)

        state = jnp.array(board_rep.to_array())
        state = jnp.expand_dims(state, 0)

        legal_mask = torch.zeros(1, 64 * 88)

        legal_moves = list(board.legal_moves)
        for m in legal_moves:
            mint = CMove.from_move(m).to_int()
            legal_mask[0, mint] = 1.0

        move_mask = MoveMask.from_mask(legal_mask)

        encoder_outputs = self.model.encode(input_ids=state, params=self.params)

        decoder_input_ids = jnp.zeros((state.shape[0], 1), dtype="i4")

        for _ in range(2):
            outputs = self.model.decode(
                decoder_input_ids, encoder_outputs, params=self.params
            )
            logits = outputs.logits

            mm = move_mask.get_mask(
                torch.LongTensor(np.array(decoder_input_ids, dtype=np.int32))
            ).numpy()

            sel_logits = logits[:, -1, :] * mm + (1 - mm) * -1e8

            if self.max_sel:
                sel = jnp.argmax(sel_logits)
            else:
                sel_logits = torch.FloatTensor(np.array(sel_logits))
                post = functional.log_softmax(sel_logits, dim=1)
                sel = jnp.array(Categorical(logits=post).sample().item())

            sel_arr = jnp.array([[sel]], dtype="i4")
            decoder_input_ids = jnp.concatenate([decoder_input_ids, sel_arr], axis=-1)

        return self.output_rep.from_numpy(np.array(decoder_input_ids[0])).to_uci()

    @property
    def name(self):
        return "ModelWrapper"
