import chess
import time
import wandb
from typing import Any
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from main import NonsharedFlaxT5ForConditionalGenerationModule
from torch.distributions.categorical import Categorical
from stonefish.eval.base import ChessEvalContext
import numpy as np
import jax.numpy as jnp

from stonefish.tokens import BoardTokenizer, MoveTokenizer, BoardMoveSeq2SeqTokenizer
from stonefish.env import RandomAgent, StockfishAgent
from stonefish.mask import MoveMask
from stonefish.rep import MoveRep, CBoard, MoveToken


@dataclass(frozen=True)
class ModelEvalWrapper:
    model: Any
    input_rep: Any = CBoard
    output_rep: Any = MoveRep

    def sample(self, board: chess.Board, max_sel=True):

        board_rep = CBoard.from_board(board)

        state = board_rep.to_array()
        legal_mask = board_rep.get_mask()
        move_mask = MoveMask.from_mask(legal_mask)

        encoder_outputs = self.model.encode(input_ids=state)

        decoder_input_ids = jnp.zeros((state.shape[0], 1), dtype="i4")

        for _ in range(2):
            outputs = self.model.decode(decoder_input_ids, encoder_outputs)
            logits = outputs.logits

            mm = move_mask.get_mask(
                torch.LongTensor(np.array(decoder_input_ids, dtype=np.int32))
            ).numpy()
            mm = jnp.concatenate([mm, jnp.zeros((1, 31998))], axis=-1)

            sel_logits = logits[:, -1, :] * mm + (1 - mm) * -1e8

            if max_sel:
                sel = jnp.argmax(sel_logits)
            else:
                sel_logits = torch.FloatTensor(np.array(sel_logits))
                post = F.log_softmax(sel_logits, dim=1)
                sel = jnp.array(Categorical(logits=post).sample().item())

            sel_arr = jnp.array([[sel]], dtype="i4")
            decoder_input_ids = jnp.concatenate([decoder_input_ids, sel_arr], axis=-1)

        return self.output_rep.from_numpy(np.array(decoder_input_ids[0])).to_uci()


# board_tokenizer = BoardTokenizer()
# move_tokenizer = MoveTokenizer()

# model = NonsharedFlaxT5ForConditionalGenerationModule.from_pretrained("t5chess")
# model = ModelEvalWrapper(model)

wandb.init()

ctx = ChessEvalContext()

ctx(StockfishAgent(1), 0)
time.sleep(60)
ctx(StockfishAgent(2), 1)
time.sleep(60)
ctx(StockfishAgent(3), 2)
