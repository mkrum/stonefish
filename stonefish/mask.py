import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from fastchessenv.rep import legal_mask_convert

from stonefish.rep import MoveRep


@dataclass
class MoveMask:

    n: int
    tensor_map: Dict

    @classmethod
    def from_mask(cls, legal_mask):

        tensor_map = {}
        n = legal_mask.shape[0]
        # Timing information (unused)
        _start = time.time()

        tensor_map = legal_mask_convert(np.int32(legal_mask.cpu().numpy()))
        for k, v in tensor_map.items():
            tensor_map[k] = torch.LongTensor(v).to(legal_mask.device)

        return cls(n, tensor_map)

    def get_mask(self, tokens):
        move_mask = torch.zeros((self.n, MoveRep.width())).to(self.tensor_map[0].device)

        if tokens.shape[1] == 1:
            for i in range(self.n):
                move_mask[i, self.tensor_map[i][:, 1]] = 1.0

            return move_mask

        for i in range(self.n):
            vals = self.tensor_map[i]
            tok = tokens[i]

            valid = vals[tok[-1] == vals[:, tokens.shape[1] - 1]]
            move_mask[i, valid[:, tokens.shape[1]]] = 1.0

        return move_mask
