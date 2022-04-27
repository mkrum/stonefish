from dataclasses import dataclass
from typing import Dict, Any
import torch

from chessenv.rep import CMove, CBoard
from stonefish.rep import MoveRep, MoveEnum


@dataclass
class MoveMask:

    move_map: Dict
    tensor_map: Dict
    masks: Any

    @classmethod
    def from_env(cls, env):
        moves = env.get_possible_moves()

        tensor_map = {i: [] for i in range(len(moves))}
        move_map = {i: [] for i in range(len(moves))}

        for (i, move_stack) in enumerate(moves):
            move_map[i] = move_stack.to_str()
            for m in move_map[i]:
                move_rep = MoveRep.from_str(m)
                tensor_map[i].append(move_rep.to_tensor())
            tensor_map[i] = torch.stack(tensor_map[i])
        return cls(move_map, tensor_map, None)

    @classmethod
    def from_data(cls, data, output):

        boards = []
        for d in data:
            boards.append(CBoard.from_arr(d))

        tensor_map = {i: [] for i in range(len(boards))}
        move_map = {i: [] for i in range(len(boards))}

        for (i, b) in enumerate(boards):
            moves = list(map(str, b.board.legal_moves))
            move_map[i] = moves
            tensor_map[i] = [MoveRep.from_str(m).to_tensor() for m in moves] + [
                output[i]
            ]

        for i in range(len(tensor_map)):
            tensor_map[i] = torch.stack(tensor_map[i])

        return cls(move_map, tensor_map, None)

    def is_valid(self, move, i):
        return move in self.move_map[i]

    def mask(self, logits, tokens):

        batch_size = logits.shape[0]

        masks = torch.zeros((batch_size, MoveRep.width())).to(logits.device)

        tensor_map = self.tensor_map
        for i in range(batch_size):

            mask_mask = (
                torch.sum(tensor_map[i][:, : tokens.shape[1]].cuda() != tokens[i], -1)
                > 0
            )
            tensor_map[i][mask_mask] = -100

            mask_idx = tokens.shape[1]
            for v in self.tensor_map[i]:
                if v[mask_idx].item() != -100:
                    masks[i, v[mask_idx].item()] = 1.0

        logits = logits * masks + -1e8 * (1 - masks)

        masks = masks.unsqueeze(1)
        if self.masks is not None:
            masks = torch.stack((self.masks, masks), dim=1)

        return logits, MoveMask(self.move_map, self.tensor_map, masks)

    def update_mask(self, tokens):

        batch_size = tokens.shape[0]

        masks = torch.zeros((batch_size, tokens.shape[1], MoveRep.width())).to(
            tokens.device
        )

        tensor_map = self.tensor_map
        for mask_idx in range(tokens.shape[1]):
            for i in range(batch_size):

                tensor_map[i] = tensor_map[i].to(tokens.device)

                mask_mask = (
                    torch.sum(tensor_map[i][:, :mask_idx] != tokens[i, :mask_idx], -1)
                    > 0
                )
                tensor_map[i][mask_mask] = -100

                for v in self.tensor_map[i]:
                    if v[mask_idx].item() != -100:
                        masks[i, mask_idx, v[mask_idx].item()] = 1.0
        return masks
