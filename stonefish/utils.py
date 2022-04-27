from dataclasses import dataclass

import chess
import torch
import numpy as np


def randomize_board(steps=range(10, 20)):
    steps = list(steps)
    board = chess.Board()
    for _ in range(np.random.choice(steps)):
        moves = list(board.legal_moves)
        board.push(np.random.choice(moves))

        # If we hit the end, just try again
        if board.is_game_over():
            return randomize_board()

    return board


@dataclass(frozen=True)
class RolloutTensor:

    state: torch.FloatTensor
    action: torch.IntTensor
    next_state: torch.FloatTensor
    reward: torch.FloatTensor
    done: torch.BoolTensor
    mask: torch.FloatTensor

    @classmethod
    def empty(cls):
        return cls(None, None, None, None, None, None)

    def __len__(self):
        if self.state is None:
            return 0
        return self.state.shape[1]

    def is_empty(self):
        return self.state == None

    def add(self, state, action, next_state, reward, done, mask) -> "RolloutTensor":

        if self.is_empty():
            return RolloutTensor(state, action, next_state, reward, done, mask)

        new_state = torch.cat((self.state, state), 1)
        new_action = torch.cat((self.action, action), 1)
        new_next_state = torch.cat((self.next_state, next_state), 1)
        new_reward = torch.cat((self.reward, reward), 1)
        new_done = torch.cat((self.done, done), 1)
        new_mask = torch.cat((self.mask, mask), 1)
        return RolloutTensor(
            new_state, new_action, new_next_state, new_reward, new_done, new_mask
        )

    def stack(self, other) -> "RolloutTensor":

        if self.is_empty():
            return other

        new_state = torch.cat((self.state, other.state), 1)
        new_action = torch.cat((self.action, other.action), 1)
        new_next_state = torch.cat((self.next_state, other.next_state), 1)
        new_reward = torch.cat((self.reward, other.reward), 1)
        new_done = torch.cat((self.done, other.done), 1)
        new_mask = torch.cat((self.mask, other.mask), 1)
        return RolloutTensor(
            new_state, new_action, new_next_state, new_reward, new_done, new_mask
        )

    def decay_(self, gamma, values) -> "RolloutTensor":

        self.reward[:, -1] += ~self.done[:, -1] * gamma * values

        for i in reversed(range(len(self) - 1)):
            self.reward[:, i] = (
                self.reward[:, i] + ~self.done[:, i] * gamma * self.reward[:, i + 1]
            )

    def raw_rewards(self):
        return self.reward[self.done]

    def to(self, device):
        new_state = self.state.to(device)
        new_action = self.action.to(device)
        new_next_state = self.next_state.to(device)
        new_reward = self.reward.to(device)
        new_done = self.done.to(device)
        new_mask = self.mask.to(device)
        return RolloutTensor(
            new_state, new_action, new_next_state, new_reward, new_done, new_mask
        )
