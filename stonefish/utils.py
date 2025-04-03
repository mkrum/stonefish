from dataclasses import dataclass

import chess
import numpy as np
import torch
from typing_extensions import Self


def randomize_board(min_steps=10, max_steps=20):
    steps = list(range(min_steps, max_steps))
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

    def __getitem__(self, idx):
        return RolloutTensor(
            self.state[idx],
            self.action[idx],
            self.next_state[idx],
            self.reward[idx],
            self.done[idx],
            self.mask[idx],
        )

    @classmethod
    def empty(cls):
        return cls(None, None, None, None, None, None)

    def __len__(self):
        if self.state is None:
            return 0
        return self.state.shape[1]

    def is_empty(self):
        return self.state is None

    def add(self, state, action, next_state, reward, done, mask) -> Self:

        state = state.unsqueeze(1)
        action = action.unsqueeze(1)
        next_state = state.unsqueeze(1)
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1).bool()
        mask = mask.unsqueeze(1)

        if self.is_empty():
            return self.__class__(state, action, next_state, reward, done, mask)

        new_state = torch.cat((self.state, state), 1)
        new_action = torch.cat((self.action, action), 1)
        new_next_state = torch.cat((self.next_state, next_state), 1)
        new_reward = torch.cat((self.reward, reward), 1)
        new_done = torch.cat((self.done, done), 1)
        new_mask = torch.cat((self.mask, mask), 1)
        return self.__class__(
            new_state, new_action, new_next_state, new_reward, new_done, new_mask
        )

    def stack(self, other) -> Self:

        if self.is_empty():
            return other  # type: ignore

        new_state = torch.cat((self.state, other.state), 1)
        new_action = torch.cat((self.action, other.action), 1)
        new_next_state = torch.cat((self.next_state, other.next_state), 1)
        new_reward = torch.cat((self.reward, other.reward), 1)
        new_done = torch.cat((self.done, other.done), 1)
        new_mask = torch.cat((self.mask, other.mask), 1)
        return self.__class__(
            new_state, new_action, new_next_state, new_reward, new_done, new_mask
        )

    def decay_(self, gamma, values) -> None:

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
        return self.__class__(
            new_state, new_action, new_next_state, new_reward, new_done, new_mask
        )

    def get_data(self):
        flat_state = self.state.view(-1, *self.state.shape[2:])
        flat_next_state = self.next_state.view(-1, *self.next_state.shape[2:])
        flat_action = self.action.view(-1, 1)
        flat_reward = self.reward.view(-1, 1)
        flat_done = self.done.view(-1, 1)
        flat_mask = self.mask.view(-1, *self.mask.shape[2:])

        return (
            flat_state,
            flat_next_state,
            flat_action,
            flat_reward,
            flat_done,
            flat_mask,
        )

    def selfplay_decay_(self, gamma, values) -> None:

        for i in reversed(range(self.reward.shape[1] - 1)):
            self.reward[:, i] -= ~self.done[:, i] * (
                self.done[:, i + 1] * self.reward[:, i + 1]
            )

        i = self.reward.shape[1] - 1

        self.reward[:, i] += ~self.done[:, i] * gamma * -1 * values

        i -= 2
        while i >= 0:

            self.done[:, i] = self.done[:, i] | self.done[:, i + 1]
            self.reward[:, i] = self.reward[:, i] + ~self.done[:, i] * gamma * (
                self.reward[:, i + 2]
            )
            i -= 2

        i = self.reward.shape[1] - 2

        self.done[:, i] = self.done[:, i] | self.done[:, i + 1]
        self.reward[:, i] += ~self.done[:, i] * gamma * values

        i -= 2
        while i >= 0:
            self.done[:, i] = self.done[:, i] | self.done[:, i + 1]
            self.reward[:, i] = self.reward[:, i] + ~self.done[:, i] * gamma * (
                self.reward[:, i + 2]
            )
            i -= 2


def ttt_state_to_str(state, action):
    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()

    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    state = state.reshape(3, 3, 3)

    ttt_str = ""
    for i in range(3):
        for j in range(3):
            if 3 * i + j == action[0]:
                ttt_str += "_"
            elif state[0, i, j] == 1:
                ttt_str += " "
            elif state[1, i, j] == 1:
                ttt_str += "x"
            elif state[2, i, j] == 1:
                ttt_str += "o"

            if j != 2:
                ttt_str += " | "
        ttt_str += "\n"
        if i != 2:
            ttt_str += "-----------\n"

    ttt_str += "\n"
    return ttt_str
