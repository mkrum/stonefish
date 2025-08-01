import numpy as np
import torch
import torch.nn.functional
import tqdm
from fastchessenv import CBoards, RandomChessEnv
from torch.distributions import Categorical

from stonefish.config import expose_modules
from stonefish.convert import board_to_lczero_tensor
from stonefish.eval.agent_loader import load_model_from_config
from stonefish.utils import RolloutTensor


def _pad_mask(mask):
    # Size of the tensor minus the size of actual valid moves
    size = 5700 - mask.shape[1]
    batch = mask.shape[0]
    pad = np.zeros((batch, size))
    return np.concatenate([mask, pad], axis=1)


def _state_to_tensor(states):
    states = states.flatten()
    boards = CBoards.from_array(states).to_board()
    boards = [board_to_lczero_tensor(b) for b in boards]
    boards_tensor = torch.Tensor(np.stack(boards))
    return boards_tensor


class FakeEnv:

    def __init__(self):
        self.envs = [RandomChessEnv(1, invert=True) for _ in range(2)]

    def step(self, actions):
        out = []
        for action, env in zip(actions, self.envs, strict=False):
            out.append(env.step(action.reshape(-1, 1)))

        out = zip(*out, strict=False)

        return map(np.concatenate, out)

    def reset(self):
        out = []
        for env in self.envs:
            out.append(env.reset())

        out = zip(*out, strict=False)

        return map(np.concatenate, out)


def main():

    model = load_model_from_config("configs/train_convnet_big.yml", "model_3.pth").model

    # Does not work
    # env = RandomChessEnv(2, invert=True)
    env = FakeEnv()

    states, mask = env.reset()

    tensor = RolloutTensor.empty()

    next_boards_tensor = _state_to_tensor(states)

    for _ in tqdm.tqdm(range(1024)):

        boards_tensor = next_boards_tensor

        mask = _pad_mask(mask)
        mask_tensor = torch.Tensor(mask)

        logits = []
        with torch.no_grad():
            for bt in boards_tensor:
                logits.append(model.forward(bt, None))

        logits = torch.concatenate(logits)

        legal_logits = logits * mask_tensor + (1 - mask_tensor) * -1e10

        probs = torch.nn.functional.softmax(legal_logits, dim=1)

        actions = Categorical(probs).sample()

        next_states, mask, rewards, done = env.step(actions.detach().numpy())

        next_boards_tensor = _state_to_tensor(next_states)

        tensor = tensor.add(
            boards_tensor,
            actions,
            next_boards_tensor,
            torch.Tensor(rewards),
            torch.Tensor(done),
            mask_tensor,
        )

        print(tensor.reward[tensor.done])


if __name__ == "__main__":
    expose_modules()
    main()
