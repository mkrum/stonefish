import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional
import torch.optim as optim
import tqdm
from fastchessenv import CBoards, RandomChessEnv
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel

from stonefish.config import expose_modules
from stonefish.convert import board_to_lczero_tensor
from stonefish.eval.agent_loader import load_model_from_config
from stonefish.train.base import cleanup_distributed, setup_distributed
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

    def __init__(self, n):
        self.envs = [RandomChessEnv(1, invert=True) for _ in range(n)]

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


def generate_rollout_batch(env, next_boards_tensor, mask, model, num_steps=16):

    tensor = RolloutTensor.empty()

    for _ in tqdm.tqdm(range(num_steps)):

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

    return tensor


def main():

    model = load_model_from_config("configs/train_convnet_big.yml", "model_3.pth")

    model.model = DistributedDataParallel(model.model)

    opt = optim.Adam(model.model.parameters(), lr=1e-4)

    # Does not work
    env = FakeEnv(1)

    states, mask = env.reset()
    next_boards_tensor = _state_to_tensor(states)

    for _ in range(100):

        tensor = generate_rollout_batch(
            env, next_boards_tensor, mask, model.model.module, num_steps=128
        )

        dist.barrier()

        tensor.reward[tensor.done & (tensor.reward == 0)] = -1.0

        print(tensor.done.sum())
        print(tensor.reward[tensor.done].mean())

        tensor.decay_(0.99)

        opt.zero_grad()

        (
            flat_state,
            flat_next_state,
            flat_action,
            flat_reward,
            flat_done,
            flat_mask,
        ) = tensor.get_data()

        completed_mask = (flat_reward != 0).flatten()
        flat_state = flat_state[completed_mask]
        flat_action = flat_action[completed_mask]
        flat_reward = flat_reward[completed_mask]

        logits = model.model(flat_state, None)

        sel_logits = logits.gather(1, flat_action)

        policy_loss = -1.0 * torch.mean(flat_reward * sel_logits)

        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
        opt.step()


if __name__ == "__main__":
    expose_modules()
    setup_distributed()
    main()
    cleanup_distributed()
