from stonefish.env import CChessEnvTorch
import torch
import numpy as np
from stonefish.utils import RolloutTensor

from stonefish.mask import MoveMask
from stonefish.rep import MoveRep


def random_action(masks):
    masks = masks.numpy()
    probs = masks / np.sum(masks, axis=1).reshape(-1, 1)
    actions = np.zeros(len(masks))
    for (i, p) in enumerate(probs):
        actions[i] = np.random.choice(len(p), p=p)
    return torch.LongTensor(actions)


def test_mask():
    n = 2
    env = CChessEnvTorch(n)
    state, legal_mask = env.reset()

    mask = MoveMask.from_mask(legal_mask)

    move_token = torch.zeros(2, 1)

    for _ in range(2):
        move_mask = mask.get_mask(move_token)
        new = random_action(move_mask)
        move_token = torch.cat((move_token, new.view(-1, 1)), dim=1)

    print(mask.get_full_mask(move_token))
    print(move_token)
    print(MoveRep.from_tensor(move_token[0]))
    print(MoveRep.from_tensor(move_token[1]))


if __name__ == "__main__":
    test_mask()
