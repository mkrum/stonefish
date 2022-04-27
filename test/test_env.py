from stonefish.env import TTTEnv
import torch
import numpy as np


def random_action(masks):
    probs = masks / np.sum(masks, axis=1).reshape(-1, 1)
    actions = np.zeros(len(masks))
    for (i, p) in enumerate(probs):
        actions[i] = np.random.choice(9, p=p)
    return torch.LongTensor(actions)


def test_tttenv():
    env = TTTEnv(8)

    states, legal_mask = env.reset()
    print(states.shape)

    dones = [False]
    while not dones[0]:
        actions = random_action(legal_mask)
        next_state, legal_mask, rewards, dones = env.step(actions)


if __name__ == "__main__":
    test_tttenv()
