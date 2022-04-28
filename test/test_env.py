from stonefish.env import TTTEnv, TTTEnvTwoPlayer
import torch
import numpy as np
from stonefish.utils import TwoPlayerRolloutTensor


def random_action(masks):
    masks = masks.numpy()
    probs = masks / np.sum(masks, axis=1).reshape(-1, 1)
    actions = np.zeros(len(masks))
    for (i, p) in enumerate(probs):
        actions[i] = np.random.choice(9, p=p)
    return torch.LongTensor(actions)


def test_tttenv():
    env = TTTEnv(8)

    states, legal_mask = env.reset()

    dones = [False]
    while not dones[0]:
        actions = random_action(legal_mask)
        next_state, legal_mask, rewards, dones = env.step(actions)

def test_twoplayer():
    env = TTTEnvTwoPlayer(2)

    state, legal_mask = env.reset()

    history = TwoPlayerRolloutTensor.empty()

    for _ in range(10):
        action = random_action(legal_mask)

        next_state, next_legal_mask, reward, done = env.step(action)

        history = history.add(
            state,
            action,
            next_state,
            reward,
            done,
            legal_mask,
        )

        state = next_state
        legal_mask = next_legal_mask

    print(history.reward)
    print(history.done)
    history.decay_(0.99, 10 * torch.ones(2))

    print(history.done)
    print(history.reward)

if __name__ == "__main__":
    test_twoplayer()
