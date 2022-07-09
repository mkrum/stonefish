from stonefish.env import TTTEnv, TTTEnvTwoPlayer, CChessEnvTorchTwoPlayer
import torch
import numpy as np
from stonefish.utils import RolloutTensor, ttt_state_to_str
from stonefish.rep import CBoardRep, MoveEnum


def random_action(masks):
    masks = masks.numpy()
    probs = masks / np.sum(masks, axis=1).reshape(-1, 1)
    actions = np.zeros(len(masks))
    for (i, p) in enumerate(probs):
        actions[i] = np.random.choice(len(p), p=p)
    return torch.LongTensor(actions)


def test_tttenv():
    env = TTTEnv(8)

    states, legal_mask = env.reset()

    dones = [False]
    while not dones[0]:
        actions = random_action(legal_mask)
        next_state, legal_mask, rewards, dones = env.step(actions)


def ttt_look_at():
    env = TTTEnv(1)

    state, legal_mask = env.reset()
    for _ in range(100):
        action = random_action(legal_mask)
        print(ttt_state_to_str(state, action))
        action = action.cpu().numpy()
        state, legal_mask, reward, done = env.step(action)
        print(f"done: {done[0]} reward: {reward[0]}")


def test_twoplayer():
    env = TTTEnvTwoPlayer(2)

    state, legal_mask = env.reset()

    history = RolloutTensor.empty()

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

    history.selfplay_decay_(0.99, 10 * torch.ones(2))


def test_twoplayer():
    env = TTTEnvTwoPlayer(2)

    state, legal_mask = env.reset()

    history = RolloutTensor.empty()

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

    history.selfplay_decay_(0.99, 10 * torch.ones(2))


def test_chess_twoplayer():
    env = CChessEnvTorchTwoPlayer(1, invert=True)

    state, legal_mask = env.reset()

    history = RolloutTensor.empty()

    for _ in range(10):
        last_state = state
        action = random_action(legal_mask)
        last_action = action

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

        env.invert_boards()
        inverted_state = torch.FloatTensor(env.get_state())
        env.invert_boards()
        test_state = torch.FloatTensor(env.get_state())
        assert (state == test_state).all()
