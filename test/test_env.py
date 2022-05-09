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

    for _ in range(int(1e5)):
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

        if not (state == test_state).all():
            print(CBoardRep.from_tensor(last_state[0]).to_fen())
            print(MoveEnum.from_tensor(last_action[0]).to_str())
            for i in range(1):
                print(CBoardRep.from_tensor(state[i]).to_fen())
                print(CBoardRep.from_tensor(test_state[i]).to_fen())
                print(CBoardRep.from_tensor(inverted_state[i]).to_fen())
                print()
            import pdb

            pdb.set_trace()

        assert (state == test_state).all()

    history.selfplay_decay_(0.99, 10 * torch.ones(2))


if __name__ == "__main__":
    test_chess_twoplayer()
