from collections import deque

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from stonefish.rep import MoveToken, MoveRep, BoardRep
from chessplotlib import plot_board, plot_move, mark_move
from mllg import TestInfo

from stonefish.env import CChessEnvTorch, TTTEnvTwoPlayer


def print_example(model, states, actions, infer):
    for (s, a, i) in list(zip(states, actions, infer))[:16]:
        example = s[s != -1]
        board_str = model.input_rep.from_tensor(example).to_str()
        pred_str = model.output_rep.from_tensor(i).to_str()
        label_str = model.output_rep.from_tensor(a).to_str()
        print(f"{board_str} {pred_str} {label_str}")


def eval_model(model, datal, train_fn, max_batch=20):

    correct = 0.0
    total = 0.0
    losses = []

    for (batch_idx, (s, a)) in enumerate(datal):
        model.eval()

        infer = model.inference(s, a.shape[1] - 1)

        for i in range(len(infer)):
            pred_str = model.output_rep.from_tensor(infer[i]).to_str()
            label_str = model.output_rep.from_tensor(a[i]).to_str()

            total += 1.0
            if pred_str == label_str:
                correct += 1.0

        with torch.no_grad():
            loss = train_fn(model, s, a)

        losses.append(loss.item())

        if batch_idx == max_batch:
            break

    print_example(model, s, a, infer)

    acc = correct / total
    m_loss = np.mean(losses)
    return [TestInfo("ACC", acc), TestInfo("loss", m_loss)]


def seq_eval_model(model, datal, train_fn, max_batch=20):

    correct = 0.0
    total = 0.0
    losses = []

    for (batch_idx, (s, a)) in enumerate(datal):
        model.eval()

        infer = model.inference(s, a.shape[1] - 1)
        flat_infer = torch.flatten(infer[:, 1:])

        labels = torch.flatten(a[:, 1:]).to(infer.device)

        flat_infer = flat_infer[labels != -1]
        labels = labels[labels != -1]

        correct += torch.sum((flat_infer == labels).float())
        total += flat_infer.shape[0]

        with torch.no_grad():
            loss = train_fn(model, s, a)

        losses.append(loss.item())

        if batch_idx == max_batch:
            break

    print_example(model, s, a, infer)

    acc = correct / total
    m_loss = np.mean(losses)
    return [TestInfo("ACC", acc.item()), TestInfo("loss", m_loss)]


def random_action(masks):
    masks = masks.numpy()
    probs = masks / np.sum(masks, axis=1).reshape(-1, 1)
    actions = np.zeros(len(masks))
    for (i, p) in enumerate(probs):
        actions[i] = np.random.choice(9, p=p)
    return torch.LongTensor(actions)


def eval_perf(model):
    env = TTTEnvTwoPlayer(1)

    N = 10
    wins = 0
    for _ in range(N):

        state, legal_mask = env.reset()

        player_id = 0
        done = [False]
        while not done[0]:

            if player_id == 0:
                action = model.sample(state, legal_mask)
            elif player_id == 1:
                action = random_action(legal_mask)
            else:
                exit()

            state, legal_mask, reward, done = env.step(action)
            player_id = (player_id + 1) % 2

        if reward[0] == 1.0 and player_id == 1:
            wins += 1.0

    print(wins / N)
