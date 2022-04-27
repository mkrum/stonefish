from collections import deque

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from stonefish.rep import MoveToken, MoveRep, BoardRep
from chessplotlib import plot_board, plot_move, mark_move
from mllg import TestInfo


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


def move_vis(model, data, N):

    for _ in range(N):
        state, action = data[np.random.choice(range(len(data)))]

        board = BoardRep.from_tensor(state).to_board()
        move = MoveRep.from_tensor(action).to_uci()

        state = state.unsqueeze(0)

        legal_moves = list(board.legal_moves)

        action_stack = torch.stack(
            [MoveRep.from_uci(m).to_tensor() for m in legal_moves]
        )
        state = state.repeat(action_stack.shape[0], 1)

        act = nn.Softmax(dim=0)
        with torch.no_grad():
            out = model.forward(state, action_stack)

            logits = torch.zeros(out.shape[0])
            for (i, o) in enumerate(out):
                logits[i] += o[0, action_stack[i, 0]] + o[1, action_stack[i, 1]]

        probs = act(logits)

        fig, ax = plt.subplots(1, 1)

        plot_board(ax, board, checkers=True)
        mark_move(ax, move)
        for i in range(len(legal_moves)):
            plot_move(ax, board, legal_moves[i], alpha=probs[i].item())

        plt.show()
