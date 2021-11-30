from collections import deque

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from stonefish.rep import MoveToken, MoveRep, BoardRep
from stonefish.vis import plot_board, square_to_grid, plot_move, mark_move
from stonefish.slogging import Logger


def eval_model(model, datal, train_fn, max_batch=20):

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

            for i in range(16):
                example = s[i]
                example = example[example != -1]
                print(model.input_rep.from_tensor(example).to_str())
                print(model.output_rep.from_tensor(infer[i]).to_str())
                print()

            break

    acc = correct / total
    m_loss = np.mean(losses)
    return acc.item(), m_loss


def prob_vis(model, data, N):

    for _ in range(N):
        state, action = data[np.random.choice(range(len(data)))]

        move = MoveRep.from_tensor(action).to_uci()

        state = state.unsqueeze(0)
        action = action.unsqueeze(0)

        with torch.no_grad():
            out = model.forward(state, action)

        act = nn.Softmax(dim=-1)
        for i in range(2):

            first_logits = out[0, i]

            probs = act(first_logits).numpy()

            prob_grid = np.zeros((8, 8))
            for (square, p) in zip(MoveToken.valid_str(), probs):
                x, y = square_to_grid(square)
                prob_grid[x, y] += p

            fig, ax = plt.subplots(1, 1)

            mark_move(ax, move)

            plot_board(ax, BoardRep.from_tensor(state[0]).to_board(), checkers=False)
            ax.imshow(prob_grid, vmax=1.0, vmin=0.0, cmap="Reds")
            plt.show()


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
