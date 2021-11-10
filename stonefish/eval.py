import os
import time
import argparse

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from stonefish.model import Model
from stonefish.dataset import ChessData
from stonefish.rep import MoveToken, MoveRep, BoardRep
from stonefish.vis import plot_board, square_to_grid, plot_move, mark_move

from rich.progress import track


def eval_model(model, data, batch_size=1024, max_batch=20):

    loss_fn = nn.CrossEntropyLoss()

    dataloader = DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=True)

    correct = 0.0
    total = 0.0
    losses = []
    for (batch_idx, (s, a)) in track(
        enumerate(dataloader), "Testing...", total=min(len(data)//batch_size, max_batch)
    ):
        infer = model.inference(s)

        infer = torch.flatten(infer)
        labels = torch.flatten(a).to(infer.device)

        correct += torch.sum((infer == labels).float())
        total += infer.shape[0]

        with torch.no_grad():
            p = model(s, a)
            p = p.view(-1, 6)
            loss = loss_fn(p, labels)

        losses.append(loss.item())
        
        if batch_idx == max_batch:
            break

    acc = correct / total
    m_loss = np.mean(losses)
    return acc.item(), m_loss


def main(dataset, load_model):

    data = ChessData(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(device, 128)
    model = model.to(model.device)
    print(model.load_state_dict(torch.load(load_model, map_location=device)))

    acc, m_loss = eval_model(model, data, batch_size=16)
    print(acc)
    print(m_loss)


def vis(dataset, load_model):
    data = ChessData(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(device, 128)
    model = model.to(model.device)
    model.load_state_dict(torch.load(load_model, map_location=device))

    state, action = data[np.random.choice(range(len(data)))]

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

        plot_board(ax, BoardRep.from_tensor(state[0]).to_board(), checkers=False)
        ax.imshow(prob_grid, vmax=1.0, vmin=0.0, cmap="Reds")
        plt.show()


def better_vis(dataset, load_model):
    data = ChessData(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(device, 128)
    model = model.to(model.device)
    model.load_state_dict(torch.load(load_model, map_location=device))

    state, action = data[np.random.choice(range(len(data)))]

    board = BoardRep.from_tensor(state).to_board()
    move = MoveRep.from_tensor(action).to_uci()

    state = state.unsqueeze(0)

    legal_moves = list(board.legal_moves)

    action_stack = torch.stack([MoveRep.from_uci(m).to_tensor() for m in legal_moves])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument("load_model")
    args = parser.parse_args()

    better_vis(args.dataset, args.load_model)
