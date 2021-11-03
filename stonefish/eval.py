import os
import time
import argparse

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from stonefish.model import Model
from stonefish.dataset import ChessData
from stonefish.rep import MoveToken

from rich.progress import track


def eval_model(model, data, batch_size=1024, max_batch=20):

    loss_fn = nn.CrossEntropyLoss()

    dataloader = DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True)

    correct = 0.0
    total = 0.0
    losses = []
    for (batch_idx, (s, a)) in track(
        enumerate(dataloader), "Testing...", total=max_batch
    ):
        infer = model.inference(s)
        infer = torch.flatten(infer)
        labels = torch.flatten(a).to(infer.device)

        correct += torch.sum((infer == labels).float())
        total += infer.shape[0]

        with torch.no_grad():
            p = model(s, a)
            p = p.view(-1, MoveToken.size())
            loss = loss_fn(p, labels)

        losses.append(loss.item())

        if batch_idx == max_batch:
            break

    acc = correct.item() / total
    m_loss = np.mean(losses)
    return acc, m_loss


def main(dataset, load_model):

    data = ChessData(dataset)

    device = torch.device("cuda")
    model = Model(device, 512)
    model = model.to(model.device)

    print(model.load_state_dict(torch.load(load_model)))

    acc, m_loss = eval_model(model, data)
    print(acc)
    print(m_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument("load_model")
    args = parser.parse_args()

    main(args.dataset, args.load_model)
