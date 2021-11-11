import os
import time
import argparse

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from stonefish.model import Model
from stonefish.dataset import ChessData, TTTData
from stonefish.eval import eval_model
from stonefish.rep import MoveToken
from stonefish.ttt import TTTBoardToken, TTTMoveToken

from stonefish.slogging import Logger


def train_step(model, state, output):
    probs = model(state, output)
    probs = probs.view(-1, probs.shape[-1])
    loss = F.cross_entropy(probs, output.flatten().cuda())
    return loss


def eval_step(model, test_data):
    out = eval_model(model, test_data, batch_size=256)
    Logger.test_output(*out)


def main(model, opt, train_data, test_data):

    dataloader = DataLoader(train_data, batch_size=256, drop_last=True, shuffle=True)

    for epoch in range(1000):
        Logger.epoch()
        eval_step(model, test_data)

        for (batch_idx, (state, output)) in enumerate(dataloader):
            opt.zero_grad()
            loss = train_step(model, state, output)
            loss.backward()
            opt.step()

            Logger.loss(batch_idx, loss.item())

        Logger.save_checkpoint(model, opt)

    eval_step(model, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("log_file")
    parser.add_argument("--load_model")
    parser.add_argument("--load_opt")
    parser.add_argument("--dir", default=".")
    parser.add_argument(
        "-o", default=False, action="store_true", help="Overwrite by default"
    )
    args = parser.parse_args()

    go_ahead = args.o
    if not go_ahead and os.path.exists(args.log_file):
        res = input(f"File {args.log_file} exists. Overwrite? (Y/n) ")
        go_ahead = res == "" or res.lower() == "y"

    if not go_ahead:
        exit()

    Logger.init(args.dir, args.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(device, 512, TTTBoardToken, TTTMoveToken)
    model = model.to(model.device)
    if args.load_model:
        model.load_state_dict(torch.load(load_model))

    opt = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    if args.load_opt:
        opt.load_state_dict(torch.load(load_opt))

    train_data = TTTData("data/ttt_train.csv")
    test_data = TTTData("data/ttt_test.csv")
    main(model, opt, train_data, test_data)
