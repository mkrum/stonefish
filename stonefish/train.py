import os
import argparse

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
    model.train()
    probs = model(state, output)
    probs = probs.view(-1, probs.shape[-1])
    loss = F.nll_loss(probs, output.flatten().cuda())
    return loss


def pg_step(model, state, output):

    with torch.no_grad():
        samples = model.sample(state).detach()

    output = output.to(samples.device)
    log_probs = model(state, samples)

    log_probs = torch.gather(log_probs, 2, samples.unsqueeze(-1))

    matches = samples == output
    matches = matches.flatten()

    reward = 1.0 * matches.float() + -1.0 * (~matches).float()

    loss = -1 * torch.mean(reward.detach() * log_probs.flatten())
    return loss


def sample_pg_step(model, state, output, N=10):
    model.train()

    loss = torch.zeros(
        1,
    ).cuda()

    for _ in range(N):
        loss += pg_step(model, state, output)

    return loss / N


def joint_fn(model, state, output):
    return sample_pg_step(model, state, output, N=10) + train_step(model, state, output)


def eval_step(model, test_data, train_fn):
    out = eval_model(model, test_data, train_fn)
    Logger.test_output(*out)


def main(model, opt, train_dl, test_dl, train_fn, eval_fn):
    for epoch in range(1000):
        Logger.epoch()
        eval_fn(model, test_dl, train_fn)

        for (batch_idx, (state, output)) in enumerate(train_dl):
            opt.zero_grad()
            loss = train_fn(model, state, output)
            loss.backward()
            opt.step()

            Logger.loss(batch_idx, loss.item())

        Logger.save_checkpoint(model, opt)

    eval_fn(model, test_data, train_fn)


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
    model = Model(device, 128, TTTBoardToken, TTTMoveToken)
    model = model.to(model.device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))

    opt = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    if args.load_opt:
        opt.load_state_dict(torch.load(args.load_opt))

    train_data = TTTData("data/ttt_train.csv")
    test_data = TTTData("data/ttt_test.csv")

    train_dl = DataLoader(train_data, batch_size=512, drop_last=True, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=256, drop_last=False, shuffle=True)

    main(model, opt, train_dl, test_dl, joint_fn, eval_step)
