import os
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich import print

from stonefish.eval import eval_model
from stonefish.slogging import Logger
from stonefish.config import load_config_and_parse_cli


def train_step(model, state, output):
    model.train()
    probs = model(state, output)
    probs = probs.view(-1, probs.shape[-1])
    loss = F.nll_loss(probs, output.flatten().to(probs.device))
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

    loss = torch.zeros(1)

    for _ in range(N):
        out = pg_step(model, state, output)
        loss = loss.to(out.device)
        loss += out

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
    config = load_config_and_parse_cli(sys.argv[1])
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config["model"](device, config["input_rep"], config["output_rep"])
    opt = config["opt"](model.parameters())

    train_dl = config["train_data"]()
    test_dl = config["test_data"]()

    main(model, opt, train_dl, test_dl, train_step, eval_step)
