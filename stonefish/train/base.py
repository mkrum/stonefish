from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from stonefish.slogging import Logger


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


def joint_step(model, state, output):

    return sample_pg_step(model, state, output, N=10) + train_step(model, state, output)


@dataclass
class TrainingContext:

    eval_fn: Any
    train_fn: Any
    train_dl: Any
    test_dl: Any
    epochs: int = 1000

    def __call__(self, model, opt):

        for epoch in range(self.epochs):
            Logger.epoch()
            out = self.eval_fn(model, self.test_dl, self.train_fn)
            Logger.test_output(*out)
            Logger.save_checkpoint(model, opt)

            for (batch_idx, (state, output)) in enumerate(self.train_dl):
                opt.zero_grad()
                loss = self.train_fn(model, state, output)
                loss.backward()
                opt.step()

                Logger.loss(model, opt, batch_idx, loss.item())

        out = self.eval_fn(model, test_data, train_fn)
        Logger.test_output(*out)
        Logger.save_checkpoint(model, opt)
