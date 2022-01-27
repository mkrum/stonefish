from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from stonefish.slogging import Logger


def train_step(model, state, output):
    model.train()

    probs = model(state, output)

    probs = probs.reshape(-1, probs.shape[-1])

    output = output[:, 1:].reshape(
        -1,
    )
    probs = probs[output != -1]
    output = output[output != -1]

    loss = F.nll_loss(probs, output.flatten().to(probs.device))
    return loss


@dataclass
class TrainingContext:

    eval_fn: Any
    train_fn: Any
    train_dl: Any
    test_dl: Any
    epochs: int = 1000
    eval_freq: int = 10000

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

                Logger.loss(model, opt, batch_idx, len(self.train_dl), loss.item())

                if batch_idx % self.eval_freq == 0 and batch_idx > 0:
                    out = self.eval_fn(model, self.test_dl, self.train_fn)
                    Logger.test_output(*out)
                    Logger.save_checkpoint(model, opt)

            out = self.eval_fn(model, self.test_dl, self.train_fn)
            Logger.test_output(*out)
            Logger.save_checkpoint(model, opt)
