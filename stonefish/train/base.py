from dataclasses import dataclass
from typing import Any

import torch.nn.functional as functional
from mllg import TrainInfo, ValidationInfo

from stonefish.mask import MoveMask


def train_step(model, state, output):
    model.train()

    probs = model(state, output)
    loss = functional.cross_entropy(probs, output.to(probs.device).flatten())
    return loss


def seq_train_step(model, state, output):
    model.train()

    probs = model(state, output)

    probs = probs.reshape(-1, probs.shape[-1])

    output = output[:, 1:].reshape(
        -1,
    )
    probs = probs[output != -1]
    output = output[output != -1]

    loss = functional.cross_entropy(probs, output.flatten().to(probs.device))
    return loss


def mask_train_step(model, state, output):
    model.train()

    mm = MoveMask.from_data(state, output)

    masks = mm.update_mask(output)
    masks = masks[:, 1:, :]

    probs = model(state, output, logit_mask=masks.cuda())

    probs = probs.reshape(-1, probs.shape[-1])

    output = output[:, 1:].reshape(
        -1,
    )
    probs = probs[output != -1]
    output = output[output != -1]

    loss = functional.cross_entropy(probs, output.flatten().to(probs.device))
    return loss


@dataclass
class PreTrainContext:

    eval_fn: Any
    train_fn: Any
    train_dl: Any
    test_dl: Any
    epochs: int = 1000
    eval_freq: int = 5000

    def __call__(self, logger, model, opt):
        out = self.eval_fn(model, self.test_dl, self.train_fn)
        logger.log_info(ValidationInfo(0, 0, out))

        for epoch in range(self.epochs):

            for batch_idx, (state, output) in enumerate(self.train_dl):
                opt.zero_grad()
                loss = self.train_fn(model, state, output)
                loss.backward()
                opt.step()

                logger.log_info(TrainInfo(epoch, batch_idx, loss.item()))

                if batch_idx % self.eval_freq == 0 and batch_idx > 0:
                    out = self.eval_fn(model, self.test_dl, self.train_fn)
                    logger.log_info(ValidationInfo(epoch, batch_idx, out))
                    logger.checkpoint(epoch, batch_idx, model)

            out = self.eval_fn(model, self.test_dl, self.train_fn)
            logger.log_info(ValidationInfo(epoch, batch_idx, out))
            logger.checkpoint(epoch, batch_idx, model)
