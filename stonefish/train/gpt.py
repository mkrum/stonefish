from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from stonefish.slogging import Logger
from datasets import load_metric

from rich import print


def compute_loss(model, state):
    preds = model(state)

    preds = preds[:, :-1, :].reshape(-1, 50257)
    target_ids = (
        state[:, 1:]
        .reshape(
            -1,
        )
        .to(preds.device)
    )
    preds = preds[target_ids != -1]
    target_ids = target_ids[target_ids != -1]
    return F.cross_entropy(preds, target_ids)


@dataclass
class GPTTrainingContext:

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

            for (batch_idx, state) in enumerate(self.train_dl):
                opt.zero_grad()
                loss = self.train_fn(model, state)
                loss.backward()
                opt.step()

                Logger.loss(model, opt, batch_idx, len(self.train_dl), loss.item())

                if batch_idx % self.eval_freq == 0 and batch_idx > 0:
                    out = self.eval_fn(model, self.test_dl, self.train_fn)
                    Logger.test_output(*out)
                    Logger.save_checkpoint(model, opt)


if __name__ == "__main__":

    from stonefish.dataset import single_default_collate_fn, default_collate_fn
    from transformers import GPT2TokenizerFast
    from stonefish.model import GPTModel
    from stonefish.language import SingleCommonGen, CommonGen, CommonGenEval
    from stonefish.rep import create_tokenizer_rep
    import torch.optim as opt
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Logger.init("/tmp", "test.txt", True, log_freq=10)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")

    GPTTokenizer = create_tokenizer_rep("GPTTokenizer", tokenizer)

    model = GPTModel(device, tokenizer, "gpt2-medium")
    model.load_state_dict(torch.load("model_2.pth"))
    opt = opt.Adam(model.parameters(), lr=5e-5)

    train_dataset = SingleCommonGen(GPTTokenizer, "train")
    test_dataset = CommonGenEval(GPTTokenizer)

    train_dl = DataLoader(
        train_dataset,
        batch_size=32,
        drop_last=True,
        shuffle=True,
        collate_fn=single_default_collate_fn,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=True,
        shuffle=True,
        collate_fn=CommonGenEval.collate_fn,
    )

    def fake_eval(model, test_dl, train_fn):

        metric = load_metric("bleu")

        step = 0
        for (s, t) in test_dl:
            out = model.inference(s)
            in_str = model.tokenizer.decode(out[0, 1 : s.shape[1]])
            out_str = model.tokenizer.decode(out[0, s.shape[1] :])
            out_str = out_str.split("<|endoftext|>")[0]

            print(model.tokenizer.decode(out[0]))

            if step <= 10:
                print(f"[red]{in_str}[blue]{out_str}")

            metric.add_batch(
                predictions=[model.tokenizer.encode(out_str)],
                references=[[model.tokenizer.encode(t_) for t_ in t[0]]],
            )

            step += 1
            if step > 100:
                print(metric.compute())
                break

        return 0.0, 0.0

    ctx = GPTTrainingContext(fake_eval, compute_loss, train_dl, test_dl)
    ctx(model, opt)
