import time
import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from stonefish.slogging import Logger
from rich import print
from datasets import load_dataset, load_metric
import jury

import multiprocessing as mp


class HS(Dataset):

    def __init__(self, type_):
        self.states = torch.load(f"{type_}_states.pth")
        self.actions = torch.load(f"{type_}_actions.pth")

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

    def __len__(self):
        return self.actions.shape[0]

def compute_loss(model, state, action):
    logits = model(state)
    loss = F.cross_entropy(logits, action)
    return loss

def eval_fn(model, test_dl, loss_fn):
    
    losses = 0.0
    total = 0
    for (batch_idx, (state, action)) in enumerate(test_dl):
        state = state.to(device)
        action = action.to(device)
        with torch.no_grad():
            losses += loss_fn(model, state, action).item()
            total += 1.0

    return 0.0, losses / total


@dataclass
class TrainingContext:

    eval_fn: Any
    train_fn: Any
    train_dl: Any
    test_dl: Any
    epochs: int = 1000
    eval_freq: int = 10000

    def __call__(self, model, opt, device):

        for epoch in range(self.epochs):
            Logger.epoch()
            start = time.time()
            out = self.eval_fn(model, self.test_dl, self.train_fn)
            end= time.time()
            print(end - start)
            Logger.test_output(*out)
            Logger.save_checkpoint(model, opt)

            for (batch_idx, (state, action)) in enumerate(self.train_dl):
                state = state.to(device)
                action = action.to(device)
                opt.zero_grad()
                loss = self.train_fn(model, state, action)
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


def run_eval(state_path, generated_path, examples_path):

    states = open(state_path, 'r').readlines()
    generated = open(generated_path, 'r').readlines()
    examples = open(examples_path, 'r').readlines()

    from jury import Jury
    jury = Jury()

    uniq_states = list(set(states))

    grouped_generated = {u: [] for u in uniq_states}
    grouped_examples = {u : [] for u in uniq_states}
    for (i, s) in enumerate(states):
        grouped_generated[s].append(generated[i])
        grouped_examples[s].append(examples[i])

    generated = [grouped_generated[u] for u in uniq_states] 
    examples = [grouped_examples[u] for u in uniq_states] 
    scorer = Jury()
    print("Running...")
    print(scorer(predictions=generated, references=examples))

def write_output(model, test_dl, train_fn):
    
    states = []
    generated = []
    examples = []

    for (batch_idx, (encoding, targets)) in enumerate(test_dl):
        input_ids, attention_mask = encoding.input_ids.to(
            device
        ), encoding.attention_mask.to(device)
        labels = targets.input_ids.to(device)

        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=32, do_sample=True
        )
        
        states += tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        generated += tokenizer.batch_decode(outputs, skip_special_tokens=True)
        examples += tokenizer.batch_decode(labels, skip_special_tokens=True)

        sample_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sample_in = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        print(f"[red]{sample_in} [green]-> [blue]{sample_out}")

    with open('states.txt', 'w') as f:
        for i in range(len(generated)):
            f.write(states[i] + '\n')

    with open('generated.txt', 'w') as f:
        for i in range(len(generated)):
            f.write(generated[i] + '\n')

    with open('examples.txt', 'w') as f:
        for i in range(len(generated)):
            f.write(examples[i] + '\n')

    p = mp.Process(target=run_eval, args=("states.txt", "generated.txt", "examples.txt"))
    p.daemon = True
    p.start()
    return 0.0, 0.0


if __name__ == "__main__":
    from transformers import T5TokenizerFast
    import torch.optim as opt
    import torch.nn as nn

    from transformers import T5Tokenizer, T5ForConditionalGeneration

    Logger.init("/tmp", "test.txt", True, log_freq=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    full = T5ForConditionalGeneration.from_pretrained("t5-small")
    full.load_state_dict(torch.load("base.pth"))
    model = copy.deepcopy(full.lm_head).to(device)
    model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32128)
    ).to(device) 
    del full

    opt = opt.Adam(model.parameters(), lr=1e-3)

    train_dl = DataLoader(
        HS("train"),
        batch_size=2048,
        drop_last=True,
        shuffle=True,
    )
    test_dl = DataLoader(
        HS("test"),
        batch_size=512,
        drop_last=False,
        shuffle=False,
    )

    ctx = TrainingContext(eval_fn, compute_loss, train_dl, test_dl)
    ctx(model, opt, device)
