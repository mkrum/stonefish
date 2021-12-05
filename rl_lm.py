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
from generate import get_lm_input

from sacrebleu.metrics import BLEU, CHRF, TER


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

def sample_rollout(model, encoding, targets):

    input_ids, attention_mask = encoding.input_ids.to(
        device
    ), encoding.attention_mask.to(device)

    with torch.no_grad():
        actions = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=32, do_sample=True
        )
        states = get_lm_input(model, input_ids=input_ids, attention_mask=attention_mask, labels=actions)

        actions = actions[:, 1:]
        states = states[:, :-1, :]

        bleu = BLEU()
        rewards = torch.zeros(states.shape[0], actions.shape[1])
        
        for j in range(actions.shape[1]):
            generated = tokenizer.batch_decode(actions[:, :j+1], skip_special_tokens=True)
            for i in range(len(generated)):
                rewards[i, j] = bleu.corpus_score([generated[i]], targets[i]).score
        
    mean_reward = torch.mean(rewards[:, -1])

    for i in reversed(range(1, rewards.shape[1])): 
        rewards[:, i] = rewards[:, i] - rewards[:, i-1]

    gamma = 0.99
    for i in reversed(range(rewards.shape[1] - 1)): 
        rewards[:, i] = rewards[:, i] + gamma * rewards[:, i+1]
    
    rewards = rewards.to(device)
    mask = actions != 0

    states = states.reshape(-1, states.shape[-1])
    actions = actions.flatten()
    rewards = rewards.flatten()
    mask = mask.flatten()

    states = states[mask]
    actions = actions[mask]
    rewards = rewards[mask]

    return states, actions, rewards, mean_reward


if __name__ == "__main__":
    from transformers import T5TokenizerFast
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch.optim as opt
    import torch.nn as nn
    from stonefish.language import CommonGenEval

    Logger.init("/tmp", "test.txt", True, log_freq=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    full = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    full.load_state_dict(torch.load("base.pth"))

    model = copy.deepcopy(full.lm_head)

    value_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
    ).to(device) 

    opt = opt.Adam(list(model.parameters()) + list(value_model.parameters()), lr=1e-2)

    dataset = load_dataset("common_gen")

    expert_train_dl = DataLoader(
        HS("train"),
        batch_size=2048,
        drop_last=True,
        shuffle=True,
    )
    expert_test_dl = DataLoader(
        HS("test"),
        batch_size=512,
        drop_last=False,
        shuffle=False,
    )

    def collate_fn(batch):
        concepts = [" ".join(b["concepts"]) for b in batch]
        targets = [b["target"] for b in batch]
        concepts = tokenizer(concepts, padding=True, return_tensors="pt")
        targets = tokenizer(targets, padding=True, return_tensors="pt")
        return concepts, targets

    train_dl = DataLoader(
        CommonGenEval("train", tokenizer),
        batch_size=256,
        drop_last=True,
        shuffle=True,
        collate_fn=CommonGenEval.collate_fn,
    )
    test_dl = DataLoader(
        dataset["validation"],
        batch_size=512,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    encoding, targets = next(iter(train_dl))
    
    from collections import deque
    import numpy as np
    
    N = 100
    steps = 0
    
    rew = deque(maxlen=N)
    policy_losses = deque(maxlen=N)
    value_losses = deque(maxlen=N)
    
    for _ in range(20000000):
        for encoding, targets in train_dl:

            full.lm_head = copy.deepcopy(model)
            states, actions, rewards, mr = sample_rollout(full, encoding, targets)
            rew.append(mr)
    
            opt.zero_grad()

            logits = model(states)
            logits = F.log_softmax(logits, dim=-1)
            logits = torch.gather(logits, 1, actions.unsqueeze(1)).squeeze(0)

            values = value_model(states).flatten()

            value_loss = F.mse_loss(values, rewards)

            policy_loss = -1 * torch.mean((rewards.view(-1, 1) - values.detach().view(-1, 1)) * logits)
            
            policy_loss.backward() 
            value_loss.backward()

            value_losses.append(value_loss.item())
            policy_losses.append(policy_loss.item())
            
            print(np.mean(rew))
            print(np.mean(value_losses))
            print(np.mean(policy_losses))
            print()
            opt.step()

            if steps % 100 == 0:
                print("Saving...")
                torch.save(model.state_dict(), "rl_head.pth")

            steps += 1
