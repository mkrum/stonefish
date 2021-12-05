import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from stonefish.slogging import Logger
from rich import print
from datasets import load_dataset, load_metric
import jury

import multiprocessing as mp


def compute_loss(model, encoding, target_encoding):
    input_ids, attention_mask = encoding.input_ids.to(
        device
    ), encoding.attention_mask.to(device)
    labels = target_encoding.input_ids.to(device)
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


METRIC = None #load_metric("meteor")


def compute_reward(outputs, target):

    reward = torch.ones(len(outputs), 1)
    for i in range(len(outputs)):
        METRIC.add(prediction=outputs[i], reference=target[i])
        reward[i, 0] = METRIC.compute()["meteor"]

    return reward


def pg_loss(model, encoding, target_encoding):
    input_ids, attention_mask = encoding.input_ids.to(
        device
    ), encoding.attention_mask.to(device)
    labels = target_encoding.input_ids.to(device)
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_length=32
    )

    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)

    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=outputs
    ).logits
    logits = F.log_softmax(logits, dim=-1)
    logits = torch.gather(logits, 2, outputs.unsqueeze(-1)).squeeze(-1)
    reward = compute_reward(generated, references)
    print(torch.mean(reward))
    reward = reward.repeat(1, logits.shape[1]).cuda()
    loss = -1 * torch.mean(reward * logits * (outputs != 0))
    return loss


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

            for (batch_idx, (encoding, target_encoding)) in enumerate(self.train_dl):
                opt.zero_grad()

                loss = self.train_fn(model, encoding, target_encoding)

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
    out = scorer(predictions=generated, references=examples)
    print(out)

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
    from torch.utils.data import DataLoader

    from transformers import T5Tokenizer, T5ForConditionalGeneration

    Logger.init("/tmp", "test.txt", True, log_freq=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    model.load_state_dict(torch.load("base.pth"))

    opt = opt.Adam(model.parameters(), lr=5e-5)

    dataset = load_dataset("common_gen")

    def collate_fn(batch):
        concepts = [" ".join(b["concepts"]) for b in batch]
        targets = [b["target"] for b in batch]
        concepts = tokenizer(concepts, padding=True, return_tensors="pt")
        targets = tokenizer(targets, padding=True, return_tensors="pt")
        return concepts, targets

    train_dl = DataLoader(
        dataset["train"],
        batch_size=256,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        dataset["validation"],
        batch_size=512,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model.lm_head.load_state_dict(torch.load("rl_head_8.4.pth"))
    write_output(model, test_dl, compute_loss)
    import time
    time.sleep(1000)
