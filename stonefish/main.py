import os
import time
import itertools
import argparse

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical

from constants import MOVE_TOKENS, BOARD_TOKENS, BTOKEN_ID, MTOKEN_ID


class TokenizedChess(Dataset):
    def __init__(self, path):
        super().__init__()

        with open(path, "r") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        raw = self.data[i].rstrip().split(",")
        board_state = raw[:65]
        board_tokens = torch.LongTensor(list(map(BTOKEN_ID.__getitem__, board_state)))
        action = torch.LongTensor(list(map(MTOKEN_ID.__getitem__, raw[65:])))
        return board_tokens, action


class Model(nn.Module):
    def __init__(self, device, emb_dim):
        super().__init__()
        self.device = device

        self.board_embed = nn.Embedding(len(BOARD_TOKENS), emb_dim)
        self.move_embed = nn.Embedding(len(MOVE_TOKENS), emb_dim)

        self.pos_encoding = nn.Parameter(torch.zeros(65, emb_dim))
        self.start_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.transformer = nn.Transformer(
            batch_first=True,
        )

        self.to_dist = nn.Sequential(nn.Linear(emb_dim, len(MOVE_TOKENS)))

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)

        embed_state = self.board_embed(state)
        pos_embed_state = embed_state + self.pos_encoding
        tgt_embed = self.move_embed(action)

        repeat_token = self.start_token.repeat(action.shape[0], 1, 1).to(self.device)
        tgt_embed_token = torch.cat((repeat_token, tgt_embed), dim=1)

        tgt_mask = self.transformer.generate_square_subsequent_mask(3).to(self.device)
        out = self.transformer(pos_embed_state, tgt_embed_token, tgt_mask=tgt_mask)
        out = out[:, :2]
        logits = self.to_dist(out)
        return logits

    @torch.no_grad()
    def inference(self, state):
        state = state.to(self.device)
        embed_state = self.board_embed(state)
        pos_embed_state = embed_state + self.pos_encoding

        decode = self.start_token.repeat(state.shape[0], 1, 1).to(self.device)
        tokens = torch.zeros((state.shape[0], 1)).to(self.device)

        for i in range(2):
            tgt_mask = self.transformer.generate_square_subsequent_mask(i + 1).to(
                self.device
            )

            out = self.transformer(pos_embed_state, decode, tgt_mask=tgt_mask)
            logits = self.to_dist(out)[:, -1, :]
            next_value = Categorical(logits=logits).sample().view(-1, 1)
            tokens = torch.cat((tokens, next_value), dim=1)
            embed_next = self.move_embed(next_value)
            decode = torch.cat((decode, embed_next), dim=1)

        return tokens[:, 1:]


def main(log_file_path: str, load_model, load_opt):
    data = TokenizedChess("data.csv")
    dataloader = DataLoader(data, batch_size=256, drop_last=True)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device("cuda")
    model = Model(device, 512)
    model = model.to(model.device)
    if load_model:
        model.load_state_dict(torch.load(load_model))

    opt = optim.Adam(model.parameters(), lr=1e-4)
    if load_opt:
        opt.load_state_dict(torch.load(load_opt))

    log_file = open(log_file_path, "w")

    losses = deque(maxlen=100)

    for epoch in range(100):
        for (batch_idx, (s, a)) in enumerate(dataloader):
            s = s.to(device)
            labels = torch.flatten(a).to(device)

            opt.zero_grad()
            p = model(s, a)
            p = p.view(-1, len(MOVE_TOKENS))
            loss = loss_fn(p, labels)
            loss.backward()
            opt.step()

            loss = loss.item()
            losses.append(loss)

            acc = None
            if batch_idx % 100 == 0:
                infer = model.inference(s)
                acc = torch.mean((infer.flatten() == labels).float()).item()
                print(f"({epoch}/{batch_idx}) Loss (Avg): {np.mean(losses)} Acc: {acc}")

            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), "model.pth")
                torch.save(opt.state_dict(), "opt.pth")

            log_file.write(f"{epoch} {batch_idx} {time.time()} {loss} {acc}\n")

        torch.save(model.state_dict(), f"model_{epoch}.pth")
        torch.save(opt.state_dict(), f"opt_{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("log_file")
    parser.add_argument("--load_model")
    parser.add_argument("--load_opt")
    args = parser.parse_args()

    if os.path.exists(args.log_file):
        res = input(f"File {args.log_file} exists. Overwrite? (Y/n) ")
        go_ahead = res == "" or res.lower() == "y"

    if not os.path.exists(args.log_file) or go_ahead:
        main(args.log_file, args.load_model, args.load_opt)
