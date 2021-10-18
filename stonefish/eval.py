import os
import time
import argparse

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from stonefish.model import Model
from stonefish.dataset import TokenizedChess
from constants import MOVE_TOKENS, BOARD_TOKENS

from rich.progress import track

def eval_model(model, data):

    dataloader = DataLoader(data, batch_size=1024, drop_last=True, shuffle=True)

    correct = 0.0
    total = 0.0
    for (batch_idx, (s, a)) in track(enumerate(dataloader), "Testing...", total=20):
        infer = model.inference(s)
        labels = torch.flatten(a).to(infer.device)
        
        correct += torch.sum((infer.flatten() == labels)).float()
        total += (2.0 * s.shape[0])

        if batch_idx == 20:
            break

    return correct.item() / total

def main(load_model):

    data = TokenizedChess("test.csv")

    device = torch.device("cuda")
    model = Model(device, 512)
    model = model.to(model.device)

    model.load_state_dict(torch.load(load_model))

    acc = eval_model(model, data)
    print(acc)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("load_model")
    args = parser.parse_args()
    
    main(args.load_model)
