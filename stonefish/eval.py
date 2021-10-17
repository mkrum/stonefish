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

def main(load_model):

    data = TokenizedChess("test.csv")
    dataloader = DataLoader(data, batch_size=256, drop_last=True)

    device = torch.device("cuda")
    model = Model(device, 512)
    model = model.to(model.device)

    model.load_state_dict(torch.load(load_model))
    
    correct = 0.0
    total = 0.0
    for (batch_idx, (s, a)) in enumerate(dataloader):
        s = s.to(device)
        labels = torch.flatten(a).to(device)
        infer = model.inference(s)
        
        correct += torch.sum((infer.flatten() == labels)).float()
        total += (2.0 * s.shape[0])
        print(correct / total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("load_model")
    args = parser.parse_args()
    
    main(args.load_model)
