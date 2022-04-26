import sys
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Any, Dict

import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from yamlargs import expose_module

from stonefish.dataset import default_collate_fn
import stonefish.dataset
import stonefish.model
import stonefish.rep
import stonefish.ttt

from yamlargs import make_lazy_constructor, make_lazy_function, YAMLConfig

expose_module(optim)
expose_module(stonefish.dataset)
expose_module(stonefish.model)
expose_module(stonefish.rep)
expose_module(stonefish.ttt)


def load_model(config, load=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config["model"](device, config["input_rep"](), config["output_rep"]())
    model = model.to(device)

    if load:
        model.load_state_dict(torch.load(load, map_location=device))

    return model


make_lazy_constructor(DataLoader, {"collate_fn": default_collate_fn})
