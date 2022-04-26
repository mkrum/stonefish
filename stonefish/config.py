import sys
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Any, Dict

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from rich import print
from transformers import AutoTokenizer

from stonefish.dataset import CEChessData, ChessData, TTTData, default_collate_fn
from stonefish.slogging import Logger
from stonefish.model import BaseModel
from stonefish.rep import BoardRep, MoveRep, create_tokenizer_rep
from stonefish.ttt import TTTBoardRep, TTTMoveRep

from yamlargs import make_lazy_constructor, make_lazy_function, YAMLConfig


def load_model(config, load=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config["model"](device, config["input_rep"](), config["output_rep"]())
    model = model.to(device)

    if load:
        model.load_state_dict(torch.load(load, map_location=device))

    return model


def logging_constructor(loader, node):
    """
    YAML constructor for the Logging object

    Only difference here is that it will call the "init" for the global logger
    instead of really intializing anything.
    """
    value = loader.construct_mapping(node)
    Logger.init(**value)
    return Logger


# Logger YAML configuration
yaml.add_constructor("!Logger", logging_constructor)

# Lazy Objects
make_lazy_constructor(TTTData)
make_lazy_constructor(ChessData)
make_lazy_constructor(CEChessData)
make_lazy_constructor(BaseModel)
make_lazy_constructor(DataLoader, {"collate_fn": default_collate_fn})

for o in [
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "SparseAdam",
    "Adamax",
    "ASGD",
    "LBFGS",
    "RMSprop",
    "Rprop",
    "SGD",
]:
    make_lazy_constructor(getattr(optim, o))

# Type objects, interpreted as literal type
make_lazy_function(TTTBoardRep)
make_lazy_function(TTTMoveRep)
make_lazy_function(BoardRep)
make_lazy_function(MoveRep)
