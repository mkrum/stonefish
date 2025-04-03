import argparse

import chessenv.env
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from yamlargs import expose_module, make_lazy_constructor

import stonefish.dataset
import stonefish.env
import stonefish.eval.base
import stonefish.model
import stonefish.rep
import stonefish.rl
import stonefish.train.base
import stonefish.ttt
from stonefish.dataset import default_collate_fn

expose_module(optim)
expose_module(stonefish.dataset)
expose_module(stonefish.model)
expose_module(stonefish.rep)
expose_module(stonefish.ttt)
expose_module(stonefish.train.base)
expose_module(stonefish.eval.base)
expose_module(chessenv.env)
expose_module(stonefish.env)
expose_module(stonefish.rl)


def load_config_and_create_parser():
    parser = argparse.ArgumentParser(description="Stonefish training")
    parser.add_argument("--config", type=str, help="Configuration file")

    config = {}
    return config, parser


def parse_args_into_config(config, args):
    if args.config:
        with open(args.config, "r") as f:
            config.update(yaml.safe_load(f))

    # Update config with any command line arguments that match config keys
    for key, value in vars(args).items():
        if key in config and value is not None:
            config[key] = value

    return config


def load_model(config, load=None):
    model_config = config.get("model", {})
    model = stonefish.model.BaseModel(**model_config)

    if load:
        model.load_state_dict(torch.load(load))

    return model


make_lazy_constructor(nn.Linear)
make_lazy_constructor(DataLoader, {"collate_fn": default_collate_fn})
