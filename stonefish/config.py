import fastchessenv.env
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from yamlargs import expose_module, make_lazy_constructor

import stonefish.dataset
import stonefish.env
import stonefish.eval.agent
import stonefish.eval.base
import stonefish.policy
import stonefish.rep
import stonefish.resnet
import stonefish.rl
import stonefish.train
import stonefish.train.base
from stonefish.dataset import default_collate_fn


def expose_modules():
    expose_module(optim)
    expose_module(stonefish.dataset)
    expose_module(stonefish.policy)
    expose_module(stonefish.rep)
    expose_module(stonefish.resnet)
    expose_module(stonefish.train)
    expose_module(stonefish.train.base)
    expose_module(stonefish.eval.base)
    expose_module(stonefish.eval.agent)
    expose_module(fastchessenv.env)
    expose_module(stonefish.env)
    expose_module(stonefish.rl)
    make_lazy_constructor(nn.Linear)
    make_lazy_constructor(DataLoader, {"collate_fn": default_collate_fn})
