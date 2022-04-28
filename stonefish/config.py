import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from yamlargs import expose_module
from yamlargs import make_lazy_constructor

from stonefish.dataset import default_collate_fn

import stonefish.train.base
import stonefish.eval.base
import stonefish.dataset
import stonefish.model
import stonefish.rep
import stonefish.ttt
import stonefish.rl
import stonefish.env
import chessenv.env

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

make_lazy_constructor(nn.Linear)
make_lazy_constructor(DataLoader, {"collate_fn": default_collate_fn})
