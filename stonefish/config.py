import importlib
import logging

import torch.optim as optim
from yamlargs import expose_module

import stonefish.binary_dataset
import stonefish.eval.base
import stonefish.resnet
import stonefish.train
import stonefish.train.base

logger = logging.getLogger(__name__)

# Modules that depend on fastchessenv C libs (may not be available locally).
# They are loaded lazily so training with binary data works without fastchessenv.
_OPTIONAL_MODULES = [
    "stonefish.rep",
    "stonefish.env",
    "stonefish.rl",
    "stonefish.policy",
    "stonefish.eval.agent",
]


def expose_modules():
    expose_module(optim)
    expose_module(stonefish.binary_dataset)
    expose_module(stonefish.resnet)
    expose_module(stonefish.train)
    expose_module(stonefish.train.base)
    expose_module(stonefish.eval.base)

    for mod_name in _OPTIONAL_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            expose_module(mod)
        except ImportError:
            logger.warning(f"Skipping {mod_name} (fastchessenv C libs not available)")
