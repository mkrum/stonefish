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

from stonefish.dataset import ChessData, TTTData, default_collate_fn
from stonefish.slogging import Logger
from stonefish.model import BaseModel
from stonefish.rep import BoardRep, MoveRep, create_tokenizer_rep
from stonefish.ttt import TTTBoardRep, TTTMoveRep
from stonefish.language import CommonGen


def load_model(config, load=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config["model"](device, config["input_rep"], config["output_rep"])
    model = model.to(device)

    if load:
        model.load_state_dict(torch.load(load, map_location=device))

    return model


@dataclass(frozen=True)
class LazyConstructor:
    """
    An object that wraps a constructor in a lazy, configurable way.

    It turns the constructor into a dictionary like object that allows you to
    set and overwrite the default values like a dictionary, and then intialize
    kind of like a factory.

    LazyConstructor should be intialized with a class and a dictionary of key
    word arguemnts. It will then represent a lazy version of lambda *args:
    class(*args, **kwargs).

    If a LazyConstructor depends on a key word argument that is also a
    LazyConstructor, it will recursively intitialize it before intializing
    itself.

    >>> class Dice:
    ...       def __init__(self, value, max_value=6):
    ...               self.value = value
    ...               if self.value > max_value:
    ...                       print("Bad value")
    ...       def __str__(self):
    ...               return f"Dice({self.value})"
    ...       def __repr__(self):
    ...               return self.__str__()
    ...
    >>> c = LazyConstructor(Dice, {"max_value": 6})
    >>> c(1)
    Dice(1)
    >>> c(10)
    Bad value
    Dice(10)
    >>> c['max_value'] = 12
    >>> c(10)
    Dice(10)
    >>> print(c)
    {'class': <class '__main__.Dice'>, 'max_value': 12}
    >>> c = LazyConstructor(Dice, {"max_value": 6, "value": 1})
    >>> c()
    Dice(1)
    >>> c['value'] = 4
    >>> c()
    Dice(4)
    """

    _fn: Any
    _kwargs: Dict

    def __getitem__(self, key):
        if key == "class":
            return self._fn
        else:
            return self._kwargs.__getitem__(key)

    def __setitem__(self, key, value):
        if key == "class":
            self._fn = value
        else:
            self._kwargs.__setitem__(key, value)

    def __call__(self, *args):

        # If there are any LazyConstructors in the keyword arguments, intialize
        # them before intializing itself. Allows for a recursive chain of intialization.
        for (k, v) in self._kwargs.items():
            if isinstance(v, LazyConstructor):
                self._kwargs[k] = v()

        return self._fn(*args, **self._kwargs)

    def __repr__(self):
        d = {"class": self._fn}
        d.update(self._kwargs)
        return str(d)

    def __str__(self):
        return self.__repr__()

    def keys(self):
        """Fake keys to act like a dictionary"""
        return ["class"] + list(self._kwargs.keys())

    def values(self):
        """Fake values to act like a dictionary"""
        return [self._fn] + list(self._kwargs.values())

    def items(self):
        """Fake items to act like a dictionary"""
        keys = self.keys()
        return [(k, self.__getitem__(k)) for k in keys]


def make_lazy_constructor(type_, name, default_kwargs=None):
    """
    Exposes to YAML a lazy constructor for the object, so it can be referenced
    in a config.

    >>> make_lazy_constructor(Dice, "Dice")
    >>> data = yaml.load("dice: !Dice")
    >>> data["dice"]()
    Dice()
    """

    def _constructor(loader, node):
        kwargs = loader.construct_mapping(node)

        # Surely, there must be a better way to do this, right? I tried .update
        # but that was casuing weird issues.

        if default_kwargs is not None:
            for (k, v) in default_kwargs.items():
                if k not in kwargs.keys():
                    kwargs[k] = v

        return LazyConstructor(type_, kwargs)

    yaml.add_constructor("!" + name, _constructor)


def logging_constructor(loader, node):
    """
    YAML constructor for the Logging object

    Only difference here is that it will call the "init" for the global logger
    instead of really intializing anything.
    """
    value = loader.construct_mapping(node)
    Logger.init(**value)
    return Logger


def make_type_constructor(type_, name):
    """
    Exposes to YAML the unwrapped class

    >>> make_type_constructor(Dice, "Dice")
    >>> data = yaml.load("dice: !Dice")
    >>> data["dice"]
     <class '__main__.Dice'>
    """

    def _constructor(loader, node):
        return type_

    yaml.add_constructor("!" + name, _constructor)


def load_config(path):
    out = yaml.load(open(path, "r"), yaml.UnsafeLoader)
    return out


def get_all_keys(config):
    keys = []
    for (k, v) in config.items():
        if hasattr(v, "keys"):
            subkeys = get_all_keys(v)
            for sk in subkeys:
                keys.append(".".join([k, sk]))
        else:
            keys.append(k)

    return keys


def dot_access(nested_dict, dot_key):
    keys = dot_key.split(".")
    d = nested_dict
    for k in keys:
        d = d[k]
    return d


def dot_set(nested_dict, dot_key, value):
    keys = dot_key.split(".")
    d = nested_dict
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    return d


def create_parser(path, config):
    key_pairs = get_all_keys(config)

    parser = ArgumentParser(
        description=f"""
This argument parser was autogenerated from the file {path}. This allows you to
overwrite specific YAML values on the fly. The options listed here do not
entail an exhaustive list of the things that you can configure. For more
information on possible kwargs, refer to the class definition of the object in
question. 
    """
    )

    parser.add_argument(f"config_file", help="YAML config file")
    for k in key_pairs:
        current = dot_access(config, k)
        parser.add_argument(f"--{k}", type=type(current))

    return parser


def load_config_and_create_parser():
    path = sys.argv[1]
    config = load_config(path)
    parser = create_parser(path, config)
    return config, parser


def parse_args_into_config(config, args, verbose=True):

    for (k, v) in vars(args).items():
        if v and k != "config_file":
            dot_set(config, k, v)

    if verbose:
        print(config)

    return config


def load_config_and_parse_cli(verbose=True):
    config, parser = load_config_and_create_parser()
    args = parser.parse_args()
    config = parse_args_into_config(config, args, verbose=verbose)
    return config


# Lazy Objects
make_lazy_constructor(TTTData, "TTTData")
make_lazy_constructor(ChessData, "ChessData")
make_lazy_constructor(BaseModel, "BaseModel")
make_lazy_constructor(DataLoader, "DataLoader", {"collate_fn": default_collate_fn})
make_lazy_constructor(CommonGen, "CommonGen")
make_lazy_constructor(create_tokenizer_rep, "BertBasedCase")

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
    make_lazy_constructor(getattr(optim, o), o)

# Logger YAML configuration
yaml.add_constructor("!Logger", logging_constructor)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", local_files_only=True)
BertBaseCased = create_tokenizer_rep("BertBaseCased", tokenizer)

# Type objects, interpreted as literal type
make_type_constructor(BertBaseCased, "BertBaseCase")
make_type_constructor(TTTBoardRep, "TTTBoardRep")
make_type_constructor(TTTMoveRep, "TTTMoveRep")
make_type_constructor(BoardRep, "BoardRep")
make_type_constructor(MoveRep, "MoveRep")
make_type_constructor(None, "None")
