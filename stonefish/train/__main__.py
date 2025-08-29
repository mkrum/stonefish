import os

import torch
import wandb
from mllg import LogWriter
from yamlargs.parser import load_config_and_create_parser, parse_args_into_config

import stonefish.config

if __name__ == "__main__":

    stonefish.config.expose_modules()

    config, parser = load_config_and_create_parser()
    parser.add_argument("log_path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    config = parse_args_into_config(config, args)

    # Set logging level based on debug flag
    import logging

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Determine device based on distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        if local_rank >= 0:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use Metal Performance Shaders (Apple Silicon GPU) if available
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for training")
    else:
        device = torch.device("cpu")

    # Only create logger on main process for distributed training
    is_main_process = local_rank == 0
    if is_main_process:
        # Initialize wandb
        wandb.init(project="stonefish")

        logger = LogWriter(args.log_path)
        config_data = config.to_json()
        config_data["type"] = "config"
        logger.log_str(str(config_data))

        with open(f"{args.log_path}/config.yml", "w") as cfg_save:
            cfg_save.write(config.to_yaml())
    else:
        # Create a dummy logger for non-main processes
        logger = LogWriter(args.log_path, log_proc=False)

    model = config["model"]().to(device)
    opt = config["opt"](model.parameters())

    ctx = config["pretrain_context"]()
    ctx(logger, model, opt)
