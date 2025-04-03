import torch
from mllg import LogWriter
from yamlargs.parser import load_config_and_create_parser, parse_args_into_config

if __name__ == "__main__":

    config, parser = load_config_and_create_parser()
    parser.add_argument("log_path")
    args = parser.parse_args()

    config = parse_args_into_config(config, args)

    logger = LogWriter(args.log_path)
    config_data = config.to_json()
    config_data["type"] = "config"
    logger.log_str(str(config_data))

    with open(f"{args.log_path}/config.yml", "w") as cfg_save:
        cfg_save.write(config.to_yaml())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config["model"](device).to(device)
    opt = config["opt"](model.parameters())

    ctx = config["pretrain_context"]()
    ctx(logger, model, opt)
