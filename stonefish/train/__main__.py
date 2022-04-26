from yamlargs.parser import load_config_and_create_parser, parse_args_into_config
from mllg import LogWriter

from stonefish.train import TrainingContext, train_step
from stonefish.config import load_model
from stonefish.eval import eval_model

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

    model = load_model(config)
    opt = config["opt"](model.parameters())

    train_dl = config["train_dl"]()
    test_dl = config["test_dl"]()

    ctx = TrainingContext(eval_model, train_step, train_dl, test_dl)
    ctx(logger, model, opt)
