from stonefish.train import TrainingContext, train_step
from stonefish.config import load_model, load_config_and_parse_cli
from stonefish.eval import eval_model

if __name__ == "__main__":
    config = load_config_and_parse_cli()

    model = load_model(config)
    opt = config["opt"](model.parameters())

    train_dl = config["train_dl"]()
    test_dl = config["test_dl"]()

    ctx = TrainingContext(eval_model, train_step, train_dl, test_dl)
    ctx(model, opt)
