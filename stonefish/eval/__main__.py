from stonefish.config import (
    load_config_and_create_parser,
    load_model,
    parse_args_into_config,
)
from stonefish.eval import eval_model
from stonefish.train import train_step

if __name__ == "__main__":
    config, parser = load_config_and_create_parser()

    parser.add_argument("-N", default=1, type=int, help="Number of samples")
    parser.add_argument("-load", type=str, help="Path to model weights")
    args = parser.parse_args()
    config = parse_args_into_config(config, args)

    model = load_model(config, load=args.load)

    config["test_dl"]["batch_size"] = 16
    test_data = config["test_dl"]()
    print(eval_model(model, test_data, train_step))
