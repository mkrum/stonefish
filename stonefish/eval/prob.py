import torch
from stonefish.eval import prob_vis
from stonefish.config import (
    load_model,
    load_config_and_create_parser,
    parse_args_into_config,
)

if __name__ == "__main__":
    config, parser = load_config_and_create_parser()

    parser.add_argument("-N", default=1, type=int, help="Number of samples")
    parser.add_argument("-load", type=str, help="Path to model weights")
    args = parser.parse_args()
    config = parse_args_into_config(config, args)

    model = load_model(config, load=args.load)

    config["test_dl"]["batch_size"] = 16
    test_data = config["test_dl"]["dataset"]()
    prob_vis(model, test_data, args.N)
