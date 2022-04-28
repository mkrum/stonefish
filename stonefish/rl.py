from dataclasses import dataclass
from collections import deque
from typing import Any
import stonefish.config

import torch
import torch.nn.functional as F
import numpy as np

from mllg import LogWriter
from yamlargs.parser import load_config_and_create_parser, parse_args_into_config

from stonefish.mask import MoveMask
from stonefish.utils import RolloutTensor

from stonefish.rep import MoveRep


def batched_index_select(input, dim, index):
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def generate_rollout(env, model, n_steps, initial_state, legal_mask):

    state = initial_state

    history = RolloutTensor.empty()

    for _ in range(n_steps):
        action = model.sample(state, move_mask=legal_mask)

        next_state, next_legal_mask, reward, done = env.step(action)

        reward = reward.view(-1, 1)
        done = done.view(-1, 1).bool()

        action = action.unsqueeze(1)
        history = history.add(
            state.unsqueeze(1),
            action,
            next_state.unsqueeze(1),
            reward,
            done,
            legal_mask.unsqueeze(1),
        )

        state = next_state
        legal_mask = next_legal_mask

    return history, state, legal_mask


@dataclass
class RLContext:

    steps: int = 16
    iters: int = int(1e5)

    def __call__(self, logger, model, opt, env):

        state, legal_mask = env.reset()
        progress = deque(maxlen=100)
        pl_hist = deque(maxlen=100)
        vl_hist = deque(maxlen=100)

        for it in range(int(self.iters)):

            history, state, legal_mask = generate_rollout(
                env, model, self.steps, state, legal_mask
            )

            history = history.to(model.device)

            decay_values = model.Q_value(history.state[:, -1], history.action[:, -1])

            history.decay_(0.99, decay_values.flatten())

            term_rewards = history.reward[history.done]
            for r in term_rewards:
                progress.append(r.item())

            flat_state = history.state.view(-1, history.state.shape[-1])
            flat_next_state = history.next_state.view(-1, history.next_state.shape[-1])
            flat_action = history.action.view(-1, *history.action.shape[2:])
            flat_reward = history.reward.view(-1)

            flat_mask = history.mask.view(-1, *history.mask.shape[2:])

            full_logits, values = model(flat_state, flat_action, logit_mask=flat_mask)

            logits = torch.gather(full_logits, 1, flat_action.unsqueeze(1))

            opt.zero_grad()

            value_loss = F.mse_loss(values, flat_reward.unsqueeze(-1))
            policy_loss = -1.0 * torch.mean(flat_reward.unsqueeze(-1).cuda() * logits)

            loss = value_loss + policy_loss
            loss.backward()
            opt.step()

            pl_hist.append(policy_loss.item())
            vl_hist.append(value_loss.item())

            print(
                f"{it}: PL: {np.mean(pl_hist)} VL: {np.mean(vl_hist)} R: {np.mean(progress)} W: {np.sum(np.array(progress) == 1.0)/len(progress)}",
                flush=True,
            )


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

    env = config["env"]()
    ctx = RLContext()
    ctx(logger, model, opt, env)
