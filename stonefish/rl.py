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
from stonefish.eval.base import eval_perf


def generate_rollout(env, model, n_steps, initial_state, legal_mask):

    state = initial_state

    history = RolloutTensor.empty()

    for _ in range(n_steps):
        with torch.no_grad():
            action = model.sample(state, move_mask=legal_mask)

        next_state, next_legal_mask, reward, done = env.step(action)

        history = history.add(
            state,
            action,
            next_state,
            reward,
            done,
            legal_mask,
        )

        state = next_state
        legal_mask = next_legal_mask

    return history, state, legal_mask


@dataclass
class RLContext:

    steps: int
    iters: int = int(1e5)
    selfplay: bool = False
    eval_freq: int = 100

    def __call__(self, logger, model, opt, env):

        eval_perf(model)

        state, legal_mask = env.reset()
        progress = deque(maxlen=1000)
        pl_hist = deque(maxlen=1000)
        vl_hist = deque(maxlen=1000)

        for it in range(int(self.iters)):

            history, state, legal_mask = generate_rollout(
                env, model, self.steps, state, legal_mask
            )

            history = history.to(model.device)

            decay_values = model.Q_value(
                history.next_state[:, -1], history.action[:, -1]
            )

            if self.selfplay:
                history.selfplay_decay_(0.99, decay_values.flatten())
            else:
                history.decay_(0.99, decay_values.flatten())

            (
                flat_state,
                flat_next_state,
                flat_action,
                flat_reward,
                flat_done,
                flat_mask,
            ) = history.get_data()

            full_logits, values = model(flat_state, flat_action, logit_mask=flat_mask)
            logits = torch.gather(full_logits, 1, flat_action)

            opt.zero_grad()

            value_loss = F.mse_loss(values, flat_reward)
            policy_loss = -1.0 * torch.mean(flat_reward * logits)

            loss = value_loss + policy_loss
            loss.backward()
            opt.step()

            term_rewards = history.reward[history.done]
            for r in term_rewards:
                progress.append(r.item())

            pl_hist.append(policy_loss.item())
            vl_hist.append(value_loss.item())

            print(
                f"{it}: PL: {np.mean(pl_hist)} VL: {np.mean(vl_hist)} R: {np.mean(progress)} W: {np.sum(np.array(progress) == 1.0)/len(progress)}",
                flush=True,
            )
            if (it % self.eval_freq == 0) and (it > 0):
                eval_perf(model)
                logger.checkpoint(it, 0, model)


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
    ctx = config["rl_context"]()
    ctx(logger, model, opt, env)
