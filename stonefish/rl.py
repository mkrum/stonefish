import os
from dataclasses import dataclass
from collections import deque
from typing import Any
import stonefish.config

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from mllg import LogWriter
from yamlargs.parser import load_config_and_create_parser, parse_args_into_config

from stonefish.mask import MoveMask
from stonefish.utils import RolloutTensor
from stonefish.rep import MoveRep


def generate_rollout(env, model, n_steps, initial_state, legal_mask):

    state = initial_state

    history = RolloutTensor.empty()

    for _ in range(n_steps):
        with torch.no_grad():
            action, legal_mask = model.sample(state, legal_mask)

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
    eval_fn: Any
    iters: int = int(1e5)
    selfplay: bool = False
    eval_freq: int = 100

    def __call__(self, logger, model, opt, env, rank, world_size):

        # if rank == 0:
        #    self.eval_fn(model, 0)

        dist.barrier()

        state, legal_mask = env.reset()
        progress = deque(maxlen=1000)
        pl_hist = deque(maxlen=100)
        vl_hist = deque(maxlen=100)

        for it in range(int(self.iters)):

            history, state, legal_mask = generate_rollout(
                env, model, self.steps, state, legal_mask
            )

            history = history.to(model.device)

            decay_values = model.value(state)

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

            logits, values = model(flat_state, flat_action, flat_mask)

            opt.zero_grad()

            value_loss = F.mse_loss(values, flat_reward)
            policy_loss = -1.0 * torch.mean((flat_reward - values) * logits)

            loss = value_loss + policy_loss
            loss.backward()

            if world_size > 1:
                for param in model.parameters():
                    dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
                    param.grad.data /= world_size

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
                if rank == 0:
                    self.eval_fn(model, it)
                    logger.checkpoint(it, 0, model)

                dist.barrier()


def run(rank, world_size, config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    model = config["model"](device).to(device)

    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.reduce_op.SUM)
        param.data /= world_size

    opt = optim.Adam(
        [
            {"params": model.policy.parameters(), "lr": 1e-4},
            {"params": model.V.parameters(), "lr": 1e-3},
        ],
    )

    env = config["env"]()
    ctx = config["rl_context"]()
    logger = LogWriter("/tmp/garbo")
    ctx(logger, model, opt, env, rank, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    config, parser = load_config_and_create_parser()
    parser.add_argument("log_path")
    parser.add_argument("--np", default=1)
    args = parser.parse_args()

    config = parse_args_into_config(config, args)

    logger = LogWriter(args.log_path)
    config_data = config.to_json()
    config_data["type"] = "config"
    logger.log_str(str(config_data))

    with open(f"{args.log_path}/config.yml", "w") as cfg_save:
        cfg_save.write(config.to_yaml())

    world_size = args.np
    if world_size > 1:
        mp.spawn(run, args=(world_size, config), nprocs=world_size, join=True)
    else:
        run(0, 1, config)
