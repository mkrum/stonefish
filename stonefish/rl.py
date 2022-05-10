import os
import copy
import random
from typing import Any
from dataclasses import dataclass
from collections import deque
import stonefish.config

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

from mllg import LogWriter, LossInfo, TrainStepInfo
from yamlargs.parser import load_config_and_create_parser, parse_args_into_config
from chessenv.sfa import SFArray

from stonefish.mask import MoveMask
from stonefish.utils import RolloutTensor
from stonefish.rep import MoveRep
from stonefish.display import RLDisplay


class RandomModel:

    def sample(self, state, tmasks):
        masks = tmasks.numpy()
        probs = masks / np.sum(masks, axis=1).reshape(-1, 1)
        actions = np.zeros(len(masks))
        for (i, p) in enumerate(probs):
            actions[i] = np.random.choice(9, p=p)
        return torch.LongTensor(actions).cuda(), tmasks.cuda()

class SFModel:

    def __init__(self, sfa):
        self.sfa = sfa

    def sample(self, state, tmasks):
        moves = self.sfa.get_move_ints(state)

        import pdb; pdb.set_trace()

        masks = tmasks.numpy()
        probs = masks / np.sum(masks, axis=1).reshape(-1, 1)
        actions = np.zeros(len(masks))
        for (i, p) in enumerate(probs):
            actions[i] = np.random.choice(9, p=p)
        return torch.LongTensor(actions).cuda(), tmasks.cuda()



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


def mulit_player_generate_rollout(
    env, model_one, model_two, n_steps, initial_state, legal_mask, player_id
):

    state = initial_state

    history = RolloutTensor.empty()

    for _ in range(n_steps):

        player_id = (player_id + 1) % 2

        assert torch.sum(state[:, 64] == 14.0) == state.shape[0]

        if player_id == 0:
            model = model_one
        elif player_id == 1:
            model = model_two

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

    return history, state, legal_mask, player_id


def mulit_player_sample_generate_rollout(
    env, model_one, models, n_steps, initial_state, legal_mask, player_id
):

    state = initial_state

    history = RolloutTensor.empty()

    for _ in range(n_steps):

        player_id = (player_id + 1) % 2

        if player_id == 0:
            model = model_one
        elif player_id == 1:
            model = random.sample(models, 1)[0]

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

    return history, state, legal_mask, player_id


@dataclass
class RLContext:

    steps: int
    eval_fn: Any
    iters: int = int(1e5)
    selfplay: bool = False
    eval_freq: int = 100
    gamma: float = 0.99
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    policy_weight: float = 0.01


    def get_data(self, env, model, state, legal_mask):
        history, state, legal_mask = generate_rollout(
            env, model, self.steps, state, legal_mask
        )

        history = history.to(model.device)

        decay_values = model.value(state)

        if self.selfplay:
            history.selfplay_decay_(self.gamma, decay_values.flatten().detach())
        else:
            history.decay_(self.gamma, decay_values.flatten().detach())

        return history, state, legal_mask

    def compute_loss(self, opt,  model, state, mask, action, reward):
        opt.zero_grad()

        logits, values = model(state, action, mask)

        value_loss = F.mse_loss(values, reward)
        policy_loss = -1.0 * torch.mean((reward - values.detach()) * logits)
        entropy_loss = -torch.mean(torch.sum(logits * torch.exp(logits), dim=-1))

        loss = (
            policy_loss
            + self.value_weight * value_loss
            + self.entropy_weight * entropy_loss
        )

        losses = [
            LossInfo("policy", policy_loss.item()),
            LossInfo("value", value_loss.item()),
            LossInfo("entropy", entropy_loss.item()),
        ]

        loss.backward()
        self.sync_gradients(model, world_size)
        opt.step()

        return losses

    def sync_gradients(self, model, world_size):

        if world_size > 1:
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
                param.grad.data /= world_size

    def __call__(self, logger, model, opt, env, rank, world_size):

        #if rank == 0:
        #    out = self.eval_fn(model, 0)
        #    logger.log_info(out)

        dist.barrier()

        state, legal_mask = env.reset()

        for it in range(int(self.iters)):
            history, state, legal_mask = self.get_data(env, model, state, legal_mask)

            (
                flat_state,
                flat_next_state,
                flat_action,
                flat_reward,
                flat_done,
                flat_mask,
            ) = history.get_data()


            loss_info = self.compute_loss(
                opt, model, flat_state, flat_mask, flat_action, flat_reward
            )
            reward_info = compute_reward_info(history)
            info = TrainStepInfo(0, it, loss_info + reward_info)


            logger.log_info(info)

            if (it % self.eval_freq == 0) and (it > 0):
                if rank == 0:
                    out = self.eval_fn(model, it)
                    logger.log_info(out)
                    logger.checkpoint(it, 0, model)

                dist.barrier()


def polyak_update(polyak_factor, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(
            polyak_factor * param.data + target_param.data * (1.0 - polyak_factor)
        )


@dataclass
class TwoModelRLContext(RLContext):

    steps: int
    eval_fn: Any
    iters: int = int(1e5)
    eval_freq: int = 100
    gamma: float = 0.99
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    polyak_factor: float = 0.001

    def get_data(self, env, model, twin_model, state, legal_mask, player_id):
        history, state, legal_mask, player_id = mulit_player_generate_rollout(
            env, model, twin_model, self.steps, state, legal_mask, player_id
        )

        history = history.to(model.device)

        decay_values = model.value(state)

        history.selfplay_decay_(self.gamma, decay_values.flatten().detach())

        history = history[:, ::2]
        return history, state, legal_mask, player_id

    def __call__(self, logger, model, opt, env, rank, world_size):

        twin_model = copy.deepcopy(model)

        if rank == 0:
            out = self.eval_fn(model, 0)
            logger.log_info(out)

        dist.barrier()

        state, legal_mask = env.reset()

        player_id = 1
        for it in range(int(self.iters)):
            history, state, legal_mask, player_id = self.get_data(
                env, model, twin_model, state, legal_mask, player_id
            )

            (
                flat_state,
                flat_next_state,
                flat_action,
                flat_reward,
                flat_done,
                flat_mask,
            ) = history.get_data()

            assert torch.sum(flat_state[:, 64] == 14.0) == flat_state.shape[0]

            opt.zero_grad()

            loss, loss_info = self.compute_loss(
                model, flat_state, flat_mask, flat_action, flat_reward
            )
            reward_info = compute_reward_info(history)
            info = TrainStepInfo(0, it, loss_info + reward_info)

            loss.backward()
            self.sync_gradients(model, world_size)

            opt.step()

            logger.log_info(info)

            polyak_update(self.polyak_factor, model, twin_model)

            if (it % self.eval_freq == 0) and (it > 0):
                if rank == 0:
                    out = self.eval_fn(model, it)
                    logger.log_info(out)
                    logger.checkpoint(it, 0, model)

                dist.barrier()


def run(rank, world_size, config, log_path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12347"

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

    log_proc = rank == 0
    logger = LogWriter(log_path, log_proc=log_proc, display=RLDisplay)

    ctx(logger, model, opt, env, rank, world_size)

    dist.destroy_process_group()

@dataclass
class SLRLContext(RLContext):

    steps: int
    eval_fn: Any
    iters: int = int(1e5)
    eval_freq: int = 100
    gamma: float = 0.99
    value_weight: float = 0.5
    policy_weight: float = 1.0
    sl_weight: float = 1.0
    entropy_weight: float = 0.01
    polyak_factor: float = 0.001

    _sfa: Any = SFArray(1)

    def compute_loss(self, opt, model, state, mask, action, reward):

        opt.zero_grad()

        labels = self._sfa.get_moves(state.cpu().numpy())
        labels = torch.LongTensor(labels).cuda()

        logits, values = model(state, action, mask)

        value_loss = F.mse_loss(values, reward)
        policy_loss = -1.0 * torch.mean((reward - values.detach()) * logits)
        entropy_loss = -torch.mean(torch.sum(logits * torch.exp(logits), dim=-1))

        first_loss = (
            self.policy_weight * policy_loss
            + self.value_weight * value_loss
            + self.entropy_weight * entropy_loss
        )
        first_loss.backward()

        labels = torch.clip(labels, 0, 128)
        full_logits = model.policy(state, labels)

        sl_logits = full_logits.reshape(-1, 129)
        labels = labels[:, 1:].flatten()

        sl_loss = self.sl_weight * F.cross_entropy(sl_logits, labels)
        sl_loss.backward()

        losses = [
            LossInfo("policy", self.policy_weight * policy_loss.item()),
            LossInfo("value", self.value_weight * value_loss.item()),
            LossInfo("entropy", self.entropy_weight * entropy_loss.item()),
            LossInfo("sl", sl_loss.item()),
        ]
        
        self.sync_gradients(model, 1)
        opt.step()

        return losses

class SLRLAgainstSFContext(SLRLContext):

    steps: int
    eval_fn: Any
    iters: int = int(1e5)
    eval_freq: int = 100
    gamma: float = 0.99
    value_weight: float = 0.5
    policy_weight: float = 1.0
    sl_weight: float = 1.0
    entropy_weight: float = 0.01
    polyak_factor: float = 0.001

    _sfa: Any = SFArray(1)

    def get_data(self, env, model, twin_model, state, legal_mask, player_id):
        sf_model = SFModel(self._sfa)

        history, state, legal_mask, player_id = mulit_player_generate_rollout(
            env, model, sf_model, self.steps, state, legal_mask, player_id
        )

        history = history.to(model.device)

        decay_values = model.value(state)

        history.selfplay_decay_(self.gamma, decay_values.flatten().detach())

        history = history[:, ::2]
        return history, state, legal_mask, player_id


def compute_reward_info(history):

    term_rewards = history.reward[history.done]

    wins = 0.0
    ties = 0.0
    losses = 0.0
    for r in term_rewards:

        if r.item() == 0.0:
            ties += 1.0
        elif r.item() == 1.0:
            wins += 1.0
        elif r.item() == -1.0:
            losses += 1.0

    outcomes = [
        LossInfo("wins", wins),
        LossInfo("losses", losses),
        LossInfo("ties", ties),
    ]
    return outcomes


if __name__ == "__main__":
    config, parser = load_config_and_create_parser()
    parser.add_argument("log_path")
    parser.add_argument("--np", type=int, default=1)
    args = parser.parse_args()

    config = parse_args_into_config(config, args)

    logger = LogWriter(args.log_path, log_proc=False)
    config_data = config.to_json()
    config_data["type"] = "config"
    logger.log_str(str(config_data))

    with open(f"{args.log_path}/config.yml", "w") as cfg_save:
        cfg_save.write(config.to_yaml())

    world_size = args.np
    if world_size > 1:
        mp.spawn(
            run, args=(world_size, config, args.log_path), nprocs=world_size, join=True
        )
    else:
        run(0, 1, config, args.log_path)
