import gc

import numpy as np
import torch
import torch.nn.functional
import torch.optim as optim
import tqdm
import wandb
from fastchessenv import CBoards, RandomChessEnv
from torch.distributions import Categorical

from stonefish.config import expose_modules
from stonefish.convert import board_to_lczero_tensor
from stonefish.eval.agent_loader import load_model_from_config
from stonefish.utils import RolloutTensor


def _pad_mask(mask):
    # Size of the tensor minus the size of actual valid moves
    size = 5700 - mask.shape[1]
    batch = mask.shape[0]
    pad = np.zeros((batch, size))
    return np.concatenate([mask, pad], axis=1)


def _state_to_tensor(states):
    states = states.flatten()
    boards = CBoards.from_array(states).to_board()
    boards = [board_to_lczero_tensor(b) for b in boards]
    boards_tensor = torch.Tensor(np.stack(boards))
    return boards_tensor


class FakeEnv:

    def __init__(self, n):
        self.envs = [RandomChessEnv(1, invert=True) for _ in range(n)]

    def step(self, actions):
        out = []
        for action, env in zip(actions, self.envs, strict=False):
            out.append(env.step(action.reshape(-1, 1)))

        out = zip(*out, strict=False)

        return map(np.concatenate, out)

    def reset(self):
        out = []
        for env in self.envs:
            out.append(env.reset())

        out = zip(*out, strict=False)

        return map(np.concatenate, out)


def generate_rollout_batch(env, next_boards_tensor, mask, model, num_steps=16):

    tensor = RolloutTensor.empty()
    device = next_boards_tensor.device

    for _ in tqdm.tqdm(range(num_steps)):

        boards_tensor = next_boards_tensor

        mask = _pad_mask(mask)
        mask_tensor = torch.Tensor(mask).to(device)

        with torch.no_grad():
            logits = model.forward(boards_tensor, None)

        legal_logits = logits * mask_tensor + (1 - mask_tensor) * -1e10

        probs = torch.nn.functional.softmax(legal_logits, dim=1)

        actions = Categorical(probs).sample()

        next_states, mask, rewards, done = env.step(actions.detach().cpu().numpy())

        next_boards_tensor = _state_to_tensor(next_states).to(device)

        tensor = tensor.add(
            boards_tensor,
            actions,
            next_boards_tensor,
            torch.Tensor(rewards).to(device),
            torch.Tensor(done).to(device),
            mask_tensor,
        )

    return tensor, next_boards_tensor, mask


def main():
    # Initialize wandb
    wandb.init(
        project="stonefish-rl",
        config={
            "lr": 1e-4,
            "num_envs": 32,
            "num_steps": 128,
            "gamma": 0.99,
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
        },
    )

    model = load_model_from_config("configs/train_convnet_big.yml", "model_3.pth")

    # Use MPS if available (Mac), otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.model = model.model.to(device)

    # Commented out for local testing
    # model.model = DistributedDataParallel(model.model)

    opt = optim.Adam(model.model.parameters(), lr=1e-4)

    env = FakeEnv(48)

    states, mask = env.reset()
    next_boards_tensor = _state_to_tensor(states).to(device)

    for step in range(10000):

        tensor, next_boards_tensor, mask = generate_rollout_batch(
            env, next_boards_tensor, mask, model.model, num_steps=128
        )

        # Commented out for local testing
        # dist.barrier()

        tensor.reward[tensor.done & (tensor.reward == 0)] = -1.0

        games_completed = tensor.done.sum().item()
        avg_reward = (
            tensor.reward[tensor.done].mean().item() if games_completed > 0 else 0.0
        )

        print(
            f"Step {step}: {games_completed} games completed, avg reward: {avg_reward:.3f}"
        )

        tensor.decay_(0.99)

        opt.zero_grad()

        (
            flat_state,
            flat_next_state,
            flat_action,
            flat_reward,
            flat_done,
            flat_mask,
        ) = tensor.get_data()

        completed_mask = (flat_reward != 0).flatten()
        flat_state = flat_state[completed_mask]
        flat_action = flat_action[completed_mask]
        flat_reward = flat_reward[completed_mask]
        flat_mask = flat_mask[completed_mask]

        if len(flat_state) == 0:
            print("No completed games, skipping update")
            continue

        logits = model.model(flat_state, None)

        # Apply mask to logits
        legal_logits = logits * flat_mask + (1 - flat_mask) * -1e10

        log_probs = torch.log_softmax(legal_logits, dim=1)
        sel_log_probs = log_probs.gather(1, flat_action).squeeze(1)

        policy_loss = -1.0 * torch.mean(flat_reward * sel_log_probs)

        policy_loss.backward()

        opt.step()

        # Log to wandb
        wandb.log(
            {
                "step": step,
                "games_completed": games_completed,
                "avg_reward": avg_reward,
                "policy_loss": policy_loss.item(),
                "completed_samples": len(flat_state),
                "wins": (tensor.reward[tensor.done] == 1).sum(),
                "losses": (tensor.reward[tensor.done] == -1).sum(),
                "avg_log_prob": sel_log_probs.mean().item(),
            }
        )

        # Save checkpoint every 100 steps
        if step % 100 == 0:
            checkpoint = {
                "step": step,
                "model_state_dict": model.model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "avg_reward": avg_reward,
            }
            torch.save(checkpoint, f"rl_checkpoint_{step}.pth")
            print(f"Saved checkpoint at step {step}")

        # Explicitly delete tensors to free memory
        del (
            tensor,
            flat_state,
            flat_action,
            flat_reward,
            flat_mask,
            logits,
            log_probs,
            sel_log_probs,
        )

        gc.collect()
        torch.mps.empty_cache()


if __name__ == "__main__":
    expose_modules()
    # Commented out for local testing
    # setup_distributed()
    main()
    # cleanup_distributed()
