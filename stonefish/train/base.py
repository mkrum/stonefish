import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from mllg import TestInfo, TrainInfo, ValidationInfo
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from stonefish.mask import MoveMask


def train_step(model, state, output):
    model.train()

    probs = model(state, output)
    loss = functional.cross_entropy(probs, output.to(probs.device).flatten())

    # Calculate accuracy
    predictions = probs.argmax(dim=-1)
    targets = output.to(probs.device).flatten()
    accuracy = (predictions == targets).float().mean()

    return loss, accuracy


def seq_train_step(model, state, output):
    model.train()

    probs = model(state, output)

    probs = probs.reshape(-1, probs.shape[-1])

    output = output[:, 1:].reshape(
        -1,
    )
    probs = probs[output != -1]
    output = output[output != -1]

    loss = functional.cross_entropy(probs, output.flatten().to(probs.device))
    return loss


def mask_train_step(model, state, output):
    model.train()

    mm = MoveMask.from_data(state, output)

    masks = mm.update_mask(output)
    masks = masks[:, 1:, :]

    probs = model(state, output, logit_mask=masks.cuda())

    probs = probs.reshape(-1, probs.shape[-1])

    output = output[:, 1:].reshape(
        -1,
    )
    probs = probs[output != -1]
    output = output[output != -1]

    loss = functional.cross_entropy(probs, output.flatten().to(probs.device))
    return loss


def setup_distributed():
    """Initialize distributed training if available."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True

    return 0, 1, False


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class PreTrainContext:

    eval_fn: Any
    train_fn: Any
    train_dl: Any
    test_dl: Any
    agent_eval_fn: Any
    epochs: int = 1000
    eval_freq: int = 5000
    gradient_clip: float = 1.0

    def __call__(self, logger, model, opt):
        # Setup distributed training if available
        local_rank, world_size, is_distributed = setup_distributed()
        is_main_process = local_rank == 0

        # Wrap model in DDP if distributed
        if is_distributed:
            model = DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )

        # Create distributed samplers if needed
        train_sampler = None
        if is_distributed and hasattr(self.train_dl.dataset, "__len__"):
            # Create distributed sampler for training data
            train_sampler = DistributedSampler(
                self.train_dl.dataset,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=True,
            )
            # Create new DataLoader with distributed sampler
            self.train_dl = DataLoader(
                dataset=self.train_dl.dataset,
                batch_size=self.train_dl.batch_size,
                sampler=train_sampler,
                num_workers=getattr(self.train_dl, "num_workers", 0),
                collate_fn=self.train_dl.collate_fn,
                pin_memory=getattr(self.train_dl, "pin_memory", False),
                drop_last=getattr(self.train_dl, "drop_last", False),
            )

            # Create distributed sampler for test data if available
            if self.test_dl is not None:
                test_sampler = DistributedSampler(
                    self.test_dl.dataset,
                    num_replicas=world_size,
                    rank=local_rank,
                    shuffle=False,
                )
                self.test_dl = DataLoader(
                    dataset=self.test_dl.dataset,
                    batch_size=self.test_dl.batch_size,
                    sampler=test_sampler,
                    num_workers=getattr(self.test_dl, "num_workers", 0),
                    collate_fn=self.test_dl.collate_fn,
                    pin_memory=getattr(self.test_dl, "pin_memory", False),
                    drop_last=getattr(self.test_dl, "drop_last", False),
                )

        # Initial evaluation
        if is_main_process and self.eval_fn:
            out = self.eval_fn(model, self.test_dl, self.train_fn)
            logger.log_info(ValidationInfo(0, 0, out))

        if is_distributed:
            dist.barrier()

        for epoch in range(self.epochs):
            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch_idx, (state, output) in enumerate(self.train_dl):
                opt.zero_grad()

                loss, accuracy = self.train_fn(model, state, output)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                opt.step()

                # Only log from main process
                if is_main_process:
                    logger.log_info(TrainInfo(epoch, batch_idx, loss.item()))

                    # Log training metrics to wandb
                    wandb.log(
                        {"train_loss": loss.item(), "train_acc": accuracy.item()},
                        step=epoch * len(self.train_dl) + batch_idx,
                    )

                    if (
                        batch_idx % self.eval_freq == 0
                        and batch_idx > 0
                        and self.eval_fn
                    ):
                        out = self.eval_fn(model, self.test_dl, self.train_fn)
                        logger.log_info(ValidationInfo(epoch, batch_idx, out))
                        logger.checkpoint(epoch, batch_idx, model)

                        # Log validation metrics to wandb
                        val_metrics = {}
                        for test_info in out:
                            val_metrics[f"val_{test_info.loss_type}"] = test_info.loss
                        wandb.log(
                            val_metrics, step=epoch * len(self.train_dl) + batch_idx
                        )

                # Synchronize processes if distributed
                if is_distributed and batch_idx % self.eval_freq == 0:
                    dist.barrier()

            # End of epoch evaluation
            if is_main_process:
                if self.eval_fn is not None:
                    out = self.eval_fn(model, self.test_dl, self.train_fn)
                    logger.log_info(ValidationInfo(epoch, batch_idx, out))

                    # Log validation metrics to wandb at end of epoch
                    val_metrics = {}
                    for test_info in out:
                        val_metrics[f"val_{test_info.loss_type}"] = test_info.loss
                    wandb.log(val_metrics, step=epoch * len(self.train_dl) + batch_idx)

                logger.checkpoint(epoch, batch_idx, model)

                # Agent evaluation at end of epoch
                # Unwrap DistributedDataParallel before passing to agent eval
                model_unwrapped = model.module if hasattr(model, "module") else model
                agent_results = self.agent_eval_fn(model_unwrapped, epoch)

                # Separate regular metrics from HTML content
                regular_metrics = {}
                html_content = {}

                for key, value in agent_results.items():
                    if key.startswith("eval/PGN"):
                        html_content[key] = value
                    else:
                        regular_metrics[key] = value
                        logger.log_info(TestInfo(key, value))

                # Log to wandb (using same step as validation)
                if regular_metrics:
                    wandb.log(
                        regular_metrics, step=epoch * len(self.train_dl) + batch_idx
                    )
                if html_content:
                    wandb.log(html_content, step=epoch * len(self.train_dl) + batch_idx)

            if is_distributed:
                dist.barrier()

        # Cleanup distributed training
        if is_distributed:
            cleanup_distributed()
