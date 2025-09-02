import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn
import torch.nn.functional as functional
from mllg import TestInfo, TrainInfo, ValidationInfo
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from stonefish.mask import MoveMask

# Set up logger
train_logger = logging.getLogger(__name__)
# Default to WARNING level - will be overridden if debug is enabled
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


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

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
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
    compile_model: bool = False
    compile_mode: str = "default"

    def __call__(self, logger, model, opt):
        # Setup distributed training if available
        local_rank, world_size, is_distributed = setup_distributed()
        is_main_process = local_rank == 0

        # Compile model if requested
        if self.compile_model:
            train_logger.info(f"Compiling model with mode='{self.compile_mode}'")
            model = torch.compile(model, mode=self.compile_mode)

        # Inject the model's board_tokenizer into the datasets
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.train_dl.dataset.board_tokenizer = base_model.board_tokenizer
        if self.test_dl:
            self.test_dl.dataset.board_tokenizer = base_model.board_tokenizer

        # Log dataset configuration
        train_logger.info(
            f"Training dataset: streaming={self.train_dl.dataset.streaming}, "
            f"type={type(self.train_dl.dataset).__name__}, "
            f"num_workers={self.train_dl.num_workers}, "
            f"prefetch_factor={self.train_dl.prefetch_factor}, "
            f"batch_size={self.train_dl.batch_size}"
        )

        # Wrap model in DDP if distributed
        if is_distributed:
            torch.cuda.set_device(local_rank)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )

        # Create distributed samplers if needed
        train_sampler = None

        if (
            is_distributed
            and hasattr(self.train_dl.dataset, "__len__")
            and not self.train_dl.dataset.streaming
        ):
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
            train_logger.info(f"Starting epoch {epoch}")
            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_logger.info("About to iterate over training dataloader")
            train_logger.info(
                f"DataLoader config: num_workers={self.train_dl.num_workers}, "
                f"prefetch_factor={self.train_dl.prefetch_factor}, "
                f"batch_size={self.train_dl.batch_size}"
            )

            # Track if this is first iteration (when workers spawn)
            epoch_start = time.time()
            first_batch_time = None

            train_logger.info("Creating dataloader iterator...")
            iter_create_start = time.time()
            train_iter = iter(self.train_dl)
            iter_create_time = time.time() - iter_create_start
            train_logger.info(f"Iterator created in {iter_create_time:.2f}s")

            batch_end = time.time()  # Initialize for first iteration

            for batch_idx, (state, output) in enumerate(train_iter):
                batch_start = time.time()
                data_time = batch_start - batch_end  # Time spent waiting for data

                # Track first batch timing
                if batch_idx == 0 and first_batch_time is None:
                    first_batch_time = batch_start - epoch_start
                    train_logger.info(
                        f"First batch took {first_batch_time:.2f}s to load (includes worker startup)"
                    )

                # Log detailed timing for first 5 batches
                if batch_idx < 5:
                    train_logger.info(
                        f"Batch {batch_idx}: data loading took {data_time*1000:.1f}ms"
                    )

                train_logger.debug(f"Successfully got batch {batch_idx}")
                train_logger.debug(
                    f"Processing batch {batch_idx}, state shape: {state.shape}, output shape: {output.shape}"
                )

                zero_grad_start = time.time()
                opt.zero_grad()
                zero_grad_time = time.time() - zero_grad_start
                train_logger.debug("Gradients zeroed")

                forward_start = time.time()
                loss, accuracy = self.train_fn(model, state, output)
                forward_time = time.time() - forward_start
                train_logger.debug(
                    f"Forward pass complete, loss: {loss.item():.6f}, accuracy: {accuracy.item():.4f}"
                )

                backward_start = time.time()
                loss.backward()
                backward_time = time.time() - backward_start
                train_logger.debug("Backward pass complete")

                clip_start = time.time()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                clip_time = time.time() - clip_start
                train_logger.debug("Gradient clipping complete")

                optimizer_start = time.time()
                opt.step()
                optimizer_time = time.time() - optimizer_start
                train_logger.debug("Optimizer step complete")

                batch_end = time.time()
                batch_time = batch_end - batch_start

                # Only log from main process
                if is_main_process:
                    total_time = (
                        data_time + batch_time
                    )  # Total time including data wait
                    effective_samples_per_sec = state.shape[0] / total_time
                    compute_samples_per_sec = state.shape[0] / batch_time
                    timing = {
                        "timing/data_ms": data_time * 1000,
                        "timing/zero_grad_ms": zero_grad_time * 1000,
                        "timing/forward_ms": forward_time * 1000,
                        "timing/backward_ms": backward_time * 1000,
                        "timing/clip_ms": clip_time * 1000,
                        "timing/optimizer_ms": optimizer_time * 1000,
                        "timing/batch_ms": batch_time * 1000,
                        "timing/total_ms": total_time * 1000,
                        "timing/compute_samples_per_sec": compute_samples_per_sec,
                        "timing/effective_samples_per_sec": effective_samples_per_sec,
                        "timing/gpu_utilization": (batch_time / total_time) * 100,
                    }

                    logger.log_info(TrainInfo(epoch, batch_idx, loss.item()))

                    # Log training metrics to wandb
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "train_acc": accuracy.item(),
                            **timing,
                        },
                        step=epoch * len(self.train_dl) + batch_idx,
                    )

                    if (
                        batch_idx % self.eval_freq == 0
                        and batch_idx > 0
                        and self.eval_fn
                    ):
                        train_logger.info(
                            f"Starting mid-epoch evaluation at batch {batch_idx}"
                        )
                        out = self.eval_fn(model, self.test_dl, self.train_fn)
                        train_logger.info("Mid-epoch evaluation complete")
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
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            train_logger.info(
                f"Finished training for epoch {epoch} in {epoch_time:.2f}s"
            )
            train_logger.info(
                f"Average batch time: {epoch_time / (batch_idx + 1):.3f}s"
            )
            if is_main_process:
                train_logger.info(f"Starting end-of-epoch evaluation for epoch {epoch}")
                if self.eval_fn is not None:
                    train_logger.info("Running model evaluation")
                    out = self.eval_fn(model, self.test_dl, self.train_fn)
                    train_logger.info("Model evaluation complete")
                    logger.log_info(ValidationInfo(epoch, batch_idx, out))

                    # Log validation metrics to wandb at end of epoch
                    val_metrics = {}
                    for test_info in out:
                        val_metrics[f"val_{test_info.loss_type}"] = test_info.loss
                    wandb.log(val_metrics, step=epoch * len(self.train_dl) + batch_idx)

                train_logger.info("Saving checkpoint")
                logger.checkpoint(epoch, batch_idx, model)
                train_logger.info("Checkpoint saved")

                # Agent evaluation at end of epoch
                train_logger.info("Starting agent evaluation")
                # Unwrap DistributedDataParallel before passing to agent eval
                model_unwrapped = model.module if hasattr(model, "module") else model
                agent_results = self.agent_eval_fn(model_unwrapped, epoch)
                train_logger.info("Agent evaluation complete")

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
