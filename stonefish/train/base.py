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

import wandb

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


def setup_distributed():
    """Initialize distributed training if available.

    Returns (local_rank, world_size, is_distributed).
    Skips process group init for single-process runs (e.g. MPS on Mac).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size <= 1:
        return 0, 1, False

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, world_size, True


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _log_agent_results(agent_results, logger, step):
    """Separate agent eval results into metrics vs HTML and log to wandb."""
    regular_metrics = {}
    html_content = {}
    for key, value in agent_results.items():
        if key.startswith("eval/PGN"):
            html_content[key] = value
        else:
            regular_metrics[key] = value
            logger.log_info(TestInfo(key, value))
    if regular_metrics:
        wandb.log(regular_metrics, step=step)
    if html_content:
        wandb.log(html_content, step=step)


@dataclass
class PreTrainContext:

    eval_fn: Any
    train_fn: Any
    train_dl: Any
    test_dl: Any
    agent_eval_fn: Any = None
    epochs: int = 1000
    eval_freq: int = 5000
    gradient_clip: float = 1.0
    compile_model: bool = False
    compile_mode: str = "default"
    use_amp: bool = False
    amp_dtype: str = "float16"
    warmup_steps: int = 0
    lr_min: float = 0.0

    def __call__(self, logger, model, opt):
        device = next(model.parameters()).device

        # Setup distributed training if available
        local_rank, world_size, is_distributed = setup_distributed()
        is_main_process = local_rank == 0

        # Setup LR scheduler (linear warmup + cosine decay)
        total_steps = self.epochs * len(self.train_dl)
        scheduler = None
        if self.warmup_steps > 0 or self.lr_min > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-8, end_factor=1.0, total_iters=self.warmup_steps
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps - self.warmup_steps, eta_min=self.lr_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps],
            )
            train_logger.info(
                f"LR schedule: linear warmup for {self.warmup_steps} steps, "
                f"cosine decay to {self.lr_min} over {total_steps - self.warmup_steps} steps"
            )

        # Setup AMP
        amp_dtype = getattr(torch, self.amp_dtype) if self.use_amp else None
        use_grad_scaler = self.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler() if use_grad_scaler else None
        if self.use_amp:
            train_logger.info(
                f"AMP enabled with dtype={self.amp_dtype}, scaler={use_grad_scaler}"
            )

        # Compile model if requested
        if self.compile_model:
            train_logger.info(f"Compiling model with mode='{self.compile_mode}'")
            model = torch.compile(model, mode=self.compile_mode)

        # Log dataset configuration
        train_logger.info(
            f"Training dataset: type={type(self.train_dl).__name__}, "
            f"num_workers={getattr(self.train_dl, 'num_workers', 0)}, "
            f"prefetch_factor={getattr(self.train_dl, 'prefetch_factor', None)}, "
            f"batch_size={self.train_dl.batch_size}"
        )

        # Wrap model in DDP if distributed
        if is_distributed:
            torch.cuda.set_device(local_rank)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )

        # Initial evaluation
        if is_main_process and self.eval_fn:
            out = self.eval_fn(model, self.test_dl, self.train_fn)
            logger.log_info(ValidationInfo(0, 0, out))

        if is_distributed:
            dist.barrier()

        for epoch in range(self.epochs):
            train_logger.info(f"Starting epoch {epoch}")

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

                opt.zero_grad()
                if self.use_amp:
                    with torch.autocast(device.type, dtype=amp_dtype):
                        loss, accuracy = self.train_fn(model, state, output)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.gradient_clip
                        )
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.gradient_clip
                        )
                        opt.step()
                else:
                    loss, accuracy = self.train_fn(model, state, output)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.gradient_clip
                    )
                    opt.step()

                if scheduler is not None:
                    scheduler.step()

                # Sync device so timing reflects actual compute, not async queue
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()

                batch_end = time.time()
                batch_time = batch_end - batch_start

                # Only log from main process
                if is_main_process:
                    total_time = data_time + batch_time
                    samples_per_sec = state.shape[0] / total_time
                    timing = {
                        "timing/data_ms": data_time * 1000,
                        "timing/batch_ms": batch_time * 1000,
                        "timing/total_ms": total_time * 1000,
                        "timing/samples_per_sec": samples_per_sec,
                    }

                    logger.log_info(TrainInfo(epoch, batch_idx, loss.item()))

                    current_lr = opt.param_groups[0]["lr"]
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "train_acc": accuracy.item(),
                            "lr": current_lr,
                            **timing,
                        },
                        step=epoch * len(self.train_dl) + batch_idx,
                    )

                    if batch_idx % self.eval_freq == 0 and batch_idx > 0:
                        step = epoch * len(self.train_dl) + batch_idx

                        if self.eval_fn:
                            train_logger.info(
                                f"Starting mid-epoch evaluation at batch {batch_idx}"
                            )
                            out = self.eval_fn(model, self.test_dl, self.train_fn)
                            train_logger.info("Mid-epoch evaluation complete")
                            logger.log_info(ValidationInfo(epoch, batch_idx, out))

                            val_metrics = {}
                            for test_info in out:
                                val_metrics[f"val_{test_info.loss_type}"] = (
                                    test_info.loss
                                )
                            wandb.log(val_metrics, step=step)

                        logger.checkpoint(epoch, batch_idx, model)

                        if self.agent_eval_fn is not None:
                            train_logger.info(
                                f"Starting mid-epoch agent evaluation at batch {batch_idx}"
                            )
                            model_unwrapped = (
                                model.module if hasattr(model, "module") else model
                            )
                            agent_results = self.agent_eval_fn(model_unwrapped, epoch)
                            train_logger.info("Mid-epoch agent evaluation complete")
                            _log_agent_results(agent_results, logger, step)

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
                if self.agent_eval_fn is not None:
                    train_logger.info("Starting agent evaluation")
                    model_unwrapped = (
                        model.module if hasattr(model, "module") else model
                    )
                    agent_results = self.agent_eval_fn(model_unwrapped, epoch)
                    train_logger.info("Agent evaluation complete")
                    _log_agent_results(
                        agent_results, logger, epoch * len(self.train_dl) + batch_idx
                    )

            if is_distributed:
                dist.barrier()

        # Cleanup distributed training
        if is_distributed:
            cleanup_distributed()
