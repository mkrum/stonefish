"""
Tensor-based evaluation metrics for training.

This module provides fast tensor-based evaluation functions for use during training,
without requiring conversion to chess objects.
"""

import torch
import torch.nn.functional
import tqdm
from mllg import TestInfo


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy.

    Args:
        logits: Model output of shape (batch_size, num_classes)
        targets: Target indices of shape (batch_size,)

    Returns:
        Accuracy as a float between 0 and 1
    """
    preds = logits.argmax(dim=1)
    correct = float((preds == targets).float().sum().item())
    total = targets.size(0)
    return float(correct / total) if total > 0 else 0.0


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute cross-entropy loss.

    Args:
        logits: Model output of shape (batch_size, num_classes)
        targets: Target indices of shape (batch_size,)

    Returns:
        Cross-entropy loss as a float
    """
    return float(torch.nn.functional.cross_entropy(logits, targets).item())


def eval_model_tensors(model, dataloader, train_fn, max_batch=20):
    """
    Evaluate model on tensor data during training.

    Args:
        model: PyTorch model
        dataloader: DataLoader yielding (input_tensor, target_tensor) pairs
        train_fn: Training function for loss computation
        max_batch: Maximum number of batches to evaluate

    Returns:
        List of TestInfo objects with metrics
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(dataloader)):
            # Move inputs and targets to model device
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get model outputs - use inference for evaluation
            # Handle DistributedDataParallel wrapper
            model_unwrapped = model.module if hasattr(model, "module") else model
            logits = model_unwrapped.inference(inputs)

            # Compute metrics
            loss = cross_entropy(logits, targets)
            acc = accuracy(logits, targets)

            # Accumulate
            total_loss += loss
            total_acc += acc
            batch_count += 1

            if batch_idx >= max_batch:
                break

    # Average metrics
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    avg_acc = total_acc / batch_count if batch_count > 0 else 0.0

    return [
        TestInfo("loss", float(avg_loss)),
        TestInfo("ACC", float(avg_acc)),
    ]
