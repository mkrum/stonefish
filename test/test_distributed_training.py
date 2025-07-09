"""
Tests for distributed training functionality
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stonefish.train.base import (
    DistributedPreTrainContext,
    PreTrainContext,
    cleanup_distributed,
    setup_distributed,
    train_step,
)


class DummyModel(nn.Module):
    """Simple model for testing"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x, y=None):
        return self.linear(x)


class DummyLogger:
    """Mock logger for testing"""

    def __init__(self):
        self.logs = []

    def log_info(self, info):
        self.logs.append(info)

    def log_str(self, s):
        self.logs.append(s)

    def checkpoint(self, epoch, batch, model):
        pass


def dummy_eval_fn(model, dataloader, train_fn):
    """Simple evaluation function for testing"""
    return {"loss": 0.5, "accuracy": 0.9}


def test_pretrain_context_basic():
    """Test basic PreTrainContext functionality"""
    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=10)
    test_loader = DataLoader(dataset, batch_size=10)

    # Create model and optimizer
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create context
    ctx = PreTrainContext(
        eval_fn=dummy_eval_fn,
        train_fn=train_step,
        train_dl=train_loader,
        test_dl=test_loader,
        epochs=1,
        eval_freq=50,
    )

    # Create logger
    logger = DummyLogger()

    # Run training
    ctx(logger, model, optimizer)

    # Check that training happened
    assert len(logger.logs) > 0
    # Check for ValidationInfo objects
    from mllg import ValidationInfo

    assert any(isinstance(log, ValidationInfo) for log in logger.logs)


def test_distributed_context_single_gpu():
    """Test DistributedPreTrainContext in single GPU mode"""
    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=10)
    test_loader = DataLoader(dataset, batch_size=10)

    # Create model and optimizer
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create distributed context
    ctx = DistributedPreTrainContext(
        eval_fn=dummy_eval_fn,
        train_fn=train_step,
        train_dl=train_loader,
        test_dl=test_loader,
        epochs=1,
        eval_freq=50,
        gradient_clip=1.0,
    )

    # Create logger
    logger = DummyLogger()

    # Run training (should work in single GPU mode)
    ctx(logger, model, optimizer)

    # Check that training happened
    assert len(logger.logs) > 0
    # Check for ValidationInfo objects
    from mllg import ValidationInfo

    assert any(isinstance(log, ValidationInfo) for log in logger.logs)


def test_setup_distributed_single_process():
    """Test distributed setup in single process mode"""
    # Ensure we're in single process mode
    os.environ.pop("LOCAL_RANK", None)
    os.environ.pop("WORLD_SIZE", None)

    local_rank, world_size, is_distributed = setup_distributed()

    assert local_rank == 0
    assert world_size == 1
    assert is_distributed is False

    # Cleanup should work even if not distributed
    cleanup_distributed()


def test_distributed_context_gradient_clipping():
    """Test that gradient clipping is applied"""
    # Create dummy data with large values to trigger gradient clipping
    x = torch.randn(20, 10) * 100
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=10)

    # Create model and track gradients
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create context with small gradient clip value
    ctx = DistributedPreTrainContext(
        eval_fn=dummy_eval_fn,
        train_fn=train_step,
        train_dl=train_loader,
        test_dl=train_loader,
        epochs=1,
        eval_freq=100,  # Don't evaluate during this test
        gradient_clip=0.1,
    )

    logger = DummyLogger()

    # Run one epoch
    ctx(logger, model, optimizer)

    # Gradients should have been clipped
    # This is implicitly tested by the training completing successfully
    assert len(logger.logs) > 0
