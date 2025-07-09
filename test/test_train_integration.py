"""
Integration test for training a small model locally
"""

import os
import tempfile

import torch
from mllg import LogWriter
from torch.utils.data import DataLoader, TensorDataset

from stonefish.eval.tensor import eval_model_tensors
from stonefish.resnet import ChessResNet
from stonefish.train import PreTrainContext, train_step


def test_train_small_resnet():
    """Test training a small ResNet model on synthetic data"""
    # Create synthetic chess-like data
    num_samples = 100
    input_dim = 69  # Standard chess board representation
    output_dim = 5700  # Number of possible moves

    # Generate random data
    x = torch.randn(num_samples, input_dim)
    # Generate random move indices
    y = torch.randint(0, output_dim, (num_samples,))

    # Create datasets
    train_dataset = TensorDataset(x[:80], y[:80])
    test_dataset = TensorDataset(x[80:], y[80:])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Create a small ResNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessResNet(
        input_dim=input_dim,
        hidden_dim=256,  # Small hidden dimension
        num_blocks=2,  # Few blocks for fast training
        output_dim=output_dim,
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create training context
    ctx = PreTrainContext(
        eval_fn=eval_model_tensors,
        train_fn=train_step,
        train_dl=train_loader,
        test_dl=test_loader,
        epochs=2,  # Just 2 epochs
        eval_freq=2,  # Evaluate every 2 batches
    )

    # Create temporary log directory
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LogWriter(tmpdir, log_proc=False)

        # Get initial loss
        initial_loss = None
        for batch in train_loader:
            x_batch, y_batch = batch
            with torch.no_grad():
                output = model.inference(x_batch.to(device))
                initial_loss = torch.nn.functional.cross_entropy(
                    output, y_batch.to(device).flatten()
                ).item()
            break

        # Train the model
        ctx(logger, model, optimizer)

        # Get final loss
        final_loss = None
        for batch in train_loader:
            x_batch, y_batch = batch
            with torch.no_grad():
                output = model.inference(x_batch.to(device))
                final_loss = torch.nn.functional.cross_entropy(
                    output, y_batch.to(device).flatten()
                ).item()
            break

        # Check that loss decreased
        assert (
            final_loss < initial_loss
        ), f"Loss did not decrease: {initial_loss} -> {final_loss}"

        # Check that checkpoints were created
        checkpoint_files = [f for f in os.listdir(tmpdir) if f.endswith(".pth")]
        assert len(checkpoint_files) > 0, "No checkpoints were created"
