"""
Tests for ResNet model architectures
"""

import torch

from stonefish.resnet import ChessConvNet, ChessResNet, ConvResBlock, ResBlock


def test_res_block():
    """Test basic ResBlock functionality"""
    block = ResBlock(256)
    x = torch.randn(8, 256)
    output = block(x)
    assert output.shape == (8, 256)
    assert not torch.allclose(
        output, x
    )  # Should be different due to residual connection


def test_conv_res_block():
    """Test ConvResBlock functionality"""
    block = ConvResBlock(64)
    x = torch.randn(4, 64, 8, 8)
    output = block(x)
    assert output.shape == (4, 64, 8, 8)


def test_chess_resnet_creation():
    """Test ChessResNet model creation and basic forward pass"""
    model = ChessResNet(input_dim=69, hidden_dim=512, num_blocks=4, output_dim=5700)

    # Test parameter count
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0

    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 69)
    output = model.inference(x)

    assert output.shape == (batch_size, 5700)
    assert not torch.isnan(output).any()


def test_chess_convnet_creation():
    """Test ChessConvNet model creation and basic forward pass"""
    model = ChessConvNet(
        input_channels=20, num_filters=128, num_blocks=4, output_dim=5700
    )

    # Test parameter count
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0

    # Test forward pass with standard input shape (batch, height, width, channels)
    batch_size = 8
    x = torch.randn(batch_size, 8, 8, 20)
    output = model.inference(x)

    assert output.shape == (batch_size, 5700)
    assert not torch.isnan(output).any()


def test_chess_convnet_input_shapes():
    """Test ChessConvNet handles different input shapes correctly"""
    model = ChessConvNet(
        input_channels=20, num_filters=64, num_blocks=2, output_dim=5700
    )

    # Test batch input: (batch, height, width, channels)
    x1 = torch.randn(4, 8, 8, 20)
    output1 = model.inference(x1)
    assert output1.shape == (4, 5700)

    # Test single input: (height, width, channels)
    x2 = torch.randn(8, 8, 20)
    output2 = model.inference(x2)
    assert output2.shape == (1, 5700)

    # Test already permuted input: (batch, channels, height, width)
    x3 = torch.randn(4, 20, 8, 8)
    output3 = model.inference(x3)
    assert output3.shape == (4, 5700)


def test_model_reproducibility():
    """Test that models produce consistent outputs with same inputs"""
    torch.manual_seed(42)
    model1 = ChessResNet(hidden_dim=256, num_blocks=2)

    torch.manual_seed(42)
    model2 = ChessResNet(hidden_dim=256, num_blocks=2)

    x = torch.randn(4, 69)

    # Models should be identical
    output1 = model1.inference(x)
    output2 = model2.inference(x)

    assert torch.allclose(output1, output2, atol=1e-6)


def test_different_configurations():
    """Test models with different hyperparameters"""
    configs = [
        {"hidden_dim": 512, "num_blocks": 2},
        {"hidden_dim": 1024, "num_blocks": 4},
        {"hidden_dim": 2048, "num_blocks": 8},
    ]

    for config in configs:
        model = ChessResNet(**config)
        x = torch.randn(2, 69)
        output = model.inference(x)
        assert output.shape == (2, 5700)

    conv_configs = [
        {"num_filters": 64, "num_blocks": 2},
        {"num_filters": 128, "num_blocks": 4},
        {"num_filters": 256, "num_blocks": 8},
    ]

    for config in conv_configs:
        model = ChessConvNet(**config)
        x = torch.randn(2, 8, 8, 20)
        output = model.inference(x)
        assert output.shape == (2, 5700)
