"""
ResNet model architectures for chess move prediction.

This module contains ResNet-style models adapted from new_repo, integrated
with stonefish's yamlargs configuration system.
"""

import torch.nn as nn

from stonefish.tokenizers import (
    FlatBoardTokenizer,
    FlatMoveTokenizer,
    LCZeroBoardTokenizer,
)


class ResBlock(nn.Module):
    """Residual block with linear layers"""

    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)


class ConvResBlock(nn.Module):
    """Residual block with convolutional layers"""

    def __init__(self, channels):
        super(ConvResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ChessResNet(nn.Module):
    """Standard ResNet model for chess using flat board representation"""

    def __init__(self, input_dim=69, hidden_dim=4096, num_blocks=8, output_dim=5700):
        super(ChessResNet, self).__init__()

        # Tokenizers
        self.board_tokenizer = FlatBoardTokenizer()
        self.move_tokenizer = FlatMoveTokenizer()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.output_dim = output_dim

        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.policy_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, action):
        """Forward pass for training - action parameter ignored for ResNet"""
        # Move input to same device as model
        device = next(self.parameters()).device
        x = x.to(device)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x)

    def inference(self, x):
        """Inference pass for move selection"""
        return self.forward(x, None)


class ChessConvNet(nn.Module):
    """Convolutional ResNet for chess using Leela Chess Zero board representation"""

    def __init__(
        self, input_channels=20, num_filters=256, num_blocks=8, output_dim=5700
    ):
        super(ChessConvNet, self).__init__()

        # Tokenizers
        self.board_tokenizer = LCZeroBoardTokenizer()
        self.move_tokenizer = FlatMoveTokenizer()

        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.output_dim = output_dim

        # Initial convolution block
        self.conv_input = nn.Sequential(
            nn.Conv2d(
                input_channels, num_filters, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ConvResBlock(num_filters) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(32 * 8 * 8, output_dim)  # 8x8 is the board size

    def forward(self, x, action):
        """Forward pass for training - action parameter ignored for ResNet"""
        # Handle different input shapes
        # Expected: [batch_size, 8, 8, 20] -> [batch_size, 20, 8, 8]
        if len(x.shape) == 4 and x.shape[3] == self.input_channels:
            x = x.permute(
                0, 3, 1, 2
            )  # [batch, height, width, channels] -> [batch, channels, height, width]
        elif len(x.shape) == 3 and x.shape[2] == self.input_channels:
            x = x.permute(2, 0, 1).unsqueeze(0)  # Single item handling

        # Forward pass
        x = self.conv_input(x)
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_relu(policy)
        policy = policy.reshape(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        return policy

    def inference(self, x):
        """Inference pass for move selection"""
        return self.forward(x, None)
