"""
ResNet model architectures for chess move prediction.

This module contains ResNet-style models adapted from new_repo, integrated
with stonefish's yamlargs configuration system.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional

from stonefish.convert import board_to_lczero_tensor
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
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class ChessResNet(nn.Module):
    """Standard ResNet model for chess using flat board representation"""

    def __init__(self, input_dim=69, hidden_dim=4096, num_blocks=8, output_dim=5632):
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
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ConvResBlock(num_filters) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=3, padding=1)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(32 * 8 * 8, output_dim)  # 8x8 is the board size

    def forward(self, x, action):
        """Forward pass for training - action parameter ignored for ResNet"""
        # Handle different input shapes
        # Expected: [batch_size, 8, 8, 20] -> [batch_size, 20, 8, 8]

        device = next(self.parameters()).device
        x = x.to(device)

        if len(x.shape) == 4 and x.shape[3] == self.input_channels:
            x = x.permute(
                0, 3, 1, 2
            ).contiguous()  # [batch, height, width, channels] -> [batch, channels, height, width]
        elif len(x.shape) == 3 and x.shape[2] == self.input_channels:
            x = x.permute(2, 0, 1).unsqueeze(0).contiguous()  # Single item handling

        # Forward pass
        x = self.conv_input(x)
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_relu(policy)
        policy = policy.reshape(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        return policy

    def inference(self, x):
        """Inference pass for move selection"""
        return self.forward(x, None)


class ChessConvNetRL(ChessConvNet):
    """ConvNet with value head for RL fine-tuning.

    Extends ChessConvNet with:
    - A value head for state evaluation
    - sample() for rollout generation
    - forward() returning (log_probs, values) for PPO-style training
    - Optional pretrained checkpoint loading
    """

    MASK_DIM = 5632

    def __init__(
        self,
        input_channels=20,
        num_filters=256,
        num_blocks=8,
        output_dim=5700,
        pretrained_checkpoint=None,
    ):
        super().__init__(
            input_channels=input_channels,
            num_filters=num_filters,
            num_blocks=num_blocks,
            output_dim=output_dim,
        )

        # Value head: Conv(1x1) -> ReLU -> Flatten -> Linear -> scalar
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc = nn.Linear(64, 1)

        if pretrained_checkpoint is not None:
            self._load_pretrained(pretrained_checkpoint)

    def _load_pretrained(self, path):
        """Load pretrained weights, ignoring missing value head keys."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        # Strip DDP 'module.' prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                cleaned[k[7:]] = v
            else:
                cleaned[k] = v
        self.load_state_dict(cleaned, strict=False)

    def _convert_state(self, x):
        """Ensure input is spatial (N, 8, 8, 20).

        If input is flat (N, 69) from CChessEnv, convert via CBoard -> chess.Board -> lczero tensor.
        """
        from fastchessenv import CBoard

        if x.dim() == 2 and x.shape[1] == 69:
            boards = []
            x_np = x.cpu().numpy()
            for i in range(x_np.shape[0]):
                cboard = CBoard.from_array(np.int32(x_np[i]))
                py_board = cboard.to_board()
                boards.append(board_to_lczero_tensor(py_board))
            return torch.from_numpy(np.stack(boards)).to(x.device)
        return x

    def _backbone(self, x):
        """Shared trunk: convert state -> conv_input -> res_blocks.

        Returns feature map (N, num_filters, 8, 8).
        """
        x = self._convert_state(x)
        device = next(self.parameters()).device
        x = x.to(device)

        # Permute (N, 8, 8, 20) -> (N, 20, 8, 8)
        if x.dim() == 4 and x.shape[3] == self.input_channels:
            x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv_input(x)
        for block in self.res_blocks:
            x = block(x)
        return x

    def _policy_head(self, features):
        """Policy head: features -> raw logits (N, output_dim)."""
        policy = self.policy_conv(features)
        policy = self.policy_relu(policy)
        policy = policy.reshape(policy.size(0), -1)
        return self.policy_fc(policy)

    def _value_head(self, features):
        """Value head: features -> scalar values (N,)."""
        v = self.value_conv(features)
        v = self.value_relu(v)
        v = v.reshape(v.size(0), -1)  # (N, 64)
        return self.value_fc(v).squeeze(-1)  # (N,)

    def forward(self, x, action=None, mask=None):
        """Forward for RL training.

        Returns (selected_log_prob, values) when action is provided.
        Returns (raw_logits, values) when action is None.
        """
        features = self._backbone(x)
        raw_logits = self._policy_head(features)
        values = self._value_head(features)

        if action is None:
            return raw_logits, values

        # Slice to mask dim, apply mask, log_softmax, gather
        logits = raw_logits[:, : self.MASK_DIM]
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        log_probs = functional.log_softmax(logits, dim=-1)
        selected_log_prob = log_probs.gather(1, action.long().view(-1, 1)).squeeze(-1)
        return selected_log_prob, values

    def value(self, x):
        """Compute state values only (for bootstrapping)."""
        features = self._backbone(x)
        return self._value_head(features)

    def sample(self, state, legal_mask):
        """Sample actions for rollout generation.

        Args:
            state: (N, ...) board state tensor
            legal_mask: (N, MASK_DIM) float tensor of legal moves

        Returns:
            (actions, legal_mask) as tensors on model device
        """
        device = self.device
        features = self._backbone(state)
        raw_logits = self._policy_head(features)

        logits = raw_logits[:, : self.MASK_DIM]
        mask = legal_mask.to(device)
        logits = logits.masked_fill(mask == 0, float("-inf"))

        probs = functional.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze(-1)
        return actions.to(device), mask

    def inference(self, x):
        """Inference pass for eval compatibility (raw policy logits only)."""
        features = self._backbone(x)
        return self._policy_head(features)

    @property
    def device(self):
        return next(self.parameters()).device
