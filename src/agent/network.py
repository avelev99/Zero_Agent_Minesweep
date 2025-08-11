"""EfficientZero-style neural network architecture.

This module implements a small but fully-fledged EfficientZero style model with
separate representation, dynamics and prediction networks.  It also includes
value and reward *supports* as used in the original MuZero/EfficientZero
implementations.

While greatly simplified compared to large scale research codebases, it
captures the key ideas: residual representations, action conditioned dynamics
and categorical value/reward heads with helper utilities for converting between
scalar values and support distributions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block used across EfficientZero modules."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class RepresentationNetwork(nn.Module):
    def __init__(self, in_channels: int, channels: int, num_blocks: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(obs)))
        return self.blocks(x)


class DynamicsNetwork(nn.Module):
    """Learned dynamics predicting next hidden state and immediate reward."""

    def __init__(
        self,
        channels: int,
        action_space_size: int,
        board_size,
        reward_support_size: int,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_emb = nn.Embedding(
            action_space_size, board_size[0] * board_size[1]
        )
        self.conv = nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.block = ResidualBlock(channels)
        flat = channels * board_size[0] * board_size[1]
        self.reward_head = nn.Linear(flat, reward_support_size)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        b = state.size(0)
        plane = self.action_emb(action).view(b, 1, self.board_size[0], self.board_size[1])
        x = torch.cat([state, plane], dim=1)
        x = F.relu(self.bn(self.conv(x)))
        x = self.block(x)
        reward_logits = self.reward_head(x.view(b, -1))
        return x, reward_logits


class PredictionNetwork(nn.Module):
    """Predict policy and state value from hidden representation."""

    def __init__(
        self, channels: int, board_size, action_space_size: int, value_support_size: int
    ) -> None:
        super().__init__()
        self.board_size = board_size
        flat = channels * board_size[0] * board_size[1]
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.block = ResidualBlock(channels)
        self.policy_head = nn.Linear(flat, action_space_size)
        self.value_head = nn.Linear(flat, value_support_size)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.bn(self.conv(state)))
        x = self.block(x)
        flat = x.view(x.size(0), -1)
        policy = self.policy_head(flat)
        value_logits = self.value_head(flat)
        return policy, value_logits


class EzAgent(nn.Module):
    """Small EfficientZero style network.

    Parameters mirror the original algorithm: the network outputs categorical
    distributions (supports) for value and reward.  Helper methods convert these
    logits into scalar estimates which are easier to reason about in tests and
    simple training loops.
    """

    def __init__(
        self,
        board_size,
        channels: int = 64,
        num_blocks: int = 2,
        support_size: int = 5,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.action_space_size = board_size[0] * board_size[1] * 3
        self.support_size = support_size
        support_dim = 2 * support_size + 1

        self.repr = RepresentationNetwork(1, channels, num_blocks)
        self.dynamics = DynamicsNetwork(
            channels, self.action_space_size, board_size, support_dim
        )
        self.prediction = PredictionNetwork(
            channels, board_size, self.action_space_size, support_dim
        )

        self.register_buffer(
            "support", torch.arange(-support_size, support_size + 1, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # Utilities for converting between scalars and support distributions
    def _support_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        return torch.sum(probs * self.support.view(1, -1), dim=1, keepdim=True)

    def _scalar_to_support(self, scalar: torch.Tensor) -> torch.Tensor:
        scalar = torch.clamp(scalar, -self.support_size, self.support_size)
        floor = scalar.floor()
        rest = scalar - floor
        one_hot = F.one_hot(floor.long() + self.support_size, self.support.numel()).float()
        next_one_hot = F.one_hot(
            torch.clamp(floor.long() + self.support_size + 1, 0, self.support.numel() - 1),
            self.support.numel(),
        ).float()
        return one_hot * (1 - rest) + next_one_hot * rest

    # ------------------------------------------------------------------
    def initial_inference(self, obs: torch.Tensor):
        state = self.repr(obs)
        policy, value_logits = self.prediction(state)
        value = self._support_to_scalar(value_logits)
        reward_logits = torch.zeros_like(value_logits)
        reward = self._support_to_scalar(reward_logits)
        return {
            "policy_logits": policy,
            "value": value,
            "reward": reward,
            "value_logits": value_logits,
            "reward_logits": reward_logits,
            "state": state,
        }

    # ------------------------------------------------------------------
    def recurrent_inference(self, state: torch.Tensor, action: torch.Tensor):
        next_state, reward_logits = self.dynamics(state, action)
        policy, value_logits = self.prediction(next_state)
        value = self._support_to_scalar(value_logits)
        reward = self._support_to_scalar(reward_logits)
        return {
            "policy_logits": policy,
            "value": value,
            "reward": reward,
            "value_logits": value_logits,
            "reward_logits": reward_logits,
            "state": next_state,
        }

    # ------------------------------------------------------------------
    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return self.initial_inference(obs)["policy_logits"]
