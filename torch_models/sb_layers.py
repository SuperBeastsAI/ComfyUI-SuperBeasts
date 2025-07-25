"""SuperBeasts – Shared utility layers used by ContextualResidualUNetV4

This module collects all the small custom layers that appear in the ONNX
export so we can reference them directly from the PyTorch implementation.

Only inference behaviour is required for ComfyUI, therefore layers that are
mainly for regularisation (FeatureNoise, DropBlock2D, StochasticDepth) act as
no-ops when the module is in `.eval()` mode.
"""

from __future__ import annotations

from typing import Optional, Tuple
import math

try:
    import torch
    from torch import nn, Tensor
except Exception as _err:  # pragma: no cover
    raise RuntimeError("PyTorch is required – please install torch before using SuperBeasts nodes") from _err

__all__ = [
    "GELU",
    "FeatureNoise",
    "DropBlock2D",
    "StochasticDepth",
    "MeanAcrossChannels",
    "MaxAcrossChannels",
    "SpatialAttention",
]


# -----------------------------------------------------------------------------
# Simple activation wrapper so the op appears by name in state_dict ordering
# -----------------------------------------------------------------------------
class GELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return torch.nn.functional.gelu(x)


# -----------------------------------------------------------------------------
# Regularisation helpers – act as identity when model.eval()
# -----------------------------------------------------------------------------
class FeatureNoise(nn.Module):
    """Adds channel-wise Gaussian noise during training; pass-through in eval."""

    def __init__(self, stddev: float = 0.02):
        super().__init__()
        self.stddev = float(stddev)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        if self.training and self.stddev > 0:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x


class DropBlock2D(nn.Module):
    """Implements DropBlock; falls back to identity in eval."""

    def __init__(self, keep_prob: float = 0.9, block_size: int = 7):
        super().__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        if not self.training or self.keep_prob == 1.0:
            return x
        gamma = (1.0 - self.keep_prob) / (self.block_size ** 2)
        mask = (torch.rand_like(x[:, :1, :, :]) < gamma).float()
        mask = torch.nn.functional.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = 1.0 - mask
        return x * mask * (mask.numel() / mask.sum())


class StochasticDepth(nn.Module):
    """Randomly drops entire residual branch during training (identity in eval)."""

    def __init__(self, keep_prob: float = 0.9):
        super().__init__()
        self.keep_prob = float(keep_prob)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        if not self.training or self.keep_prob == 1.0:
            return x
        if torch.rand(1).item() < self.keep_prob:
            return x / self.keep_prob  # rescale to keep expected value
        return torch.zeros_like(x)


# -----------------------------------------------------------------------------
# Attention utility layers (channel reductions)
# -----------------------------------------------------------------------------
class MeanAcrossChannels(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return torch.mean(x, dim=1, keepdim=True)


class MaxAcrossChannels(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return torch.max(x, dim=1, keepdim=True).values


class SpatialAttention(nn.Module):
    """Spatial attention similar to CBAM: concat(avg, max) → Conv(7×7) → Sigmoid."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv = torch.max(x, dim=1, keepdim=True).values
        attn = torch.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))
        return x * attn


# -----------------------------------------------------------------------------
# Quick verification when executed directly – ensures the first spatial
# attention conv matches the checkpoint shapes.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import safetensors.torch as st
    import csv, pathlib

    ckpt = st.load_file("models/SuperBeasts_ColorAdjustment_512px_V1.safetensors")
    print("Checkpoint tensors:", len(ckpt))

    # Build a tiny model to reveal param ordering
    sa = SpatialAttention()
    state = sa.state_dict()
    for k, v in state.items():
        shape = tuple(v.shape)
        print(k, shape, "->", "OK" if any(tuple(t.shape) == shape for t in ckpt.values()) else "missing") 