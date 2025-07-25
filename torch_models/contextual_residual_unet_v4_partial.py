from __future__ import annotations

"""Partial rewrite of ContextualResidualUNetV4 – stem + first encoder block
Used only to validate that our PyTorch layers reproduce the parameter shapes
found in SuperBeasts_ColorAdjustment_512px_V1.safetensors.
"""

import torch
from torch import nn, Tensor
from .sb_layers import GELU, MeanAcrossChannels, MaxAcrossChannels, SpatialAttention, FeatureNoise, DropBlock2D, StochasticDepth


class ConvGNReLU(nn.Module):
    """Conv → GroupNorm → GELU → Conv → GroupNorm with residual"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.act1 = GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        res = x
        x = self.conv1(x)
        x = self.act1(self.gn1(x))
        x = self.conv2(x)
        x = self.gn2(x)
        return x + res


class UNetV4StemPartial(nn.Module):
    def __init__(self, in_ch: int = 7, base_ch: int = 32):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.stem_gn = nn.GroupNorm(8, base_ch)
        self.stem_act = GELU()
        self.enc_block0 = ConvGNReLU(base_ch)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        x = self.stem_act(self.stem_gn(self.stem_conv(x)))
        x = self.enc_block0(x)
        return x


if __name__ == "__main__":
    import safetensors.torch as st
    ckpt = st.load_file("models/SuperBeasts_ColorAdjustment_512px_V1.safetensors")

    model = UNetV4StemPartial()
    state = model.state_dict()

    missing = []
    for k, v in state.items():
        shape = tuple(v.shape)
        if not any(tuple(t.shape) == shape for t in ckpt.values()):
            missing.append((k, shape))
    if missing:
        print("Missing shapes:")
        for k, s in missing:
            print(" ", k, s)
    else:
        print("✓ all", len(state), "parameter shapes found in checkpoint") 