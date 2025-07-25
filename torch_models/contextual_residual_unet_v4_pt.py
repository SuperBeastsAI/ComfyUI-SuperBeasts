from __future__ import annotations

"""Minimal inference-only PyTorch implementation of **ContextualResidualUNetV4**
that matches the tensor *shapes* of the original Keras/ONNX model so we can
load the weights stored in `SuperBeasts_ColorAdjustment_… .safetensors` without
using onnx2pytorch at run-time.

Important notice
----------------
*Parameter names* will not match those inside the safetensors checkpoint – that
conversion tool encodes paths like `Conv_contextual_residual_unet_1/conv2d_1/…`.
We therefore rely on the loader to copy values **by shape & order**, not by
name.  See SBLoadModel where we remap the state_dict.

The architecture is a UNet-style encoder/decoder with skip connections, group
normalisation and a lightweight context attention block.  All dropout /
regularisation layers are omitted for simplicity (weights for them are not
used during inference anyway).
"""

from typing import Sequence

# Conditional import so static linters don’t fail when torch isn’t available.
try:
    import torch  # type: ignore
    from torch import nn, Tensor
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    Tensor = object  # type: ignore
    def _stub(*_, **__):
        raise RuntimeError("PyTorch is required to use ContextualResidualUNetV4Torch")
    F = type("_F", (), {"gelu": staticmethod(_stub)})()  # type: ignore

# -----------------------------------------------------------------------------
# Utility layers
# -----------------------------------------------------------------------------
class GELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 (simple one-liner)
        return F.gelu(x)


class ConvGNReLU(nn.Module):
    """Conv → GroupNorm → GELU x2 with residual (similar to original _conv_block)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.need_proj = in_ch != out_ch
        self.proj = nn.Conv2d(in_ch, out_ch, 1, padding=0) if self.need_proj else nn.Identity()
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.act1 = GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        res = self.proj(x)
        x = self.conv1(res)
        x = self.act1(self.gn1(x))
        x = self.conv2(x)
        x = self.gn2(x)
        return x + res


class SpatialAttention(nn.Module):
    """2-D Spatial attention similar to CBAM – avg & max pool along channels."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat((avg, maxv), dim=1)))
        return x * attn


# -----------------------------------------------------------------------------
# Main UNet V4
# -----------------------------------------------------------------------------
class ContextualResidualUNetV4Torch(nn.Module):
    """Inference-only forward pass returning *residual* tensor (RGB, same H×W)."""

    # Default hyperparameters (must match training config)
    FILTER_LEVELS: Sequence[int] = (32, 64, 96, 128)

    def __init__(self, in_channels: int = 7, ctx_channels: int = 7):
        super().__init__()
        f = self.FILTER_LEVELS

        # Initial stem
        self.stem_conv = nn.Conv2d(in_channels, f[0], 3, padding=1)
        self.stem_gn = nn.GroupNorm(8, f[0])
        self.stem_act = GELU()

        # Encoder blocks + pools
        self.enc_blocks = nn.ModuleList([
            ConvGNReLU(f[i], f[i]) for i in range(len(f))
        ])
        self.enc_pools = nn.ModuleList([
            nn.MaxPool2d(2) for _ in range(len(f) - 1)
        ])

        # Bottleneck
        self.bottleneck = ConvGNReLU(f[-1], f[-1])

        # Context embedding (global average pool + dense)
        self.ctx_conv1 = nn.Conv2d(ctx_channels, f[0], 3, padding=1)
        self.ctx_pool1 = nn.MaxPool2d(2)
        self.ctx_conv2 = nn.Conv2d(f[0], f[1], 3, padding=1)
        self.ctx_pool2 = nn.MaxPool2d(2)
        self.ctx_gap = nn.AdaptiveAvgPool2d(1)
        self.ctx_fc = nn.Linear(f[-1], f[-1])

        # Decoder blocks
        self.dec_ups = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode="nearest") for _ in range(len(f) - 1)
        ])
        self.dec_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=f[::-1][i], num_heads=4, batch_first=True)
            for i in range(len(f))
        ])
        self.dec_spatial = nn.ModuleList([
            SpatialAttention() for _ in range(len(f))
        ])
        self.dec_blocks = nn.ModuleList([
            ConvGNReLU(f[::-1][i] * (2 if i > 0 else 1), f[::-1][i]) for i in range(len(f))
        ])

        # Output conv (residual)
        self.out_conv = nn.Conv2d(f[0], 3, 3, padding=1)

    # ------------------------------------------------------------------
    def forward(self, tile: Tensor, ctx: Tensor) -> Tensor:  # noqa: D401
        # Encoder
        x = self.stem_act(self.stem_gn(self.stem_conv(tile)))
        skips = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            skips.append(x)
            if i < len(self.enc_pools):
                x = self.enc_pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Context embedding
        c = F.gelu(self.ctx_conv1(ctx))
        c = self.ctx_pool1(c)
        c = F.gelu(self.ctx_conv2(c))
        c = self.ctx_pool2(c)
        c = self.ctx_gap(c).flatten(1)  # (N, C)
        c = F.gelu(self.ctx_fc(c))

        # Decoder
        skips = skips[::-1]
        for i in range(len(self.dec_blocks)):
            if i > 0:
                x = self.dec_ups[i - 1](x)
                x = torch.cat([x, skips[i]], dim=1)

            # Token-wise attention with context vector
            N, C, H, W = x.shape
            q = x.reshape(N, C, H * W).permute(0, 2, 1)  # (N, HW, C)
            k = c.unsqueeze(1)  # (N, 1, C)
            attn_out, _ = self.dec_attn[i](q, k, k)
            attn_out = attn_out.permute(0, 2, 1).reshape(N, C, H, W)
            x = x + attn_out
            x = self.dec_spatial[i](x)
            x = self.dec_blocks[i](x)

        residual = self.out_conv(x)
        return residual 