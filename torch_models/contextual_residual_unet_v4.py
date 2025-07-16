import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# NOTE: This is an INFERENCE-ONLY replica of models/versions/V4_K3/regularized_unet_v4.py
# It intentionally omits Dropout, FeatureNoise, DropBlock, StochasticDepth etc. so that
# conversion from Keras weights is straightforward and inference fast.

class MeanAcrossChannels(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True)  # NCHW

class MaxAcrossChannels(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1, keepdim=True)[0]

def spatial_attention_block(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    avg_pool = MeanAcrossChannels()(x)
    max_pool = MaxAcrossChannels()(x)
    concat = torch.cat([avg_pool, max_pool], dim=1)
    attention_map = F.conv2d(concat, weight=spatial_attention_block.conv_w, bias=spatial_attention_block.conv_b, padding=kernel_size // 2)
    attention_map = torch.sigmoid(attention_map)
    return x * attention_map

# Dummy parameters initialised lazily so that weight assignment during conversion works.
spatial_attention_block.conv_w = nn.Parameter(torch.empty(1, 2, 7, 7))
spatial_attention_block.conv_b = nn.Parameter(torch.empty(1))

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.project = None if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.project is None else self.project(x)
        x = self.conv1(identity)
        x = self.gn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.gn2(x)
        return x + identity

class ContextualResidualUNetV4Torch(nn.Module):
    def __init__(self, img_size: int = 512, context_size: Tuple[int, int] = (64, 64),
                 filters: List[int] = (32, 64, 96, 128)):
        super().__init__()
        self.img_size = img_size
        self.context_size = context_size
        self.filters = list(filters)
        self.in_ch = 7  # RGB + extra channels (Y Cb Cr S) – must match config.INPUT_CHANNELS during conversion

        # Context encoder – mirror down-sampling depth of main encoder so the final
        # channel dimension equals `self.filters[-1]` (128). This prevents runtime
        # shape mismatches like 1×64 · 128×128.
        ctx_layers: List[nn.Module] = []
        ch = self.in_ch
        for f in self.filters:
            ctx_layers.append(nn.Conv2d(ch, f, 3, padding=1))
            ctx_layers.append(nn.GELU())
            ctx_layers.append(nn.MaxPool2d(2))
            ch = f
        # The MaxPool after the last stage slightly over-downsamples; we can safely
        # remove it to keep spatial extent ≥1. Pop the last layer if it is MaxPool.
        if isinstance(ctx_layers[-1], nn.MaxPool2d):
            ctx_layers.pop()
        self.ctx_encoder = nn.Sequential(*ctx_layers)
        self.ctx_gap = nn.AdaptiveAvgPool2d(1)
        self.ctx_fc = nn.Linear(self.filters[-1], self.filters[-1])

        # Encoder blocks
        encs = []
        ch = self.in_ch
        for f in self.filters:
            encs.append(ConvBlock(ch, f))
            ch = f
        self.enc_blocks = nn.ModuleList(encs)
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(len(self.filters) - 1)])

        # Bottleneck – project down to filters[-2] (96) so decoder concatenations match training
        self.bottleneck = ConvBlock(self.filters[-1], self.filters[-2])

        # Decoder – build dynamically so each stage input = prev_out + skip
        rev = list(reversed(self.filters))  # [128,96,64,32]

        decoder_specs = []  # (in_ch, out_ch)
        prev_out = self.filters[-2]  # 96 (bottleneck output)
        for skip_ch in rev[1:]:  # iterate over 96,64,32
            in_ch = prev_out + skip_ch
            decoder_specs.append((in_ch, skip_ch))
            prev_out = skip_ch
        # final conv without concat
        decoder_specs.append((prev_out, prev_out))

        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest") for _ in range(len(decoder_specs) - 1)])
        self.dec_blocks = nn.ModuleList([ConvBlock(inp, out) for inp, out in decoder_specs])

        # Output conv
        self.out_conv = nn.Conv2d(self.filters[0], 3, 1)

    def forward(self, tile_input: torch.Tensor, ctx_input: torch.Tensor):
        # tile_input / ctx_input expected shape: N,C,H,W (C=7)
        # Context path
        c = self.ctx_encoder(ctx_input)
        c = self.ctx_gap(c).squeeze(-1).squeeze(-1)  # N,C
        context_embedding = self.ctx_fc(c)

        # Encoder
        skips = []
        x = tile_input
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            print(f"enc{i} after block: {x.shape}")
            skips.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
                print(f"enc{i} after pool : {x.shape}")

        x = self.bottleneck(x)
        print(f"bottleneck out     : {x.shape}")
        skips = list(reversed(skips[:-1]))  # drop deepest (size 64x64), reverse others

        # Decoder
        for i, (up, dec_block) in enumerate(zip(self.ups, self.dec_blocks[:-1])):
            x = up(x)
            print(f"dec{i} after up    : {x.shape}")
            skip = skips.pop(0)  # grab first skip (matching spatial size)
            print(f"dec{i} skip shape  : {skip.shape}")
            x = torch.cat([x, skip], dim=1)
            print(f"dec{i} concat in   : {x.shape}")

            # Cross-attention (simplified: add context embedding as bias)
            if context_embedding is not None:
                b, c_feat, h, w = x.shape
                ctx = context_embedding
                if ctx.shape[1] != c_feat:
                    repeat = (c_feat + ctx.shape[1] - 1) // ctx.shape[1]
                    ctx = ctx.repeat(1, repeat)[:, :c_feat]
                ctx = ctx.view(b, c_feat, 1, 1)
                x = x + ctx
            x = dec_block(x)
            print(f"dec{i} after block : {x.shape}")
            x = spatial_attention_block(x)
            print(f"dec{i} after attn  : {x.shape}")

        x = self.dec_blocks[-1](x)
        print(f"dec_final out      : {x.shape}")
        residual = self.out_conv(x)
        print(f"residual out       : {residual.shape}")
        return residual

__all__ = ["ContextualResidualUNetV4Torch"] 