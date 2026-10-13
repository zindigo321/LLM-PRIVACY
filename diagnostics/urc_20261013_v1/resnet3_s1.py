"""URC-20261013-v1 model: ``ResNet3S1AutoEncoder``, an isolated stride-1 diagnostic variant of the ResNet-3.

The production ``cg_fedllm.compression.autoencoder.ResNetAutoEncoder`` (HEAD blob cf38fd3f7749a2cdd45864e41f22d3ab0dc9e3db,
unchanged since the P5-A execution commit 6603bd8) hardcodes stride 2 in its six encoder convolutions and stride 2 with
output_padding 1 in its six decoder transposed convolutions, and accepts only sides divisible by 2^6 = 64. This file
re-declares that network layer by layer, in the same construction order, with one intervention: every stride is 1 and
every output_padding is 0. It imports nothing from ``cg_fedllm``; the production model is neither used nor modified.
``checks/test_urc_model.py`` compares the two.

Encoder:  sampling = ReflectionPad(k//2), Conv k x k (1 -> stem_ch, no bias), BN, ReLU,
                     6 x [Conv3x3 stride 1 pad 1 (no bias), BN, ReLU], channels stem_ch -> 2 -> 4 -> ... -> 64
          core     = 3 x ResidualBlock(64)
Decoder:  core     = 3 x ResidualBlock(64)
          sampling = 6 x [ConvTranspose3x3 stride 1 pad 1 output_pad 0 (no bias), BN, ReLU], 64 -> ... -> stem_ch,
                     ReflectionPad(3), Conv7x7 (stem_ch -> 1, bias), Tanh
ResidualBlock(c): x + [ReflectionPad(1), Conv3x3, BN, ReLU, ReflectionPad(1), Conv3x3, BN](x)

Submodules are named ``encoder.sampling`` / ``encoder.core`` / ``decoder.core`` / ``decoder.sampling``, as in the UID
Uformer auto-encoder, so the r1 measurement groups apply unchanged. Name mapping to the production state dict:
encoder.sampling.<i> = encoder.net.<i> (i < 22), encoder.core.<j> = encoder.net.<22 + j>, decoder.core.<j> =
decoder.net.<j>, decoder.sampling.<i> = decoder.net.<3 + i>.

The geometry is explicit, not a bypassed validator: the spatial factor is 1, so the latent grid is the input grid; the
input must be exactly [B, in_channels, *input_hw]; reflection padding needs every input side to exceed the stem and
head paddings, and a latent grid of at least 2 x 2 for the residual blocks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

VARIANT = "resnet3_stride1_diagnostic"
STRUCTURE_ATTRS = (
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "output_padding",
    "dilation",
    "groups",
    "padding_mode",
    "num_features",
    "eps",
    "momentum",
    "affine",
    "track_running_stats",
    "inplace",
)


@dataclass(frozen=True)
class ResNet3S1Config:
    """Every architecture-defining value; strides are fixed at 1 by the variant, not configurable."""

    input_hw: tuple[int, int] = (32, 32)
    in_channels: int = 1
    stem_kernel: int = 3
    stem_channels: int = 1
    down_channels: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    num_res_blocks: int = 3
    final_kernel: int = 7

    def __post_init__(self) -> None:
        for name in ("input_hw", "down_channels"):
            object.__setattr__(self, name, tuple(int(v) for v in getattr(self, name)))
        h, w = self.input_hw
        pad = max(self.stem_kernel // 2, self.final_kernel // 2)
        if min(h, w) <= pad:
            raise ValueError(f"input {h}x{w}: reflection padding {pad} needs every side > {pad}")
        if min(h, w) < 2:
            raise ValueError(f"latent grid {h}x{w} too small for the residual blocks' reflection padding")
        if len(self.down_channels) != 6:
            raise ValueError("the ResNet-3 variant has exactly six sampling stages")

    @property
    def strides(self) -> tuple[int, ...]:
        return (1,) * len(self.down_channels)

    @property
    def output_paddings(self) -> tuple[int, ...]:
        return tuple(s - 1 for s in self.strides)

    @property
    def latent_channels(self) -> int:
        return self.down_channels[-1]

    @property
    def latent_hw(self) -> tuple[int, int]:
        return self.input_hw  # spatial factor 1

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_channels, *self.latent_hw)

    @property
    def latent_element_ratio(self) -> float:
        c, h, w = self.latent_shape
        return (c * h * w) / (self.in_channels * self.input_hw[0] * self.input_hw[1])

    def to_dict(self) -> dict[str, Any]:
        d = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
        return {
            "variant": VARIANT,
            **d,
            "strides": list(self.strides),
            "output_paddings": list(self.output_paddings),
        }


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNet3S1Encoder(nn.Module):
    def __init__(self, cfg: ResNet3S1Config) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(cfg.stem_kernel // 2),
            nn.Conv2d(cfg.in_channels, cfg.stem_channels, kernel_size=cfg.stem_kernel, bias=False),
            nn.BatchNorm2d(cfg.stem_channels),
            nn.ReLU(inplace=False),
        ]
        chans = [cfg.stem_channels, *cfg.down_channels]
        for c_in, c_out, stride in zip(chans[:-1], chans[1:], cfg.strides):
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=False),
            ]
        self.sampling = nn.Sequential(*layers)
        self.core = nn.Sequential(*[ResidualBlock(cfg.latent_channels) for _ in range(cfg.num_res_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(self.sampling(x))


class ResNet3S1Decoder(nn.Module):
    def __init__(self, cfg: ResNet3S1Config) -> None:
        super().__init__()
        # construction order of the production decoder: residual blocks first, then the sampling stack
        self.core = nn.Sequential(*[ResidualBlock(cfg.latent_channels) for _ in range(cfg.num_res_blocks)])
        chans = [cfg.stem_channels, *cfg.down_channels]
        layers: list[nn.Module] = []
        for c_in, c_out, stride, out_pad in zip(
            reversed(chans[1:]), reversed(chans[:-1]), reversed(cfg.strides), reversed(cfg.output_paddings)
        ):
            layers += [
                nn.ConvTranspose2d(
                    c_in, c_out, kernel_size=3, stride=stride, padding=1, output_padding=out_pad, bias=False
                ),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=False),
            ]
        layers += [
            nn.ReflectionPad2d(cfg.final_kernel // 2),
            nn.Conv2d(cfg.stem_channels, cfg.in_channels, kernel_size=cfg.final_kernel, bias=True),
            nn.Tanh(),
        ]
        self.sampling = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.sampling(self.core(z))


class ResNet3S1AutoEncoder(nn.Module):
    """x -> Encoder -> z [B, 64, H, W] -> Decoder -> x_hat; no skip, no global shortcut, PyTorch default init."""

    def __init__(self, cfg: ResNet3S1Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ResNet3S1Config()
        self.encoder = ResNet3S1Encoder(self.cfg)
        self.decoder = ResNet3S1Decoder(self.cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.cfg.in_channels, *self.cfg.input_hw)
        if x.dim() != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"input shape {tuple(x.shape)} != [B, {expected}]")
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        expected = (self.cfg.latent_channels, *self.cfg.latent_hw)
        if z.dim() != 4 or tuple(z.shape[1:]) != expected:
            raise ValueError(f"latent shape {tuple(z.shape)} != [B, {expected}]")
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def num_parameters(self) -> dict[str, int]:
        parts = {
            f"{side}.{part}": sum(p.numel() for p in getattr(getattr(self, side), part).parameters())
            for side in ("encoder", "decoder")
            for part in ("sampling", "core")
        }
        parts["total"] = sum(parts.values())
        return parts


def structure(seq: nn.Module) -> list[dict[str, Any]]:
    """Layer types and hyper-parameters of a sequential stack (used to compare corresponding stacks)."""
    out = []
    for layer in seq:
        d: dict[str, Any] = {"type": type(layer).__name__}
        for a in STRUCTURE_ATTRS:
            if hasattr(layer, a):
                v = getattr(layer, a)
                d[a] = list(v) if isinstance(v, tuple) else v
        if isinstance(getattr(layer, "bias", None), (torch.Tensor, type(None))) and hasattr(layer, "bias"):
            d["bias"] = layer.bias is not None
        out.append(d)
    return out


def production_name(name: str, num_res_blocks: int = 3) -> str:
    """The production ``ResNetAutoEncoder`` state-dict key of a key of this variant (fixed mapping)."""
    side, part, idx, rest = (name.split(".", 3) + [""])[:4]
    i = int(idx)
    offset = {("encoder", "sampling"): 0, ("encoder", "core"): 22, ("decoder", "core"): 0}
    j = i + offset.get((side, part), num_res_blocks)
    return ".".join(x for x in (side, "net", str(j), rest) if x)
