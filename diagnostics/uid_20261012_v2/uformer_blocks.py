"""LeWinTransformerBlock / BasicUformerLayer for the UID-20261012-v2 diagnostic.

Adapted from ZhendongWang6/Uformer, commit 65fc970a8ffc09605faca74ed016ee93c9ad8a36, ``model.py``
(SHA-256 03a2c6e50b4342e03a457dd10537c14d0f832ce186da49c781fbd9d7027904a9; byte copy in ``third_party/uformer_65fc970``).
Copyright (c) 2022 Zhendong Wang, MIT License (``third_party/uformer_65fc970/LICENSE``).

Adaptations; none changes the computation of the configured path (``checks/test_upstream_equivalence.py`` compares
outputs and gradients with the pinned file):

* einops ``rearrange``/``repeat`` become the equivalent ``view``/``permute``/``reshape``; timm ``to_2tuple`` and
  ``trunc_normal_`` become a tuple and ``torch.nn.init.trunc_normal_`` (same absolute bounds -2, 2);
* only the configured options are kept: linear token projection, LeFF without ECA, no modulator, no cross modulator,
  no input mask, self-attention only (the key/value ``ratio`` repeat is 1);
* dropout, attention dropout and stochastic depth are omitted. They are 0 in this diagnostic, where ``nn.Dropout(0)``
  and ``DropPath(0)`` are the identity; the upstream ``Uformer`` class defaults to ``drop_path_rate=0.1``;
* explicit guards: the token grid must be square, equal ``input_resolution`` and be divisible by the window
  (upstream silently infers ``H = W = sqrt(L)``).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def window_partition(x: torch.Tensor, win_size: int) -> torch.Tensor:
    """[B, H, W, C] -> [B * nW, win, win, C] (upstream with ``dilation_rate = 1``)."""
    b, h, w, c = x.shape
    x = x.view(b, h // win_size, win_size, w // win_size, win_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, c)


def window_reverse(windows: torch.Tensor, win_size: int, h: int, w: int) -> torch.Tensor:
    """Inverse of :func:`window_partition`: [B * nW, win, win, C] -> [B, H, W, C]."""
    b = windows.shape[0] // ((h // win_size) * (w // win_size))
    x = windows.view(b, h // win_size, w // win_size, win_size, win_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


def shift_attention_mask(h: int, w: int, win_size: int, shift_size: int, like: torch.Tensor) -> torch.Tensor:
    """The SW-MSA mask of upstream ``LeWinTransformerBlock.forward``: [nW, win*win, win*win], 0 within a region, -100
    between tokens that the cyclic shift brought from different regions."""
    shift_mask = torch.zeros((1, h, w, 1)).type_as(like)
    slices = (slice(0, -win_size), slice(-win_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for hs in slices:
        for ws in slices:
            shift_mask[:, hs, ws, :] = cnt
            cnt += 1
    windows = window_partition(shift_mask, win_size).view(-1, win_size * win_size)
    mask = windows.unsqueeze(1) - windows.unsqueeze(2)
    return mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))


class LinearProjection(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, bias: bool = True) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b_, n, c = x.shape
        q = self.to_q(x).reshape(b_, n, 1, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(x).reshape(b_, n, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        return q[0], kv[0], kv[1]


class WindowAttention(nn.Module):
    """W-MSA with a learned relative position bias (the only positional information in the block)."""

    def __init__(self, dim: int, win_size: tuple[int, int], num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        wh, ww = win_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * wh - 1) * (2 * ww - 1), num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :]).permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += wh - 1
        relative_coords[:, :, 1] += ww - 1
        relative_coords[:, :, 0] *= 2 * ww - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        q, k, v = self.qkv(x)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n, n, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)
        return self.proj((attn @ v).transpose(1, 2).reshape(b_, n, c))


class LeFF(nn.Module):
    """Locally-enhanced feed-forward: Linear-GELU, depthwise 3x3 conv-GELU on the token grid, Linear."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), nn.GELU()
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, hw, _ = x.shape
        hh = int(math.sqrt(hw))
        x = self.linear1(x)
        x = self.dwconv(x.view(bs, hh, hh, -1).permute(0, 3, 1, 2))
        return self.linear2(x.permute(0, 2, 3, 1).reshape(bs, hh * hh, -1))


class LeWinTransformerBlock(nn.Module):
    """Pre-norm W-MSA/SW-MSA and LeFF, each with an internal residual (upstream defaults otherwise)."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        win_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        # upstream rule: a grid no larger than the window is one unshifted window
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        if not 0 <= self.shift_size < self.win_size:
            raise ValueError("shift_size must be in [0, win_size)")
        h, w = self.input_resolution
        if h != w or h % self.win_size:
            raise ValueError(f"token grid {h}x{w} must be square and divisible by the window {self.win_size}")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, (self.win_size, self.win_size), num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = LeFF(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, c = x.shape
        h, w = self.input_resolution
        if length != h * w:
            raise ValueError(f"{length} tokens do not form the {h}x{w} grid")
        s, win = self.shift_size, self.win_size
        attn_mask = shift_attention_mask(h, w, win, s, x) if s > 0 else None
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if s > 0:
            x = torch.roll(x, shifts=(-s, -s), dims=(1, 2))
        windows = window_partition(x, win).view(-1, win * win, c)
        attn_windows = self.attn(windows, mask=attn_mask).view(-1, win, win, c)
        x = window_reverse(attn_windows, win, h, w)
        if s > 0:
            x = torch.roll(x, shifts=(s, s), dims=(1, 2))
        x = shortcut + x.view(b, h * w, c)
        return x + self.mlp(self.norm2(x))


class BasicUformerLayer(nn.Module):
    """``depth`` LeWin blocks; with ``shift_flag`` every odd block is shifted by ``win_size // 2``."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        win_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        shift_flag: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                LeWinTransformerBlock(
                    dim,
                    input_resolution,
                    num_heads,
                    win_size=win_size,
                    shift_size=win_size // 2 if shift_flag and i % 2 == 1 else 0,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


def init_weights(m: nn.Module) -> None:
    """Upstream ``Uformer._init_weights``: Linear trunc-normal(std 0.02) with zero bias, LayerNorm (1, 0)."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
