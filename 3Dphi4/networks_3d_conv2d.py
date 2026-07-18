"""3D score network whose spatial convolutions use only ``nn.Conv2d``.

This is a controlled dimensional-reduction port of ``4Dphi4/networks_4d.py``.
The U-Net topology, time embedding, non-residual blocks, and periodic
boundaries are unchanged; channel widths and the output bias are explicit
configuration options.  A periodic 3D
convolution is represented exactly by rolling the first spatial axis, stacking
the three taps on the channel axis, folding that spatial axis into the batch,
and applying one ordinary 2D convolution to the remaining two axes.

Inputs and outputs use ``(B, C, Lx, Ly, Lz)``.  No Gaussian score baseline or
other score decomposition is implemented in this module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CHANNELS_3D_CONV2D = (4, 8, 16, 32)


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale, requires_grad=False
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        projected = t[:, None] * self.W[None, :] * (2.0 * np.pi)
        return torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)


class Conv3dCircularVia2d(nn.Module):
    """Exact periodic 3D convolution backed by a single ``nn.Conv2d``.

    ``kernel_size=3`` stacks the first-axis taps ``(-1, 0, +1)`` as three
    channel blocks.  ``stride=2`` subsamples that first axis explicitly while
    the backing Conv2d applies the same stride to the other two axes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str = "circular",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        if kernel_size == 3:
            if padding != 1 or padding_mode != "circular":
                raise ValueError(
                    "kernel_size=3 requires padding=1 and circular padding"
                )
            self.conv = nn.Conv2d(
                3 * in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                padding_mode="circular",
                bias=bias,
            )
        elif kernel_size == 1:
            if padding != 0:
                raise ValueError("kernel_size=1 requires padding=0")
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=bias,
            )
        else:
            raise NotImplementedError(
                f"only kernel_size 1 and 3 are supported, got {kernel_size}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected (B,C,Lx,Ly,Lz), got {tuple(x.shape)}")
        batch = x.shape[0]
        if self.kernel_size == 3:
            # At output coordinate i these blocks supply x[i-1], x[i], x[i+1].
            h = torch.cat(
                [torch.roll(x, shift, dims=2) for shift in (1, 0, -1)], dim=1
            )
        else:
            h = x
        if self.stride != 1:
            h = h[:, :, :: self.stride]
        first_out = h.shape[2]
        h = h.movedim(2, 1).reshape(
            batch * first_out, h.shape[1], *h.shape[3:]
        )
        h = self.conv(h)
        return h.reshape(batch, first_out, *h.shape[1:]).movedim(1, 2)


def convert_conv2d_wrappers_to_native(module: nn.Module) -> nn.Module:
    """In-place conversion to an exactly weight-matched native Conv3d control.

    This is used only for the scientific A/B training control.  The requested
    Conv2d-only model never calls this function.
    """
    for name, child in list(module.named_children()):
        if not isinstance(child, Conv3dCircularVia2d):
            convert_conv2d_wrappers_to_native(child)
            continue
        backing = child.conv
        has_bias = backing.bias is not None
        if child.kernel_size == 3:
            in_channels = backing.in_channels // 3
            native = nn.Conv3d(
                in_channels,
                backing.out_channels,
                kernel_size=3,
                stride=child.stride,
                padding=1,
                padding_mode="circular",
                bias=has_bias,
                device=backing.weight.device,
                dtype=backing.weight.dtype,
            )
            mapped_weight = (
                backing.weight.reshape(
                    backing.out_channels, 3, in_channels, 3, 3
                )
                .permute(0, 2, 1, 3, 4)
                .contiguous()
            )
        else:
            native = nn.Conv3d(
                backing.in_channels,
                backing.out_channels,
                kernel_size=1,
                stride=child.stride,
                bias=has_bias,
                device=backing.weight.device,
                dtype=backing.weight.dtype,
            )
            mapped_weight = backing.weight.unsqueeze(2)
        with torch.no_grad():
            native.weight.copy_(mapped_weight)
            if has_bias:
                native.bias.copy_(backing.bias)
        setattr(module, name, native)
    return module


def upsample_nearest_3d(x: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    for axis, target in enumerate(size, start=2):
        current = x.shape[axis]
        if target % current != 0:
            raise ValueError(f"cannot nearest-upsample axis {axis}: {current} -> {target}")
        x = x.repeat_interleave(target // current, dim=axis)
    return x


class Block3DVia2D(nn.Module):
    """Two convolutions with additive time conditioning, matching the 4D block."""

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int) -> None:
        super().__init__()
        conv = Conv3dCircularVia2d
        self.conv1 = conv(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(max(1, out_ch // 8), out_ch)
        self.temb_proj = nn.Linear(embed_dim, out_ch)
        self.conv2 = conv(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(max(1, out_ch // 8), out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.temb_proj(temb)[(...,) + (None,) * 3]
        h = self.act(self.norm1(h))
        return self.act(self.norm2(self.conv2(h)))


class Attention3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(max(1, channels // 8), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        spatial_shape = x.shape[2:]
        h = self.norm(x).reshape(batch, channels, -1).permute(0, 2, 1)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(batch, channels, *spatial_shape)
        return x + h


class NCSNpp3DConv2d(nn.Module):
    """Current 4D architecture reduced to 3D, using Conv2d exclusively."""

    def __init__(
        self,
        marginal_prob_std_fn,
        embed_dim: int = 256,
        use_attention: bool = False,
        in_ch: int = 1,
        channels: tuple[int, int, int, int] = DEFAULT_CHANNELS_3D_CONV2D,
        final_bias: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or any(int(channel) <= 0 for channel in channels):
            raise ValueError(f"channels must contain four positive widths, got {channels}")
        self.marginal_prob_std = marginal_prob_std_fn
        self.channels = tuple(int(channel) for channel in channels)
        self.final_bias = bool(final_bias)
        c1, c2, c3, c4 = self.channels
        conv = Conv3dCircularVia2d

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim), nn.Linear(embed_dim, embed_dim)
        )

        self.conv_in = conv(in_ch, c1, 3, padding=1)
        self.res1a = Block3DVia2D(c1, c1, embed_dim)
        self.res1b = Block3DVia2D(c1, c1, embed_dim)
        self.down1 = conv(c1, c1, 3, stride=2, padding=1)

        self.res2a = Block3DVia2D(c1, c2, embed_dim)
        self.res2b = Block3DVia2D(c2, c2, embed_dim)
        self.down2 = conv(c2, c2, 3, stride=2, padding=1)

        self.res3a = Block3DVia2D(c2, c3, embed_dim)
        self.res3b = Block3DVia2D(c3, c3, embed_dim)
        self.down3 = conv(c3, c3, 3, stride=2, padding=1)

        self.res4a = Block3DVia2D(c3, c4, embed_dim)
        self.res4b = Block3DVia2D(c4, c4, embed_dim)
        self.mid1 = Block3DVia2D(c4, c4, embed_dim)
        self.mid_attn = Attention3D(c4) if use_attention else nn.Identity()
        self.mid2 = Block3DVia2D(c4, c4, embed_dim)

        self.tres4a = Block3DVia2D(c4 + c4, c4, embed_dim)
        self.tres4b = Block3DVia2D(c4 + c4, c3, embed_dim)
        self.up3 = conv(c3, c3, 3, padding=1)

        self.tres3a = Block3DVia2D(c3 + c3, c3, embed_dim)
        self.tres3b = Block3DVia2D(c3 + c3, c2, embed_dim)
        self.up2 = conv(c2, c2, 3, padding=1)

        self.tres2a = Block3DVia2D(c2 + c2, c2, embed_dim)
        self.tres2b = Block3DVia2D(c2 + c2, c1, embed_dim)
        self.up1 = conv(c1, c1, 3, padding=1)

        self.tres1a = Block3DVia2D(c1 + c1, c1, embed_dim)
        self.tres1b = Block3DVia2D(c1 + c1, c1, embed_dim)
        self.norm_out = nn.GroupNorm(max(1, c1 // 8), c1)
        self.conv_out = conv(c1, in_ch, 3, padding=1, bias=self.final_bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = F.silu(self.embed(t))

        h = self.conv_in(x)
        h1a = self.res1a(h, temb)
        h1b = self.res1b(h1a, temb)
        h = self.down1(h1b)

        h2a = self.res2a(h, temb)
        h2b = self.res2b(h2a, temb)
        h = self.down2(h2b)

        h3a = self.res3a(h, temb)
        h3b = self.res3b(h3a, temb)
        h = self.down3(h3b)

        h4a = self.res4a(h, temb)
        h4b = self.res4b(h4a, temb)
        h = self.mid1(h4b, temb)
        h = self.mid_attn(h)
        h = self.mid2(h, temb)

        h = self.tres4a(torch.cat((h, h4b), dim=1), temb)
        h = self.tres4b(torch.cat((h, h4a), dim=1), temb)
        h = self.up3(upsample_nearest_3d(h, h3b.shape[2:]))

        h = self.tres3a(torch.cat((h, h3b), dim=1), temb)
        h = self.tres3b(torch.cat((h, h3a), dim=1), temb)
        h = self.up2(upsample_nearest_3d(h, h2b.shape[2:]))

        h = self.tres2a(torch.cat((h, h2b), dim=1), temb)
        h = self.tres2b(torch.cat((h, h2a), dim=1), temb)
        h = self.up1(upsample_nearest_3d(h, h1b.shape[2:]))

        h = self.tres1a(torch.cat((h, h1b), dim=1), temb)
        h = self.tres1b(torch.cat((h, h1a), dim=1), temb)
        h = self.conv_out(F.silu(self.norm_out(h)))
        std = self.marginal_prob_std(t).view(-1, 1, 1, 1, 1)
        return h / std


__all__ = [
    "Attention3D",
    "Block3DVia2D",
    "Conv3dCircularVia2d",
    "DEFAULT_CHANNELS_3D_CONV2D",
    "GaussianFourierProjection",
    "NCSNpp3DConv2d",
    "convert_conv2d_wrappers_to_native",
    "upsample_nearest_3d",
]
