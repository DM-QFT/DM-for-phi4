"""
EDM2-style network (Karras 2024 *Analyzing and Improving the Training Dynamics
of Diffusion Models*) for 2D lattice field configurations. Mirrors the
structure of NCSNpp2D in networks.py but uses magnitude-preserving (MP) layers
throughout, with forced weight normalisation applied each forward (during
training).

Design choices for this codebase:
- Pixel-wise normalisation is OMITTED (kept GroupNorm) per user preference,
  since batch_size=2048 in our setup makes GroupNorm robust.
- All Conv/Linear bias terms removed.
- All activations are mp_silu (SiLU / 0.596 to preserve unit variance).
- Residual sums use mp_sum (t=0.3 default — slight bias toward identity branch).
- Skip concatenations in U-Net up-path use mp_cat.
- Strided downsampling uses MPConv2dStrided (replaces stride-2 conv).
- Default kept as periodic BC (matches NCSNpp2D for lattice φ⁴).
- The final output conv is also MPConv2d. The external EDM preconditioning
  (c_out, c_skip, etc.) handles output scaling; the network itself learns
  a roughly unit-variance F_θ.

Forward signature is identical to NCSNpp2D:
    out = net(x, t)
so this is a drop-in replacement.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mp_layers import (
    MPConv2d, MPConv2dStrided, MPLinear,
    MPGaussianFourierProjection,
    mp_silu, mp_cat, mp_sum,
)


# ----------------------- ResBlock with MP layers -----------------------

class MPResnetBlock2D(nn.Module):
    """EDM2-style residual block: GroupNorm + MP-SiLU + MPConv2d, with time
    embedding gated additively. Residual merge via mp_sum.

    Ablation flags:
      use_bias: re-introduce bias to conv/linear layers
      use_mp_silu: use mp_silu (SiLU/0.596) if True, else plain SiLU
      use_mp_sum: use mp_sum for residual; else plain (skip + h)
    """

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int,
                 periodic: bool = True, residual_t: float = 0.3,
                 use_bias: bool = False, use_mp_silu: bool = True,
                 use_mp_sum: bool = True):
        super().__init__()
        pad_mode = "circular" if periodic else "zeros"
        self.norm1 = nn.GroupNorm(max(1, min(8, in_ch // 4)), in_ch)
        self.conv1 = MPConv2d(in_ch, out_ch, 3, padding_mode=pad_mode,
                              use_bias=use_bias)
        self.dense_t = MPLinear(embed_dim, out_ch, use_bias=use_bias)
        self.norm2 = nn.GroupNorm(max(1, min(8, out_ch // 4)), out_ch)
        self.conv2 = MPConv2d(out_ch, out_ch, 3, padding_mode=pad_mode,
                              use_bias=use_bias)
        self.skip = (MPConv2d(in_ch, out_ch, 1, padding=0, use_bias=use_bias)
                     if in_ch != out_ch else None)
        self.residual_t = residual_t
        self._act = mp_silu if use_mp_silu else F.silu
        self._use_mp_sum = use_mp_sum

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self._act(self.norm1(x)))
        h = h + self.dense_t(self._act(temb))[..., None, None]
        h = self.conv2(self._act(self.norm2(h)))
        skip = x if self.skip is None else self.skip(x)
        if self._use_mp_sum:
            return mp_sum(skip, h, t=self.residual_t)
        return skip + h


# ----------------------- Attention (optional) -----------------------

class MPAttnBlock2D(nn.Module):
    """Optional self-attention with magnitude-preserving Q/K/V projections."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(max(1, min(8, channels // 4)), channels)
        self.qkv = MPConv2d(channels, channels * 3, 1, padding=0)
        self.proj = MPConv2d(channels, channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(dim=1)
        attn = torch.softmax(
            torch.einsum("bhcn,bhcm->bhnm", q, k) / np.sqrt(C // self.num_heads),
            dim=-1,
        )
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(B, C, H, W)
        return mp_sum(x, self.proj(out), t=0.3)


# ----------------------- MPNCSNpp2D -----------------------

class MPNCSNpp2D(nn.Module):
    """Magnitude-preserving NCSN++ 2D U-Net. Drop-in replacement for NCSNpp2D
    when training with the EDM2 recipe.

    Forward(x, t):
        x: (B, 1, L, L) — noisy lattice configuration (after c_in scaling done
            externally by EDMPrecond)
        t: (B,)         — c_noise = ¼·ln σ (handed in by EDMPrecond)
        returns: (B, 1, L, L) — raw F_θ output (EDM preconditioning applied
            externally to produce the denoiser D_θ).

    Note: unlike NCSNpp2D, this network does NOT divide its output by
    marginal_prob_std(t). It just returns F_θ directly, leaving all the
    sigma-aware scaling to the EDM preconditioning wrapper.
    """

    def __init__(self, channels=(16, 32, 64, 128), embed_dim: int = 256,
                 use_attention: bool = False, periodic: bool = True,
                 l_cond: bool = False,
                 use_bias: bool = False, use_mp_silu: bool = True,
                 use_mp_sum: bool = True):
        super().__init__()
        self.l_cond = l_cond
        self.use_mp_silu = use_mp_silu
        self._act = mp_silu if use_mp_silu else F.silu
        pad_mode = "circular" if periodic else "zeros"

        kw = dict(use_bias=use_bias, use_mp_silu=use_mp_silu, use_mp_sum=use_mp_sum)

        # Time embedding
        self.t_fourier = MPGaussianFourierProjection(embed_dim)
        self.t_lin1 = MPLinear(embed_dim, embed_dim, use_bias=use_bias)
        self.t_lin2 = MPLinear(embed_dim, embed_dim, use_bias=use_bias)
        if l_cond:
            self.L_fourier = MPGaussianFourierProjection(embed_dim)
            self.L_lin = MPLinear(embed_dim, embed_dim, use_bias=use_bias)

        c = list(channels)
        self.conv_in = MPConv2d(1, c[0], 3, padding_mode=pad_mode, use_bias=use_bias)

        self.res1a = MPResnetBlock2D(c[0], c[0], embed_dim, periodic=periodic, **kw)
        self.res1b = MPResnetBlock2D(c[0], c[0], embed_dim, periodic=periodic, **kw)
        self.down1 = MPConv2dStrided(c[0], c[0], 3, stride=2, padding_mode=pad_mode, use_bias=use_bias)

        self.res2a = MPResnetBlock2D(c[0], c[1], embed_dim, periodic=periodic, **kw)
        self.res2b = MPResnetBlock2D(c[1], c[1], embed_dim, periodic=periodic, **kw)
        self.down2 = MPConv2dStrided(c[1], c[1], 3, stride=2, padding_mode=pad_mode, use_bias=use_bias)

        self.res3a = MPResnetBlock2D(c[1], c[2], embed_dim, periodic=periodic, **kw)
        self.res3b = MPResnetBlock2D(c[2], c[2], embed_dim, periodic=periodic, **kw)
        self.down3 = MPConv2dStrided(c[2], c[2], 3, stride=2, padding_mode=pad_mode, use_bias=use_bias)

        self.res4a = MPResnetBlock2D(c[2], c[3], embed_dim, periodic=periodic, **kw)
        self.res4b = MPResnetBlock2D(c[3], c[3], embed_dim, periodic=periodic, **kw)

        self.mid1 = MPResnetBlock2D(c[3], c[3], embed_dim, periodic=periodic, **kw)
        self.mid_attn = (MPAttnBlock2D(c[3]) if use_attention else nn.Identity())
        self.mid2 = MPResnetBlock2D(c[3], c[3], embed_dim, periodic=periodic, **kw)

        self.tres4a = MPResnetBlock2D(c[3] + c[3], c[3], embed_dim, periodic=periodic, **kw)
        self.tres4b = MPResnetBlock2D(c[3] + c[3], c[2], embed_dim, periodic=periodic, **kw)
        self.up3 = MPConv2d(c[2], c[2], 3, padding_mode=pad_mode, use_bias=use_bias)

        self.tres3a = MPResnetBlock2D(c[2] + c[2], c[2], embed_dim, periodic=periodic, **kw)
        self.tres3b = MPResnetBlock2D(c[2] + c[2], c[1], embed_dim, periodic=periodic, **kw)
        self.up2 = MPConv2d(c[1], c[1], 3, padding_mode=pad_mode, use_bias=use_bias)

        self.tres2a = MPResnetBlock2D(c[1] + c[1], c[1], embed_dim, periodic=periodic, **kw)
        self.tres2b = MPResnetBlock2D(c[1] + c[1], c[0], embed_dim, periodic=periodic, **kw)
        self.up1 = MPConv2d(c[0], c[0], 3, padding_mode=pad_mode, use_bias=use_bias)

        self.tres1a = MPResnetBlock2D(c[0] + c[0], c[0], embed_dim, periodic=periodic, **kw)
        self.tres1b = MPResnetBlock2D(c[0] + c[0], c[0], embed_dim, periodic=periodic, **kw)

        self.norm_out = nn.GroupNorm(max(1, min(8, c[0] // 4)), c[0])
        self.conv_out = MPConv2d(c[0], 1, 3, padding_mode=pad_mode, use_bias=use_bias)

    def _temb(self, t: torch.Tensor, spatial_size: int) -> torch.Tensor:
        h = self.t_lin1(self._act(self.t_fourier(t)))
        if self.l_cond:
            L_inv = torch.full_like(t, 1.0 / spatial_size)
            h2 = self.L_lin(self._act(self.L_fourier(L_inv)))
            h = mp_sum(h, h2, t=0.5)
        return self._act(self.t_lin2(self._act(h)))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self._temb(t, x.shape[-1])

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

        # Up path: nearest-neighbour upsample + mp_cat skip
        h = self.tres4a(mp_cat(h, h4b, dim=1), temb)
        h = self.tres4b(mp_cat(h, h4a, dim=1), temb)
        h = self.up3(F.interpolate(h, size=h3b.shape[-2:], mode="nearest"))

        h = self.tres3a(mp_cat(h, h3b, dim=1), temb)
        h = self.tres3b(mp_cat(h, h3a, dim=1), temb)
        h = self.up2(F.interpolate(h, size=h2b.shape[-2:], mode="nearest"))

        h = self.tres2a(mp_cat(h, h2b, dim=1), temb)
        h = self.tres2b(mp_cat(h, h2a, dim=1), temb)
        h = self.up1(F.interpolate(h, size=h1b.shape[-2:], mode="nearest"))

        h = self.tres1a(mp_cat(h, h1b, dim=1), temb)
        h = self.tres1b(mp_cat(h, h1a, dim=1), temb)

        h = self._act(self.norm_out(h))
        # NOTE: no division by marginal_prob_std — EDM preconditioning applies
        # c_out externally. This network just returns raw F_θ.
        return self.conv_out(h)
