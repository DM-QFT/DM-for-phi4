"""
Magnitude-preserving (MP) primitives from EDM2 (Karras 2024).

The key building blocks:
- normalize(): unit-norm projection (used for forced weight norm)
- MPConv2d / MPLinear: bias-free conv/linear with forced weight normalisation
  during training, plus magnitude-preserving scaling (1/sqrt(fan_in))
- mp_silu(): SiLU rescaled so output preserves unit variance
- mp_cat() / mp_sum(): magnitude-preserving concatenation / weighted sum
- MPGaussianFourierProjection: same Fourier embedding as before but with
  fixed (non-learnable) freqs/phases, output scaled to unit variance.

Pixel-wise normalisation is intentionally omitted (per user preference for
this codebase): with batch_size=2048 GroupNorm is fine, and per ablation
in Karras 2024 Fig. 2 the MP-layer + forced-weight-norm change alone gives
most of the training-stability benefit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, dim=None, eps=1e-4):
    """Project to unit Frobenius norm along all dims except the first one.

    For a conv weight of shape (out, in, kh, kw), this normalises each output
    channel's flattened weight vector to unit length (modulo the sqrt(fan_in)
    scale absorbed in MPConv.forward).
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = eps + norm * np.sqrt(norm.numel() / max(x.numel(), 1))
    return x / norm.to(x.dtype)


def mp_silu(x):
    """SiLU divided by 0.596 so output has unit variance on unit-variance input."""
    return F.silu(x) / 0.596


def mp_cat(a, b, dim=1, t=0.5):
    """Magnitude-preserving concatenation. t in [0,1] biases toward a or b."""
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


def mp_sum(a, b, t=0.5):
    """Magnitude-preserving (1-t)*a + t*b, normalised so unit-variance inputs
    give a unit-variance output."""
    return ((1 - t) * a + t * b) / np.sqrt((1 - t) ** 2 + t ** 2)


class MPConv2d(nn.Module):
    """Bias-free Conv2d with forced unit-norm weights (per output channel) +
    magnitude-preserving scaling 1/sqrt(fan_in). EDM2 Eq. (3)-(4).

    For ablation experiments, supports `use_bias` (re-introduces bias param).
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=None, padding_mode="zeros", use_bias=False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.padding = padding if padding is not None else kernel_size[0] // 2
        self.padding_mode = padding_mode
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if use_bias else None

    def forward(self, x, gain: float = 1.0):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalisation
        w = normalize(w)
        w = w * (gain / np.sqrt(self.weight[0].numel()))
        w = w.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        if self.padding_mode != "zeros":
            x = F.pad(x, [self.padding] * 4, mode=self.padding_mode)
            return F.conv2d(x, w, bias=b, padding=0)
        return F.conv2d(x, w, bias=b, padding=self.padding)


class MPConv2dStrided(MPConv2d):
    """Strided MPConv2d for downsampling."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=2,
                 padding=None, padding_mode="zeros", use_bias=False):
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=padding, padding_mode=padding_mode,
                         use_bias=use_bias)
        self.stride = stride

    def forward(self, x, gain: float = 1.0):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))
        w = normalize(w)
        w = w * (gain / np.sqrt(self.weight[0].numel()))
        w = w.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        if self.padding_mode != "zeros":
            x = F.pad(x, [self.padding] * 4, mode=self.padding_mode)
            return F.conv2d(x, w, bias=b, padding=0, stride=self.stride)
        return F.conv2d(x, w, bias=b, padding=self.padding, stride=self.stride)


class MPLinear(nn.Module):
    """Bias-free Linear with forced unit-norm weights + 1/sqrt(fan_in) scaling."""

    def __init__(self, in_features, out_features, use_bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None

    def forward(self, x, gain: float = 1.0):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))
        w = normalize(w)
        w = w * (gain / np.sqrt(self.weight[0].numel()))
        w = w.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class MPGaussianFourierProjection(nn.Module):
    """Karras 2024-style fixed (non-learnable) Fourier embedding.

    Output dimension = embed_dim. Uses cos-only with uniform phase to give
    unit-variance output on scalar input. Matches Karras EDM2 `MPFourier`.
    """

    def __init__(self, embed_dim, bandwidth: float = 1.0):
        super().__init__()
        self.register_buffer(
            "freqs", 2 * np.pi * torch.randn(embed_dim) * bandwidth
        )
        self.register_buffer(
            "phases", 2 * np.pi * torch.rand(embed_dim)
        )

    def forward(self, x):
        y = x.float().outer(self.freqs.float()) + self.phases.float()
        return (torch.cos(y) * np.sqrt(2)).to(x.dtype)


# Convenience helper: applied once after model construction to ensure the
# initial weights satisfy unit-norm constraint (saves the first forced-norm
# step during training).
def apply_forced_weight_norm(module: nn.Module):
    """In-place: normalise every MPConv2d/MPLinear weight to unit Frobenius."""
    for m in module.modules():
        if isinstance(m, (MPConv2d, MPLinear)):
            with torch.no_grad():
                m.weight.copy_(normalize(m.weight.to(torch.float32)).to(m.weight.dtype))
