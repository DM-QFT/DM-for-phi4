# Score-based diffusion with a beta-dependent noise schedule for 2D SU(2) gauge configurations from su2_2d_heatbath.jl, the SU(2) version of
# ../SU3/train_su3_dm.py.  Links are unit quaternions U = a0 + i a⃗·σ stored as real (…, 4) tensors, su(2) elements
# Y = ξ·σ/2 as real 3-vectors ξ (T^a = σ^a/2, |Y|² = Σ ξ_a²).  Brownian motion on SU(2)^links is a random walk of exact
# heat-kernel increments U ← exp(iY) U, and the DSM target is the analytic heat-kernel score of the last increment; for
# SU(2) both are closed-form functions of θ = |ξ| (no eigh, no matrix_exp).  The score transforms like Ω(x) Y Ω(x)†, an
# SO(3) rotation of ξ; the network is the reduced L-CNN of the SU(3) code in quaternion arithmetic:
# plaquette → parallel-transport convolution → quaternion bilinear products → scalar-part gating → per-link readout.
import os
import time
import math
import h5py
import numpy as np
import torch
import triton
import triton.language as tl
from torch import nn
from torch_ema import ExponentialMovingAverage

torch.set_default_device("cuda")

L = 128
BETA = 200.0
DATA = f"/work/gw97/w97000/project/DM/2DSUN/trainningdata/cfgs_su2_beta={BETA}_{L}^2_n5120.h5"
SIGMA = 14.0  # Retain the old endpoint Σ(1)^2 = (14^2 - 1)/(2 log 14) = 36.945, close to Haar.
DELTA_MAX = 1.0  # variance per heat-kernel increment; exp(iY) = 1 first at θ = 4π, so the nearest wrapping image weighs e^{-79/Δ}
CH = 32
NLAYERS = 3
BATCH = 64
EPOCHS = 500
SAVE_EPOCHS = np.unique(np.rint(np.geomspace(1, EPOCHS, 10)).astype(int)).tolist()
OUT = f"/work/gw97/w97000/project/DM/2DSUN/results/su2_L{L}_beta{BETA}_lcnn_hk_c{CH}l{NLAYERS}_n5120_b{BATCH}_e{EPOCHS}_betanoise"
LR = 2e-4
EMA_DECAY = 0.999
EPS = 1e-5
SEED = 0


def marginal_prob_std(t, sigma, noise_beta=None):
    if noise_beta is None:
        return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma))
    # Σ(t)^2 = expm1(t log(1 + beta Σ(1)^2)) / beta.
    variance_max = (sigma**2 - 1) / (2 * math.log(sigma))
    rate = math.log1p(noise_beta * variance_max)
    return torch.sqrt(torch.expm1(rate * t) / noise_beta)


def diffusion_coeff(t, sigma, noise_beta=None):
    if noise_beta is None:
        return sigma**t
    # g(t)^2 = dΣ(t)^2/dt, using the same schedule as the training noise.
    variance_max = (sigma**2 - 1) / (2 * math.log(sigma))
    rate = math.log1p(noise_beta * variance_max)
    return math.sqrt(rate / noise_beta) * torch.exp(0.5 * rate * t)


def bc(s):
    return s.view(-1, 1, 1, 1, 1)


def roll(A, mu, k):
    # A[..., x, y, :] evaluated at x + k e_mu; lattice axes are the two before the component axis
    return torch.roll(A, -k, dims=A.ndim - 3 + mu)


def qmul(p, q):
    # (p0 + i p⃗·σ)(q0 + i q⃗·σ) = p0 q0 − p⃗·q⃗ + i [p0 q⃗ + q0 p⃗ − p⃗×q⃗]·σ from σ_a σ_b = δ_ab + i ε_abc σ_c
    p0, pv, q0, qv = p[..., :1], p[..., 1:], q[..., :1], q[..., 1:]
    return torch.cat([p0 * q0 - (pv * qv).sum(-1, keepdim=True), p0 * qv + q0 * pv - torch.linalg.cross(pv, qv)], -1)


@triton.jit
def _mm(a, b):
    # three TF32 tensor-core products of the split operands, float32 accuracy
    return tl.dot(a, b, input_precision="tf32x3")


@triton.jit
def _split4(X, C: tl.constexpr, S: tl.constexpr):
    # (C, 4S) tile in memory order (site, q) → its four (C, S) quaternion components
    x02, x13 = tl.split(tl.reshape(X, (C, S, 2, 2)))
    x0, x2 = tl.split(x02)
    x1, x3 = tl.split(x13)
    return x0, x1, x2, x3


@triton.jit
def _join4(x0, x1, x2, x3, C: tl.constexpr, S: tl.constexpr):
    return tl.reshape(tl.join(tl.join(x0, x2), tl.join(x1, x3)), (C, 4 * S))


@triton.jit
def _qbil_fwd(X, Wb, Wl, Wc, Out, N: tl.constexpr, C: tl.constexpr, S: tl.constexpr):
    # out_k = Σ_j A_kj W_j, A_kj = Σ_i w_kij W_i, one j at a time on S sites
    b = tl.program_id(1)
    n = tl.program_id(0) * S + tl.arange(0, S)
    c = tl.arange(0, C)
    X += b * C * N * 4
    Out += b * C * N * 4
    off = c[None, :] * N * 4 + n[:, None] * 4
    x0 = tl.load(X + off)
    x1 = tl.load(X + off + 1)
    x2 = tl.load(X + off + 2)
    x3 = tl.load(X + off + 3)
    w = tl.load(Wl + c[:, None] * C + c[None, :])
    o0 = _mm(x0, w) + tl.load(Wc + c)[None, :]
    o1 = _mm(x1, w)
    o2 = _mm(x2, w)
    o3 = _mm(x3, w)
    for j in range(C):
        w = tl.load(Wb + j * C * C + c[:, None] * C + c[None, :])
        a0 = _mm(x0, w)
        a1 = _mm(x1, w)
        a2 = _mm(x2, w)
        a3 = _mm(x3, w)
        y0 = tl.load(X + j * N * 4 + n * 4)[:, None]
        y1 = tl.load(X + j * N * 4 + n * 4 + 1)[:, None]
        y2 = tl.load(X + j * N * 4 + n * 4 + 2)[:, None]
        y3 = tl.load(X + j * N * 4 + n * 4 + 3)[:, None]
        o0 += a0 * y0 - a1 * y1 - a2 * y2 - a3 * y3
        o1 += a0 * y1 + a1 * y0 - a2 * y3 + a3 * y2
        o2 += a0 * y2 + a2 * y0 - a3 * y1 + a1 * y3
        o3 += a0 * y3 + a3 * y0 - a1 * y2 + a2 * y1
    tl.store(Out + off, o0)
    tl.store(Out + off + 1, o1)
    tl.store(Out + off + 2, o2)
    tl.store(Out + off + 3, o3)


@triton.jit
def _qbil_bwd_x(X, G, Wp, Wm, Wl, Gx, N: tl.constexpr, C: tl.constexpr, S: tl.constexpr):
    # ∂/∂W_m = Σ_l (H_ml W̄_l + W̄_l E_lm), H_ml = Σ_k w_kml g_k, E_lm = Σ_k w_klm g_k, since <p, a b> = <p b̄, a> = <ā p, b>.
    # Only H ± E enter: the scalar part is x̄₀(H+E)₀ − x̄⃗·(H+E)⃗, the vector part x̄₀(H+E)⃗ + (H+E)₀x̄⃗ − (H−E)⃗×x̄⃗.
    b = tl.program_id(1)
    n = tl.program_id(0) * S + tl.arange(0, S)
    c = tl.arange(0, C)
    X += b * C * N * 4
    G += b * C * N * 4
    Gx += b * C * N * 4
    off = c[None, :] * N * 4 + n[:, None] * 4
    g0 = tl.load(G + off)
    g1 = tl.load(G + off + 1)
    g2 = tl.load(G + off + 2)
    g3 = tl.load(G + off + 3)
    w = tl.load(Wl + c[:, None] * C + c[None, :])
    o0 = _mm(g0, w)
    o1 = _mm(g1, w)
    o2 = _mm(g2, w)
    o3 = _mm(g3, w)
    for l in range(C):
        w = tl.load(Wp + l * C * C + c[:, None] * C + c[None, :])
        s0 = _mm(g0, w)
        s1 = _mm(g1, w)
        s2 = _mm(g2, w)
        s3 = _mm(g3, w)
        w = tl.load(Wm + l * C * C + c[:, None] * C + c[None, :])
        d1 = _mm(g1, w)
        d2 = _mm(g2, w)
        d3 = _mm(g3, w)
        y0 = tl.load(X + l * N * 4 + n * 4)[:, None]
        y1 = tl.load(X + l * N * 4 + n * 4 + 1)[:, None]
        y2 = tl.load(X + l * N * 4 + n * 4 + 2)[:, None]
        y3 = tl.load(X + l * N * 4 + n * 4 + 3)[:, None]
        o0 += y0 * s0 + y1 * s1 + y2 * s2 + y3 * s3
        o1 += y0 * s1 - s0 * y1 + d2 * y3 - d3 * y2
        o2 += y0 * s2 - s0 * y2 + d3 * y1 - d1 * y3
        o3 += y0 * s3 - s0 * y3 + d1 * y2 - d2 * y1
    tl.store(Gx + off, o0)
    tl.store(Gx + off + 1, o1)
    tl.store(Gx + off + 2, o2)
    tl.store(Gx + off + 3, o3)


@triton.jit
def _qbil_bwd_w(X, G, P, N: tl.constexpr, C: tl.constexpr, S: tl.constexpr, JB: tl.constexpr):
    # ∂/∂w_kij = Σ_x <g_k W̄_j, W_i>, JB values of j per program; tiles are read in memory order, so (site, q) is one
    # contraction index of length 4S
    b = tl.program_id(1)
    c = tl.arange(0, C)
    j = tl.program_id(0) * JB + tl.arange(0, JB)
    r = tl.arange(0, 4 * S)
    X += b * C * N * 4
    G += b * C * N * 4
    acc = tl.zeros((JB * C, C), tl.float32)
    for t in range(N // S):
        # splitting after the broadcast reshape keeps the register layout of the products (38 spills otherwise)
        g02, g13 = tl.split(tl.reshape(tl.load(G + c[:, None] * N * 4 + t * S * 4 + r[None, :]), (1, C, S, 2, 2)))
        g0, g2 = tl.split(g02)
        g1, g3 = tl.split(g13)
        y02, y13 = tl.split(tl.reshape(tl.load(X + j[:, None] * N * 4 + t * S * 4 + r[None, :]), (JB, 1, S, 2, 2)))
        y0, y2 = tl.split(y02)
        y1, y3 = tl.split(y13)
        z = _join4(g0 * y0 + g1 * y1 + g2 * y2 + g3 * y3, g1 * y0 - g0 * y1 + g2 * y3 - g3 * y2,
                   g2 * y0 - g0 * y2 + g3 * y1 - g1 * y3, g3 * y0 - g0 * y3 + g1 * y2 - g2 * y1, JB * C, S)
        acc += _mm(z, tl.load(X + c[None, :] * N * 4 + t * S * 4 + r[:, None]))
    k = tl.arange(0, JB * C)
    tl.store(P + ((b * C + tl.program_id(0) * JB) * C + k[:, None]) * C + c[None, :], acc)


class QBilinear(torch.autograd.Function):
    # The GEMM form A = W·w stores (B L² 4) × 32 × 33 floats, 17.7 GB per layer at B=64, L=128; here A lives in registers.
    @staticmethod
    def forward(ctx, X, wb, wl, wc):
        X = X.contiguous()
        B, C, Lx, Ly, _ = X.shape
        out = torch.empty_like(X)
        _qbil_fwd[(Lx * Ly // 32, B)](X, wb.permute(2, 1, 0).contiguous(), wl.t().contiguous(), wc.contiguous(), out,
                                      Lx * Ly, C, 32, num_warps=2)
        ctx.save_for_backward(X, wb, wl)
        return out

    @staticmethod
    def backward(ctx, g):
        X, wb, wl = ctx.saved_tensors
        B, C, Lx, Ly, _ = X.shape
        g = g.contiguous()
        gx = torch.empty_like(X)
        w1, w2 = wb.permute(2, 0, 1), wb.permute(1, 0, 2)
        _qbil_bwd_x[(Lx * Ly // 16, B)](X, g, (w1 + w2).contiguous(), (w1 - w2).contiguous(), wl.contiguous(), gx,
                                        Lx * Ly, C, 16, num_warps=2)
        P = torch.empty(B, C, C, C, device=X.device)
        _qbil_bwd_w[(C // 8, B)](X, g, P, Lx * Ly, C, 8, 8, num_warps=4)
        return gx, P.sum(0).permute(1, 2, 0), torch.einsum("bkxyq,bixyq->ki", g, X), g[..., 0].sum((0, 2, 3))


def qbilinear(weight, W):
    # Σ_ij weight[k,i,j] W_i W_j over W_0 = 1 and the channels of W: constant w_k00, linear w_k0i + w_ki0, bilinear rest
    return QBilinear.apply(W, weight[:, 1:, 1:], weight[:, 0, 1:] + weight[:, 1:, 0], weight[:, 0, 0])


def conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def sinc_half(th):
    return torch.where(th > 1e-2, torch.sin(th / 2) / (th / 2), 1 - th**2 / 24)


def dlog_sinc_half_over(th):
    # [½ cot(θ/2) − 1/θ] / θ = θ⁻¹ d/dθ log(sin(θ/2)/(θ/2))
    return torch.where(th > 5e-2, (0.5 / torch.tan(th / 2) - 1 / th) / th, -1 / 12 - th**2 / 720)


def qexp(xi):
    # exp(i ξ·σ/2) = cos(θ/2) + i sin(θ/2) ξ̂·σ
    th = xi.norm(dim=-1, keepdim=True)
    return torch.cat([torch.cos(th / 2), sinc_half(th) * xi / 2], -1)


def plaquette(U):
    U0, U1 = U[:, 0], U[:, 1]
    return qmul(qmul(qmul(U0, roll(U1, 0, 1)), conj(roll(U0, 1, 1))), conj(U1))


def transport(W, U):
    # loops based at the four neighbours, brought to x through the connecting link (L-Conv, kernel ±1)
    out = [W]
    for mu in range(2):
        Um = U[:, mu, None]
        out.append(qmul(qmul(Um, roll(W, mu, 1)), conj(Um)))
        Ub = roll(Um, mu, -1)
        out.append(qmul(qmul(conj(Ub), roll(W, mu, -1)), Ub))
    return torch.cat(out, 1)


@triton.jit
def _shift(n, d: tl.constexpr, L: tl.constexpr):
    # site index of x + d for d = 0, +e₀, −e₀, +e₁, −e₁ (the order of transport), n = x L + y, periodic
    m = n
    if d == 1:
        m = (n + L) % (L * L)
    elif d == 2:
        m = (n + L * L - L) % (L * L)
    elif d == 3:
        m = n - n % L + (n + 1) % L
    elif d == 4:
        m = n - n % L + (n + L - 1) % L
    return m


@triton.jit
def _ad(U, n, d: tl.constexpr, L: tl.constexpr, z0, z1, z2, z3):
    # u z ū = (|u|² z₀, (u₀² − |u⃗|²) z⃗ + 2(u⃗·z⃗) u⃗ − 2u₀ u⃗×z⃗) with the link of transport, u = U_μ(x) or Ū_μ(x − μ)
    U += (d - 1) // 2 * L * L * 4
    if d % 2 == 1:
        U += n * 4
    else:
        U += _shift(n, d, L) * 4
    s = 1 - 2 * ((d - 1) % 2)
    u0 = tl.load(U)[None, :]
    u1 = s * tl.load(U + 1)[None, :]
    u2 = s * tl.load(U + 2)[None, :]
    u3 = s * tl.load(U + 3)[None, :]
    uu = u1 * u1 + u2 * u2 + u3 * u3
    a = u0 * u0 - uu
    p = 2 * (u1 * z1 + u2 * z2 + u3 * z3)
    return ((u0 * u0 + uu) * z0, a * z1 + p * u1 - 2 * u0 * (u2 * z3 - u3 * z2),
            a * z2 + p * u2 - 2 * u0 * (u3 * z1 - u1 * z3), a * z3 + p * u3 - 2 * u0 * (u1 * z2 - u2 * z1))


@triton.jit
def _tconv(W, U, Wt, Out, L: tl.constexpr, C: tl.constexpr, S: tl.constexpr):
    # Out(x) = Σ_d w_d Ad[u_d(x)] W(x + d) on S sites, the einsum of the conv weights with transport(W, U).  Ad acts on
    # the quaternion index only and commutes with the channel weights.
    b = tl.program_id(1)
    n0 = tl.program_id(0) * S
    n = n0 + tl.arange(0, S)
    r = tl.arange(0, 4 * S)
    c = tl.arange(0, C)
    W += b * C * L * L * 4
    U += b * 2 * L * L * 4
    Out += b * C * L * L * 4
    o0 = tl.zeros((C, S), tl.float32)
    o1 = tl.zeros((C, S), tl.float32)
    o2 = tl.zeros((C, S), tl.float32)
    o3 = tl.zeros((C, S), tl.float32)
    for d in tl.static_range(5):
        x0, x1, x2, x3 = _split4(tl.load(W + c[:, None] * L * L * 4 + (_shift(n0 + r // 4, d, L) * 4 + r % 4)[None, :]),
                                 C, S)
        if d > 0:
            x0, x1, x2, x3 = _ad(U, n, d, L, x0, x1, x2, x3)
        o0 += _mm(tl.load(Wt + 2 * d * C * C + c[:, None] * C + c[None, :]), x0)
        w = tl.load(Wt + (2 * d + 1) * C * C + c[:, None] * C + c[None, :])
        o1 += _mm(w, x1)
        o2 += _mm(w, x2)
        o3 += _mm(w, x3)
    tl.store(Out + c[:, None] * L * L * 4 + n0 * 4 + r[None, :], _join4(o0, o1, o2, o3, C, S))


@triton.jit
def _tconv_bwd_w(W, U, G, P, L: tl.constexpr, C: tl.constexpr, S: tl.constexpr, T: tl.constexpr, d: tl.constexpr):
    # ∂/∂w_d = Σ_x <g(x), Ad[u_d(x)] W(x + d)> for the scalar and the vector weights over T tiles of S sites
    b = tl.program_id(1)
    c = tl.arange(0, C)
    r = tl.arange(0, 4 * S)
    W += b * C * L * L * 4
    U += b * 2 * L * L * 4
    G += b * C * L * L * 4
    accs = tl.zeros((C, C), tl.float32)
    accv = tl.zeros((C, C), tl.float32)
    for t in range(tl.program_id(0) * T, (tl.program_id(0) + 1) * T):
        n0 = t * S
        x0, x1, x2, x3 = _split4(tl.load(W + c[:, None] * L * L * 4 + (_shift(n0 + r // 4, d, L) * 4 + r % 4)[None, :]),
                                 C, S)
        if d > 0:
            x0, x1, x2, x3 = _ad(U, n0 + tl.arange(0, S), d, L, x0, x1, x2, x3)
        z = tl.zeros_like(x0)
        g = tl.load(G + c[:, None] * L * L * 4 + n0 * 4 + r[None, :])
        accs += _mm(g, tl.trans(_join4(x0, z, z, z, C, S)))
        accv += _mm(g, tl.trans(_join4(z, x1, x2, x3, C, S)))
    P += ((d * tl.num_programs(0) + tl.program_id(0)) * tl.num_programs(1) + b) * 2 * C * C + c[:, None] * C + c[None, :]
    tl.store(P, accs)
    tl.store(P + C * C, accv)


class TConv(torch.autograd.Function):
    # The einsum over transport(W, U) stores 5 ch transported loops per site, 2.7 GB at B=64, L=128, ch=32.
    @staticmethod
    def forward(ctx, W, U, conv):
        W, U = W.contiguous(), U.contiguous()
        B, C, L, _, _ = W.shape
        out = torch.empty_like(W)
        _tconv[(L * L // 64, B)](W, U, conv.view(2, C, 5, C).permute(2, 0, 1, 3).contiguous(), out, L, C, 64, num_warps=4)
        ctx.save_for_backward(W, U, conv)
        return out

    @staticmethod
    def backward(ctx, g):
        W, U, conv = ctx.saved_tensors
        B, C, L, _, _ = W.shape
        g = g.contiguous()
        gW = torch.empty_like(W)
        # transport to x from x + μ is the adjoint of transport to x from x − μ: swap the ±μ blocks, transpose the weights
        w = conv.view(2, C, 5, C)[:, :, [0, 2, 1, 4, 3]].permute(2, 0, 3, 1).contiguous()
        _tconv[(L * L // 64, B)](g, U, w, gW, L, C, 64, num_warps=4)
        ns = max(1, L * L // 2048)
        P = torch.empty(5, ns * B, 2, C, C, device=W.device)
        for d in range(5):
            _tconv_bwd_w[(ns, B)](W, U, g, P, L, C, 16, L * L // 16 // ns, d, num_warps=2)
        return gW, None, P.sum(1).permute(1, 2, 0, 3).reshape(2, C, 5 * C)


@triton.jit
def _heat_kernel_draw(Y, Delta, Seed, N: tl.constexpr, REPEAT: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    delta = tl.load(Delta + idx // REPEAT, idx < N, other=0)
    sd = tl.sqrt(delta)
    seed = tl.load(Seed)
    accepted = (idx >= N) | (delta == 0)
    x = tl.full((BLOCK,), 0, tl.float32)
    y = tl.full((BLOCK,), 0, tl.float32)
    z = tl.full((BLOCK,), 0, tl.float32)
    attempt = 0
    while tl.sum((~accepted).to(tl.int32), 0) > 0:
        # Separate Philox counters for the proposal and acceptance uniform, including every retry.
        counter = 2 * (idx.to(tl.uint64) + attempt.to(tl.uint64) * N)
        a, b, c, _ = tl.randn4x(seed, counter)
        a, b, c = sd * a, sd * b, sd * c
        th = tl.sqrt(a * a + b * b + c * c)
        prob = tl.where(th > 1e-2, tl.sin(th / 2) / (th / 2), 1 - th * th / 24)
        ok = (~accepted) & (tl.rand(seed, counter + 1) < prob)
        x, y, z = tl.where(ok, a, x), tl.where(ok, b, y), tl.where(ok, c, z)
        accepted = accepted | ok
        attempt += 1
    tl.store(Y + 3 * idx, x, idx < N)
    tl.store(Y + 3 * idx + 1, y, idx < N)
    tl.store(Y + 3 * idx + 2, z, idx < N)


def heat_kernel_step(shape, delta):
    # Y = √Δ ξ, ξ ~ N(0, 1₃), accepted with probability √j(Y), j = [sin(θ/2)/(θ/2)]² (Weyl density): the Gaussian
    # pushforward has density Gauss(Y)/j w.r.t. Haar, the heat kernel Gauss(Y)/√j, so exp(iY) is an exact Brownian
    # increment of variance Δ per generator up to wrapping images of weight e^{-79/Δ}.  Draws with θ > 2π (outside the
    # alcove, probability e^{-2π²/Δ}) have √j < 0 and are rejected. Δ = 0 gives Y = 0.
    # delta holds one variance per leading group, shared by all trailing link/site axes.
    # Each GPU block retries until all its draws are accepted. No CPU synchronization or retry cutoff.
    n = math.prod(shape)
    Y = torch.empty((*shape, 3), device=delta.device)
    seed = torch.randint(0, 2**63 - 1, (), device=delta.device, dtype=torch.int64)
    _heat_kernel_draw[(triton.cdiv(n, 128),)](Y, delta, seed, n, n // delta.numel(), 128)
    return Y


@triton.jit
def _brownian(U0, Delta, Kn, Seed, Ut, Y, NL, NT, BLOCK: tl.constexpr):
    # K increments per link drawn as in _heat_kernel_draw and multiplied from the left in registers. K and Δ belong to
    # the configuration, and a block never spans two configurations (NL ≥ BLOCK links each).
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.program_id(0) * BLOCK // NL
    sd = tl.sqrt(tl.load(Delta + b))
    K = tl.load(Kn + b)
    seed = tl.load(Seed)
    g0 = tl.full((BLOCK,), 1, tl.float32)
    g1 = tl.zeros((BLOCK,), tl.float32)
    g2 = tl.zeros((BLOCK,), tl.float32)
    g3 = tl.zeros((BLOCK,), tl.float32)
    x = tl.zeros((BLOCK,), tl.float32)
    y = tl.zeros((BLOCK,), tl.float32)
    z = tl.zeros((BLOCK,), tl.float32)
    for k in range(K):
        accepted = idx < 0
        attempt = 0
        while tl.sum((~accepted).to(tl.int32), 0) > 0:
            # Philox counters unique in (link, substep, retry), separate ones for proposal and acceptance
            counter = 2 * (idx.to(tl.uint64) + (attempt * K + k).to(tl.uint64) * NT)
            a, bb, c, _ = tl.randn4x(seed, counter)
            a, bb, c = sd * a, sd * bb, sd * c
            th = tl.sqrt(a * a + bb * bb + c * c)
            prob = tl.where(th > 1e-2, tl.sin(th / 2) / (th / 2), 1 - th * th / 24)
            ok = (~accepted) & (tl.rand(seed, counter + 1) < prob)
            x, y, z = tl.where(ok, a, x), tl.where(ok, bb, y), tl.where(ok, c, z)
            accepted = accepted | ok
            attempt += 1
        # G ← exp(iY·σ/2) G with exp(iY·σ/2) = (cos(θ/2), sin(θ/2) Y/θ)
        th = tl.sqrt(x * x + y * y + z * z)
        e0 = tl.cos(th / 2)
        s = tl.where(th > 1e-2, tl.sin(th / 2) / th, 0.5 - th * th / 48)
        e1, e2, e3 = s * x, s * y, s * z
        g0, g1, g2, g3 = (e0 * g0 - e1 * g1 - e2 * g2 - e3 * g3, e0 * g1 + g0 * e1 - e2 * g3 + e3 * g2,
                          e0 * g2 + g0 * e2 - e3 * g1 + e1 * g3, e0 * g3 + g0 * e3 - e1 * g2 + e2 * g1)
    u0 = tl.load(U0 + 4 * idx)
    u1 = tl.load(U0 + 4 * idx + 1)
    u2 = tl.load(U0 + 4 * idx + 2)
    u3 = tl.load(U0 + 4 * idx + 3)
    tl.store(Ut + 4 * idx, g0 * u0 - g1 * u1 - g2 * u2 - g3 * u3)
    tl.store(Ut + 4 * idx + 1, g0 * u1 + u0 * g1 - g2 * u3 + g3 * u2)
    tl.store(Ut + 4 * idx + 2, g0 * u2 + u0 * g2 - g3 * u1 + g1 * u3)
    tl.store(Ut + 4 * idx + 3, g0 * u3 + u0 * g3 - g1 * u2 + g2 * u1)
    tl.store(Y + 3 * idx, x)
    tl.store(Y + 3 * idx + 1, y)
    tl.store(Y + 3 * idx + 2, z)


def brownian(U0, delta, K):
    # U_t = exp(iY_K)…exp(iY_1) U₀ with K heat-kernel increments of variance δ per configuration, and the last one Y_K
    Ut = torch.empty_like(U0)
    Y = torch.empty((*U0.shape[:-1], 3), device=U0.device)
    seed = torch.randint(0, 2**63 - 1, (), device=U0.device, dtype=torch.int64)
    _brownian[(U0.numel() // 512,)](U0, delta, K.int(), seed, Ut, Y, U0[0].numel() // 4, U0.numel() // 4, 128)
    return Ut, Y


def heat_kernel_score(Y, delta):
    # ∇log K_Δ(exp iY) = −Y/Δ − ½ ∇log j(Y) = −Y [1/Δ + (½ cot(θ/2) − 1/θ)/θ], using ∇θ = Y/θ
    return -Y * (1 / delta + dlog_sinc_half_over(Y.norm(dim=-1)))[..., None]


class LCNN(nn.Module):
    def __init__(self, ch, nlayers, sigma, noise_beta=None):
        super().__init__()
        self.ch, self.sigma = ch, sigma
        self.noise_beta = noise_beta
        self.conv = nn.ParameterList()  # (2, ch, 5 cin): one real weight for the scalar part of a loop, one for its vector part
        self.bilin = nn.ParameterList()
        self.gate_a = nn.ParameterList()
        self.gate_b = nn.ModuleList()
        cin = 1
        for _ in range(nlayers):
            self.conv.append(nn.Parameter(torch.randn(2, ch, 5 * cin) / np.sqrt(5 * cin)))
            self.bilin.append(nn.Parameter(torch.randn(ch, ch + 1, ch + 1) / (ch + 1)))
            self.gate_a.append(nn.Parameter(torch.ones(ch)))
            self.gate_b.append(nn.Linear(64, ch))
            cin = ch
        self.temb = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        # per-site real coefficients for (mu, channel) from the scalar parts of the local loops (gauge invariant) and time
        self.readout = nn.Sequential(nn.Linear(ch + 64, 64), nn.SiLU(), nn.Linear(64, 2 * ch))

    def forward(self, U, t):
        Sigma = marginal_prob_std(t, self.sigma, self.noise_beta)
        h = self.temb(torch.log(Sigma)[:, None])
        W = plaquette(U)[:, None]
        B, _, Lx, Ly, _ = W.shape
        for conv, bilin, a, b in zip(self.conv, self.bilin, self.gate_a, self.gate_b):
            if W.shape[1] == 1:
                # the plaquette channel alone is narrower than a tensor-core tile (16)
                # Keep batch separate so the weight gradient uses many small reductions in parallel.
                weights = conv[[0, 1, 1, 1]].expand(B, -1, -1, -1)
                W = torch.einsum("bqoc,bcxyq->boxyq", weights, transport(W, U))
            else:
                W = TConv.apply(W, U, conv)
            W = qbilinear(bilin, W)
            # bounded gate: with SiLU the bilinear product times the gate grows cubically per layer and the q0
            # fluctuations of SU(2) loops (std 1/2) blow up the tails at initialisation (|W| ~ 300 at some sites)
            gate = torch.sigmoid(a[None, :, None, None] * W[..., 0] + b(h)[:, :, None, None])
            W = W * gate[..., None]
        feat = torch.cat([W[..., 0], h[:, :, None, None].expand(-1, -1, Lx, Ly)], 1).permute(0, 2, 3, 1)
        c = self.readout(feat).view(B, Lx, Ly, 2, self.ch).permute(0, 3, 4, 1, 2)
        # a broadcast sum is one fused kernel; the einsum becomes B L² batched 2×ch·ch×3 GEMMs in 16 cuBLAS calls
        return (c[..., None] * W[:, None, ..., 1:]).sum(2) / bc(Sigma)


def loss_fn(model, U0):
    # Retain the time density 2(1−t), giving more samples to lower noise.
    B = U0.shape[0]
    t = EPS + (1.0 - EPS) * (1 - torch.sqrt(1 - torch.rand(B)))
    Sigma = marginal_prob_std(t, model.sigma, model.noise_beta)
    K = torch.ceil(Sigma**2 / DELTA_MAX).clamp_min(1)
    delta = Sigma**2 / K
    Ut, Y_last = brownian(U0, delta, K)
    # the 1/K weight only reweights t, the minimiser per t is unchanged; the all-zero score gives loss 3
    residual = bc(Sigma) * (model(Ut, t) - heat_kernel_score(Y_last, delta[:, None, None, None]))
    return (residual.pow(2).sum(-1).mean((1, 2, 3)) / K).mean()


def haar_su2(shape):
    # uniform on S³
    q = torch.randn(*shape, 4)
    return q / q.norm(dim=-1, keepdim=True)


def gauge_transform(U, Om):
    return torch.stack([qmul(qmul(Om, U[:, mu]), conj(roll(Om, mu, 1))) for mu in range(2)], 1)


def rotate(Om, xi):
    # Ω (ξ·σ) Ω† as the vector part of Ω (0, ξ) Ω̄
    return qmul(qmul(Om, torch.cat([torch.zeros_like(xi[..., :1]), xi], -1)), conj(Om))[..., 1:]


def check_equivariance(model, U):
    Om = haar_su2(U.shape[:1] + U.shape[2:4])
    t = torch.full((U.shape[0],), 0.3)
    S = model(U, t)
    S2 = model(gauge_transform(U, Om), t)
    err = (S2 - rotate(Om[:, None], S)).abs().max() / S.abs().max()
    print(f"gauge equivariance check: max relative deviation {err:.1e}")


def load_cfgs(n=None, path=DATA, dtype=torch.float32):
    with h5py.File(path) as f:
        U = f["cfgs"][:n]  # (n, y, x, mu, a, b)
    # U = a0 + i a⃗·σ, so (a0, a1, a2, a3) = (Re U₁₁, Im U₁₂, Re U₁₂, Im U₁₁)
    q = np.stack([U[..., 0, 0].real, U[..., 0, 1].imag, U[..., 0, 1].real, U[..., 0, 0].imag], -1)
    return torch.tensor(q.transpose(0, 3, 2, 1, 4), dtype=dtype).contiguous()


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    print(f"Data {DATA}\nOutput {OUT}\nEpochs {EPOCHS}, batch {BATCH}, lr {LR}, "
          f"EMA decay {EMA_DECAY}, eps {EPS}, seed {SEED}\nSave epochs {SAVE_EPOCHS}", flush=True)
    U_data = load_cfgs()
    N = U_data.shape[0]
    model = LCNN(CH, NLAYERS, SIGMA, noise_beta=BETA)
    params = list(model.parameters())
    ema = ExponentialMovingAverage(params, decay=EMA_DECAY)
    opt = torch.optim.Adam(params, lr=LR, fused=True)
    train_loss = torch.compile(loss_fn, mode="reduce-overhead", fullgraph=True, dynamic=False)
    print(f"{N} configs, L={L}, β={BETA}, noise_beta={model.noise_beta}, "
          f"Sigma(1)={marginal_prob_std(torch.tensor(1.0), SIGMA, model.noise_beta):.6f}, "
          f"{sum(p.numel() for p in params)} parameters")
    check_equivariance(model, U_data[:2])
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(N)
        tot = torch.zeros(())
        nb = 0
        for i in range(0, N - BATCH + 1, BATCH):
            torch.compiler.cudagraph_mark_step_begin()
            opt.zero_grad(set_to_none=True)
            loss = train_loss(model, U_data[perm[i:i + BATCH]])
            loss.backward()
            opt.step()
            # Preserve torch_ema's warmup decay while updating all parameter tensors in one operation.
            ema.num_updates += 1
            decay = min(ema.decay, (1 + ema.num_updates) / (10 + ema.num_updates))
            with torch.no_grad():
                torch._foreach_lerp_(ema.shadow_params, params, 1 - decay)
            tot.add_(loss.detach())
            nb += 1
        print(f"epoch {epoch:4d}  loss {(tot / nb).item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if epoch in SAVE_EPOCHS:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "ch": CH, "nlayers": NLAYERS,
                        "sigma": SIGMA, "noise_beta": model.noise_beta,
                        "L": L, "beta": BETA, "epoch": epoch}, f"{OUT}/ckpt_epoch{epoch:04d}.pt")


if __name__ == "__main__":
    main()
