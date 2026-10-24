# Residual L-CNN score model with dilated parallel-transport kernels for 2D SU(2) gauge configurations.  Forward process,
# loss, bilinear kernels, data and readout are those of ../SU2/train_su2_dm.py (imported unchanged).  Two changes.
#   1. Every layer is residual,  W_{l+1} = W_l + α_l g_l(x,t) bilin_l(conv_l(W̃_l)),  W̃ = W / sqrt(1 + |W|²),  with the
#      plaquette broadcast to all channels as W_0.  α_l starts at zero (identity map at initialisation, which keeps deep
#      stacks trainable without the per-layer shrinking of the default init); the soft-normalised branch input bounds the
#      bilinear growth for any weights.  Both terms transform as Ω(x) W Ω(x)†.
#   2. conv_l transports the loops at x ± 2^j μ̂, j < DILATIONS, along the straight Wilson line T_{μ,2^j}(x) =
#      U_μ(x) U_μ(x+μ̂) ... (2^j links), the L-Conv kernel of size 2^DILATIONS - 1 along each axis (cross-shaped), so the
#      receptive field per layer is 2^DILATIONS - 1 instead of 1.  DILATIONS = 1 is the ±1 kernel of ../SU2.
import os
import sys
import time
import numpy as np
import torch
import triton
import triton.language as tl
from torch import nn
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SU2"))
from train_su2_dm import (_mm, _split4, _join4, qbilinear, qmul, conj, roll, plaquette, marginal_prob_std, bc, loss_fn,
                          load_cfgs, check_equivariance, L, BETA, SIGMA, LR, EMA_DECAY, EPS, SEED)

CH = 32
NLAYERS = 16
DILATIONS = 3  # taps 0, ±1, ±2, ±4 along each axis
SKIP = True  # False: W_{l+1} = g_l bilin_l(conv_l(W_l)) as in ../SU2 with the near-identity init, to isolate the skip connection
BATCH = 64
EPOCHS = 50
SAVE_EPOCHS = sorted(set(np.rint(np.geomspace(1, EPOCHS, 10)).astype(int).tolist()) | {10})
OUT = f"/work/gw97/w97000/project/DM/2DSUN/results/su2res_L{L}_beta{BETA}_lcnn_hk_c{CH}l{NLAYERS}d{DILATIONS}s{int(SKIP)}_n5120_b{BATCH}_e{EPOCHS}_betanoise"


def transporters(U, nd):
    # T[:, j, mu](x) = U_mu(x) U_mu(x+mu) ... over 2^j links, by doubling: T_{2k}(x) = T_k(x) T_k(x + k mu)
    T = [U]
    for j in range(1, nd):
        k = 2 ** (j - 1)
        T.append(torch.stack([qmul(T[-1][:, mu], roll(T[-1][:, mu], mu, k)) for mu in range(2)], 1))
    return torch.stack(T, 1).contiguous()


@triton.jit
def _site(n, dx: tl.constexpr, dy: tl.constexpr, L: tl.constexpr):
    # site index of x + (dx, dy), n = x L + y, periodic
    return ((n // L + (dx + L)) % L) * L + (n % L + (dy + L)) % L


@triton.jit
def _adT(T, n, j: tl.constexpr, mu: tl.constexpr, sgn: tl.constexpr, dx: tl.constexpr, dy: tl.constexpr, L: tl.constexpr,
         z0, z1, z2, z3):
    # u z ū with u = T_{mu,k}(x) for the +k tap and T_{mu,k}(x - k mu)† for the -k tap, T offset to the configuration
    T += (j * 2 + mu) * L * L * 4
    if sgn > 0:
        T += n * 4
    else:
        T += _site(n, dx, dy, L) * 4
    u0 = tl.load(T)[None, :]
    u1 = sgn * tl.load(T + 1)[None, :]
    u2 = sgn * tl.load(T + 2)[None, :]
    u3 = sgn * tl.load(T + 3)[None, :]
    uu = u1 * u1 + u2 * u2 + u3 * u3
    a = u0 * u0 - uu
    p = 2 * (u1 * z1 + u2 * z2 + u3 * z3)
    return ((u0 * u0 + uu) * z0, a * z1 + p * u1 - 2 * u0 * (u2 * z3 - u3 * z2),
            a * z2 + p * u2 - 2 * u0 * (u3 * z1 - u1 * z3), a * z3 + p * u3 - 2 * u0 * (u1 * z2 - u2 * z1))


@triton.jit
def _tap(W, T, c, n0, r, d: tl.constexpr, L: tl.constexpr, C: tl.constexpr, S: tl.constexpr):
    # tap d of the loops W on the S sites from n0, transported to the sites: d = 0 on site, then (+x, -x, +y, -y) for
    # each distance 2^j, j = (d - 1) // 4
    if d == 0:
        x0, x1, x2, x3 = _split4(tl.load(W + c[:, None] * L * L * 4 + (n0 * 4 + r)[None, :]), C, S)
    else:
        j: tl.constexpr = (d - 1) // 4
        q: tl.constexpr = (d - 1) % 4
        mu: tl.constexpr = q // 2
        sgn: tl.constexpr = 1 - 2 * (q % 2)
        dx: tl.constexpr = sgn * (1 << j) * (1 - mu)
        dy: tl.constexpr = sgn * (1 << j) * mu
        x0, x1, x2, x3 = _split4(tl.load(W + c[:, None] * L * L * 4 + (_site(n0 + r // 4, dx, dy, L) * 4 + r % 4)[None, :]),
                                 C, S)
        x0, x1, x2, x3 = _adT(T, n0 + tl.arange(0, S), j, mu, sgn, dx, dy, L, x0, x1, x2, x3)
    return x0, x1, x2, x3


@triton.jit
def _tconvd(W, T, Wt, Out, L: tl.constexpr, C: tl.constexpr, S: tl.constexpr, ND: tl.constexpr):
    # Out(x) = Σ_d w_d Ad[T_d(x)] W(x + s_d) over the 1 + 4 ND taps on S sites.  The taps run in a runtime loop with the
    # tap decoded at run time: unrolling 13 taps spills 42 registers and is twice as slow (one tap live at a time here).
    b = tl.program_id(1)
    n0 = tl.program_id(0) * S
    n = n0 + tl.arange(0, S)
    r = tl.arange(0, 4 * S)
    c = tl.arange(0, C)
    W += b * C * L * L * 4
    T += b * ND * 2 * L * L * 4
    Out += b * C * L * L * 4
    x0, x1, x2, x3 = _split4(tl.load(W + c[:, None] * L * L * 4 + (n0 * 4 + r)[None, :]), C, S)
    o0 = _mm(tl.load(Wt + c[:, None] * C + c[None, :]), x0)
    w = tl.load(Wt + C * C + c[:, None] * C + c[None, :])
    o1 = _mm(w, x1)
    o2 = _mm(w, x2)
    o3 = _mm(w, x3)
    nr = n0 + r // 4
    for d in range(1, 1 + 4 * ND):
        j = (d - 1) // 4
        q = (d - 1) % 4
        mu = q // 2
        sgn = 1 - 2 * (q % 2)
        dx = sgn * (1 << j) * (1 - mu)
        dy = sgn * (1 << j) * mu
        sr = ((nr // L + dx + L) % L) * L + (nr % L + dy + L) % L
        z0, z1, z2, z3 = _split4(tl.load(W + c[:, None] * L * L * 4 + (sr * 4 + r % 4)[None, :]), C, S)
        site = ((n // L + dx + L) % L) * L + (n % L + dy + L) % L
        Tp = T + (j * 2 + mu) * L * L * 4 + tl.where(sgn > 0, n, site) * 4
        s = sgn.to(tl.float32)
        u0 = tl.load(Tp)[None, :]
        u1 = s * tl.load(Tp + 1)[None, :]
        u2 = s * tl.load(Tp + 2)[None, :]
        u3 = s * tl.load(Tp + 3)[None, :]
        uu = u1 * u1 + u2 * u2 + u3 * u3
        a = u0 * u0 - uu
        p = 2 * (u1 * z1 + u2 * z2 + u3 * z3)
        y1 = a * z1 + p * u1 - 2 * u0 * (u2 * z3 - u3 * z2)
        y2 = a * z2 + p * u2 - 2 * u0 * (u3 * z1 - u1 * z3)
        y3 = a * z3 + p * u3 - 2 * u0 * (u1 * z2 - u2 * z1)
        o0 += _mm(tl.load(Wt + 2 * d * C * C + c[:, None] * C + c[None, :]), (u0 * u0 + uu) * z0)
        w = tl.load(Wt + (2 * d + 1) * C * C + c[:, None] * C + c[None, :])
        o1 += _mm(w, y1)
        o2 += _mm(w, y2)
        o3 += _mm(w, y3)
    tl.store(Out + c[:, None] * L * L * 4 + n0 * 4 + r[None, :], _join4(o0, o1, o2, o3, C, S))


@triton.jit
def _tconvd_bwd_w(W, T, G, P, L: tl.constexpr, C: tl.constexpr, S: tl.constexpr, NTILE: tl.constexpr, ND: tl.constexpr,
                  d: tl.constexpr):
    # ∂/∂w_d = Σ_x <g(x), Ad[T_d(x)] W(x + s_d)> for the scalar and the vector weights over NTILE tiles of S sites
    b = tl.program_id(1)
    c = tl.arange(0, C)
    r = tl.arange(0, 4 * S)
    W += b * C * L * L * 4
    T += b * ND * 2 * L * L * 4
    G += b * C * L * L * 4
    accs = tl.zeros((C, C), tl.float32)
    accv = tl.zeros((C, C), tl.float32)
    for t in range(tl.program_id(0) * NTILE, (tl.program_id(0) + 1) * NTILE):
        n0 = t * S
        x0, x1, x2, x3 = _tap(W, T, c, n0, r, d, L, C, S)
        z = tl.zeros_like(x0)
        g = tl.load(G + c[:, None] * L * L * 4 + n0 * 4 + r[None, :])
        accs += _mm(g, tl.trans(_join4(x0, z, z, z, C, S)))
        accv += _mm(g, tl.trans(_join4(z, x1, x2, x3, C, S)))
    P += ((d * tl.num_programs(0) + tl.program_id(0)) * tl.num_programs(1) + b) * 2 * C * C + c[:, None] * C + c[None, :]
    tl.store(P, accs)
    tl.store(P + C * C, accv)


class TConvD(torch.autograd.Function):
    # L-Conv with the taps 0, ±2^j μ̂ (j < ND) using the precomputed straight Wilson lines T of transporters(U, ND)
    @staticmethod
    def forward(ctx, W, T, conv):
        W = W.contiguous()
        B, C, L, _, _ = W.shape
        ND = T.shape[1]
        NT = 1 + 4 * ND
        out = torch.empty_like(W)
        _tconvd[(L * L // 64, B)](W, T, conv.view(2, C, NT, C).permute(2, 0, 1, 3).contiguous(), out, L, C, 64, ND, num_warps=4)
        ctx.save_for_backward(W, T, conv)
        return out

    @staticmethod
    def backward(ctx, g):
        W, T, conv = ctx.saved_tensors
        B, C, L, _, _ = W.shape
        ND = T.shape[1]
        NT = 1 + 4 * ND
        g = g.contiguous()
        gW = torch.empty_like(W)
        # transport to x from x + s is the adjoint of transport to x from x - s: swap the ± blocks, transpose the weights
        perm = [0] + [1 + 4 * j + i for j in range(ND) for i in (1, 0, 3, 2)]
        w = conv.view(2, C, NT, C)[:, :, perm].permute(2, 0, 3, 1).contiguous()
        _tconvd[(L * L // 64, B)](g, T, w, gW, L, C, 64, ND, num_warps=4)
        ns = max(1, L * L // 2048)
        P = torch.empty(NT, ns * B, 2, C, C, device=W.device)
        for d in range(NT):
            _tconvd_bwd_w[(ns, B)](W, T, g, P, L, C, 16, L * L // 16 // ns, ND, d, num_warps=2)
        return gW, None, P.sum(1).permute(1, 2, 0, 3).reshape(2, C, NT * C)


class LCNN(nn.Module):
    def __init__(self, ch, nlayers, sigma, noise_beta=None, dilations=DILATIONS, skip=SKIP):
        super().__init__()
        self.ch, self.sigma, self.nd, self.skip = ch, sigma, dilations, skip
        self.noise_beta = noise_beta
        nt = 1 + 4 * dilations
        self.conv = nn.ParameterList([nn.Parameter(torch.randn(2, ch, nt * ch) / np.sqrt(nt * ch)) for _ in range(nlayers)])
        self.bilin = nn.ParameterList([nn.Parameter(torch.randn(ch, ch + 1, ch + 1) / (ch + 1)) for _ in range(nlayers)])
        self.gate_a = nn.ParameterList([nn.Parameter(torch.ones(ch)) for _ in range(nlayers)])
        self.gate_b = nn.ModuleList([nn.Linear(64, ch) for _ in range(nlayers)])
        self.alpha = nn.Parameter(torch.zeros(nlayers)) if skip else None
        self.temb = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(ch + 64, 64), nn.SiLU(), nn.Linear(64, 2 * ch))
        if not skip:
            # without a skip path the default init shrinks |W| per layer; start each layer near the identity on channels
            with torch.no_grad():
                for w, b in zip(self.bilin, self.gate_b):
                    w[:, 1:, 1:] *= 0.3
                    w[:, 0, 0] = 0
                    w[:, 0, 1:] = 0.5 * torch.eye(ch)
                    w[:, 1:, 0] = 0.5 * torch.eye(ch)
                    b.bias += 3.0

    def forward(self, U, t):
        Sigma = marginal_prob_std(t, self.sigma, self.noise_beta)
        h = self.temb(torch.log(Sigma)[:, None])
        T = transporters(U, self.nd)
        W = plaquette(U)[:, None].expand(-1, self.ch, -1, -1, -1).contiguous()
        B, _, Lx, Ly, _ = W.shape
        for l, (conv, bilin, a, b) in enumerate(zip(self.conv, self.bilin, self.gate_a, self.gate_b)):
            X = W / torch.sqrt(1 + W.pow(2).sum(-1, keepdim=True)) if self.skip else W
            V = qbilinear(bilin, TConvD.apply(X, T, conv))
            gate = torch.sigmoid(a[None, :, None, None] * V[..., 0] + b(h)[:, :, None, None])
            W = W + self.alpha[l] * gate[..., None] * V if self.skip else gate[..., None] * V
        feat = torch.cat([W[..., 0], h[:, :, None, None].expand(-1, -1, Lx, Ly)], 1).permute(0, 2, 3, 1)
        c = self.readout(feat).view(B, Lx, Ly, 2, self.ch).permute(0, 3, 4, 1, 2)
        return (c[..., None] * W[:, None, ..., 1:]).sum(2) / bc(Sigma)


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    print(f"Output {OUT}\nEpochs {EPOCHS}, batch {BATCH}, lr {LR}, EMA decay {EMA_DECAY}, eps {EPS}, seed {SEED}\n"
          f"Save epochs {SAVE_EPOCHS}", flush=True)
    U_data = load_cfgs()
    N = U_data.shape[0]
    model = LCNN(CH, NLAYERS, SIGMA, noise_beta=BETA, dilations=DILATIONS, skip=SKIP)
    params = list(model.parameters())
    ema = ExponentialMovingAverage(params, decay=EMA_DECAY)
    opt = torch.optim.Adam(params, lr=LR, fused=True)
    train_loss = torch.compile(loss_fn, mode="reduce-overhead", fullgraph=True, dynamic=False)
    print(f"{N} configs, L={L}, β={BETA}, L-CNN {CH} channels {NLAYERS} layers, taps 0, ±2^j for j < {DILATIONS}, "
          f"skip {SKIP}, {sum(p.numel() for p in params)} parameters", flush=True)
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
            ema.num_updates += 1
            decay = min(ema.decay, (1 + ema.num_updates) / (10 + ema.num_updates))
            with torch.no_grad():
                torch._foreach_lerp_(ema.shadow_params, params, 1 - decay)
            tot.add_(loss.detach())
            nb += 1
        print(f"epoch {epoch:4d}  loss {(tot / nb).item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if epoch in SAVE_EPOCHS:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "ch": CH, "nlayers": NLAYERS,
                        "dilations": DILATIONS, "skip": SKIP, "sigma": SIGMA, "noise_beta": model.noise_beta, "L": L,
                        "beta": BETA, "epoch": epoch, "arch": "residual"}, f"{OUT}/ckpt_epoch{epoch:04d}.pt")


if __name__ == "__main__":
    main()
