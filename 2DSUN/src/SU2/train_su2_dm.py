# Score-based diffusion (VE-SDE, sigma^t) for 2D SU(2) gauge configurations from su2_2d_heatbath.jl, the SU(2) version of
# ../SU3/train_su3_dm.py.  Links are unit quaternions U = a0 + i a⃗·σ stored as real (…, 4) tensors, su(2) elements
# Y = ξ·σ/2 as real 3-vectors ξ (T^a = σ^a/2, |Y|² = Σ ξ_a²).  Brownian motion on SU(2)^links is a random walk of exact
# heat-kernel increments U ← exp(iY) U, and the DSM target is the analytic heat-kernel score of the last increment; for
# SU(2) both are closed-form functions of θ = |ξ| (no eigh, no matrix_exp).  The score transforms like Ω(x) Y Ω(x)†, an
# SO(3) rotation of ξ; the network is the reduced L-CNN of the SU(3) code in quaternion arithmetic:
# plaquette → parallel-transport convolution → quaternion bilinear products → scalar-part gating → per-link readout.
import os
import time
import h5py
import numpy as np
import torch
from torch import nn
from torch_ema import ExponentialMovingAverage

torch.set_default_device("cuda")

L = 16
BETA = 2.0
DATA = f"/work/gw97/w97000/project/DM/2DSUN/trainningdata/cfgs_su2_beta={BETA}_{L}^2.h5"
SIGMA = 14.0  # Σ(1) = 36.9: the fundamental character has decayed to e^{-3Σ/8} = 1e-6 (C_F = 3/4), the t=1 marginal is Haar
DELTA_MAX = 1.0  # variance per heat-kernel increment; exp(iY) = 1 first at θ = 4π, so the nearest wrapping image weighs e^{-79/Δ}
CH = 16
NLAYERS = 3
OUT = f"/work/gw97/w97000/project/DM/2DSUN/results/su2_L{L}_beta{BETA}_lcnn_hk_c{CH}l{NLAYERS}"
BATCH = 64
EPOCHS = 100
LR = 2e-4
EMA_DECAY = 0.999
EPS = 1e-5
SEED = 0


def marginal_prob_std(t, sigma):
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma))


def diffusion_coeff(t, sigma):
    return sigma**t


def bc(s):
    return s.view(-1, 1, 1, 1, 1)


def roll(A, mu, k):
    # A[..., x, y, :] evaluated at x + k e_mu; lattice axes are the two before the component axis
    return torch.roll(A, -k, dims=A.ndim - 3 + mu)


def qmul(p, q):
    # (p0 + i p⃗·σ)(q0 + i q⃗·σ) = p0 q0 − p⃗·q⃗ + i [p0 q⃗ + q0 p⃗ − p⃗×q⃗]·σ from σ_a σ_b = δ_ab + i ε_abc σ_c
    p0, pv, q0, qv = p[..., :1], p[..., 1:], q[..., :1], q[..., 1:]
    return torch.cat([p0 * q0 - (pv * qv).sum(-1, keepdim=True), p0 * qv + q0 * pv - torch.linalg.cross(pv, qv)], -1)


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


def heat_kernel_step(shape, delta):
    # Y = √Δ ξ, ξ ~ N(0, 1₃), accepted with probability √j(Y), j = [sin(θ/2)/(θ/2)]² (Weyl density): the Gaussian
    # pushforward has density Gauss(Y)/j w.r.t. Haar, the heat kernel Gauss(Y)/√j, so exp(iY) is an exact Brownian
    # increment of variance Δ per generator up to wrapping images of weight e^{-79/Δ}.  Draws with θ > 2π (outside the
    # alcove, probability e^{-2π²/Δ}) have √j < 0 and are rejected.  delta broadcasts to shape; Δ = 0 gives Y = 0.
    sd = torch.sqrt(delta)[..., None]
    Y = sd * torch.randn(*shape, 3)
    acc = torch.zeros(shape, dtype=torch.bool)
    while not acc.all():
        acc = acc | (torch.rand(shape) < sinc_half(Y.norm(dim=-1)))
        Y = torch.where(acc[..., None], Y, sd * torch.randn(*shape, 3))
    return Y


def heat_kernel_score(Y, delta):
    # ∇log K_Δ(exp iY) = −Y/Δ − ½ ∇log j(Y) = −Y [1/Δ + (½ cot(θ/2) − 1/θ)/θ], using ∇θ = Y/θ
    return -Y * (1 / delta + dlog_sinc_half_over(Y.norm(dim=-1)))[..., None]


class LCNN(nn.Module):
    def __init__(self, ch, nlayers, sigma):
        super().__init__()
        self.ch, self.sigma = ch, sigma
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
        s = marginal_prob_std(t, self.sigma)
        h = self.temb(torch.log(s)[:, None])
        W = plaquette(U)[:, None]
        B, _, Lx, Ly, _ = W.shape
        one = torch.zeros(B, 1, Lx, Ly, 4)
        one[..., 0] = 1
        for conv, bilin, a, b in zip(self.conv, self.bilin, self.gate_a, self.gate_b):
            W = torch.einsum("qoc,bcxyq->boxyq", conv[[0, 1, 1, 1]], transport(W, U))
            Wi = torch.cat([one, W], 1)
            W = qmul(torch.einsum("kij,bixyq->bkjxyq", bilin, Wi), Wi[:, None]).sum(2)
            # bounded gate: with SiLU the bilinear product times the gate grows cubically per layer and the q0
            # fluctuations of SU(2) loops (std 1/2) blow up the tails at initialisation (|W| ~ 300 at some sites)
            gate = torch.sigmoid(a[None, :, None, None] * W[..., 0] + b(h)[:, :, None, None])
            W = W * gate[..., None]
        feat = torch.cat([W[..., 0], h[:, :, None, None].expand(-1, -1, Lx, Ly)], 1).permute(0, 2, 3, 1)
        c = self.readout(feat).view(B, Lx, Ly, 2, self.ch)
        return torch.einsum("bxymc,bcxyv->bmxyv", c, W[..., 1:]) / bc(s)


def loss_fn(model, U0):
    # t with density 2(1−t): the score is only learnable for t ≲ 0.5, beyond that it is ≈ 0
    B = U0.shape[0]
    t = EPS + (1.0 - EPS) * (1 - torch.sqrt(1 - torch.rand(B)))
    s = marginal_prob_std(t, model.sigma)
    K = torch.ceil(s**2 / DELTA_MAX).clamp_min(1)
    Kmax = 1 << (int(K.max()) - 1).bit_length()  # power of two for the pairwise product below; padding substeps have Δ = 0, Y = 0
    delta = s**2 / K
    active = torch.arange(1, Kmax + 1)[None, :] <= K[:, None]
    # all increments of all substeps are drawn at once (they do not depend on U); the ordered product exp(iY_K)…exp(iY_1)
    # is formed pairwise in log₂ Kmax rounds, 28% faster per batch than Kmax sequential left multiplications
    Y = heat_kernel_step((B, Kmax) + U0.shape[1:4], (delta[:, None] * active)[:, :, None, None, None])
    G = qexp(Y)
    while G.shape[1] > 1:
        G = qmul(G[:, 1::2], G[:, ::2])
    Ut = qmul(G[:, 0], U0)
    Y_last = Y[torch.arange(B), (K - 1).long()]
    # the 1/K weight only reweights t, the minimiser per t is unchanged; the all-zero score gives loss 3
    residual = bc(s) * (model(Ut, t) - heat_kernel_score(Y_last, delta[:, None, None, None]))
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


def load_cfgs(n=None):
    with h5py.File(DATA) as f:
        U = f["cfgs"][:n]  # (n, y, x, mu, a, b)
    # U = a0 + i a⃗·σ, so (a0, a1, a2, a3) = (Re U₁₁, Im U₁₂, Re U₁₂, Im U₁₁)
    q = np.stack([U[..., 0, 0].real, U[..., 0, 1].imag, U[..., 0, 1].real, U[..., 0, 0].imag], -1)
    return torch.tensor(q.transpose(0, 3, 2, 1, 4), dtype=torch.float32)


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    U_data = load_cfgs()
    N = U_data.shape[0]
    model = LCNN(CH, NLAYERS, SIGMA)
    ema = ExponentialMovingAverage(model.parameters(), decay=EMA_DECAY)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"{N} configs, L={L}, β={BETA}, σ={SIGMA} (std(1)={marginal_prob_std(torch.tensor(1.0), SIGMA):.2f}), "
          f"{sum(p.numel() for p in model.parameters())} parameters")
    check_equivariance(model, U_data[:2])
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(N)
        tot = 0.0
        nb = 0
        for i in range(0, N - BATCH + 1, BATCH):
            loss = loss_fn(model, U_data[perm[i:i + BATCH]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update()
            tot += loss.item()
            nb += 1
        print(f"epoch {epoch:4d}  loss {tot / nb:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if epoch % 10 == 0 or epoch == EPOCHS:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "ch": CH, "nlayers": NLAYERS,
                        "sigma": SIGMA, "L": L, "beta": BETA, "epoch": epoch}, f"{OUT}/ckpt_epoch{epoch:04d}.pt")


if __name__ == "__main__":
    main()
