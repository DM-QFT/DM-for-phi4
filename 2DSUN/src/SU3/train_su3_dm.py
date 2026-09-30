# Score-based diffusion (VE-SDE, sigma^t) for 2D SU(3) gauge configurations from su3_2d_heatbath.jl.
# Brownian motion on SU(3)^links is realised as a random walk of exact heat-kernel increments U ← exp(iY) U, Y ∈ su(3)
# (a plain Gaussian step exp(i s Ξ) is not the heat kernel once s ≳ 0.5).  The score of the marginal equals
# E[score of the last increment | U_t], so the DSM target is the analytic heat-kernel score of the last increment.
# The score is an su(3) element per link transforming like Ω(x) S Ω(x)†; the network is a reduced L-CNN:
# plaquettes → parallel-transport convolution → bilinear loop products → trace gating → per-link readout.
import os
import time
import h5py
import numpy as np
import torch
from torch import nn
from torch_ema import ExponentialMovingAverage

L = 16
BETA = 6.0
DATA = f"/work/gw97/w97000/project/DM/2DSUN/trainningdata/cfgs_su3_beta={BETA}_{L}^2.h5"
SIGMA = 10.0  # Σ(1) = 21.5: the fundamental character has decayed to e^{-2Σ/3} = 1e-6, the t=1 marginal is Haar
DELTA_MAX = 1.0  # variance per heat-kernel increment; the nearest wrapping image of the kernel weighs e^{-79/Δ}
CH = 16
NLAYERS = 3
OUT = f"/work/gw97/w97000/project/DM/2DSUN/results/su3_L{L}_beta{BETA}_lcnn_hk_c{CH}l{NLAYERS}"
BATCH = 64
EPOCHS = 100
LR = 2e-4
EMA_DECAY = 0.999
EPS = 1e-5
SEED = 0
dev = "cuda"


def marginal_prob_std(t, sigma):
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma))


def diffusion_coeff(t, sigma):
    return sigma**t


def gellmann_half():
    l = np.zeros((8, 3, 3), complex)
    l[0][0, 1] = l[0][1, 0] = 1
    l[1][0, 1], l[1][1, 0] = -1j, 1j
    l[2][0, 0], l[2][1, 1] = 1, -1
    l[3][0, 2] = l[3][2, 0] = 1
    l[4][0, 2], l[4][2, 0] = -1j, 1j
    l[5][1, 2] = l[5][2, 1] = 1
    l[6][1, 2], l[6][2, 1] = -1j, 1j
    l[7][0, 0] = l[7][1, 1] = 1 / np.sqrt(3)
    l[7][2, 2] = -2 / np.sqrt(3)
    return torch.tensor(l / 2, dtype=torch.complex64)  # Tr(T^a T^b) = δ^{ab}/2, so |Ξ|² = Σ_a ξ_a² = 2 Tr Ξ²


T = gellmann_half().to(dev)


def bc(s):
    return s.view(-1, 1, 1, 1, 1, 1)


def roll(A, mu, k):
    # A[..., x, y, :, :] evaluated at x + k e_mu; lattice axes are the two before the 3x3 block
    return torch.roll(A, -k, dims=A.ndim - 4 + mu)


def plaquette(U):
    U0, U1 = U[:, 0], U[:, 1]
    return U0 @ roll(U1, 0, 1) @ roll(U0, 1, 1).mH @ U1.mH


def transport(W, U):
    # loops based at the four neighbours, brought to x through the connecting link (L-Conv, kernel ±1)
    out = [W]
    for mu in range(2):
        Um = U[:, mu].unsqueeze(1)
        out.append(Um @ roll(W, mu, 1) @ Um.mH)
        Ub = roll(Um, mu, -1)
        out.append(Ub.mH @ roll(W, mu, -1) @ Ub)
    return torch.cat(out, 1)


def herm(M):
    # traceless Hermitian part: the su(3) coordinates of exp(iX)
    H = 0.5 * (M + M.mH)
    tr = H.diagonal(dim1=-2, dim2=-1).sum(-1)
    return H - (tr / 3)[..., None, None] * torch.eye(3, dtype=M.dtype, device=M.device)


def xi_matrix(shape):
    xi = torch.randn(*shape, 8, device=dev)
    return torch.einsum("...a,aij->...ij", xi.to(torch.complex64), T)


def eigvals_h3(Y):
    # eigenvalues of a traceless Hermitian 3x3 matrix: λ³ − pλ − r = 0 solved trigonometrically
    p = (Y @ Y).diagonal(dim1=-2, dim2=-1).sum(-1).real / 2
    r = torch.linalg.det(Y).real
    a = torch.sqrt(p / 3).clamp_min(1e-12)
    phi = torch.acos((r / (2 * a**3)).clamp(-1, 1))
    k = torch.arange(3, device=Y.device)
    return 2 * a[..., None] * torch.cos((phi[..., None] - 2 * np.pi * k) / 3)


def sinc_half(d):
    return torch.where(d.abs() > 1e-2, torch.sin(d / 2) / (d / 2), 1 - d**2 / 24)


def dlog_sinc_half(d):
    # d/dd log(sin(d/2)/(d/2)) = ½ cot(d/2) − 1/d
    return torch.where(d.abs() > 5e-2, 0.5 / torch.tan(d / 2) - 1 / d, -d / 12 - d**3 / 720)


def heat_kernel_step(shape, delta):
    # Y = √Δ Ξ accepted with probability √j(Y), j = Π_{i<j} [sin(d_ij/2)/(d_ij/2)]², d_ij = θ_i − θ_j (Weyl density):
    # the Gaussian pushforward has density Gauss(Y)/j w.r.t. Haar, the heat kernel Gauss(Y)/√j, so exp(iY) is an
    # exact Brownian increment of variance Δ per generator up to wrapping images of weight e^{-79/Δ}.
    # delta is broadcastable to shape; Δ = 0 gives Y = 0.
    sd = torch.sqrt(delta)[..., None, None]
    Y = sd * xi_matrix(shape)
    acc = torch.zeros(shape, dtype=torch.bool, device=dev)
    while not acc.all():
        th = eigvals_h3(Y)
        d = th[..., :, None] - th[..., None, :]
        jhalf = (sinc_half(d[..., 0, 1]) * sinc_half(d[..., 0, 2]) * sinc_half(d[..., 1, 2])).abs()
        acc = acc | (torch.rand(shape, device=dev) < jhalf)
        Y = torch.where(acc[..., None, None], Y, sd * xi_matrix(shape))
    return Y


def heat_kernel_score(Y, delta):
    # ∇ log K_Δ(exp iY) = −Y/Δ − ½ ∇log j(Y) = V diag(−θ/Δ − c) V†, c_i = ½ Σ_{j≠i} [½cot(d_ij/2) − 1/d_ij]
    th, V = torch.linalg.eigh(Y)
    d = th[..., :, None] - th[..., None, :]
    lam = -th / delta[..., None] - 0.5 * dlog_sinc_half(d).sum(-1)
    return V @ torch.diag_embed(lam.to(V.dtype)) @ V.mH


class LCNN(nn.Module):
    def __init__(self, ch, nlayers, sigma):
        super().__init__()
        self.ch, self.sigma = ch, sigma
        self.conv = nn.ParameterList()
        self.bilin = nn.ParameterList()
        self.gate_a = nn.ParameterList()
        self.gate_b = nn.ModuleList()
        cin = 2
        for _ in range(nlayers):
            self.conv.append(nn.Parameter(torch.randn(ch, 5 * cin, dtype=torch.complex64) / np.sqrt(10 * cin)))
            self.bilin.append(nn.Parameter(torch.randn(ch, ch + 1, ch + 1, dtype=torch.complex64) / (ch + 1)))
            self.gate_a.append(nn.Parameter(torch.ones(ch)))
            self.gate_b.append(nn.Linear(64, ch))
            cin = ch
        self.temb = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        # per-site complex coefficients for (mu, channel) from the local loop traces (gauge invariant) and time
        self.readout = nn.Sequential(nn.Linear(2 * ch + 64, 64), nn.SiLU(), nn.Linear(64, 2 * 2 * ch))

    def forward(self, U, t):
        s = marginal_prob_std(t, self.sigma)
        h = self.temb(torch.log(s)[:, None])
        P = plaquette(U)
        W = torch.stack([P, P.mH], 1)
        B, _, Lx, Ly = W.shape[:4]
        eye = torch.eye(3, dtype=W.dtype, device=W.device).expand(B, 1, Lx, Ly, 3, 3)
        for conv, bilin, a, b in zip(self.conv, self.bilin, self.gate_a, self.gate_b):
            W = torch.einsum("oc,bcxyij->boxyij", conv, transport(W, U))
            Wi = torch.cat([eye, W], 1)
            W = torch.einsum("kij,nixyab,njxybc->nkxyac", bilin, Wi, Wi)
            tr = W.diagonal(dim1=-2, dim2=-1).sum(-1).real / 3
            gate = nn.functional.silu(a[None, :, None, None] * tr + b(h)[:, :, None, None])
            W = W * gate[..., None, None]
        tr = W.diagonal(dim1=-2, dim2=-1).sum(-1) / 3
        feat = torch.cat([tr.real, tr.imag, h[:, :, None, None].expand(-1, -1, Lx, Ly)], 1).permute(0, 2, 3, 1)
        c = self.readout(feat).view(B, Lx, Ly, 2, self.ch, 2)
        c = torch.complex(c[..., 0], c[..., 1])
        F = torch.einsum("bxymc,bcxyij->bmxyij", c, W).contiguous()
        return herm(F) / bc(s)


def loss_fn(model, U0):
    # t with density 2(1−t): the score is only learnable for t ≲ 0.5 (partial-reverse diagnostic), beyond that it is ≈ 0
    t = EPS + (1.0 - EPS) * (1 - torch.sqrt(1 - torch.rand(U0.shape[0], device=dev)))
    s = marginal_prob_std(t, model.sigma)
    K = torch.ceil(s**2 / DELTA_MAX).clamp_min(1)
    Kmax = int(K.max())
    delta = s**2 / K
    B = U0.shape[0]
    active = torch.arange(1, Kmax + 1, device=dev)[None, :] <= K[:, None]
    # all increments of all substeps are drawn at once (they do not depend on U), then applied in order
    Y = heat_kernel_step((B, Kmax) + U0.shape[1:4], (delta[:, None] * active)[:, :, None, None, None])
    Ut = U0
    for k in range(Kmax):
        Ut = torch.linalg.matrix_exp(1j * Y[:, k]) @ Ut
    Y_last = Y[torch.arange(B, device=dev), (K - 1).long()]
    # the 1/K weight only reweights t, the minimiser per t is unchanged
    residual = bc(s) * (model(Ut, t) - heat_kernel_score(Y_last, delta[:, None, None, None]))
    return (2 * residual.abs().pow(2).sum((-2, -1)).mean((1, 2, 3)) / K).mean()


def haar_su3(shape):
    # Gram–Schmidt of a complex Ginibre matrix is the QR with positive diagonal, hence Haar on U(3) (Mezzadri); the
    # batched torch.linalg.qr takes 1.4 s for 32k matrices, this takes a few ms.  The det phase is then removed.
    Q = torch.randn(*shape, 3, 3, dtype=torch.complex64, device=dev)
    for _ in range(2):  # the second pass restores float32 orthogonality for ill-conditioned draws (1e-3 → 1e-6)
        cols = []
        for k in range(3):
            v = Q[..., k]
            for q in cols:
                v = v - (q.conj() * v).sum(-1, keepdim=True) * q
            cols.append(v / v.norm(dim=-1, keepdim=True))
        Q = torch.stack(cols, -1)
    return Q / torch.linalg.det(Q)[..., None, None] ** (1 / 3)


def gauge_transform(U, Om):
    return torch.stack([Om @ U[:, mu] @ roll(Om, mu, 1).mH for mu in range(2)], 1)


def check_equivariance(model, U):
    Om = haar_su3(U.shape[:1] + U.shape[2:4])
    t = torch.full((U.shape[0],), 0.3, device=dev)
    S = model(U, t)
    S2 = model(gauge_transform(U, Om), t)
    err = (S2 - Om[:, None] @ S @ Om[:, None].mH).abs().max() / S.abs().max()
    print(f"gauge equivariance check: max relative deviation {err:.1e}")


def main(loss_fn=loss_fn, out=OUT, sigma=SIGMA):
    torch.manual_seed(SEED)
    os.makedirs(out, exist_ok=True)
    with h5py.File(DATA) as f:
        cfgs = f["cfgs"][:]  # (n, y, x, mu, a, b)
    U_data = torch.tensor(np.ascontiguousarray(cfgs.transpose(0, 3, 2, 1, 4, 5)), dtype=torch.complex64, device=dev)
    N = U_data.shape[0]
    model = LCNN(CH, NLAYERS, sigma).to(dev)
    ema = ExponentialMovingAverage(model.parameters(), decay=EMA_DECAY)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"{N} configs, L={L}, β={BETA}, σ={sigma} (std(1)={marginal_prob_std(torch.tensor(1.0), sigma):.2f}), "
          f"{sum(p.numel() for p in model.parameters())} parameters")
    check_equivariance(model, U_data[:2])
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(N, device=dev)
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
                        "sigma": sigma, "L": L, "beta": BETA, "epoch": epoch}, f"{out}/ckpt_epoch{epoch:04d}.pt")


if __name__ == "__main__":
    main()
