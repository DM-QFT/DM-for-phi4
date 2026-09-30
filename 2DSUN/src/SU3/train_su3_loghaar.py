# One-step noising on SU(3)^links along the geodesic to Haar: U_t = exp(iτΞ) U_0 with Ξ = −i log G, G Haar-random,
# τ = s(t)/s(1), so τ = 1 is exactly Haar.  The coupling (U_0, Ξ) does not depend on t, hence the marginals p_t are
# transported by v = τ̇ E[Ξ|U_t] (conditional flow matching on the group), and the conditional density of U_t,
# j(Y/τ)/(τ⁸ j(Y)) w.r.t. Haar at U_t = exp(iY) U_0, has the score S = (1/τ) ∇log j(Ξ) − ∇log j(Y).  For any g̃ the
# reverse-time SDE dU = [−v + ½ g̃² S] dt̄ + g̃ dB̄ transports p_t back to p_0 (README 7.2).  With g̃ = σ^t this is
# the sampler of sample_su3_em.py, and −v + ½ g² S is one conditional expectation, so a single head suffices:
# S_θ regresses (−v + ½ g² S)/g² = −Y/2s² + ½ [(s₁/s) ∇log j(Ξ) − ∇log j(Y)], s₁ = s(1).  No eigh: θ and the
# frame V are drawn directly.  The target is finite on the support edge since the density of Ξ vanishes like j.
import numpy as np
import torch
from torch import nn
from train_su3_dm import marginal_prob_std, dlog_sinc_half, haar_su3, bc, main, L, BETA, CH, NLAYERS, EPS, dev

SIGMA = 2.15  # s(1) = 1.54 = per-generator rms of Ξ, so τΞ has the rms s(t) of the Gaussian noise in train_su3_dm.py
S1 = float(marginal_prob_std(torch.tensor(1.0), SIGMA))
OUT = f"/work/gw97/w97000/project/DM/2DSUN/results/su3_L{L}_beta{BETA}_lcnn_loghaar_c{CH}l{NLAYERS}"


def log_haar(shape):
    # eigenangles with the Weyl density Π_{i<j} sin²(d_ij/2) (maximum 27/64) on the traceless torus, reduced to the
    # alcove |d_ij| ≤ 2π where exp is injective (angles into (−π, π], then the extreme one shifted by 2π so that
    # Σθ = 0), and an independent Haar-random frame: Ξ = V diag(θ) V† is the principal log of a Haar element
    th = torch.empty(shape + (3,), device=dev)
    todo = torch.ones(shape, dtype=torch.bool, device=dev)
    while todo.any():
        c = (torch.rand(shape + (2,), device=dev) * 2 - 1) * np.pi
        c = torch.cat([c, -c.sum(-1, keepdim=True)], -1)
        d = c[..., :, None] - c[..., None, :]
        w = (torch.sin(d[..., 0, 1] / 2) * torch.sin(d[..., 0, 2] / 2) * torch.sin(d[..., 1, 2] / 2)) ** 2 / (27 / 64)
        hit = todo & (torch.rand(shape, device=dev) < w)
        th = torch.where(hit[..., None], c, th)
        todo &= ~hit
    th = th - 2 * np.pi * torch.round(th / (2 * np.pi))
    m = torch.round(th.sum(-1) / (2 * np.pi))
    k = torch.where(m > 0, th.argmax(-1), th.argmin(-1))
    th = th - 2 * np.pi * m[..., None] * nn.functional.one_hot(k, 3)
    return th, haar_su3(shape)


def loss_fn(model, U0):
    t = EPS + (1.0 - EPS) * (1 - torch.sqrt(1 - torch.rand(U0.shape[0], device=dev)))
    s = marginal_prob_std(t, model.sigma)
    sv = s.view(-1, 1, 1, 1, 1)
    tau = sv / S1
    th, V = log_haar(U0.shape[:4])
    d = th[..., :, None] - th[..., None, :]
    # ∇log j(Y) = V diag(Σ_{j≠i} [½cot(d_ij/2) − 1/d_ij]) V†, evaluated at Ξ and at Y = τΞ in the same frame
    lam = -tau * th / (2 * sv**2) + 0.5 * (dlog_sinc_half(d).sum(-1) / tau - dlog_sinc_half(tau[..., None] * d).sum(-1))
    target = V @ torch.diag_embed(lam.to(V.dtype)) @ V.mH
    Y = V @ torch.diag_embed((tau * th).to(V.dtype)) @ V.mH
    Ut = torch.linalg.matrix_exp(1j * Y) @ U0
    residual = bc(s) * (model(Ut, t) - target)
    return (2 * residual.abs().pow(2).sum((-2, -1))).mean()


if __name__ == "__main__":
    main(loss_fn, OUT, SIGMA)
