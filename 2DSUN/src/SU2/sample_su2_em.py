# Reverse-time Euler–Maruyama on SU(2)^links for the score model of train_su2_dm.py.
# U_{t-Δt} = exp(i [g² S_θ(U,t) Δt + Z]·σ/2) U_t with g = σ^t and Z a heat-kernel increment of variance g² Δt, started
# from Haar-random links (the t=1 marginal).  Checks the average plaquette and the 1x2, 2x2 Wilson loops against the
# exact 2D values ⟨P⟩ = I₂(β)/I₁(β), ⟨P⟩², ⟨P⟩⁴ and against the training set.
import os
import sys
import h5py
import numpy as np
import torch
from scipy.special import ive
from torch_ema import ExponentialMovingAverage
from train_su2_dm import LCNN, OUT, EPOCHS, diffusion_coeff, roll, qmul, conj, qexp, heat_kernel_step, haar_su2, load_cfgs

CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/ckpt_epoch{EPOCHS:04d}.pt"
N_CFGS = 1000
BATCH = 250
N_STEPS = 2000
EPS = 1e-3
SEED = 1


def line(U, mu, n):
    M = U[:, mu]
    for k in range(1, n):
        M = qmul(M, roll(U[:, mu], mu, k))
    return M


def wilson(U, R, T):
    # R x T rectangle from x: R links along 0, T along 1, back along 0, back along 1; returns Re Tr / 2 per site
    W = qmul(qmul(qmul(line(U, 0, R), roll(line(U, 1, T), 0, R)), conj(roll(line(U, 0, R), 1, T))), conj(line(U, 1, T)))
    return W[..., 0]


@torch.no_grad()
def em_sample(model, n, L):
    ts = torch.linspace(1.0, EPS, N_STEPS + 1)
    shape = (n, 2, L, L)
    U = haar_su2(shape)
    for i in range(N_STEPS):
        tt = ts[i].expand(n)
        dt = ts[i] - ts[i + 1]
        g2 = diffusion_coeff(tt, model.sigma) ** 2
        X = g2.view(-1, 1, 1, 1, 1) * model(U, tt) * dt
        if i < N_STEPS - 1:
            X = X + heat_kernel_step(shape, (g2 * dt).view(-1, 1, 1, 1))
        U = qmul(qexp(X), U)
    return U


def to_matrix(U):
    # (n, mu, x, y, 4) → (n, y, x, mu, a, b) complex like the heat-bath file, U = a0 + i a⃗·σ
    a = U.permute(0, 3, 2, 1, 4).cpu().numpy().astype(np.float64)
    return np.stack([np.stack([a[..., 0] + 1j * a[..., 3], a[..., 2] + 1j * a[..., 1]], -1),
                     np.stack([-a[..., 2] + 1j * a[..., 1], a[..., 0] - 1j * a[..., 3]], -1)], -2)


def report(name, U):
    p = wilson(U, 1, 1)
    w12 = wilson(U, 1, 2).mean((1, 2))
    w22 = wilson(U, 2, 2).mean((1, 2))
    pc = p.mean((1, 2))
    n = U.shape[0]
    print(f"{name:10s} ⟨P⟩ = {pc.mean():.5f} ± {pc.std() / np.sqrt(n):.5f}   site std {p.std():.4f}   "
          f"W(1x2) = {w12.mean():.5f} ± {w12.std() / np.sqrt(n):.5f}   W(2x2) = {w22.mean():.5f} ± {w22.std() / np.sqrt(n):.5f}")


def main():
    torch.manual_seed(SEED)
    ck = torch.load(CKPT)
    model = LCNN(ck["ch"], ck["nlayers"], ck["sigma"])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    ema.load_state_dict(ck["ema"])
    ema.copy_to(model.parameters())
    model.eval()

    U = torch.cat([em_sample(model, min(BATCH, N_CFGS - i), ck["L"]) for i in range(0, N_CFGS, BATCH)])
    unit = (U.norm(dim=-1) - 1).abs().max()
    U = U / U.norm(dim=-1, keepdim=True)
    print(f"epoch {ck['epoch']}, {N_CFGS} configs, {N_STEPS} EM steps, unitarity drift before projection {unit:.1e}")

    P = ive(2, ck["beta"]) / ive(1, ck["beta"])
    print(f"{'exact':10s} ⟨P⟩ = {P:.5f}                             W(1x2) = {P**2:.5f}             W(2x2) = {P**4:.5f}")
    report("training", load_cfgs(N_CFGS))
    report("generated", U)

    out = f"{os.path.dirname(CKPT)}/samples_epoch{ck['epoch']:04d}_steps{N_STEPS}.h5"
    with h5py.File(out, "w") as f:
        f["cfgs"] = to_matrix(U)
        f["plaq"] = wilson(U, 1, 1).mean((1, 2)).cpu().numpy()
        for k, v in (("beta", ck["beta"]), ("L", ck["L"]), ("sigma", ck["sigma"]), ("epoch", ck["epoch"]), ("n_steps", N_STEPS), ("plaq_exact", P)):
            f.attrs[k] = v
    print(f"saved {out}")


if __name__ == "__main__":
    main()
