# Reverse-time Euler–Maruyama on SU(3)^links for the score model of train_su3_dm.py.
# U_{t-Δt} = exp(i [g² S_θ(U,t) Δt + g √Δt Z]) U_t with g = σ^t, Z = z^a T^a, z ~ N(0,1), started from Haar-random
# links (the t=1 marginal).  Checks the average plaquette and the 1x2, 2x2 Wilson loops against the exact 2D
# values ⟨P⟩, ⟨P⟩², ⟨P⟩⁴ and against the training set.
import os
import sys
import h5py
import numpy as np
import torch
from scipy.special import ive
from torch_ema import ExponentialMovingAverage
from train_su3_dm import LCNN, DATA, OUT, EPOCHS, dev, diffusion_coeff, roll, heat_kernel_step, bc, haar_su3

CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/ckpt_epoch{EPOCHS:04d}.pt"
N_CFGS = 1000
BATCH = 250
N_STEPS = 2000  # 500 steps bias ⟨P⟩ by +0.5% (3σ with 1000 configs)
EPS = 1e-3
SCHEDULE = "linear"  # or "log": t_i geometric from 1 to EPS, as diffusion_lightning's 'log'
SEED = 1


def time_grid(n):
    if SCHEDULE == "log":
        return torch.exp(torch.linspace(0.0, np.log(EPS), n + 1, device=dev))
    return torch.linspace(1.0, EPS, n + 1, device=dev)


def plaq_exact(beta, h=1e-3):
    def lnZ1(b):
        x = b / 3
        Q = 20 + int(np.ceil(6 * np.sqrt(x)))
        return np.log(sum(np.linalg.det([[ive(i - j + q, x) for j in range(3)] for i in range(3)]) for q in range(-Q, Q + 1)))
    return 1 + (lnZ1(beta + h) - lnZ1(beta - h)) / (2 * h)


def line(U, mu, n):
    M = torch.eye(3, dtype=U.dtype, device=U.device).expand(U.shape[0], *U.shape[2:4], 3, 3)
    for k in range(n):
        M = M @ roll(U[:, mu], mu, k)
    return M


def wilson(U, R, Tt):
    # R x Tt rectangle from x: R links along 0, Tt along 1, back along 0, back along 1; returns Re Tr / 3 per site
    W = line(U, 0, R) @ roll(line(U, 1, Tt), 0, R) @ roll(line(U, 0, R), 1, Tt).mH @ line(U, 1, Tt).mH
    return W.diagonal(dim1=-2, dim2=-1).sum(-1).real / 3


def polar_su3(U):
    A, _, Bh = torch.linalg.svd(U)
    P = A @ Bh
    return P / torch.linalg.det(P)[..., None, None] ** (1 / 3)


@torch.no_grad()
def em_sample(model, n, L):
    ts = time_grid(N_STEPS)
    shape = (n, 2, L, L)
    U = haar_su3(shape)
    for i in range(N_STEPS):
        tt = ts[i].expand(n)
        dt = ts[i] - ts[i + 1]
        g = diffusion_coeff(tt, model.sigma)
        X = bc(g**2) * model(U, tt) * dt
        if i < N_STEPS - 1:
            X = X + heat_kernel_step(shape, (g**2 * dt).view(-1, 1, 1, 1))
        U = torch.linalg.matrix_exp(1j * X) @ U
    return U


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
    ck = torch.load(CKPT, map_location=dev)
    model_L = ck["L"]
    model = LCNN(ck["ch"], ck["nlayers"], ck["sigma"]).to(dev)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    ema.load_state_dict(ck["ema"])
    ema.copy_to(model.parameters())
    model.eval()

    U = torch.cat([em_sample(model, min(BATCH, N_CFGS - i), model_L) for i in range(0, N_CFGS, BATCH)])
    unit = (U.mH @ U - torch.eye(3, dtype=U.dtype, device=dev)).abs().amax()
    U = polar_su3(U)
    print(f"epoch {ck['epoch']}, {N_CFGS} configs, {N_STEPS} EM steps, unitarity drift before projection {unit:.1e}")

    P = plaq_exact(ck["beta"])
    print(f"{'exact':10s} ⟨P⟩ = {P:.5f}                             W(1x2) = {P**2:.5f}             W(2x2) = {P**4:.5f}")
    with h5py.File(DATA) as f:
        ref = f["cfgs"][:N_CFGS]
    report("training", torch.tensor(np.ascontiguousarray(ref.transpose(0, 3, 2, 1, 4, 5)), dtype=torch.complex64, device=dev))
    report("generated", U)

    out = f"{os.path.dirname(CKPT)}/samples_epoch{ck['epoch']:04d}_steps{N_STEPS}_{SCHEDULE}.h5"
    with h5py.File(out, "w") as f:
        f["cfgs"] = U.permute(0, 3, 2, 1, 4, 5).cpu().numpy().astype(np.complex128)  # (n, y, x, mu, a, b) like the heat-bath file
        f["plaq"] = wilson(U, 1, 1).mean((1, 2)).cpu().numpy()
        for k, v in (("beta", ck["beta"]), ("L", model_L), ("sigma", ck["sigma"]), ("epoch", ck["epoch"]), ("n_steps", N_STEPS), ("plaq_exact", P)):
            f.attrs[k] = v
    print(f"saved {out}")


if __name__ == "__main__":
    main()
