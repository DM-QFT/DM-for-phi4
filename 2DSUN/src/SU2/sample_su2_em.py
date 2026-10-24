# Reverse-time Euler–Maruyama on SU(2)^links for the score model of train_su2_dm.py.
# U_{t-Δt} = exp(i [g² S_θ(U,t) Δt + Z]·σ/2) U_t with g² = dΣ²/dt and Z a heat-kernel increment of variance g² Δt, started
# from Haar-random links (the t=1 marginal).  Checks the average plaquette and the 1x2, 2x2 Wilson loops against the
# infinite-volume 2D values ⟨P⟩ = I₂(β)/I₁(β), ⟨P⟩², ⟨P⟩⁴ and against the training set.
import os
import sys
import h5py
import numpy as np
import torch
from scipy.special import ive
from torch_ema import ExponentialMovingAverage
from train_su2_dm import LCNN, DATA, OUT, EPOCHS, DELTA_MAX, diffusion_coeff, roll, qmul, conj, qexp, heat_kernel_step, haar_su2, load_cfgs

CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/ckpt_epoch{EPOCHS:04d}.pt"
N_CFGS = 1000
N_STEPS = 2000
EPS = 1e-5
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


@torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
def em_step(model, U, t, dt, noise):
    tt = t.expand(U.shape[0])
    g2 = diffusion_coeff(tt, model.sigma, model.noise_beta) ** 2
    X = g2.view(-1, 1, 1, 1, 1) * model(U, tt) * dt
    X = X + heat_kernel_step(U.shape[:-1], (g2 * dt * noise).view(-1, 1, 1, 1))
    return qmul(qexp(X), U)


def em_time_grid(t_start=1.0, full_steps=None, eps=None):
    full_steps = N_STEPS if full_steps is None else full_steps
    eps = EPS if eps is None else eps
    if not 0 < eps < t_start <= 1 or full_steps < 1:
        raise ValueError("Reverse diffusion requires 0 < eps < t_start <= 1 and positive steps")
    full = torch.linspace(1.0, eps, full_steps + 1)
    if t_start == 1.0:
        return full
    start = full.new_tensor(t_start)
    # All subsequent times coincide with the full sampler. The first step may be partial.
    return torch.cat((start[None], full[full < start]))


@torch.no_grad()
def em_sample(model, n, L, U_start=None, t_start=1.0, full_steps=None, eps=None):
    ts = em_time_grid(t_start, full_steps, eps)
    n_steps = len(ts)-1
    variance = diffusion_coeff(ts[:-1], model.sigma, model.noise_beta)**2 * (ts[:-1] - ts[1:])
    if variance.max().item() > DELTA_MAX:
        raise ValueError("Reverse step exceeds the heat-kernel variance limit")
    noise = torch.ones(n_steps)
    noise[-1] = 0  # Preserve the final drift-only denoising step.
    shape = (n, 2, L, L)
    if U_start is not None and U_start.shape != (*shape, 4):
        raise ValueError("Initial configuration shape does not match n and L")
    U = haar_su2(shape) if U_start is None else U_start.clone()
    for i in range(n_steps):
        torch.compiler.cudagraph_mark_step_begin()
        # Keep U outside the graph so later replays cannot overwrite retained samples.
        U.copy_(em_step(model, U, ts[i], ts[i] - ts[i + 1], noise[i]))
        if (i+1) % 250 == 0:
            print(f"Reverse steps {i+1}/{n_steps}", flush=True)
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
    model = LCNN(ck["ch"], ck["nlayers"], ck["sigma"], noise_beta=ck.get("noise_beta"))
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    ema.load_state_dict(ck["ema"])
    ema.copy_to(model.parameters())
    model.eval()

    U = em_sample(model, N_CFGS, ck["L"])
    unit = (U.norm(dim=-1) - 1).abs().max()
    U = U / U.norm(dim=-1, keepdim=True)
    print(f"epoch {ck['epoch']}, {N_CFGS} configs, {N_STEPS} EM steps, unitarity drift before projection {unit:.1e}")

    P = ive(2, ck["beta"]) / ive(1, ck["beta"])
    print(f"{'exact':10s} ⟨P⟩ = {P:.5f}                             W(1x2) = {P**2:.5f}             W(2x2) = {P**4:.5f}")
    reference = f"{os.path.dirname(DATA)}/cfgs_su2_beta={ck['beta']}_{ck['L']}^2.h5"
    report("training", load_cfgs(N_CFGS, path=reference))
    report("generated", U)

    out = f"{os.path.dirname(CKPT)}/samples_epoch{ck['epoch']:04d}_steps{N_STEPS}.h5"
    with h5py.File(out, "w") as f:
        f["cfgs"] = to_matrix(U)
        f["plaq"] = wilson(U, 1, 1).mean((1, 2)).cpu().numpy()
        for k, v in (("beta", ck["beta"]), ("L", ck["L"]), ("sigma", ck["sigma"]), ("epoch", ck["epoch"]), ("n_steps", N_STEPS), ("plaq_exact", P)):
            f.attrs[k] = v
        if model.noise_beta is not None:
            f.attrs["noise_beta"] = model.noise_beta
    print(f"saved {out}")


if __name__ == "__main__":
    main()
