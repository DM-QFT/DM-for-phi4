"""Langevin reference of Dieball et al. (Soft Matter 21, 2170), Eq. (2), from the same origins as the DM rollouts.
Usage: python src/langevin_ref.py eq|driven"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.interpolate import CubicSpline

EXP_DIR = Path(__file__).resolve().parents[1]
GROUP = sys.argv[1]
DATA = EXP_DIR / "data/trajectories.h5"
OUT = EXP_DIR / "runs" / f"langevin_{GROUP}" / "samples/rollout.h5"
D = 0.04            # um^2/s, the paper's value from var(dx)/2dt (Eq. 14)
V0 = 0.053          # um/s, prescribed stage velocity; drift is toward -x after the b-file reflection
NBINS = 60          # p_eq histogram bins per period, as in the paper (Fig. 4 caption)
NGRID = 4096
DT_SIM = 0.01       # s, Euler-Maruyama substep, 4 per frame
SAVE_EVERY = 125    # frames = 5 s; a 10 s comparison takes every second step
R, SEED = 32, 20260916


def potential():
    """U(u)/kT = -ln p_eq(u) + const from the equilibrium training windows, min U = 0 (paper, Sec. 3.1)."""
    with h5py.File(DATA, "r") as h:
        g, W = h["eq"], int(h.attrs["window_frames"])
        pos = g["pos"][:, :W]
        fi = g["file_index"][:]
        x0, tilt, lam = (g[k][:][fi][:, None] for k in ("x0", "tilt", "lam"))
    u = ((pos[..., 0] - x0) / lam - tilt * pos[..., 1]) % 1.0
    p, edges = np.histogram(u, bins=NBINS, range=(0, 1), density=True)
    U = -np.log(p)
    U -= U.min()
    centres = 0.5 * (edges[1:] + edges[:-1])
    spline = CubicSpline(np.r_[centres, centres[0] + 1], np.r_[U, U[0]],
                         bc_type="periodic", extrapolate="periodic")
    grid = np.arange(NGRID) / NGRID
    return centres, U, spline(grid), spline(grid, 1)


def ness(U_grid, lam, v0):
    """Paper Eqs. (10)-(13) on the grid: steady-state density p_s(u), its maximum in cycles, and the mean velocity L j_s."""
    x = lam * np.arange(NGRID) / NGRID
    psi = np.exp(U_grid[0] - U_grid + v0 * x / D)
    psi_L = np.exp(v0 * lam / D)
    inv = 1 / psi
    r = np.cumsum(inv) / inv.sum()
    ps = psi * (1 - r * (1 - 1 / psi_L))
    ps /= ps.sum() * lam / NGRID
    js = D * ps[0] * (1 / psi[0] - 1 / psi_L) / (inv.sum() * lam / NGRID)
    return ps, np.argmax(ps) / NGRID, lam * js


def main():
    assert not OUT.exists(), OUT
    centres, U, U_grid, dU = potential()
    print(f"potential from eq windows: barrier {U.max():.3f} kT, {NBINS} bins", flush=True)
    dev = torch.device("cuda:0")
    with h5py.File(DATA, "r") as h:
        g, W, dt_s = h[GROUP], int(h.attrs["window_frames"]), float(h.attrs["dt_s"])
        T, origin = g["pos"].shape[1], W - 1
        r0 = g["pos"][:, origin]
        fi = g["file_index"][:]
        x0, tilt, lam = (torch.from_numpy(g[k][:][fi]).to(dev).repeat(R) for k in ("x0", "tilt", "lam"))
    N = r0.shape[0]
    K = (T - 1 - origin) // SAVE_EVERY
    r = torch.from_numpy(r0.astype(np.float64)).to(dev).repeat(R, 1)     # row = replica * N + track
    force = torch.from_numpy(-dU).to(dev)                                  # -dU/du in kT
    v0 = -V0 if GROUP == "driven" else 0.0
    # the data's u = 0 is the maximum of the measured density; in the NESS that maximum sits at u_max, not at
    # the bare minimum, so the model potential is shifted by s to put its own density maximum at u = 0
    ps, u_max, v_bar = ness(U_grid, float(lam.mean()), v0)
    s = -((u_max + 0.5) % 1 - 0.5)
    print(f"model NESS: density maximum at u = {u_max:.4f} -> potential shift {s:+.4f} cycles, "
          f"mean velocity L j_s = {v_bar:+.5f} um/s", flush=True)
    n_sub = round(SAVE_EVERY * dt_s / DT_SIM)
    noise = np.sqrt(2 * D * DT_SIM)
    torch.manual_seed(SEED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    h = h5py.File(OUT, "w")
    h.create_dataset("r_gen", (K, R, N, 2), dtype="f4", chunks=(1, R, N, 2))
    h["r0"] = r0
    h.create_dataset("times", (K,), dtype="f8")
    h["potential_u"], h["potential_kT"] = centres, U
    h.attrs.update(group=GROUP, D=D, v0=v0, nbins=NBINS, dt_sim=DT_SIM, R=R, seed=SEED, phase_shift_cycles=s,
                   v_bar_model=v_bar, origin_index=origin, dt_frames=SAVE_EVERY, layout="r_gen[step, replica, track, xy]",
                   model="Dieball et al. Eq. (2); U by Boltzmann inversion of the eq training windows")
    print(f"{GROUP}: N={N} R={R} K={K} saved steps of {SAVE_EVERY} frames, {n_sub} substeps each, v0={v0}", flush=True)
    start = time.perf_counter()
    for k in range(K):
        for _ in range(n_sub):
            u = (r[:, 0] - x0) / lam - tilt * r[:, 1] - s
            idx = ((u - torch.floor(u)) * NGRID).long().clamp_(max=NGRID - 1)
            r[:, 0] += (D * force[idx] / lam + v0) * DT_SIM + noise * torch.randn(r.shape[0], device=dev, dtype=torch.float64)
            r[:, 1] += noise * torch.randn(r.shape[0], device=dev, dtype=torch.float64)
        h["r_gen"][k] = r.view(R, N, 2).float().cpu().numpy()
        h["times"][k] = dt_s * (origin + SAVE_EVERY * (k + 1))
        h.flush()
        if (k + 1) % 50 == 0 or k + 1 == K:
            dx = float((r[:, 0] - torch.from_numpy(r0[:, 0].astype(np.float64)).to(dev).repeat(R)).mean())
            print(f"step {k + 1}/{K}: {time.perf_counter() - start:.0f} s, <dx>={dx:+.3f} um", flush=True)
    h.close()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
