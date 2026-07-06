"""Dump raw (m, S/V) point pairs per (κ, source) as 2-column .dat for
Mathematica. Density estimation / contouring is done on the MMA side.

Outputs (under results/dm_vs_hmc_L128/dat_cache/):
  mS_pts_k{kappa}_{HMC|DM}.dat   — two columns: m  S_over_V
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))
sys.path.append(str(HERE.parents[2]))

import h5py, numpy as np
from phi4_action import phi4_action as _phi4
import torch

RUNS = [
    ("0.26",   "phi4_L128_k0.26_l0.022_ncsnpp_sigma300"),
    ("0.2705", "phi4_L128_k0.2705_l0.022_ncsnpp_sigma450"),
    ("0.28",   "phi4_L128_k0.28_l0.022_ncsnpp_sigma640"),
]
LAM = 0.022
L = 128; V = L * L
N_USE_HMC = None           # None = use all HMC cfgs (10240); only set for testing
EPOCH = 10000

OUT = HERE.parents[1] / "results" / "dm_vs_hmc_L128" / "dat_cache"
OUT.mkdir(parents=True, exist_ok=True)


def load_hmc(k, n):
    p = HERE.parents[1] / "trainingdata" / f"cfgs_wolff_fahmc_k={k}_l=0.022_128^2.jld2"
    with h5py.File(p, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    if n is None:
        return cfgs
    return cfgs[-n:]


def load_dm(run, z2=True):
    p = HERE.parents[1] / "runs" / run / "data" / f"samples_em_steps2000_epoch={EPOCH:04d}.npy"
    a = np.load(p); a = np.transpose(a, (2, 0, 1)).astype(np.float64)
    if z2:
        a = np.stack([a, -a], axis=1).reshape(-1, *a.shape[1:])
    return a


def action_per_V(cfgs, k):
    phi = torch.tensor(cfgs[:, None, :, :], dtype=torch.float64)
    S = _phi4(phi, k=k, l=LAM, phi_min=-1.0, phi_max=1.0, spatial_dims=2).numpy()
    return S / V


for k_str, run in RUNS:
    k = float(k_str)
    hmc = load_hmc(k_str, N_USE_HMC)
    dm  = load_dm(run, z2=True)
    m_h = hmc.mean(axis=(1, 2)); m_d = dm.mean(axis=(1, 2))
    sv_h = action_per_V(hmc, k); sv_d = action_per_V(dm, k)

    np.savetxt(OUT / f"mS_pts_k{k_str}_HMC.dat",
               np.column_stack([m_h, sv_h]), fmt="%.6e",
               header="m  S_over_V", comments="# ")
    np.savetxt(OUT / f"mS_pts_k{k_str}_DM.dat",
               np.column_stack([m_d, sv_d]), fmt="%.6e",
               header="m  S_over_V (Z2-augmented)", comments="# ")
    print(f"κ={k_str}: HMC N={len(m_h)} (m∈[{m_h.min():.3g},{m_h.max():.3g}], "
          f"S/V∈[{sv_h.min():.3g},{sv_h.max():.3g}]), DM N={len(m_d)}")

print(f"\noutputs in {OUT}")
