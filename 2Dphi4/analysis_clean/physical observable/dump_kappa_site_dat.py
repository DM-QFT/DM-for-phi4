"""Dump per-site cumulants kappa_n(x) on the L=128 lattice for n=4 and n=8,
HMC vs DM, at three kappa values (sym/critical/broken).

Outputs (under results/dm_vs_hmc_L128/dat_cache/):
  kappa_site_k{kappa}_{HMC|DM}_n{4|8}.dat   — 128x128 whitespace matrix
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[2]))
sys.path.append(str(HERE.parents[3]))

import h5py
import numpy as np

from dm_vs_hmc_observables_2d import (
    cumulants_from_moments, _z2_interleave)


RUNS = [
    ("0.26",   "phi4_L128_k0.26_l0.022_ncsnpp_sigma300"),
    ("0.2705", "phi4_L128_k0.2705_l0.022_ncsnpp_sigma450"),
    ("0.28",   "phi4_L128_k0.28_l0.022_ncsnpp_sigma640"),
]
EPOCH    = 10000
N_MATCH  = 2048
ORDER    = 8
N_SHOW   = (4, 8)

OUT = HERE.parents[2] / "results" / "dm_vs_hmc_L128" / "dat_cache"
OUT.mkdir(parents=True, exist_ok=True)


def load_hmc(k_str, n_match):
    p = HERE.parents[2] / "trainingdata" / f"cfgs_wolff_fahmc_k={k_str}_l=0.022_128^2.jld2"
    with h5py.File(p, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    return cfgs[-n_match:]


def load_dm(run, z2=True):
    p = HERE.parents[2] / "runs" / run / "data" / f"samples_em_steps2000_epoch={EPOCH:04d}.npy"
    a = np.transpose(np.load(p), (2, 0, 1)).astype(np.float64)
    if z2:
        a = np.stack([a, -a], axis=1).reshape(-1, *a.shape[1:])
    return a


def per_site_cumulants(cfgs, order=ORDER):
    L = cfgs.shape[1]
    moments = np.empty((order, L, L), dtype=np.float64)
    xp = cfgs.copy()
    for k in range(order):
        moments[k] = xp.mean(axis=0)
        if k != order - 1:
            xp = xp * cfgs
    return cumulants_from_moments(moments)


for k_str, run in RUNS:
    hmc_sub = load_hmc(k_str, N_MATCH)
    hmc = _z2_interleave(hmc_sub)
    dm = load_dm(run, z2=True)
    n_dm = min(N_MATCH * 2, dm.shape[0])
    dm = dm[:n_dm]

    kh = per_site_cumulants(hmc)
    kd = per_site_cumulants(dm)

    for n in N_SHOW:
        np.savetxt(OUT / f"kappa_site_k{k_str}_HMC_n{n}.dat",
                   kh[n - 1], fmt="%.6e")
        np.savetxt(OUT / f"kappa_site_k{k_str}_DM_n{n}.dat",
                   kd[n - 1], fmt="%.6e")
    print(
        f"k={k_str}: HMC N={hmc.shape[0]}, DM N={dm.shape[0]}; "
        f"k4 HMC mean={kh[3].mean():.4g} (std={kh[3].std():.3g})  "
        f"DM mean={kd[3].mean():.4g} (std={kd[3].std():.3g});  "
        f"k8 HMC mean={kh[7].mean():.4g} (std={kh[7].std():.3g})  "
        f"DM mean={kd[7].mean():.4g} (std={kd[7].std():.3g})"
    )

print(f"\noutputs in {OUT}")
