"""Compute action density <S>/V for the 3D phi^4 HMC and DM ensembles at the
three trained kappa values, in the format used by Tab 2 of the paper.

Action convention matches phi4_action.py:
    S[phi] = sum_x [ -2 kappa phi_x sum_{mu} phi_{x+mu_hat}
                     + (1 - 2 lambda) phi_x^2
                     + lambda phi_x^4 ]
with forward-only neighbour sums (one direction per axis).

DM filtering follows compare_L64_multiL_HMC_cumulants.py:
    drop any cfg with min(phi) < lo or max(phi) > hi, where [lo, hi] is the
    full HMC field-range support for that kappa.

Errors are bootstrap 1 sigma over configurations (B=2000 resamples).

Run from 3Dphi4/:
    python analysis/compute_action_density_3d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_3D = HERE.parent           # .../project/DM/3Dphi4
sys.path.append(str(REPO_3D))
from _layout import to_NLLL     # noqa: E402

LAM = 0.9
L = 64
V = L ** 3

# (kappa, sigma) pairs from the in-distribution single-L runs, matching
# _L64_SINGLE in compare_L64_multiL_HMC_cumulants.py.
KAPPAS = [
    (0.18,   2000),
    (0.1923, 2048),
    (0.2,    2760),
]
EP = 20000
NUM_STEPS = 2000
SCHEDULE = "linear"        # Tab 2 uses linear time grid
N_BOOT = 2000
RNG = np.random.default_rng(0)


def phi4_S_per_cfg(cfgs: np.ndarray, kappa: float, lam: float) -> np.ndarray:
    """Per-cfg action S[phi] for cfgs in raw field space, shape (N, L, L, L)."""
    nb = (np.roll(cfgs, 1, axis=1)
          + np.roll(cfgs, 1, axis=2)
          + np.roll(cfgs, 1, axis=3))
    local = -2.0 * kappa * cfgs * nb + (1.0 - 2.0 * lam) * cfgs ** 2 + lam * cfgs ** 4
    return local.sum(axis=(1, 2, 3))


def load_hmc(kappa: float) -> tuple[np.ndarray, float, float]:
    """Load HMC cfgs (N, L, L, L) and return (cfgs, lo, hi) field-range support.

    [lo, hi] is the actual min/max of the full HMC ensemble, equal to the
    cached cfgs_min/cfgs_max written by training scripts when available.
    """
    p = REPO_3D / f"trainingdata/cfgs_wolff_fahmc_k={kappa}_l={LAM}_{L}^3.jld2"
    with h5py.File(p, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    cfgs = to_NLLL(cfgs)
    return cfgs, float(cfgs.min()), float(cfgs.max())


def load_dm(kappa: float, sigma: int) -> np.ndarray:
    """Load DM cfgs (N, L, L, L) in raw field space (de-normalised by sample_phi4.py)."""
    run = REPO_3D / f"runs/phi4_3d_L64_k{kappa}_l{LAM}_ncsnpp_sigma{sigma}/data"
    # tag convention: see compare_L64_multiL_HMC_cumulants.py.dm_path_L64single
    cands = [
        run / f"samples_em_steps{NUM_STEPS}_{EP}.npy",
        run / f"samples_em_{SCHEDULE}_steps{NUM_STEPS}_epoch={EP}.npy",
        run / f"samples_em_steps{NUM_STEPS}_epoch={EP}.npy",
    ]
    for p in cands:
        if p.exists():
            return to_NLLL(np.load(p)).astype(np.float64)
    raise FileNotFoundError(f"None of: {[str(p) for p in cands]}")


def filter_oor(cfgs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sp = (1, 2, 3)
    keep = (cfgs.min(axis=sp) >= lo) & (cfgs.max(axis=sp) <= hi)
    return cfgs[keep]


def bootstrap_mean_err(x: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    n = len(x)
    idx = RNG.integers(0, n, size=(n_boot, n))
    return float(x.mean()), float(x[idx].mean(axis=1).std(ddof=1))


def fmt(mean: float, err: float) -> str:
    """Render mean(err) in APS style with the error sitting in the last 1-2 digits.

    Auto-picks the number of decimal digits so err_in_last_digits is in [1, 99].
    Falls back to 6 decimals if err is non-finite.
    """
    if err <= 0 or not np.isfinite(err):
        return f"{mean:.6f}"
    digits = max(0, int(np.ceil(-np.log10(err))) + 1)
    err_disp = round(err * 10 ** digits)
    return f"{mean:.{digits}f}({err_disp:d})"


def main() -> None:
    print(f"{'kappa':>8}   {'HMC <S>/V':>14}   {'N_HMC':>6}   "
          f"{'DM <S>/V':>14}   {'N_DM':>5}   N_drop  "
          f"raw HMC mean,err  raw DM mean,err")
    print("-" * 110)
    for kappa, sigma in KAPPAS:
        hmc_cfgs, lo, hi = load_hmc(kappa)
        S_hmc = phi4_S_per_cfg(hmc_cfgs, kappa, LAM) / V
        m_h, e_h = bootstrap_mean_err(S_hmc)

        dm_raw = load_dm(kappa, sigma)
        dm_cfgs = filter_oor(dm_raw, lo, hi)
        # Z_2-symmetrise DM by concatenating {phi, -phi}. The action is
        # Z_2-invariant so per-cfg S values just duplicate; the bootstrap pool
        # doubles to 2N which shrinks the naive 1-sigma error by sqrt(2),
        # matching the convention used for the other columns of Tab.~\ref{tab:indist_scalar_L128}.
        dm_cfgs_z2 = np.concatenate([dm_cfgs, -dm_cfgs], axis=0)
        S_dm = phi4_S_per_cfg(dm_cfgs_z2, kappa, LAM) / V
        m_d, e_d = bootstrap_mean_err(S_dm)

        print(f"{kappa:>8}   {fmt(m_h, e_h):>14}   {len(hmc_cfgs):>6}   "
              f"{fmt(m_d, e_d):>14}   {len(dm_cfgs_z2):>5}   {len(dm_raw) - len(dm_cfgs):>5}   "
              f"{m_h:+.6e},{e_h:.2e}   {m_d:+.6e},{e_d:.2e}")


if __name__ == "__main__":
    main()
