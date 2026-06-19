"""
Maximum pairwise distance D_max in the normalised [-1,1] field space.

Loads training configurations, maps them to [-1,1] exactly as FieldDataModule
does (per-dataset global min/max), and reports the maximum (and mean) pairwise
Euclidean distance over a random subsample. D_max^norm sets the empirical VE-SDE
sigma rule:  std(t=1) ~ D_max^norm / 2,  with std(t=1)=sqrt((sigma^2-1)/(2 ln sigma)).

Replaces the old maxdis.ipynb (which repeated the same cell per data file).
Works for any dimension (configs are flattened); pass --data_path for 3D files.

Run from 2Dphi4/:
  python "analysis/calibration/max_pairwise_distance.py" --k 0.26 --l 0.022 --L 128
  python "analysis/calibration/max_pairwise_distance.py" \
      --data_path trainingdata/cfgs_wolff_fahmc_k=0.2705_l=0.022_128^2.jld2
"""

import argparse
import math

import h5py
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import brentq


def std_t1(sigma: float) -> float:
    """VE-SDE marginal std at t=1 for g(t)=sigma^t."""
    return math.sqrt((sigma ** 2 - 1.0) / (2.0 * math.log(sigma)))


def sigma_for_target_std(target: float) -> float:
    """Invert std_t1(sigma) = target for sigma in (1.001, 1e6)."""
    return brentq(lambda s: std_t1(s) - target, 1.001, 1.0e6)


def normalize_pm1(arr, lo, hi):
    """Map to [-1, 1], the FieldDataModule convention."""
    return ((arr - lo) / (hi - lo) - 0.5) * 2.0


def max_pairwise_distance(cfgs_norm, n_subsample=512, seed=0):
    """(N, ...) configs -> (D_max, D_mean, n_used) over a random subsample.

    pdist is O(n^2) in #configs, so we subsample to keep it tractable; the
    maximum is a lower bound that converges quickly for these ensembles.
    """
    flat = cfgs_norm.reshape(cfgs_norm.shape[0], -1).astype(np.float64)
    rng = np.random.default_rng(seed)
    n = min(n_subsample, flat.shape[0])
    idx = rng.choice(flat.shape[0], size=n, replace=False)
    d = pdist(flat[idx])
    return float(d.max()), float(d.mean()), n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_path", default=None,
                    help="path to .jld2; if omitted, built from --k/--l/--L (2D)")
    ap.add_argument("--k", type=float, default=0.26)
    ap.add_argument("--l", type=float, default=0.022)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--n_subsample", type=int, default=512,
                    help="configs subsampled before pdist (keeps O(n^2) tractable)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = args.data_path or (
        f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2")
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)

    lo, hi = float(cfgs.min()), float(cfgs.max())
    cfgs_norm = normalize_pm1(cfgs, lo, hi)
    d_max, d_mean, n = max_pairwise_distance(cfgs_norm, args.n_subsample, args.seed)

    print(f"data         : {path}")
    print(f"shape        : {cfgs.shape}   raw range [{lo:.4f}, {hi:.4f}]")
    print(f"subsample    : {n} cfgs  ({n * (n - 1) // 2} pairs)")
    print(f"D_max^norm   : {d_max:.4f}")
    print(f"D_mean^norm  : {d_mean:.4f}")
    print(f"sigma rule   : std(t=1) ~ D_max/2 = {d_max / 2:.4f}"
          f"   ->   sigma ~ {sigma_for_target_std(d_max / 2):.0f}")


if __name__ == "__main__":
    main()
