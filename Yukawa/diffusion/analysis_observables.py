"""
Compare magnetization observables between HMC training data and DM samples.

Observables (V = L^2):
    M     = (1/V) sum_x phi(x)          (per config)
    <|M|> = order parameter             (paper Table I: 0.0733(1) at g=0.1)
    chi   = V (<M^2> - <|M|>^2)         (susceptibility; not reported in paper)

Errors: data blocking + jackknife (HMC data is autocorrelated; DM samples iid).

Usage:
    python analysis_observables.py --g 0.1 --dm_samples runs/yukawa_L16_g0.1_ncsnpp/data/samples_em_steps1000_0002.npy
"""

import argparse

import h5py
import numpy as np


def jackknife_binned(values, n_bins, estimator):
    """Jackknife error of `estimator` over `values` split into n_bins blocks."""
    n = len(values) // n_bins * n_bins
    blocks = values[:n].reshape(n_bins, -1)
    full = estimator(values[:n])
    jk = np.array([estimator(np.delete(blocks, i, axis=0).ravel())
                   for i in range(n_bins)])
    err = np.sqrt((n_bins - 1) / n_bins * np.sum((jk - full) ** 2))
    return full, err


def observables(M, V, n_bins):
    """Return (<|M|>, err), (chi, err) from per-config magnetization M."""
    absM = np.abs(M)
    mean_absM = jackknife_binned(absM, n_bins, np.mean)
    chi_est = lambda m: V * (np.mean(m ** 2) - np.mean(np.abs(m)) ** 2)
    chi = jackknife_binned(M, n_bins, chi_est)
    return mean_absM, chi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=float, default=0.1)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--hmc_data", type=str, default=None)
    parser.add_argument("--dm_samples", type=str,
                        default="runs/yukawa_L16_g0.1_ncsnpp/data/samples_em_steps1000_0002.npy")
    parser.add_argument("--n_bins_hmc", type=int, default=200)
    parser.add_argument("--n_bins_dm", type=int, default=32)
    args = parser.parse_args()

    if args.hmc_data is None:
        args.hmc_data = f"../samples/yukawa_g{args.g}_L{args.L}_1000000.jld2"
    V = args.L ** 2

    # HMC
    with h5py.File(args.hmc_data, "r") as f:
        cfgs = np.array(f["configs"])
        Mconf = np.array(f["Mconf"])
    M_hmc = cfgs.mean(axis=(1, 2))
    assert np.allclose(np.abs(M_hmc), Mconf, atol=1e-10), "Mconf mismatch vs configs"
    (absM_h, e_absM_h), (chi_h, e_chi_h) = observables(M_hmc, V, args.n_bins_hmc)

    # DM samples: saved as (L, L, N)
    s = np.load(args.dm_samples)
    M_dm = s.mean(axis=(0, 1))
    (absM_d, e_absM_d), (chi_d, e_chi_d) = observables(M_dm, V, args.n_bins_dm)

    print(f"g={args.g}, L={args.L}, V={V}")
    print(f"HMC: N={len(M_hmc)} configs ({args.n_bins_hmc} blocks)   "
          f"DM: N={len(M_dm)} samples ({args.n_bins_dm} blocks)  [{args.dm_samples}]")
    print()
    print(f"{'observable':<12} {'HMC':>18} {'DM':>18}")
    print(f"{'<|M|>':<12} {absM_h:>12.5f}({e_absM_h*1e5:03.0f}) {absM_d:>12.5f}({e_absM_d*1e5:03.0f})")
    print(f"{'chi':<12} {chi_h:>12.4f}({e_chi_h*1e4:03.0f}) {chi_d:>12.4f}({e_chi_d*1e4:03.0f})")


if __name__ == "__main__":
    main()
