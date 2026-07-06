"""
Supplement to dm_vs_hmc_observables_2d.py — compute the 4 observables that
the main script doesn't report:

  kappa_8(m)                  eighth connected magnetisation cumulant
                              (unscaled; matches the c4, c6 convention of
                              the main script).
  kappa_6^(site)              sixth on-site connected cumulant (pool every
                              (cfg, x) as one sample of phi(x)).
  kappa_8^(site)              eighth on-site connected cumulant.
  U_4^(site) = 1 - <phi^4>/(3 <phi^2>^2)
                              on-site Binder cumulant.

All quantities use the same data and conventions as the main script
(Z2-augmented DM, 50-bin block jackknife on global, 20-bin on site).

Usage (from 2Dphi4/):
  python analysis/dm_vs_hmc_extra_obs_2d.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))
sys.path.append(str(HERE.parents[2]))

import numpy as np

from analysis.dm_vs_hmc_observables_2d import (
    load_dm, load_hmc, magnetisation, binned_jackknife,
    per_site_cumulants_jk, _z2_interleave,
)


def kappa8(m_arr, n_bins=50):
    """kappa_8 of m (Z2-symmetric form): <m^8>_c.

    For ensembles with <m>=0 (after Z2 augmentation),
      kappa_8(m) = <m^8>
                 - 28 <m^6><m^2>
                 - 35 <m^4>^2
                 + 420 <m^4><m^2>^2
                 - 630 <m^2>^4 .
    Matches the c4, c6 convention of the main script.
    """
    m2 = m_arr ** 2
    m4 = m_arr ** 4
    m6 = m_arr ** 6
    m8 = m_arr ** 8

    stk = np.stack([m2, m4, m6, m8], axis=-1)

    def c8(b):
        m2m, m4m, m6m, m8m = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return (m8m
                - 28.0 * m6m * m2m
                - 35.0 * m4m ** 2
                + 420.0 * m4m * m2m ** 2
                - 630.0 * m2m ** 4)

    return binned_jackknife(stk, n_bins=n_bins,
                            func=lambda a, axis=0: c8(a.mean(axis=axis)))


def U4_site_jk(cfgs, n_bins=20):
    """U_4^(site) = 1 - <phi^4>/(3 <phi^2>^2), jackknifed over config blocks.
    Each (cfg, site) is one sample of phi(x).
    """
    N, L, _ = cfgs.shape
    bs = N // n_bins
    n_used = bs * n_bins
    x = cfgs[:n_used].astype(np.float64)
    binned_m2 = np.empty(n_bins, dtype=np.float64)
    binned_m4 = np.empty(n_bins, dtype=np.float64)
    for b in range(n_bins):
        xb = x[b * bs:(b + 1) * bs]
        x2 = xb * xb
        binned_m2[b] = x2.mean()
        binned_m4[b] = (x2 * x2).mean()

    def U4_site_(m2, m4):
        return 1.0 - m4 / (3.0 * m2 * m2)

    full_m2 = binned_m2.mean()
    full_m4 = binned_m4.mean()
    full = U4_site_(full_m2, full_m4)
    jk = np.empty(n_bins, dtype=np.float64)
    for i in range(n_bins):
        m2_jk = (binned_m2.sum() - binned_m2[i]) / (n_bins - 1)
        m4_jk = (binned_m4.sum() - binned_m4[i]) / (n_bins - 1)
        jk[i] = U4_site_(m2_jk, m4_jk)
    err = np.sqrt((n_bins - 1) * np.mean((jk - full) ** 2))
    return float(full), float(err)


def main():
    epoch = 10000
    n_match = 2048
    lam = 0.022
    n_bins = 50
    n_bins_site = 20
    z2 = True

    repo_root = HERE.parents[1]

    runs = [
        ("0.26",   "phi4_L128_k0.26_l0.022_ncsnpp_sigma300",   "0.26"),
        ("0.2705", "phi4_L128_k0.2705_l0.022_ncsnpp_sigma450", "0.2705"),
        ("0.28",   "phi4_L128_k0.28_l0.022_ncsnpp_sigma640",   "0.28"),
    ]

    print(f"DM vs HMC supplementary observables — 2D phi^4, L=128, λ={lam}")
    print(f"epoch={epoch}, n_match={n_match}, z2_aug_dm={z2}, jk n_bins={n_bins} (global) / {n_bins_site} (site)")
    for k_str, run_name, hmc_k in runs:
        print(f"\n========== κ={k_str} ==========")
        dm_path = repo_root / "runs" / run_name / "data" / f"samples_em_steps2000_epoch={epoch:04d}.npy"
        hmc_path = repo_root / "trainingdata" / f"cfgs_wolff_fahmc_k={hmc_k}_l=0.022_128^2.jld2"
        if not dm_path.exists():
            print(f"  [skip] no DM samples at {dm_path}")
            continue

        dm_full = load_dm(str(dm_path), z2_augment=z2)
        hmc_full = load_hmc(str(hmc_path), z2_augment=False)
        n_hmc_raw = min(n_match, hmc_full.shape[0])
        hmc_sub = hmc_full[-n_hmc_raw:]
        hmc = _z2_interleave(hmc_sub) if z2 else hmc_sub
        n_dm_cap = n_match * (2 if z2 else 1)
        n_dm = min(n_dm_cap, dm_full.shape[0])
        dm = dm_full[:n_dm]
        L = hmc.shape[1]
        V = L * L
        print(f"  HMC cfgs: {hmc.shape}, DM cfgs: {dm.shape}, V={V}")

        # --- kappa_8(m) ---
        m_h = magnetisation(hmc)
        m_d = magnetisation(dm)
        c8h, c8h_err = kappa8(m_h, n_bins=n_bins)
        c8d, c8d_err = kappa8(m_d, n_bins=n_bins)
        rd = (c8d - c8h) / (abs(c8h) + 1e-30)
        print(f"  kappa_8(m)       HMC = {c8h:+.4e} +/- {c8h_err:.2e}   "
              f"DM = {c8d:+.4e} +/- {c8d_err:.2e}   "
              f"rel = {rd:+.3%}")

        # --- kappa_6, kappa_8 on-site ---
        kh, kh_err = per_site_cumulants_jk(hmc, order=8, n_bins=n_bins_site)
        kd, kd_err = per_site_cumulants_jk(dm,  order=8, n_bins=n_bins_site)
        for nn in (6, 8):
            vh = kh[nn - 1].mean()
            vd = kd[nn - 1].mean()
            rd = (vd - vh) / (abs(vh) + 1e-30)
            print(f"  kappa_{nn}^(site)    HMC = {vh:+.4e}   "
                  f"DM = {vd:+.4e}   rel = {rd:+.3%}")

        # --- U_4^(site) ---
        U4sh, U4sh_err = U4_site_jk(hmc, n_bins=n_bins_site)
        U4sd, U4sd_err = U4_site_jk(dm,  n_bins=n_bins_site)
        rd = (U4sd - U4sh) / (abs(U4sh) + 1e-30)
        print(f"  U_4^(site)       HMC = {U4sh:+.6f} +/- {U4sh_err:.2e}   "
              f"DM = {U4sd:+.6f} +/- {U4sd_err:.2e}   "
              f"rel = {rd:+.3%}")


if __name__ == "__main__":
    main()
