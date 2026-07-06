"""
Single-site cumulants (kappa_2,4,6,8 and U4_site) with block-jackknife errors
for the L=128 2D single-L DM ensembles in Tab. tab:indist_scalar_L128.

Replicates exactly the canonical method of
analysis/compute_multiL_L128_observables.py::per_site_observables:
  - per-site raw moments m_1..m_8 over configs (per lattice site),
  - Z2 symmetrization by forcing odd raw moments to zero,
  - per-site connected cumulants kappa_n(x), then SPATIAL MEAN over x,
  - U4_site(x) = 1 - m4/(3 m2^2), spatial mean,
  - error = block-jackknife over N_BLOCKS config blocks.

This is the per-site (block-jackknife) error group. The global / zero-mode
columns of the same table use bootstrap instead (see appendix note).

Run from 2Dphi4/:  python analysis/dm_site_cumulant_errors_2d.py
"""

import math
import numpy as np
from scipy.special import binom

ORDER = 8
N_BLOCKS = 50  # appendix: 50-bin block-jackknife

RUNS = [
    (0.26,   "runs/phi4_L128_k0.26_l0.022_ncsnpp_sigma300/data/samples_em_steps2000_epoch=10000.npy"),
    (0.2705, "runs/phi4_L128_k0.2705_l0.022_ncsnpp_sigma450/data/samples_em_steps2000_epoch=10000.npy"),
    (0.28,   "runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640/data/samples_em_steps2000_epoch=10000.npy"),
]


def cumulants_from_raw_moments(m):
    """m: list/array (order,...) raw moments -> kappa same shape."""
    m = [np.asarray(x, dtype=np.float64) for x in m]
    order = len(m)
    kappa = [None] * order
    kappa[0] = m[0]
    for n in range(2, order + 1):
        v = m[n - 1].copy()
        for i in range(1, n):
            v = v - binom(n - 1, i - 1) * kappa[i - 1] * m[n - i - 1]
        kappa[n - 1] = v
    return kappa


def per_site_obs_from_moments(moments):
    cum = cumulants_from_raw_moments(moments)
    m2, m4 = moments[1], moments[3]
    U4_site = 1.0 - m4 / (3.0 * m2 * m2)
    return np.array([
        float(cum[1].mean()),   # kappa2 site
        float(cum[3].mean()),   # kappa4 site
        float(cum[5].mean()),   # kappa6 site
        float(cum[7].mean()),   # kappa8 site
        float(U4_site.mean()),  # U4 site
    ])


def block_persite_moments(cfgs, n_blocks):
    """Return per-block per-site moment sums (B,8,L,L) and counts (B,)."""
    N, L, _ = cfgs.shape
    B = max(1, min(n_blocks, N))
    sums = np.zeros((B, 8, L, L), dtype=np.float64)
    counts = np.zeros(B, dtype=np.int64)
    ranks = np.arange(N)
    block_ids = np.minimum((ranks * B) // N, B - 1)
    for b in range(B):
        xb = cfgs[block_ids == b]
        counts[b] = xb.shape[0]
        xp = xb
        for n in range(1, 9):
            sums[b, n - 1] += xp.sum(axis=0)
            if n != 8:
                xp = xp * xb
    return sums, counts


def z2_moments(total_sums, count, L):
    moments = [total_sums[n] / count for n in range(8)]
    for n in range(0, 8, 2):  # zero the odd raw moments (1st,3rd,5th,7th)
        moments[n] = np.zeros((L, L), dtype=np.float64)
    return moments


def compute(cfgs, n_blocks):
    N, L, _ = cfgs.shape
    sums, counts = block_persite_moments(cfgs, n_blocks)
    B = sums.shape[0]
    total_sums = sums.sum(axis=0)
    total_count = int(counts.sum())
    full = per_site_obs_from_moments(z2_moments(total_sums, total_count, L))
    if B <= 1:
        return full, np.full(full.shape, np.nan)
    jk = []
    for b in range(B):
        leave_count = total_count - int(counts[b])
        if leave_count <= 0:
            continue
        leave_sums = total_sums - sums[b]
        jk.append(per_site_obs_from_moments(z2_moments(leave_sums, leave_count, L)))
    jk = np.asarray(jk)
    err = np.sqrt((len(jk) - 1) * np.mean((jk - full) ** 2, axis=0))
    return full, err


def format_err_latex(value, error, sig_digits=1):
    if error is None or not np.isfinite(error) or error == 0:
        return f"{value:.6g}"
    exp_err = int(math.floor(math.log10(abs(error))))
    last = exp_err - (sig_digits - 1)
    err_digits = int(round(error / 10 ** last))
    val_round = round(value, -last) if last < 0 else round(value / 10 ** last) * 10 ** last
    if last < 0:
        s = f"{val_round:.{-last}f}({err_digits})"
    else:
        s = f"{int(round(val_round))}({err_digits}\\times10^{{{last}}})" if last > 0 else f"{int(round(val_round))}({err_digits})"
    return s


labels = ["kappa2_site", "kappa4_site", "kappa6_site", "kappa8_site", "U4_site"]
existing = {
    0.26:   ["0.8977", "-0.2565", "0.5501", "-2.916", "0.1053"],
    0.2705: ["1.416", "-1.418", "8.703", "-119.3", "0.2350"],
    0.28:   ["2.686", "-9.754", "169.9", "-6343", "0.4504"],
}
for kappa, path in RUNS:
    arr = np.load(path)
    cfgs = np.transpose(arr, (2, 0, 1)).astype(np.float64)  # (N,L,L), raw
    full, err = compute(cfgs, N_BLOCKS)
    print(f"\nkappa={kappa}  N={cfgs.shape[0]} raw, z2=True, n_blocks={N_BLOCKS}")
    for lab, v, e, old in zip(labels, full, err, existing[kappa]):
        print(f"  {lab:12s} new={v: .6g} +/- {e:.3g}   latex={format_err_latex(v, e):>14s}   (table old: {old})")
