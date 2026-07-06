"""Single-site cumulants kappa_n^(site) of the raw field.

Per site x, take the connected cumulant kappa_n(x), then average over sites
(DM.tex Eq. eq:site_cumulants_avg). Computed on RAW configs (no Z2 augmentation).

  kappa2, kappa4 : scipy.stats.kstat  (unbiased k-statistics)
  kappa6, kappa8 : moment -> cumulant recursion + delete-one-block jackknife
                   bias correction (no scipy k-statistic exists for n>4; the
                   plug-in is finite-N biased at high order, so we de-bias).
                   The point estimate is cross-checked against scipy.stats.moment.

Fast: per-block RAW power sums are additive, so each leave-one-block estimate is
(total - one block); the recursion handles the (non-zero) mean correctly, so no
per-block re-centering is needed.
"""
import numpy as np
from math import comb
from scipy import stats


def load_configs(path, L=128):
    """Return configs as (N_cfg, V). Accepts .npy or HDF5/.jld2 (dataset 'cfgs').
    The sample axis is the one whose length != L."""
    if path.endswith(".npy"):
        a = np.load(path).astype(np.float64)
    else:
        import h5py
        with h5py.File(path, "r") as f:
            a = np.asarray(f["cfgs"], np.float64)
    sample_ax = next(i for i, s in enumerate(a.shape) if s != L)
    return np.moveaxis(a, sample_ax, 0).reshape(a.shape[sample_ax], -1)


def _cumulants_from_raw(m):
    """Connected cumulants kappa_1..8 (per element) from raw moments
    m[r] = <x^(r+1)>. Standard recursion; for n>=2 identical to the
    central-moment cumulant (shift invariant)."""
    k = [m[0]]
    for n in range(2, 9):
        v = m[n - 1].copy()
        for i in range(1, n):
            v = v - comb(n - 1, i - 1) * k[i - 1] * m[n - i - 1]
        k.append(v)
    return k


def site_cumulants(data, n_blocks=50):
    """data: (N_cfg, V). Returns site-averaged cumulants.
    kappa2/kappa4 = unbiased k-statistics; kappa6/kappa8 = jackknife-debiased."""
    k2 = float(stats.kstat(data, n=2, axis=0).mean())
    k4 = float(stats.kstat(data, n=4, axis=0).mean())

    bs = data.shape[0] // n_blocks
    d = data[:bs * n_blocks]
    # per-block raw power sums S[r-1, b, x] = sum over block b of phi^r, r=1..8
    S = np.stack([(d.reshape(n_blocks, bs, -1) ** r).sum(1) for r in range(1, 9)])
    tot = S.sum(1)
    N = bs * n_blocks

    def k6k8(rawsum, n):
        k = _cumulants_from_raw(rawsum / n)
        return np.array([k[5].mean(), k[7].mean()])

    full = k6k8(tot, N)
    jk = np.array([k6k8(tot - S[:, b], N - bs) for b in range(n_blocks)])
    bc = n_blocks * full - (n_blocks - 1) * jk.mean(0)
    err = np.sqrt((n_blocks - 1) / n_blocks * ((jk - jk.mean(0)) ** 2).sum(0))

    # cross-check the point estimate against scipy central moments
    m2, m3, m4, m5, m6, m7, m8 = stats.moment(d, order=[2, 3, 4, 5, 6, 7, 8], axis=0)
    chk6 = (m6 - 15*m4*m2 - 10*m3**2 + 30*m2**3).mean()
    chk8 = (m8 - 28*m6*m2 - 56*m5*m3 - 35*m4**2
            + 420*m4*m2**2 + 560*m3**2*m2 - 630*m2**4).mean()

    return dict(k2=k2, k4=k4,
                k6=bc[0], k6_err=err[0], k6_plug=full[0], k6_scipy=float(chk6),
                k8=bc[1], k8_err=err[1], k8_plug=full[1], k8_scipy=float(chk8))


if __name__ == "__main__":
    DATA = "2Dphi4/trainingdata/cfgs_wolff_fahmc_k={k}_l=0.022_128^2.jld2"
    for k in ["0.26", "0.2705", "0.28"]:
        c = site_cumulants(load_configs(DATA.format(k=k)))
        print(f"k={k:>6} | k2={c['k2']:.4f} k4={c['k4']:.4f} | "
              f"k6 {c['k6_plug']:.3f}->{c['k6']:.3f}(±{c['k6_err']:.3f}) "
              f"k8 {c['k8_plug']:.2f}->{c['k8']:.2f}(±{c['k8_err']:.2f}) | "
              f"scipy chk: k6={c['k6_scipy']:.3f} k8={c['k8_scipy']:.2f}", flush=True)
