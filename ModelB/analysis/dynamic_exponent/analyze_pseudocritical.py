#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

path = Path(sys.argv[1])
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

with h5py.File(path) as f:
    kappas = f["kappas"][()].astype(float)
    abs_m = f["abs_m"][()].astype(float)
    L = int(f["L"][()])
    lambda_ = float(f["lambda"][()])
    n_warmup = int(f["n_warmup"][()])
if abs_m.shape[0] == len(kappas):
    abs_m = abs_m.T
assert abs_m.shape[1] == len(kappas)

nblock = 128
n = (len(abs_m) // nblock) * nblock
block_size = n // nblock
a = abs_m[:n].reshape(nblock, block_size, -1)
s1 = a.sum(axis=1)
s2 = (a * a).sum(axis=1)
chi = L**2 * (s2.sum(axis=0) / n - (s1.sum(axis=0) / n) ** 2)

rng = np.random.default_rng(202607291)
nboot = 2000
w = rng.multinomial(nblock, np.full(nblock, 1 / nblock),
                    size=(nboot, len(kappas)))
m1 = np.einsum("bki,ik->bk", w, s1) / n
m2 = np.einsum("bki,ik->bk", w, s2) / n
chi_boot = L**2 * (m2 - m1 * m1)
chi_err = chi_boot.std(axis=0, ddof=1)

i0 = int(np.argmax(chi))
assert 3 <= i0 < len(kappas) - 3
fits = {}
for width in (3, 5, 7):
    lo = min(max(i0 - width // 2, 0), len(kappas) - width)
    ind = np.arange(lo, lo + width)
    p = np.polyfit(kappas[ind], chi[ind], 2)
    peak = -p[1] / (2 * p[0])
    assert p[0] < 0 and kappas[ind[0]] <= peak <= kappas[ind[-1]]
    peaks = np.array([
        -q[1] / (2 * q[0])
        for q in (np.polyfit(kappas[ind], y[ind], 2) for y in chi_boot)
    ])
    valid = (peaks >= kappas[ind[0]]) & (peaks <= kappas[ind[-1]])
    fits[str(width)] = {
        "kappa_peak": float(peak),
        "bootstrap_std": float(peaks[valid].std(ddof=1)),
        "valid_bootstrap": int(valid.sum()),
        "fit_range": [float(kappas[ind[0]]), float(kappas[ind[-1]])],
    }

center = fits["7"]["kappa_peak"]
stat_err = fits["7"]["bootstrap_std"]
window_err = float(
    max(abs(fits[str(w)]["kappa_peak"] - center) for w in (3, 5))
)
summary = {
    "L": L,
    "lambda": lambda_,
    "susceptibility": "L^2*(<m^2>-<|m|>^2)",
    "kappa_peak": center,
    "bootstrap_std": stat_err,
    "fit_window_systematic": window_err,
    "total_uncertainty_quadrature": float(np.hypot(stat_err, window_err)),
    "n_samples_per_kappa": len(abs_m),
    "n_warmup": n_warmup,
    "block_size": block_size,
    "fit_windows": fits,
}
with (outdir / "pseudocritical.json").open("w") as f:
    json.dump(summary, f, indent=2)
with (outdir / "pseudocritical_scan.tsv").open("w") as f:
    f.write("# kappa\tchi_abs\tchi_abs_err\n")
    for kappa, value, err in zip(kappas, chi, chi_err):
        f.write(f"{kappa:.6f}\t{value:.8f}\t{err:.8f}\n")

ind = np.arange(i0 - 3, i0 + 4)
p = np.polyfit(kappas[ind], chi[ind], 2)
x = np.linspace(kappas[0], 0.3407, 500)
fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
ax.errorbar(kappas, chi, chi_err, fmt="o", color="#0072B2", capsize=2.5)
xf = x[x <= kappas[ind[-1]] + 0.0005]
ax.plot(xf, np.polyval(p, xf), color="#D55E00", lw=1.4,
        label="local quadratic")
ax.axvline(center, color="#D55E00", ls="--",
           label=rf"$\kappa_\chi={center:.6f}$")
ax.axvline(0.34034, color="0.35", ls=":", label=r"$\kappa_c=0.34034$")
ax.set(xlabel=r"$\kappa$", ylabel=r"$\chi_{|m|}$",
       title=rf"$L={L}$, $\lambda={lambda_:g}$ susceptibility peak")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.savefig(outdir / "pseudocritical_scan.pdf")
print(json.dumps(summary, indent=2))
