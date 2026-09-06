#!/usr/bin/env python3
# 多文件按逆方差合并(各文件按自身 sample_dt 判定模的可分辨性)。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from diagonal_modes import cov_per_ens, tau_1e

outdir = Path(sys.argv[1])
outdir.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260729)
NBOOT = 200
def analyze(path):
    with h5py.File(path) as f:
        L = int(f["L"][()])
        kappa = float(f["kappa"][()])
        lambda_ = float(f["lambda"][()])
        dt = float(f["dt"][()])
        n_ens = int(f["Nensemble"][()])
        times = f["times"][()].astype(float)
        ds = f["fields"]  # h5py 轴序 (t, ens, y, x)
        nt = ds.shape[0]
        assert ds.shape == (nt, n_ens, L, L) and nt == len(times)
        nmax = L // 2
        ns = np.arange(1, nmax + 1)
        ax_m = np.empty((nt, 2, nmax, n_ens), np.complex64)
        dg_m = np.empty((nt, 2, nmax, n_ens), np.complex64)
        for s in range(0, nt, 1000):
            F = (np.fft.fft2(ds[s:s + 1000]) / L**2).astype(np.complex64)
            ax_m[s:s + 1000, 0] = F[:, :, 0, 1:nmax + 1].transpose(0, 2, 1)
            ax_m[s:s + 1000, 1] = F[:, :, 1:nmax + 1, 0].transpose(0, 2, 1)
            dg_m[s:s + 1000, 0] = F[:, :, ns, ns].transpose(0, 2, 1)
            dg_m[s:s + 1000, 1] = F[:, :, L - ns, ns].transpose(0, 2, 1)

    sample_dt = times[1] - times[0]
    nlag = nt // 2 + 1
    t = times[:nlag]
    out = {}
    for cls, modes, khat in [
        ("axial", ax_m, 2 * np.sin(np.pi * ns / L)),
        ("diag", dg_m, 2 * np.sqrt(2) * np.sin(np.pi * ns / L)),
    ]:
        for j, n in enumerate(ns):
            cov = cov_per_ens(modes[:, :, j, :])
            pooled = cov.sum(axis=0)
            corr = pooled / pooled[0]
            if corr.min() > np.exp(-1.0) or corr[1] <= 0:
                continue
            tau = tau_1e(t, corr)
            if not np.isfinite(tau) or tau < 3 * sample_dt:
                continue
            boot = []
            for _ in range(NBOOT):
                w = RNG.multinomial(n_ens, np.full(n_ens, 1 / n_ens))
                c = w @ cov
                boot.append(tau_1e(t, c / c[0]))
            out[(cls, n)] = (khat[j], tau, np.std(boot, ddof=1))
            print(f"  {cls:5s} n={n:2d}  khat={khat[j]:.4f}  tau={tau:9.3f} ± {np.std(boot, ddof=1):.3f}")
    return (L, kappa, lambda_, dt), out


results, metadata = [], None
for p in sys.argv[2:]:
    print(p)
    meta, out = analyze(Path(p))
    metadata = metadata or meta
    assert meta == metadata
    results.append(out)
L, kappa, lambda_, dt = metadata

combined = {}
for key in sorted(set().union(*results), key=lambda k: (k[0], k[1])):
    ests = [r[key] for r in results if key in r]
    w = np.array([1 / e[2] ** 2 for e in ests])
    tau = np.sum(w * [e[1] for e in ests]) / w.sum()
    combined[key] = (ests[0][0], tau, 1 / np.sqrt(w.sum()))

with (outdir / "tau_k.tsv").open("w") as f:
    f.write(f"# L={L}\tkappa={kappa:.12g}\tlambda={lambda_:.12g}\tdt={dt:.12g}\n")
    f.write("# class\tn\tkhat\ttau\ttau_err\tnfiles\n")
    for (cls, n), (kh, tau, err) in combined.items():
        f.write(f"{cls}\t{n}\t{kh:.6f}\t{tau:.6f}\t{err:.6f}\t"
                f"{sum(1 for r in results if (cls, n) in r)}\n")
print("combined:")
for (cls, n), (kh, tau, err) in combined.items():
    print(f"  {cls:5s} n={n:2d}  khat={kh:.4f}  tau={tau:9.3f} ± {err:.3f}")

for cls, color, marker, name in [("axial", "#0072B2", "o", "axial (n,0)"),
                                 ("diag", "#E69F00", "s", "diagonal (n,n)")]:
    pts = [(v[0], v[1], v[2]) for (c, n), v in combined.items() if c == cls]
    k, tau, err = map(np.array, zip(*pts))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 7.6), sharex=True,
                                 height_ratios=[2.2, 1], constrained_layout=True)
    ak, at = k[0], tau[0]
    kk = np.geomspace(ak * 0.85, k[-1] * 1.6, 100)
    a1.plot(kk, at * (kk / ak) ** -4, "--", color="0.55", lw=1.2)
    a1.plot(kk, at * (kk / ak) ** -3.75, "-.", color="0.30", lw=1.2)
    ke = k[-1] * 1.25
    a1.text(ke, at * (ke / ak) ** -4 * 0.5, r"$\hat k^{-4}$", color="0.45")
    a1.text(ke, at * (ke / ak) ** -3.75 * 1.7, r"$\hat k^{-3.75}$", color="0.30")
    a1.errorbar(k, tau, err, fmt=marker, color=color, ms=6, mec="white", mew=0.6,
                lw=0, elinewidth=1.2, capsize=2.5, zorder=3)
    a2.errorbar(k, tau * k**3.75, err * k**3.75, fmt=marker, color=color, ms=6,
                mec="white", mew=0.6, lw=0, elinewidth=1.2, capsize=2.5, zorder=3)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_ylabel(r"$\tau_{1/e}(k)$")
    a1.set_title(
        rf"Model B {name}: $L={L}$, canonical" "\n"
        rf"$\lambda={lambda_:g}$, $\kappa={kappa:.6f}$, $dt={dt:g}$",
        fontsize=13,
    )
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.axhline(at * ak**3.75, color="0.30", ls="-.", lw=1.0)
    a2.set_xlabel(r"$\hat k$  ($\hat k^2 = \sum_\mu 4\sin^2(k_\mu/2)$)")
    a2.set_ylabel(r"$\tau\,\hat k^{3.75}$")
    for a in (a1, a2):
        a.grid(alpha=0.25, which="both", lw=0.5)
        a.tick_params(direction="in", which="both")
    fig.savefig(outdir / f"tau_k_{cls}.pdf")
    plt.close(fig)
print(f"saved {outdir}/tau_k_axial.pdf tau_k_diag.pdf tau_k.tsv")
