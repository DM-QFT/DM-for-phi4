"""
Paper-grade joint (m, S/V) distribution figures for DM vs HMC, three κ.

Outputs to results/dm_vs_hmc_L128/:
  fig_joint_paper.pdf      — 2x3 hexbin (HMC top / DM bottom) × 3 κ
  fig_joint_residual.pdf   — 1x3 log-ratio map + marginal P(m), P(S/V) overlay
  joint_metrics_table.txt  — κ_{2,1}(m²,s) jackknife + sliced Wasserstein + MMD

All DM samples are Z2-augmented (φ, −φ); m is signed (Z2 symmetry visible).
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[2]))
sys.path.append(str(HERE.parents[3]))

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter


REPO = HERE.parents[2]
RUNS = [
    ("0.26",   "sym",    "phi4_L128_k0.26_l0.022_ncsnpp_sigma300"),
    ("0.2705", "crit",   "phi4_L128_k0.2705_l0.022_ncsnpp_sigma450"),
    ("0.28",   "broken", "phi4_L128_k0.28_l0.022_ncsnpp_sigma640"),
]
LAM = 0.022
N_HMC = 2048
N_DM = 2048          # post-Z2 augment cap; 1024 raw → 2048 Z2
JK_BINS = 32
OUT = REPO / "results" / "dm_vs_hmc_L128"
OUT.mkdir(parents=True, exist_ok=True)


# ───────── data ─────────
def load_hmc(path, n=N_HMC):
    with h5py.File(path, "r") as f:
        c = np.array(f["cfgs"], dtype=np.float64)
    return c[-n:]


def load_dm(path, n=N_DM, z2=True):
    arr = np.load(path)
    c = np.transpose(arr, (2, 0, 1)).astype(np.float64)
    if z2:
        c = np.stack([c, -c], axis=1).reshape(-1, *c.shape[1:])
    return c[:n]


def action(cfgs, k):
    nb = np.roll(cfgs, 1, axis=1) + np.roll(cfgs, 1, axis=2)
    return (-2 * k * cfgs * nb + (1 - 2 * LAM) * cfgs ** 2
            + LAM * cfgs ** 4).sum(axis=(1, 2))


def features(cfgs, k):
    V = cfgs.shape[-1] ** 2
    m = cfgs.mean(axis=(1, 2))                 # signed magnetisation
    s = action(cfgs, k) / V
    return m, s


# ───────── metrics ─────────
def jackknife(x, func, n_bins=JK_BINS):
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]; n_bins = min(n_bins, N); bs = N // n_bins
    x = x[: bs * n_bins]
    binned = x.reshape(n_bins, bs, *x.shape[1:]).mean(axis=1)
    full = float(func(binned))
    jk = np.array([func(np.delete(binned, i, axis=0)) for i in range(n_bins)])
    err = float(np.sqrt((n_bins - 1) * np.mean((jk - full) ** 2)))
    return full, err


def kappa21(m2, s):
    """κ_{2,1} = ⟨m² s⟩ − ⟨m²⟩⟨s⟩  (mixed cumulant)."""
    arr = np.stack([m2, s, m2 * s], axis=-1)         # (N, 3)
    f = lambda a: a.mean(axis=0)[2] - a.mean(axis=0)[0] * a.mean(axis=0)[1]
    return jackknife(arr, f)


def sliced_wasserstein(X, Y, n_proj=512, seed=0):
    """Sliced W1: average 1D W1 over random projections of standardised 2D data."""
    rng = np.random.default_rng(seed)
    # Standardise jointly so distances scale invariantly
    XY = np.vstack([X, Y])
    mu = XY.mean(0); sd = XY.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd; Ys = (Y - mu) / sd
    # Random unit directions on S^1
    theta = rng.uniform(0, 2 * np.pi, size=n_proj)
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    px = Xs @ dirs.T; py = Ys @ dirs.T              # (N, n_proj)
    px.sort(axis=0); py.sort(axis=0)
    if px.shape[0] != py.shape[0]:
        # interpolate the longer to match the shorter
        n = min(px.shape[0], py.shape[0])
        idx = np.linspace(0, px.shape[0] - 1, n).astype(int); px = px[idx]
        idx = np.linspace(0, py.shape[0] - 1, n).astype(int); py = py[idx]
    w1_per_proj = np.abs(px - py).mean(axis=0)      # 1D W1 each direction
    return float(w1_per_proj.mean()), float(w1_per_proj.std())


def mmd_gauss(X, Y, n_sub=1024, seed=1):
    """Unbiased MMD² with Gaussian kernel, median-heuristic bandwidth."""
    rng = np.random.default_rng(seed)
    XY = np.vstack([X, Y])
    mu = XY.mean(0); sd = XY.std(0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd; Ys = (Y - mu) / sd
    if Xs.shape[0] > n_sub:
        Xs = Xs[rng.choice(Xs.shape[0], n_sub, replace=False)]
    if Ys.shape[0] > n_sub:
        Ys = Ys[rng.choice(Ys.shape[0], n_sub, replace=False)]
    Z = np.vstack([Xs, Ys])
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    sigma2 = max(np.median(D2[D2 > 0]), 1e-12)
    Kxx = np.exp(-((Xs[:, None] - Xs[None]) ** 2).sum(-1) / sigma2)
    Kyy = np.exp(-((Ys[:, None] - Ys[None]) ** 2).sum(-1) / sigma2)
    Kxy = np.exp(-((Xs[:, None] - Ys[None]) ** 2).sum(-1) / sigma2)
    nx, ny = Xs.shape[0], Ys.shape[0]
    # unbiased MMD² (Gretton 2012)
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1)) \
         + (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1)) \
         - 2.0 * Kxy.mean()
    return float(mmd2)


# ───────── plotting helpers ─────────
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def per_kappa_range(m_h, sv_h, m_d, sv_d, pad=0.05):
    """Make m range symmetric around 0 (Z2-aug), s range from data percentiles."""
    m = np.concatenate([m_h, m_d]); sv = np.concatenate([sv_h, sv_d])
    m_abs_p = np.percentile(np.abs(m), 99.5)
    p_sv = np.percentile(sv, [0.5, 99.5])
    rx = m_abs_p * pad; ry = (p_sv[1] - p_sv[0]) * pad
    return (-m_abs_p - rx, m_abs_p + rx), (p_sv[0] - ry, p_sv[1] + ry)


def kde_levels(Z, mass_fracs=(0.5, 0.68, 0.9, 0.95)):
    """Return contour level values that enclose the given probability fractions
    of the 2D density Z. Z is a (nx, ny) density grid; areas sum is normalised."""
    z_flat = np.sort(Z.ravel())[::-1]
    csum = np.cumsum(z_flat); csum /= csum[-1]
    levels = []
    for f in mass_fracs:
        idx = np.searchsorted(csum, f)
        idx = min(idx, len(z_flat) - 1)
        levels.append(z_flat[idx])
    return np.array(sorted(set(levels)))


def eval_kde_grid(x, y, xr, yr, ng=120):
    """Fit Gaussian KDE on (x, y), evaluate on a (ng, ng) grid covering (xr, yr).
    Returns (XX, YY, Z) with Z shape (ng, ng)."""
    xx = np.linspace(*xr, ng); yy = np.linspace(*yr, ng)
    XX, YY = np.meshgrid(xx, yy, indexing="xy")
    pts = np.vstack([XX.ravel(), YY.ravel()])
    kde = gaussian_kde(np.vstack([x, y]))
    Z = kde(pts).reshape(XX.shape)
    return XX, YY, Z


# ───────── main ─────────
def main():
    data = []
    for k_str, phase, run in RUNS:
        k = float(k_str)
        hmc = load_hmc(REPO / "trainingdata" / f"cfgs_wolff_fahmc_k={k_str}_l=0.022_128^2.jld2")
        dm = load_dm(REPO / "runs" / run / "data" / "samples_em_steps2000_epoch=10000.npy")
        m_h, sv_h = features(hmc, k)
        m_d, sv_d = features(dm,  k)
        data.append(dict(k=k, k_str=k_str, phase=phase,
                         m_h=m_h, sv_h=sv_h, m_d=m_d, sv_d=sv_d))
        print(f"κ={k_str} ({phase})  HMC N={len(m_h)}, DM N={len(m_d)} (Z2-aug)")

    # =============================================================
    # Figure 1: side-by-side hexbin (|m|, S/V), 2×3 grid
    # =============================================================
    # 1×3 layout: HMC and DM overlaid in same axes per κ, using KDE contours
    # at 50/68/90/95% mass. HMC = filled contours (semi-transparent), DM = lines.
    mass_fracs = (0.5, 0.68, 0.9, 0.95)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7))
    for j, d in enumerate(data):
        ax = axes[j]
        xr, yr = per_kappa_range(d["m_h"], d["sv_h"], d["m_d"], d["sv_d"])
        XX, YY, Zh = eval_kde_grid(d["m_h"], d["sv_h"], xr, yr)
        _,  _,  Zd = eval_kde_grid(d["m_d"], d["sv_d"], xr, yr)
        lvl_h = kde_levels(Zh, mass_fracs)
        lvl_d = kde_levels(Zd, mass_fracs)
        # HMC: filled, low-opacity blue gradient
        ax.contourf(XX, YY, Zh, levels=[*lvl_h, Zh.max() * 1.001],
                    colors=["#deebf7", "#9ecae1", "#4292c6", "#08519c"],
                    alpha=0.85)
        # DM: dashed black contour lines on top
        ax.contour(XX, YY, Zd, levels=lvl_d, colors="k",
                   linewidths=1.0, linestyles="dashed")
        # raw scatter underlay at very low opacity for visual anchoring
        ax.scatter(d["m_d"], d["sv_d"], s=1.5, color="C3", alpha=0.25,
                   zorder=0, rasterized=True)
        ax.scatter(d["m_h"], d["sv_h"], s=1.5, color="C0", alpha=0.25,
                   zorder=0, rasterized=True)
        ax.axvline(0, color="0.5", lw=0.5, alpha=0.5)
        ax.set_xlim(*xr); ax.set_ylim(*yr)
        ax.set_xlabel(r"$m$"); ax.set_ylabel(r"$S/V$")
        ax.set_title(rf"$\kappa={d['k_str']}$ ({d['phase']})")
        ax.grid(alpha=0.25)
    # legend (manual, since contour/contourf legend handles are awkward)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#4292c6", edgecolor="none", alpha=0.85,
              label="HMC (50/68/90/95%)"),
        Line2D([0], [0], color="k", lw=1.0, ls="dashed",
               label="DM  (50/68/90/95%)"),
    ]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)
    fig.suptitle(r"Joint density of $(m, S/V)$ — HMC vs DM, $L=128$",
                 y=1.02, fontsize=13)
    out = OUT / "fig_joint_paper.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"saved {out}")

    # =============================================================
    # Figure 2: log-ratio map + marginals
    #   3 columns × 3 rows = (log-ratio, P(|m|) overlay, P(S/V) overlay)
    # =============================================================
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.4, 1.0, 1.0],
                          hspace=0.45, wspace=0.30)
    for j, d in enumerate(data):
        xr, yr = per_kappa_range(d["m_h"], d["sv_h"], d["m_d"], d["sv_d"])
        # ----- (a) log P_DM / P_HMC heatmap -----
        nb = 36
        H, ex, ey = np.histogram2d(d["m_h"], d["sv_h"], bins=nb, range=[xr, yr])
        D, _, _   = np.histogram2d(d["m_d"], d["sv_d"], bins=nb, range=[xr, yr])
        H = H / H.sum(); D = D / D.sum()
        mask = H < 5e-4 * H.max()
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.log10((D + 1e-12) / (H + 1e-12))
        R[mask] = np.nan
        v = np.nanmax(np.abs(R)); v = min(v, 1.0)
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(R.T, origin="lower", aspect="auto",
                       extent=(xr[0], xr[1], yr[0], yr[1]),
                       cmap="seismic", norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v))
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                     label=r"$\log_{10}(P_{\rm DM}/P_{\rm HMC})$")
        ax.set_xlim(*xr); ax.set_ylim(*yr)
        ax.axvline(0, color="0.3", lw=0.6, alpha=0.5)
        ax.set_xlabel(r"$m$"); ax.set_ylabel(r"$S/V$")
        ax.set_title(rf"$\kappa={d['k_str']}$ ({d['phase']})  log-ratio")
        ax.grid(alpha=0.25)

        # ----- (b) P(m) marginal -----
        ax = fig.add_subplot(gs[1, j])
        bins = np.linspace(*xr, 60)
        ax.hist(d["m_h"], bins=bins, density=True, color="C0", alpha=0.55, label="HMC", histtype="stepfilled")
        ax.hist(d["m_d"], bins=bins, density=True, color="C3", alpha=0.55, label="DM",  histtype="stepfilled")
        ax.set_xlim(*xr); ax.set_xlabel(r"$m$"); ax.set_ylabel("density")
        ax.axvline(0, color="0.5", lw=0.6, alpha=0.5)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)

        # ----- (c) P(S/V) marginal -----
        ax = fig.add_subplot(gs[2, j])
        bins = np.linspace(*yr, 60)
        ax.hist(d["sv_h"], bins=bins, density=True, color="C0", alpha=0.55, label="HMC", histtype="stepfilled")
        ax.hist(d["sv_d"], bins=bins, density=True, color="C3", alpha=0.55, label="DM",  histtype="stepfilled")
        ax.set_xlim(*yr); ax.set_xlabel(r"$S/V$"); ax.set_ylabel("density")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.suptitle(r"Joint residual + marginals — DM/HMC at three $\kappa$",
                 y=1.005, fontsize=13)
    out = OUT / "fig_joint_residual.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"saved {out}")

    # =============================================================
    # Table: κ_{2,1}(m², s), sliced W1, MMD
    # =============================================================
    lines = ["Joint-distribution diagnostics, DM vs HMC, L=128",
             f"N_HMC={N_HMC}, N_DM={N_DM} (Z2-augmented), JK bins={JK_BINS}",
             "",
             f"{'κ':>8s} {'phase':>8s}   "
             f"{'κ_{2,1}(m²,s) HMC':>26s}   {'DM':>22s}   "
             f"{'sliced W1':>12s} ± {'std':>5s}   {'MMD²':>10s}"]
    for d in data:
        m2_h = d["m_h"] ** 2; m2_d = d["m_d"] ** 2
        k21_h = kappa21(m2_h, d["sv_h"])
        k21_d = kappa21(m2_d, d["sv_d"])
        Xh = np.stack([d["m_h"], d["sv_h"]], axis=-1)
        Xd = np.stack([d["m_d"], d["sv_d"]], axis=-1)
        sw_mean, sw_std = sliced_wasserstein(Xh, Xd)
        mmd = mmd_gauss(Xh, Xd)
        lines.append(
            f"{d['k_str']:>8s} {d['phase']:>8s}   "
            f"{k21_h[0]:>12.4g} ± {k21_h[1]:.2g}   "
            f"{k21_d[0]:>12.4g} ± {k21_d[1]:.2g}   "
            f"{sw_mean:>12.4g}  {sw_std:>5.2g}    {mmd:>10.3g}"
        )
    table = "\n".join(lines)
    (OUT / "joint_metrics_table.txt").write_text(table)
    print("\n" + table)


if __name__ == "__main__":
    main()
