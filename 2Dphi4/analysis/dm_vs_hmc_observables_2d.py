"""
DM vs HMC observables for 2D phi^4 at L=128, across three kappa
(0.26 symmetric, 0.2705 critical, 0.28 broken).

Computes, with jackknife errors (binned), and side-by-side HMC / DM panels:

  Global (per cfg → ensemble):
    M       = <|m|>
    chi     = V * (<m^2> - <|m|>^2)
    U_4     = 1 - <m^4> / (3 <m^2>^2)
    c_4/V   = (V^3 / N) * <m_c^4>      connected 4th magnetisation cumulant
    c_6/V   = ...                       connected 6th magnetisation cumulant
    <S>     mean action

  Single-site cumulants kappa_n(x), n=1..6, treating each lattice site as
  a scalar RV over the ensemble; report:
    - spatial mean kappa_bar_n and spatial std
    - heatmap kappa_n(x) (n=2 and n=4) HMC / DM / diff
    - diagonal cut kappa_n(x_d, x_d) HMC vs DM

  Action distribution:
    - histogram S(phi) HMC vs DM
    - 2D joint histogram (m, S) HMC vs DM

  Geometric +sign clusters (scipy.ndimage.label, 4-connectivity, open BC):
    - log-binned cluster-size distribution HMC vs DM
    - largest-cluster fraction <s_max / V>

  Tau_int (Sokal automatic windowing) on HMC time series only:
    |M|, m^2, S, single-site phi (DM is iid, so tau_int = 0.5 by convention).

Usage (from 2Dphi4/):
  python analysis/dm_vs_hmc_observables_2d.py
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))      # 2Dphi4/
sys.path.append(str(HERE.parents[2]))      # repo root

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import label as nd_label
from scipy.special import binom

from phi4_action import phi4_action as _phi4_action_unified
from dm_mc_ess_2d import acf_per_chain_fft, sokal_window


# ============================================================
#  Data loading
# ============================================================

def _z2_interleave(cfgs):
    """Interleave φ and −φ so each Z2 pair is adjacent. This prevents
    binned jackknife from putting all +φ in one bin and yields the
    correct (n_independent / 2)-bin variance estimate for Z2-invariant
    observables, since each bin then contains duplicate (cfg, −cfg)
    pairs rather than a block of one Z2 sector."""
    return np.stack([cfgs, -cfgs], axis=1).reshape(-1, *cfgs.shape[1:])


def load_hmc(path, n_max=None, z2_augment=False):
    """HMC cfgs in physical phi-space, shape (N, L, L). Default returns
    the raw chain (so τ_int analyses work). Set z2_augment=True for the
    interleaved Z2-augmented ensemble used in DM-vs-HMC comparisons."""
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    if z2_augment:
        cfgs = _z2_interleave(cfgs)
    if n_max is not None and cfgs.shape[0] > n_max:
        cfgs = cfgs[:n_max]
    return cfgs


def load_dm(path, n_max=None, z2_augment=True):
    """DM samples saved as (L, L, N) → transpose to (N, L, L).
    With z2_augment=True, interleave φ and −φ via :func:`_z2_interleave`."""
    arr = np.load(path)
    cfgs = np.transpose(arr, (2, 0, 1)).astype(np.float64)
    if z2_augment:
        cfgs = _z2_interleave(cfgs)
    if n_max is not None and cfgs.shape[0] > n_max:
        cfgs = cfgs[:n_max]
    return cfgs


# ============================================================
#  Jackknife with binning (handles HMC autocorrelation)
# ============================================================

def binned_jackknife(values, n_bins=50, func=np.mean):
    """
    values: (N,) or (N, ...) ndarray
    Bin into n_bins consecutive blocks, jackknife over bins.
    Returns (estimate, error).
    """
    x = np.asarray(values, dtype=np.float64)
    N = x.shape[0]
    n_bins = int(min(n_bins, N))
    bs = N // n_bins
    n_used = bs * n_bins
    x = x[:n_used]

    binned = x.reshape(n_bins, bs, *x.shape[1:]).mean(axis=1)
    full = np.asarray(func(binned, axis=0))
    out_shape = full.shape
    jk = np.empty((n_bins,) + out_shape, dtype=np.float64)
    for i in range(n_bins):
        jk[i] = np.asarray(func(np.delete(binned, i, axis=0), axis=0))
    err = np.sqrt((n_bins - 1) * np.mean((jk - full) ** 2, axis=0))
    if full.shape == ():
        return float(full), float(err)
    return full, err


# ============================================================
#  Global observables on (N, L, L) cfgs
# ============================================================

def magnetisation(cfgs):
    """m_t = (1/V) sum_x phi_x — signed, shape (N,)."""
    return cfgs.mean(axis=(1, 2))


def action_per_cfg(cfgs, k, l, phi_min=-1.0, phi_max=1.0):
    """S(phi) per cfg, shape (N,). phi already in physical space →
    pass (phi_min,phi_max) = (-1, 1) so the unified action's denormalisation
    is the identity."""
    phi = torch.tensor(cfgs[:, None, :, :], dtype=torch.float64)
    S = _phi4_action_unified(phi, k=k, l=l, phi_min=phi_min, phi_max=phi_max,
                             spatial_dims=2)
    return S.numpy()


def global_estimators(cfgs, k, l, n_bins=50):
    """Return dict of (mean, err) for M, chi, U4, kappa_4, kappa_6, <S>."""
    N, L, _ = cfgs.shape
    V = L * L
    m = magnetisation(cfgs)
    am = np.abs(m)
    m2 = m ** 2
    m4 = m ** 4
    m6 = m ** 6
    S = action_per_cfg(cfgs, k, l)

    out = {}
    out["abs_m"] = binned_jackknife(am, n_bins=n_bins)
    out["m2"]    = binned_jackknife(m2, n_bins=n_bins)

    def chi_func(b):
        return V * (b[..., 1] - b[..., 0] ** 2)
    out["chi"] = binned_jackknife(np.stack([am, m2], axis=-1),
                                  n_bins=n_bins, func=lambda a, axis=0: chi_func(a.mean(axis=axis)))

    def U4_func(b):
        return 1.0 - b[..., 1] / (3.0 * b[..., 0] ** 2)
    out["U4"] = binned_jackknife(np.stack([m2, m4], axis=-1),
                                 n_bins=n_bins, func=lambda a, axis=0: U4_func(a.mean(axis=axis)))

    def c4(b):
        m2m, m4m = b[..., 0], b[..., 1]
        return m4m - 3.0 * m2m ** 2
    out["c4"] = binned_jackknife(np.stack([m2, m4], axis=-1),
                                 n_bins=n_bins, func=lambda a, axis=0: c4(a.mean(axis=axis)))

    def c6(b):
        m2m, m4m, m6m = b[..., 0], b[..., 1], b[..., 2]
        return m6m - 15.0 * m2m * m4m + 30.0 * m2m ** 3
    out["c6"] = binned_jackknife(np.stack([m2, m4, m6], axis=-1),
                                 n_bins=n_bins, func=lambda a, axis=0: c6(a.mean(axis=axis)))

    out["S_mean"] = binned_jackknife(S, n_bins=n_bins)
    return out, dict(m=m, abs_m=am, m2=m2, m4=m4, S=S)


# ============================================================
#  Single-site cumulants kappa_n(x)
# ============================================================

def cumulants_from_moments(moments):
    """moments: (order, ...) → kappa: (order, ...)."""
    m = np.asarray(moments, dtype=np.float64)
    order = m.shape[0]
    kappa = np.empty_like(m)
    kappa[0] = m[0]
    for n in range(2, order + 1):
        v = m[n - 1].copy()
        for i in range(1, n):
            v -= binom(n - 1, i - 1) * kappa[i - 1] * m[n - i - 1]
        kappa[n - 1] = v
    return kappa


def per_site_cumulants(cfgs, order=6):
    """cfgs: (N, L, L). Returns kappa: (order, L, L)."""
    x = cfgs
    L = x.shape[1]
    moments = np.empty((order, L, L), dtype=np.float64)
    xp = x.copy()
    for k in range(order):
        moments[k] = xp.mean(axis=0)
        if k != order - 1:
            xp = xp * x
    return cumulants_from_moments(moments)


def per_site_cumulants_jk(cfgs, order=6, n_bins=20):
    """Binned-jackknife per-site cumulants. Returns (kappa[ord,L,L], err[ord,L,L])."""
    N, L, _ = cfgs.shape
    n_bins = int(min(n_bins, N))
    bs = N // n_bins
    n_used = bs * n_bins
    x = cfgs[:n_used]
    binned_moments = np.empty((n_bins, order, L, L), dtype=np.float64)
    for b in range(n_bins):
        xb = x[b * bs:(b + 1) * bs]
        xp = xb.copy()
        for k in range(order):
            binned_moments[b, k] = xp.mean(axis=0)
            if k != order - 1:
                xp = xp * xb
    full_m = binned_moments.mean(axis=0)
    full_k = cumulants_from_moments(full_m)
    jk = np.empty((n_bins, order, L, L), dtype=np.float64)
    for i in range(n_bins):
        m_i = (binned_moments.sum(axis=0) - binned_moments[i]) / (n_bins - 1)
        jk[i] = cumulants_from_moments(m_i)
    err = np.sqrt((n_bins - 1) * np.mean((jk - full_k) ** 2, axis=0))
    return full_k, err


# ============================================================
#  Geometric +sign clusters
# ============================================================

def cluster_size_distribution(cfgs, sign=+1):
    """For each config, list connected-component sizes of sign(phi)=sign mask
    via scipy.ndimage.label (4-connectivity, open BC)."""
    sizes = []
    max_frac = []
    for x in cfgs:
        mask = (x > 0) if sign > 0 else (x < 0)
        lab, n = nd_label(mask)
        if n == 0:
            max_frac.append(0.0)
            continue
        sz = np.bincount(lab.ravel())[1:]
        sizes.append(sz)
        max_frac.append(sz.max() / x.size)
    if not sizes:
        return np.array([]), np.array(max_frac)
    return np.concatenate(sizes), np.array(max_frac)


# ============================================================
#  Tau_int on HMC time series
# ============================================================

def tau_int_series(series, sokal_c=5.0):
    """series: (N,) 1D. Return tau_int (Sokal)."""
    s = np.asarray(series, dtype=np.float64)[:, None]
    rho = acf_per_chain_fft(s).ravel()
    tau, W, _ = sokal_window(rho, c=sokal_c)
    return tau, W


# ============================================================
#  Plotting
# ============================================================

def fmt(x, e):
    return f"{x:.4g} ± {e:.2g}"


def plot_scalar_summary(scalars, out_path):
    """
    scalars: list of (kappa, name → (val_hmc, err_hmc, val_dm, err_dm))
    Make one figure with 6 panels: M, chi, U4, c4/V, c6/V, <S>.
    """
    names = ["abs_m", "chi", "U4", "c4", "c6", "S_mean"]
    titles = [r"$\langle|m|\rangle$",
              r"$\chi = V(\langle m^2\rangle - \langle|m|\rangle^2)$",
              r"$U_4 = 1 - \langle m^4\rangle / (3\langle m^2\rangle^2)$",
              r"$\kappa_4$  (connected)",
              r"$\kappa_6$  (connected)",
              r"$\langle S\rangle$"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    kappas = [k for k, _ in scalars]
    for ax, name, title in zip(axes.ravel(), names, titles):
        hmc = np.array([d[name][0] for _, d in scalars])
        ehmc = np.array([d[name][1] for _, d in scalars])
        dm = np.array([d[name][2] for _, d in scalars])
        edm = np.array([d[name][3] for _, d in scalars])
        x = np.arange(len(kappas))
        ax.errorbar(x - 0.08, hmc, yerr=ehmc, fmt="o", color="C0", label="HMC", capsize=3)
        ax.errorbar(x + 0.08, dm,  yerr=edm,  fmt="s", color="C3", label="DM",  capsize=3)
        ax.set_xticks(x); ax.set_xticklabels([f"κ={k}" for k in kappas])
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("DM vs HMC global observables (2D φ⁴, L=128)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulant_heatmaps(per_run, out_path):
    """
    per_run: list of (kappa, kappa_hmc[ord,L,L], kappa_dm[ord,L,L]).
    Show n=2 and n=4 heatmaps as 3-col (HMC, DM, diff) × len(runs)*2-row.
    """
    nrows = len(per_run) * 2
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    if nrows == 1:
        axes = axes[None, :]
    for ridx, (k, kh, kd) in enumerate(per_run):
        for i, n_show in enumerate([2, 4]):
            r = ridx * 2 + i
            a_h = kh[n_show - 1]; a_d = kd[n_show - 1]
            vmin = min(a_h.min(), a_d.min()); vmax = max(a_h.max(), a_d.max())
            im0 = axes[r, 0].imshow(a_h, vmin=vmin, vmax=vmax, cmap="viridis")
            im1 = axes[r, 1].imshow(a_d, vmin=vmin, vmax=vmax, cmap="viridis")
            diff = a_d - a_h
            v = np.abs(diff).max()
            im2 = axes[r, 2].imshow(diff, vmin=-v, vmax=v, cmap="seismic")
            axes[r, 0].set_title(rf"HMC, $\kappa_{n_show}(x)$, κ={k}")
            axes[r, 1].set_title(rf"DM,  $\kappa_{n_show}(x)$, κ={k}")
            axes[r, 2].set_title(rf"DM − HMC, $\kappa_{n_show}$")
            for c, im in zip(range(3), [im0, im1, im2]):
                plt.colorbar(im, ax=axes[r, c], fraction=0.045)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("Single-site cumulants κₙ(x), n=2,4 (should be spatially uniform)", y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulant_diagonal(per_run, out_path):
    """3 subplots, one per kappa; each plots kappa_n along diagonal for n=1..6."""
    fig, axes = plt.subplots(1, len(per_run), figsize=(4.5 * len(per_run), 4))
    if len(per_run) == 1:
        axes = [axes]
    for ax, (k, kh, kd) in zip(axes, per_run):
        L = kh.shape[-1]
        idx = np.arange(L)
        diag_h = kh[:, idx, idx]
        diag_d = kd[:, idx, idx]
        for n_show in range(1, kh.shape[0] + 1):
            ax.plot(idx, diag_h[n_show - 1], "-",  color=f"C{n_show-1}",
                    alpha=0.9, label=f"HMC κ_{n_show}" if ax is axes[0] else None)
            ax.plot(idx, diag_d[n_show - 1], "--", color=f"C{n_show-1}",
                    alpha=0.9, label=f"DM  κ_{n_show}" if ax is axes[0] else None)
        ax.set_title(f"κ={k}, diagonal slice κₙ(x_d, x_d)")
        ax.set_xlabel("x_d"); ax.set_ylabel("κₙ")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_action_and_joint(per_run, out_path):
    """3-col × 2-row: row1 = S(phi) 1D hist (viridis-style fills);
    row2 = (m, S) HMC hexbin (viridis) with DM density contours overlaid.

    Using hexbin/contour instead of scatter avoids the "outlier envelope" effect
    where N scatter points with alpha visually look much wider than the bulk
    distribution—especially noticeable in the broken phase where the true (m,S)
    cloud is very narrow but a handful of tail samples draw a misleadingly
    large visual boundary.
    """
    from scipy.ndimage import gaussian_filter
    HMC_FILL = "#3b528b"   # viridis dark-blue
    DM_FILL  = "#fde725"   # viridis yellow
    DM_LINE  = "#d62728"   # red for contour
    ncols = len(per_run)
    fig, axes = plt.subplots(2, ncols, figsize=(4.8 * ncols, 7.5))
    if ncols == 1:
        axes = axes[:, None]
    for ci, (k, raw_h, raw_d) in enumerate(per_run):
        # 1D action histogram
        ax = axes[0, ci]
        bins = 60
        all_S = np.concatenate([raw_h["S"], raw_d["S"]])
        rng = (all_S.min(), all_S.max())
        ax.hist(raw_h["S"], bins=bins, range=rng, alpha=0.6, color=HMC_FILL,
                density=True, label="HMC")
        ax.hist(raw_d["S"], bins=bins, range=rng, alpha=0.6, color=DM_FILL,
                density=True, label="DM")
        ax.set_xlabel("S(φ)"); ax.set_ylabel("density")
        ax.set_title(f"S distribution, κ={k}")
        ax.legend()
        ax.grid(alpha=0.3)

        # 2D joint: HMC hexbin (viridis), DM density as contour overlay.
        ax = axes[1, ci]
        m_all = np.concatenate([raw_h["m"], raw_d["m"]])
        S_all = np.concatenate([raw_h["S"], raw_d["S"]])
        xmin, xmax = np.percentile(m_all, [0.25, 99.75])
        ymin, ymax = np.percentile(S_all, [0.25, 99.75])
        dx, dy = xmax - xmin, ymax - ymin
        xmin -= 0.05 * dx; xmax += 0.05 * dx
        ymin -= 0.05 * dy; ymax += 0.05 * dy
        gs = 50
        hb = ax.hexbin(raw_h["m"], raw_h["S"], gridsize=gs,
                       extent=[xmin, xmax, ymin, ymax],
                       cmap="viridis", mincnt=1)
        cbar = plt.colorbar(hb, ax=ax, fraction=0.045)
        cbar.set_label("HMC count")

        # DM density → smooth → contour at iso-mass levels (50/80/95% of mass).
        H, xed, yed = np.histogram2d(raw_d["m"], raw_d["S"], bins=gs,
                                     range=[[xmin, xmax], [ymin, ymax]])
        H = gaussian_filter(H.T, sigma=1.1)
        if H.sum() > 0:
            flat = np.sort(H.ravel())[::-1]
            cum = np.cumsum(flat) / flat.sum()
            lvls = []
            for frac in (0.5, 0.8, 0.95):
                idx = np.searchsorted(cum, frac)
                if 0 <= idx < len(flat):
                    lvls.append(float(flat[idx]))
            lvls = sorted(set(v for v in lvls if v > 0))
            if lvls:
                xc = 0.5 * (xed[:-1] + xed[1:])
                yc = 0.5 * (yed[:-1] + yed[1:])
                cs = ax.contour(xc, yc, H, levels=lvls, colors=DM_LINE,
                                linewidths=1.4)
                # label each contour by the cumulative-mass fraction it encloses
                fmt = {lvls[i]: lbl for i, lbl in
                       enumerate(["50%", "80%", "95%"][:len(lvls)])}
                ax.clabel(cs, lvls, fmt=fmt, fontsize=8, inline=True)
        # legend handle for DM contour
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], color=DM_LINE, lw=1.4,
                                  label="DM iso-density")],
                  loc="upper right", fontsize=9)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xlabel("m"); ax.set_ylabel("S(φ)")
        ax.set_title(f"(m, S): HMC density + DM contours, κ={k}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_distribution(per_run, out_path):
    ncols = len(per_run)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    for ax, (k, sz_h, mf_h, sz_d, mf_d) in zip(axes, per_run):
        # log-binned histogram
        all_max = max(sz_h.max() if sz_h.size else 1, sz_d.max() if sz_d.size else 1)
        bins = np.logspace(0, np.log10(all_max + 1), 40)
        ax.hist(sz_h, bins=bins, density=True, alpha=0.55, color="C0", label="HMC")
        ax.hist(sz_d, bins=bins, density=True, alpha=0.55, color="C3", label="DM")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cluster size s")
        ax.set_ylabel("P(s)")
        ax.set_title(f"+sign clusters, κ={k}\n"
                     rf"$\langle s_\max/V\rangle$  HMC={mf_h.mean():.3f}, DM={mf_d.mean():.3f}")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================
#  Driver
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", type=int, default=10000)
    p.add_argument("--n_match", type=int, default=2048,
                   help="Subsample HMC down to this many cfgs to match DM N")
    p.add_argument("--n_bins", type=int, default=50)
    p.add_argument("--n_bins_site", type=int, default=20)
    p.add_argument("--lam", type=float, default=0.022)
    p.add_argument("--out_dir", type=str,
                   default=str(HERE.parents[1] / "results" / "dm_vs_hmc_L128"))
    p.add_argument("--no_z2", action="store_true",
                   help="Disable Z2 augmentation of DM samples (default: on).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("0.26",   "phi4_L128_k0.26_l0.022_ncsnpp_sigma300",   "0.26",   "symmetric"),
        ("0.2705", "phi4_L128_k0.2705_l0.022_ncsnpp_sigma450", "0.2705", "critical"),
        ("0.28",   "phi4_L128_k0.28_l0.022_ncsnpp_sigma640",   "0.28",   "broken"),
    ]
    repo_root = HERE.parents[1]

    scalars = []          # for global plot
    cumul_runs = []       # for cumulant heatmap/diagonal
    raw_runs = []         # for (S, m) plot
    clust_runs = []       # for cluster plot
    tau_lines = []        # for summary text
    summary_lines = []

    for k_str, run_name, hmc_k, phase in runs:
        print(f"\n========== κ={k_str}  ({phase}) ==========")
        dm_path = repo_root / "runs" / run_name / "data" / f"samples_em_steps2000_epoch={args.epoch:04d}.npy"
        hmc_path = repo_root / "trainingdata" / f"cfgs_wolff_fahmc_k={hmc_k}_l=0.022_128^2.jld2"

        if not dm_path.exists():
            print(f"  [skip] no DM samples at {dm_path}")
            continue

        z2 = not args.no_z2
        dm_full = load_dm(str(dm_path), z2_augment=z2)
        # Load raw HMC (z2_augment=False) so τ_int on hmc_full uses the raw
        # chain. Subsample first, then interleave Z2 pairs to match DM.
        hmc_full = load_hmc(str(hmc_path), z2_augment=False)
        n_hmc_raw = min(args.n_match, hmc_full.shape[0])
        hmc_sub = hmc_full[-n_hmc_raw:]                  # last n cfgs (skip burn-in)
        hmc = _z2_interleave(hmc_sub) if z2 else hmc_sub
        # DM is iid; with Z2 augmentation, use up to 2*n_match (full augmented set)
        n_dm_cap = args.n_match * (2 if z2 else 1)
        n_dm = min(n_dm_cap, dm_full.shape[0])
        dm  = dm_full[:n_dm]
        print(f"  HMC cfgs: {hmc.shape}, DM cfgs: {dm.shape} (z2_aug={z2})")

        k = float(k_str)

        # ---- global ----
        gh, rawh = global_estimators(hmc, k=k, l=args.lam, n_bins=args.n_bins)
        gd, rawd = global_estimators(dm,  k=k, l=args.lam, n_bins=args.n_bins)
        merged = {n: (gh[n][0], gh[n][1], gd[n][0], gd[n][1]) for n in gh}
        scalars.append((k_str, merged))

        # ---- per-site cumulants ----
        kh, kh_err = per_site_cumulants_jk(hmc, order=6, n_bins=args.n_bins_site)
        kd, kd_err = per_site_cumulants_jk(dm,  order=6, n_bins=args.n_bins_site)
        cumul_runs.append((k_str, kh, kd))

        # ---- raw timeseries for joint plot ----
        raw_runs.append((k_str, rawh, rawd))

        # ---- clusters ----
        sz_h, mf_h = cluster_size_distribution(hmc, sign=+1)
        sz_d, mf_d = cluster_size_distribution(dm,  sign=+1)
        clust_runs.append((k_str, sz_h, mf_h, sz_d, mf_d))

        # ---- tau_int on full HMC chain (uses every sample, not the subset) ----
        m_hmc_full = magnetisation(hmc_full)
        S_hmc_full = action_per_cfg(hmc_full, k=k, l=args.lam)
        tau_M, _   = tau_int_series(np.abs(m_hmc_full))
        tau_m2, _  = tau_int_series(m_hmc_full ** 2)
        tau_S, _   = tau_int_series(S_hmc_full)
        tau_phi, _ = tau_int_series(hmc_full[:, 0, 0])
        tau_lines.append(
            f"  κ={k_str:>6s}  HMC N={len(m_hmc_full)}  "
            f"τ_int(|M|)={tau_M:6.2f}  τ_int(m²)={tau_m2:6.2f}  "
            f"τ_int(S)={tau_S:6.2f}  τ_int(φ_site)={tau_phi:6.2f}"
        )

        # ---- text summary ----
        summary_lines += [
            f"\n=== κ = {k_str} ({phase}) ===  HMC N={hmc.shape[0]}, DM N={dm.shape[0]}",
            f"  {'observable':>14s}    {'HMC':>18s}    {'DM':>18s}    {'rel diff (DM/HMC-1)':>22s}",
        ]
        for name in ["abs_m", "chi", "U4", "c4", "c6", "S_mean"]:
            hv, he, dv, de = merged[name]
            rd = (dv - hv) / (abs(hv) + 1e-30)
            summary_lines.append(
                f"  {name:>14s}    {fmt(hv, he):>18s}    {fmt(dv, de):>18s}    {rd:>+22.3%}"
            )
        summary_lines.append(
            f"  {'kappa_2_bar':>14s}    {kh[1].mean():18.4g}    {kd[1].mean():18.4g}    "
            f"{(kd[1].mean()/kh[1].mean()-1):>+22.3%}"
        )
        summary_lines.append(
            f"  {'kappa_4_bar':>14s}    {kh[3].mean():18.4g}    {kd[3].mean():18.4g}    "
            f"{(kd[3].mean()/(kh[3].mean()+1e-30)-1):>+22.3%}"
        )
        summary_lines.append(
            f"  spatial std of κ₂(x):  HMC={kh[1].std():.3g}  DM={kd[1].std():.3g}   "
            f"(should be ~ statistical noise)"
        )

    # -------- write summary --------
    summary = "\n".join(
        ["DM vs HMC observables — 2D phi^4, L=128, λ=0.022",
         f"epoch={args.epoch}, n_match={args.n_match}, z2_aug_dm={not args.no_z2}, jk n_bins={args.n_bins}",
         ""] + summary_lines
        + ["", "tau_int (Sokal, c=5) on full HMC chain (DM samples are iid ⇒ τ≈0.5):"]
        + tau_lines
    )
    (out_dir / "summary.txt").write_text(summary)
    print("\n" + summary)

    # -------- figures --------
    if scalars:
        plot_scalar_summary(scalars, out_dir / "scalar_summary.pdf")
        print(f"saved {out_dir/'scalar_summary.pdf'}")
    if cumul_runs:
        plot_cumulant_heatmaps(cumul_runs, out_dir / "cumulant_heatmaps.pdf")
        plot_cumulant_diagonal(cumul_runs, out_dir / "cumulant_diagonal.pdf")
        print(f"saved cumulant_*.pdf")
    if raw_runs:
        plot_action_and_joint(raw_runs, out_dir / "action_and_joint.pdf")
        print(f"saved action_and_joint.pdf")
    if clust_runs:
        plot_cluster_distribution(clust_runs, out_dir / "cluster_distribution.pdf")
        print(f"saved cluster_distribution.pdf")


if __name__ == "__main__":
    main()
