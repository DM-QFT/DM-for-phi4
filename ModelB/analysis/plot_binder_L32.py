"""Fig.2 两条链的四阶量随 t 演化：zlog DM 对 SOSRA。

站点涨落场 δφ=φ-M 的 Binder U_δ=1-<δφ⁴>/(3<δφ²>²)，最软对角模的 U_1=1-<|φ_1|⁴>/(2<|φ_1|²>²)
（复高斯为 0），以及站点 <φ⁴>。M 逐轨道守恒，全系综 M 的 Binder 不随 t 变，不画。
"""

from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT.parent / "plots"
sys.path.insert(0, str(PLOTS))
from modelb_palette import ALT, DM  # noqa: E402

plt.style.use(PLOTS / "cjp_prl_nature.mplstyle")

RUN = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip/samples"
T0 = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip/samples"
REF = ROOT / "results/modelb_reference"
OUT = ROOT / "results/dm_vs_modelb/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip/binder_L32"
ENSEMBLES = (
    ("Eq.", ALT, "o", T0 / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     RUN / "hmc_chain_ep10000_N1024/chain.h5", REF / "corrgauss_hmc_L32_N1024.h5"),
    ("Noneq.", DM, "s", T0 / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     RUN / "gaussianM_chain_ep10000_N1024/chain.h5", REF / "corrgauss_gaussianM_L32_N1024.h5"),
)
TMAX = 100
L = 32
K1 = ((1, 1), (1, L - 1), (L - 1, 1), (L - 1, L - 1))
LABELS = (r"$U_\delta(t)$", r"$U_1(t)$", r"$\langle\phi^4\rangle(t)$")


def per_config(x):
    """每构型 (<δφ²>, <δφ⁴>, <φ⁴>, |φ_1|², |φ_1|⁴)，|φ_1| 对 4 个符号模平均。"""
    x = x.astype(np.float64)
    d = x - x.mean((1, 2), keepdims=True)
    d2 = d * d
    F = np.fft.fft2(d) / L
    p = np.stack([np.abs(F[:, i, j]) ** 2 for i, j in K1], 1)
    return np.stack([d2.mean((1, 2)), (d2 * d2).mean((1, 2)), (x ** 4).mean((1, 2)),
                     p.mean(1), (p * p).mean(1)], 1)


def ratios(q):
    """(U_δ, U_1, <φ⁴>) 与构型 jackknife 误差。"""
    n = len(q)
    s = q.sum(0)
    loo = (s - q) / (n - 1)

    def f(m):
        return np.stack([1 - m[..., 1] / (3 * m[..., 0] ** 2),
                         1 - m[..., 4] / (2 * m[..., 3] ** 2), m[..., 2]], -1)

    full = f(s / n)
    jk = f(loo)
    return full, np.sqrt((n - 1) / n * ((jk - full) ** 2).sum(0))


def series(t0_path, chain_path, ref_path):
    x0 = np.load(t0_path)["phi_gen"]
    out = {}
    for key, path, name in (("dm", chain_path, "phi_gen"), ("modelb", ref_path, "cfgs")):
        with h5py.File(path, "r") as f:
            times = np.asarray(f["times"])
            keep = np.where(times <= TMAX)[0]
            vals = [ratios(per_config(x0))]
            vals += [ratios(per_config(np.asarray(f[name][i]))) for i in keep]
        out[key] = (np.concatenate([[0.0], times[keep]]),
                    np.stack([v[0] for v in vals]), np.stack([v[1] for v in vals]))
    return out


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), layout="constrained")
for label, color, marker, t0_path, chain_path, ref_path in ENSEMBLES:
    data = series(t0_path, chain_path, ref_path)
    t, mb, mb_err = data["modelb"]
    _, dm, dm_err = data["dm"]
    for k, ax in enumerate(axes):
        ax.fill_between(t, mb[:, k] - mb_err[:, k], mb[:, k] + mb_err[:, k],
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(t, mb[:, k], color=color, linestyle="-", marker="")
        sel = slice(1, None)
        ax.errorbar(t[sel], dm[sel, k], yerr=dm_err[sel, k], color=color, marker=marker,
                    fillstyle="full", markerfacecolor="white", markeredgecolor=color,
                    linestyle="none", markersize=3.5, markeredgewidth=0.9, elinewidth=0.6,
                    capsize=0, zorder=3)
    late = slice(5, None)
    for k, name in enumerate(("U_delta", "U_1", "phi4")):
        print(f"{label:7s} {name:8s} t>=50: SOSRA {mb[late, k].mean():+.4f}  DM {dm[late, k].mean():+.4f}"
              f"  diff {dm[late, k].mean() - mb[late, k].mean():+.4f}"
              f"  | t=0 {mb[0, k]:+.4f}  t=10 SOSRA {mb[1, k]:+.4f} DM {dm[1, k]:+.4f}")

for ax, label, panel in zip(axes, LABELS, "abc"):
    ax.set(xlim=(-1.5, 101.5), xlabel=r"$t$", ylabel=label)
    ax.set_xticks((0, 50, 100))
    ax.text(0.02, 0.98, f"({panel})", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
axes[0].legend(handles=[Line2D([], [], color=c, marker=m, fillstyle="full", markerfacecolor="white", label=l)
                        for l, c, m, *_ in ENSEMBLES], loc="upper right")
axes[1].legend([Line2D([], [], color="0.35"), Line2D([], [], color="0.35", linestyle="none", marker="o",
                markerfacecolor="white", markersize=3.5)], ["SOSRA", "DM"], loc="upper right")
# Noneq. 的 t=0 高斯态 <φ⁴>=5.3，一步后就落到 1.0，裁掉以看清平衡区
axes[2].set_ylim(0.95, 1.25)
axes[2].text(0.97, 0.98, "Noneq. $t=0$: 5.3", transform=axes[2].transAxes, ha="right", va="top")
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("saved", OUT.with_suffix(".pdf"))
