# PRL 正文 Fig.3:(a) DM 在 L=128 外推链的快照 (b) C(r,t) 标度坍缩 + L(t) inset。
# 另存 SM 备用图:S(k,t)/L² 坍缩(Porod k^-3)。
#   python analysis/plot_coarsening_dm.py
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "coarsening" / "prl"
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.style.use(ROOT.parent / "plots" / "cjp.mplstyles")

CHAIN = ROOT / (
    "runs/modelBcond_Lmulti8-16-32-64_k0.85_l1.0_t10_ncsnpp_quench_tanhgauss/"
    "samples/quench_chain_ep10000_L128_N32_t1000.h5"
)
TIMES = (100, 160, 250, 400, 630, 1000)
SNAP_TIMES = (10, 100, 1000)


def sign_stats(cfgs):
    L = cfgs.shape[-1]
    psi = np.sign(cfgs)
    S2 = (np.abs(np.fft.fft2(psi, axes=(1, 2))) ** 2).mean(0) / L**2
    C = np.fft.ifft2(S2).real
    C /= C[0, 0]
    d = np.minimum(np.arange(L), L - np.arange(L))
    r = np.round(np.hypot(*np.meshgrid(d, d))).astype(int)
    Cprof = np.bincount(r.ravel(), C.ravel()) / np.bincount(r.ravel())
    n = np.where(np.arange(L) > L // 2, np.arange(L) - L, np.arange(L))
    m = np.round(np.hypot(*np.meshgrid(n, n))).astype(int)
    Sprof = np.bincount(m.ravel(), S2.ravel()) / np.bincount(m.ravel())
    i = np.argmax(Cprof < 0)
    Lt = (i - 1) + Cprof[i - 1] / (Cprof[i - 1] - Cprof[i])
    return Cprof[: L // 2 + 1], Sprof[: L // 2 + 1], Lt


with h5py.File(CHAIN) as f:
    tt = f["times"][()]
    idx = [np.argmin(np.abs(tt - t)) for t in TIMES]
    Cs, Ss, Ls = zip(*(sign_stats(f["phi_gen"][i]) for i in idx))
    snaps = [f["phi_gen"][np.argmin(np.abs(tt - t)), 0] for t in SNAP_TIMES]
    t_all = tt[tt >= 30]
    L_all = [sign_stats(f["phi_gen"][i])[2] for i in np.flatnonzero(tt >= 30)]
Lbox = snaps[0].shape[-1]

# --- Fig.3:快照行 + 坍缩 ---
fig = plt.figure(figsize=(3.375, 3.85))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.45], hspace=0.14, wspace=0.06)
for c, (snap, t) in enumerate(zip(snaps, SNAP_TIMES)):
    axs = fig.add_subplot(gs[0, c])
    axs.imshow(snap.T, cmap="RdBu_r", vmin=-1.8, vmax=1.8,
               origin="lower", interpolation="nearest")
    axs.set(xticks=[], yticks=[], title=rf"$t={t}$")
    if c == 0:
        axs.text(-0.15, 1.28, "(a)", transform=axs.transAxes, fontsize=8)

ax = fig.add_subplot(gs[1, :])
colors = plt.cm.viridis(np.linspace(0, 0.85, len(TIMES)))
r = np.arange(len(Cs[0]))
for Cprof, Lt, t, c in zip(Cs, Ls, TIMES, colors):
    ax.plot(r / Lt, Cprof, "-", lw=0.9, color=c, label=rf"$t={t}$")
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel=r"$r/L(t)$", ylabel=r"$C(r,t)$", xlim=(0, 3.2), ylim=(-0.25, 1.02))
ax.legend(frameon=False, fontsize=6.5, loc=(0.72, 0.24), handlelength=1.2)
ax.text(0.03, 0.06, r"DM, $L=128$", transform=ax.transAxes, fontsize=7)
ax.text(-0.145, 1.02, "(b)", transform=ax.transAxes, fontsize=8)

axin = ax.inset_axes([0.52, 0.66, 0.44, 0.32])
axin.loglog(t_all, L_all, "o", ms=1.6, color="C2")
tg = np.array([60, 1000.0])
axin.loglog(tg, 1.55 * tg ** (1 / 3), "k--", lw=0.7)
axin.text(430, 8.2, r"$t^{1/3}$", fontsize=6)
axin.set_xlabel(r"$t$", fontsize=6, labelpad=0.5)
axin.set_ylabel(r"$L(t)$", fontsize=6, labelpad=1)
axin.set_yticks([5, 10, 20])
axin.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
axin.yaxis.set_minor_formatter(plt.NullFormatter())
axin.xaxis.set_minor_formatter(plt.NullFormatter())
axin.tick_params(labelsize=5.5, pad=1.5)

fig.savefig(OUTPUT / "fig_dm_collapse_L128.pdf", bbox_inches="tight")
# draft 引用同一张图,重跑自动同步
fig.savefig(ROOT / "draft" / "figures" / "collapse_dm_L128.pdf",
            bbox_inches="tight")

# --- SM 备用:S(k,t)/L² 坍缩 + Porod ---
fig2, ax2 = plt.subplots(figsize=(3.375, 2.55))
k = 2 * np.pi * np.arange(len(Ss[0])) / Lbox
for Sprof, Lt, t, c in zip(Ss, Ls, TIMES, colors):
    ax2.loglog(k[1:] * Lt, Sprof[1:] / Lt**2, "-", lw=0.9, color=c,
               label=rf"$t={t}$")
x = np.array([5, 25.0])
ax2.loglog(x, 1.0 * x**-3.0, "k--", lw=0.7)
ax2.text(9, 2.5e-3, r"$k^{-3}$", fontsize=7)
ax2.set(xlabel=r"$k\,L(t)$", ylabel=r"$S(k,t)/L(t)^2$")
ax2.legend(frameon=False, fontsize=6.5, loc="lower left", handlelength=1.2)
ax2.text(0.63, 0.9, r"DM, $L=128$", transform=ax2.transAxes, fontsize=7)
fig2.tight_layout()
fig2.savefig(OUTPUT / "fig_dm_Sk_collapse_L128.pdf")
fig2.savefig(ROOT / "draft" / "figures" / "sk_collapse_dm_L128.pdf")
print("saved Fig.3 and SM S(k) collapse (+draft sync)")
