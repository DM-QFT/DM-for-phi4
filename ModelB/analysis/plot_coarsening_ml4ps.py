# ML4PS Fig.3:(a) L=128 同初态两算法快照 2×4 网格, SOSRA 上 DM 下, t=0,10,100,1000, 高 2.55 in 与
# cost_accuracy_L64.pdf 同高, 同一行插入时取相同缩放即平齐;(b) C(r,t) 坍缩 + L(t) inset(稿件已不用,
# 保留可再生)。样式按 PRL 的 collapse_dm_L128.pdf 复原。PRL 原图不动。
#   python analysis/plot_coarsening_ml4ps.py
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "draft" / "figures" / "prl"
plt.style.use(ROOT.parent / "plots" / "cjp.mplstyles")
plt.rcParams["figure.constrained_layout.use"] = False

SAMPLES = ROOT / "runs/modelBcond_Lmulti8-16-32-64_k0.85_l1.0_t10_ncsnpp_quench_tanhgauss/samples"
CHAIN = SAMPLES / "quench_chain_ep10000_L128_N32_t1000.h5"
PHI0 = SAMPLES / "quench_t0_L128_N32.npz"
SOSRA = ROOT / "results/coarsening/quench_L128_k0.85.jld2"
# 同初态 SOSRA 参照 (data_generation/run_quench_L128_reference.jl), 布局 cfgs[time, ensemble, x, y]
SOSRA_SAME = ROOT / "results/modelb_reference/quench_tanhgauss_quench_L128_N32.h5"
TIMES = (100, 160, 250, 400, 630, 1000)
SNAP_TIMES = (0, 10, 100, 1000)
DM, REF, GUIDE, ZERO = "#0072B2", "#262626", "#7A7A7A", "#B0B0B0"
# cost_accuracy_L64.pdf 里坐标轴框 [x0, y0, w, h],英寸
AXBOX = (0.5055, 0.3810, 2.8445, 2.1440)


def sign_stats(cfgs):
    L = cfgs.shape[-1]
    psi = np.sign(cfgs)
    S2 = (np.abs(np.fft.fft2(psi, axes=(1, 2))) ** 2).mean(0) / L**2
    C = np.fft.ifft2(S2).real
    C /= C[0, 0]
    d = np.minimum(np.arange(L), L - np.arange(L))
    r = np.round(np.hypot(*np.meshgrid(d, d))).astype(int)
    Cprof = np.bincount(r.ravel(), C.ravel()) / np.bincount(r.ravel())
    i = np.argmax(Cprof < 0)
    Lt = (i - 1) + Cprof[i - 1] / (Cprof[i - 1] - Cprof[i])
    return Cprof[: L // 2 + 1], Lt


def snapshot(path, key, t):
    """轨道 0 在时刻 t 的场; t=0 两条链共用 PHI0。"""
    if t == 0:
        return np.load(PHI0)["phi_gen"][0]
    with h5py.File(path) as f:
        tt = f["times"][()]
        assert tt[np.argmin(np.abs(tt - t))] == t
        return f[key][np.argmin(np.abs(tt - t)), 0]


with h5py.File(CHAIN) as f:
    tt = f["times"][()]
    Cs, Ls = zip(*(sign_stats(f["phi_gen"][np.argmin(np.abs(tt - t))]) for t in TIMES))
    L_dm = np.array([sign_stats(f["phi_gen"][i])[1] for i in range(len(tt))])
with h5py.File(SOSRA) as f:
    t_ref, L_ref = f["times"][()], f["Lt"][()]
L_ref_at = np.exp(np.interp(np.log(tt), np.log(t_ref), np.log(L_ref)))

# --- (a):2×4 快照, SOSRA 上 DM 下, 高 2.55 in 与 cost 图同高 ---
W, H, s, g = 4.20, 2.55, 0.965, 0.04
X0, Y_CB = 0.22, 0.36  # 左缘留行标签, 底部留色标
rows = (("SOSRA", SOSRA_SAME, "cfgs"), ("DM", CHAIN, "phi_gen"))
fig = plt.figure(figsize=(W, H))
for r, (label, path, key) in enumerate(rows):
    y0 = Y_CB + (1 - r) * (s + g)
    for c, t in enumerate(SNAP_TIMES):
        axs = fig.add_axes([(X0 + c * (s + g)) / W, y0 / H, s / W, s / H])
        im = axs.imshow(snapshot(path, key, t).T, cmap="RdBu_r", vmin=-1.7, vmax=1.7,
                        origin="lower", interpolation="nearest")
        axs.set(xticks=[], yticks=[])
        if r == 0:
            axs.set_title(rf"$t={t}$", pad=3)
        if c == 0:
            axs.set_ylabel(label, fontsize=8, labelpad=2)
cax = fig.add_axes([(X0 + s + g) / W, 0.2 / H, (2 * s + g) / W, 0.06 / H])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both", ticks=[-1.7, 0, 1.7])
cax.tick_params(labelsize=6.5, length=0, pad=1.5)
cb.set_label(r"$\phi$", fontsize=8, labelpad=-1)
fig.text(0.002, 0.985, "(a)", fontsize=9, va="top")
fig.savefig(OUT / "snapshots_L128_ml4ps.pdf")

# --- (b):坍缩 + inset,坐标轴框复制 (c) ---
fig = plt.figure(figsize=(3.375, 2.55))
ax = fig.add_axes([AXBOX[0] / 3.375, AXBOX[1] / 2.55, AXBOX[2] / 3.375, AXBOX[3] / 2.55])
colors = plt.cm.viridis(np.linspace(0, 0.85, len(TIMES)))
r = np.arange(len(Cs[0]))
for Cprof, Lt, t, c in zip(Cs, Ls, TIMES, colors):
    ax.plot(r / Lt, Cprof, "-", lw=0.9, color=c, label=rf"$t={t}$", marker="")
ax.axhline(0, color=ZERO, lw=0.8, zorder=0)
ax.set(xlabel=r"$r/L(t)$", ylabel=r"$C(r,t)$", xlim=(0, 3.2), ylim=(-0.25, 1.02))
ax.legend(frameon=False, fontsize=6.5, loc=(0.24, 0.50), handlelength=1.2)
for y, txt in zip((0.14, 0.08, 0.02),
                  (r"DM, $L=128$", r"$\kappa=0.85$, $\lambda=1$", r"trained at $L=8$-$64$")):
    ax.text(0.033, y, txt, transform=ax.transAxes, fontsize=7.5, va="bottom")
ax.text(0.022, 0.98, "(b)", transform=ax.transAxes, fontsize=9, va="top")

axin = ax.inset_axes([0.555, 0.673, 0.415, 0.294])
axin.loglog(t_ref, L_ref, "-", color=REF, lw=0.9, marker="", label="SOSRA")
axin.loglog(tt, L_dm, "o", ms=1.8, mew=0.6, mfc="none", color=DM, label="DM")
tg = np.array([10, 1000.0])
axin.loglog(tg, 1.55 * tg ** (1 / 3), "--", color=GUIDE, lw=0.8, marker="", label=r"$t^{1/3}$")
axin.set(xlim=(10, 1000), ylim=(3, 30), yticks=[5, 10, 20], xticklabels=[])
axin.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
axin.yaxis.set_minor_formatter(plt.NullFormatter())
axin.xaxis.set_minor_formatter(plt.NullFormatter())
axin.set_ylabel(r"$L(t)$", fontsize=7, labelpad=1)
axin.tick_params(labelsize=6, pad=1.5)
axin.legend(frameon=False, fontsize=5.5, loc="upper left", handlelength=1.4,
            borderaxespad=0.3, labelspacing=0.25)

axr = ax.inset_axes([0.555, 0.423, 0.415, 0.147])
axr.semilogx(tt, L_dm / L_ref_at, "o-", ms=1.6, lw=0.6, color=DM, mfc=DM)
axr.axhline(1, ls="--", color=GUIDE, lw=0.8)
axr.set(xlim=(10, 1000), ylim=(0.93, 1.07), yticks=[0.95, 1.05])
axr.xaxis.set_minor_formatter(plt.NullFormatter())
axr.set_xlabel(r"$t$", fontsize=7, labelpad=0.5)
axr.tick_params(labelsize=6, pad=1.5)
axr.text(0.97, 0.08, r"$L_{\mathrm{DM}}/L_{\mathrm{SOSRA}}$", transform=axr.transAxes,
         fontsize=5.8, ha="right", va="bottom")
fig.savefig(OUT / "collapse_L128_ml4ps.pdf")
print("saved", OUT / "snapshots_L128_ml4ps.pdf", OUT / "collapse_L128_ml4ps.pdf")
