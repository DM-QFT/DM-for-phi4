# G_1(t) 对照图，样式同 PRL Fig.2(b)：颜色=初态系综，线+带=SOSRA，空心标记=学到的核。
# 出两张：Δt=0.5 各核迭代 40 步到 t=20（附 Δt=10 zlog 核 t=10、20 的叉），Δt=10 zlog 核到 t=1000。
#   python analysis/plot_g1_kernels.py
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT.parent / "plots"
sys.path.insert(0, str(PLOTS))
from modelb_palette import ALT, DM  # noqa: E402

plt.style.use(PLOTS / "cjp_prl_nature.mplstyle")
RES = ROOT / "results/dm_vs_modelb"
ENSEMBLES = (("hmc", "Eq.", ALT), ("gaussian", "Noneq.", DM))
RUN = "modelBcond_L32_k0.34034_l1.0_t{h}_ncsnpp_corrgauss_{tag}_noskip"
MK = dict(fillstyle="full", markerfacecolor="white", markersize=3.5, markeredgewidth=0.9, linestyle="none")


def draw(ax, kernels, pick, n_line, legend_pos):
    """kernels=(horizon 标签, run tag, 图例, marker)；SOSRA 线画到第 n_line 个时间点。"""
    handles = []
    for i, (h, tag, label, marker) in enumerate(kernels):
        run = RES / RUN.format(h=h, tag=tag)
        if not (run / "hmc_chain.npz").exists():
            continue
        for name, _, color in ENSEMBLES:
            a = np.load(run / f"{name}_chain.npz")
            t, mb, sem = a["times"], a["modelb_G"][:, 0], a["modelb_G_sem"][:, 0]
            if i == 0:  # SOSRA 参照对各 run 相同，只画一次
                s = slice(1, n_line)
                ax.fill_between(t[s], (mb - sem)[s], (mb + sem)[s], color=color, alpha=0.18, linewidth=0)
                ax.plot(t[s], mb[s], color=color, linestyle="-", marker="")
            ax.errorbar(t[pick], a["dm_G"][pick, 0], yerr=a["dm_G_sem"][pick, 0], color=color, marker=marker,
                        markeredgecolor=color, elinewidth=0.6, capsize=0, zorder=3, **MK)
        handles.append(Line2D([], [], color="0.35", marker=marker, label=fr"{label}, $\Delta t={h}$", **MK))
    ax.set(ylim=(0, 9.5), xlabel=r"$t$", ylabel=r"$G_1(t)$")
    ax.set_yticks((0, 3, 6, 9))
    ens = ax.legend(
        handles=[Line2D([], [], color=c, marker="", label=l) for _, l, c in ENSEMBLES]
        + [Line2D([], [], color="0.35", linestyle="-", marker="", label="SOSRA")],
        loc="center", bbox_to_anchor=legend_pos[0], handlelength=1.2, borderaxespad=0.15)
    ax.legend(handles=handles, loc="center", bbox_to_anchor=legend_pos[1], handlelength=1.0,
              handletextpad=0.4, borderaxespad=0.15)
    ax.add_artist(ens)
    ax.text(0.03, 0.97, r"$L=32,\ \lambda=1,\ \kappa=0.34034$", transform=ax.transAxes, ha="left", va="top")


# Δt=0.5 各核到 t=20；3<G_1<5.5 的横带全程是空的，图例放那里
fig, ax = plt.subplots(figsize=(3.375, 2.4), layout="constrained")
draw(ax, (("0.5", "zlog", "zlog", "o"), ("0.5", "zhmc", "zhmc", "^"), ("0.5", "zbox", "zbox", "s")),
     [1, 2, 4, 6, 10, 14, 20, 28, 40], 41, ((0.62, 0.46), (0.2, 0.46)))
for name, _, color in ENSEMBLES:
    a = np.load(RES / RUN.format(h="10", tag="zlog") / f"{name}_chain.npz")
    ax.plot(a["times"][1:3], a["dm_G"][1:3, 0], color=color, marker="x", linestyle="none", markersize=4,
            markeredgewidth=0.9, zorder=3)
ax.legend(handles=ax.get_legend().legend_handles + [
    Line2D([], [], color="0.35", linestyle="none", marker="x", markersize=4, markeredgewidth=0.9,
           label=r"zlog, $\Delta t=10$")], loc="center", bbox_to_anchor=(0.2, 0.46), handlelength=1.0,
    handletextpad=0.4, borderaxespad=0.15)
ax.set(xlim=(0, 20.8))
ax.set_xticks((0, 5, 10, 15, 20))
fig.savefig(RES / "g1_dt0p5_kernels.pdf", transparent=True)
fig.savefig(RES / "g1_dt0p5_kernels.png", dpi=300)

# Δt=10 核到 t=1000，点每 60 一个；t>250 后两条线都在 6.5 附近，下方空着放图例
fig, ax = plt.subplots(figsize=(3.375, 2.4), layout="constrained")
draw(ax, (("10", "zlog", "zlog", "o"), ("10", "boxa2_b128", "boxa2", "s")),
     list(range(1, 101, 6)), 101, ((0.78, 0.45), (0.72, 0.17)))
ax.set(xlim=(-15, 1015))
ax.set_xticks((0, 500, 1000))
fig.savefig(RES / "g1_dt10_zlog.pdf", transparent=True)
fig.savefig(RES / "g1_dt10_zlog.png", dpi=300)
print(f"saved {RES}/g1_dt0p5_kernels.pdf, g1_dt10_zlog.pdf")
