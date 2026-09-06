"""论文 S5(旧样式)：one_step_diagonal.tsv → one_step_diagonal_combined.pdf。"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
plt.style.use(ROOT.parent / "plots" / "cjp.mplstyles")

TABLE = ROOT / "results/multiL/analysis/one_step_diagonal.tsv"
OUT = ROOT / "results/multiL/analysis/one_step_diagonal_combined.pdf"

# 6 个 (Δt, 动力学) 组合各一色；Δt=0.5 实心实线 ms3，Δt=10 空心虚线 ms4
SERIES = (
    (0.5, "modelb", "#000000", "o"),
    (0.5, "single", "#3163CE", "s"),
    (0.5, "mixed", "#CA3300", "^"),
    (10.0, "modelb", "#00981C", "o"),
    (10.0, "single", "#9152BA", "s"),
    (10.0, "mixed", "#FF9B00", "^"),
)
LABELS = {"modelb": "SOSRA", "single": r"single-$L$", "mixed": r"multi-$L$"}

table = np.genfromtxt(TABLE, delimiter="\t", names=True, dtype=None, encoding=None)
# 轴框直接取自旧图的矢量测量值，避免 constrained layout 的微小差异
figure = plt.figure(figsize=(7.0, 2.45), layout="none")
axes = [figure.add_axes(rect) for rect in
        ((0.06431, 0.13629, 0.41443, 0.85558),
         (0.56645, 0.13629, 0.41443, 0.85558))]
for tag, axis, L in zip("ab", axes, (32, 64)):
    axis.axhline(0.0, color=(0.65, 0.65, 0.65), lw=0.5, zorder=0)
    for horizon, model, color, marker in SERIES:
        rows = table[
            (table["L"] == L)
            & np.isclose(table["horizon"], horizon)
            & (table["model"] == model)
        ]
        rows = rows[np.argsort(rows["khat"])]
        filled = horizon == 0.5
        axis.errorbar(
            rows["khat"], rows["rho"], yerr=rows["rho_sem"],
            color=color, marker=marker, linestyle="-" if filled else "--",
            lw=0.9, ms=3.0 if filled else 4.0,
            markerfacecolor=color if filled else "none",
            markeredgecolor=color, markeredgewidth=0.8,
            elinewidth=0.4, capsize=1.0, zorder=2 if filled else 3,
        )
    axis.set(
        xlim=(0.1, 2.9), ylim=(-0.13, 1.05),
        xlabel=r"$\hat k$", ylabel=r"$\rho_n(\Delta t)$",
    )
    axis.text(0.012, 0.90, f"({tag})", transform=axis.transAxes,
              ha="left", va="top", fontsize=9)
    axis.text(0.9654, 0.9601, fr"$L={L}$", transform=axis.transAxes,
              ha="right", va="top", fontsize=9)

for anchor, horizon, title in (
    ((0.4325, 0.8068), 0.5, r"$\Delta t=0.5$"),
    ((0.7225, 0.8068), 10.0, r"$\Delta t=10$"),
):
    filled = horizon == 0.5
    handles = [
        Line2D([], [], color=color, marker=marker,
               linestyle="-" if filled else "--", lw=0.9,
               ms=3.0 if filled else 4.0,
               markerfacecolor=color if filled else "none",
               markeredgecolor=color, markeredgewidth=0.8)
        for h, model, color, marker in SERIES if h == horizon
    ]
    legend = axes[0].legend(
        handles, [LABELS[m] for h, m, *_ in SERIES if h == horizon],
        title=title, loc="upper left", bbox_to_anchor=anchor,
        fontsize=8, title_fontsize=9, handlelength=1.6,
        handletextpad=0.4, labelspacing=0.2, borderaxespad=0.0,
    )
    if horizon == 0.5:
        first_legend = legend
axes[0].add_artist(first_legend)

figure.savefig(OUT, facecolor="white")
print("saved", OUT)
