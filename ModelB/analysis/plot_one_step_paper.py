"""Fig. S7. one_step_diagonal.tsv → one_step_diagonal_combined.pdf。"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT.parent / "plots"
sys.path.insert(0, str(PLOTS))
from modelb_palette import ALT, DM, GUIDE, REFERENCE, ZERO  # noqa: E402

plt.style.use(PLOTS / "cjp_prl_nature.mplstyle")

TABLE = ROOT / "results/multiL/analysis/one_step_diagonal.tsv"
OUT = ROOT / "results/multiL/analysis/one_step_diagonal_combined.pdf"

# marker 按样式表 prop_cycle 的配对：vermillion+o、blue+s
METHOD = {
    "modelb": (REFERENCE, "", "SOSRA"),
    "single": (DM, "s", r"single-$L$"),
    "mixed": (ALT, "o", r"multi-$L$"),
}
STEP = {0.5: "-", 10.0: "--"}

table = np.genfromtxt(TABLE, delimiter="\t", names=True, dtype=None, encoding=None)
# (6.61, 2.52) + wspace 0.091 使每个面板的轴框 = 单栏图的 2.92 x 2.11 in，面板间距 0.3 in；tex 里按原尺寸插入
figure, axes = plt.subplots(1, 2, figsize=(6.61, 2.52), sharey=True, gridspec_kw={"wspace": 0.091})
for tag, axis, L in zip("ab", axes, (32, 64)):
    axis.axhline(0.0, color=ZERO, linestyle=":", zorder=0)
    for horizon, linestyle in STEP.items():
        for model, (color, marker, _) in METHOD.items():
            rows = table[(table["L"] == L) & np.isclose(table["horizon"], horizon) & (table["model"] == model)]
            rows = rows[np.argsort(rows["khat"])]
            if marker:
                axis.errorbar(rows["khat"], rows["rho"], yerr=rows["rho_sem"],
                              color=color, marker=marker, linestyle=linestyle)
            else:  # SOSRA 一律点线，两条曲线靠衰减位置区分
                axis.plot(rows["khat"], rows["rho"], color=color, linestyle=":", marker="")
    axis.set(xlim=(0.1, 2.9), ylim=(-0.13, 1.18), xlabel=r"$\hat k$")
    axis.text(0.02, 0.97, f"({tag})", transform=axis.transAxes, ha="left", va="top")
    axis.text(0.98, 0.97, fr"$L={L}$", transform=axis.transAxes, ha="right", va="top")
axes[0].set_ylabel(r"$\rho_n(\delta t)$")

# SOSRA 图例用黑色点线。核的 δt 线型图例用灰色。
handles = [Line2D([], [], color=c, marker=m, linestyle="none" if m else ":", label=l) for c, m, l in METHOD.values()]
handles += [Line2D([], [], color=GUIDE, linestyle=ls, marker="", label=fr"$\delta t={h:g}$")
            for h, ls in STEP.items()]
axes[0].legend(handles=handles, ncol=2, loc="center right")
figure.savefig(OUT)
print("saved", OUT)
