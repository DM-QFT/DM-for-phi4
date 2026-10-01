"""PRL equilibrium validation and L32 mode-correlation supplement.

The archived NPZ files contain every curve, uncertainty, and snapshot used in
the figure.  This script only changes the publication layout and visual
encoding; it does not rerun either dynamics.
"""

from pathlib import Path
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT.parent / "plots"
sys.path.insert(0, str(PLOTS))

from modelb_palette import (  # noqa: E402
    ALT,
    DM,
    FIELD_CMAP,
    REFERENCE,
    ZERO,
)

plt.style.use(PLOTS / "cjp_prl_nature.mplstyle")
# 2026-09-10 用户要求：(a) 面板与色标 10.5 pt，(b)(c) 略小
FS, FS_BC = 10.5, 9
plt.rcParams.update({"font.size": FS_BC, "axes.labelsize": FS_BC, "xtick.labelsize": FS_BC,
                     "ytick.labelsize": FS_BC, "legend.fontsize": FS_BC})

# 盒 A（Z/m²/M 按 HMC 与非平衡轨迹的有效参数分布取区间）102400 对、batch 128、10000 epoch。
# MODELB_FIG2_TAG 换成别的 run 后缀时只写 results/，不碰定稿 PDF。
TAG = os.environ.get("MODELB_FIG2_TAG", "boxa3_b256")
DATA = ROOT / "results" / "dm_vs_modelb" / (
    f"modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_{TAG}_noskip"
)
# Final vivid single-column layout with the archived initial configurations.
if TAG == "boxa3_b256":
    PAPER = ROOT / "draft" / "figures" / "prl" / "validation_L32.pdf"
    RHO_SUPP = PAPER.parent / "rho_L32.pdf"
else:
    PAPER = DATA.parent / f"validation_L32_{TAG}.pdf"
    RHO_SUPP = DATA.parent / f"rho_L32_{TAG}.pdf"
ENSEMBLES = (
    ("hmc", "Eq.", ALT, "o"),
    ("gaussian", "Noneq.", DM, "s"),
)
RHO_MODES = (0, 1, 2, 3)
KHAT = np.sqrt(8.0) * np.sin(np.pi * np.arange(1, 5) / 32)
S = 201  # Fig. 2 covers the full trajectories to t=2000.
RHO_S = 101  # Supplementary correlations retain lags up to 1000.
SNAP_COLS = ((0, 0), (1, 10), (3, 50), (4, 100), (6, 500), (8, 2000))
# Use the geometric selection of Fig. S2, with interleaved drift markers.
MARKER_STEPS = np.unique(np.geomspace(1, S - 1, 26).astype(int))
DRIFT_MARKER_STEPS = np.unique(np.geomspace(1, S - 1, 52).astype(int))


def panel_label(axis, label):
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=FS,
        fontweight="bold",
        zorder=10,
    )


def truth_and_points(axis, t, mb, mb_sem, dm, dm_sem, color, marker, selected):
    """Reference line/band and learned open markers for one ensemble."""
    axis.fill_between(
        t[1:],
        (mb - mb_sem)[1:],
        (mb + mb_sem)[1:],
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    axis.plot(t[1:], mb[1:], color=color, linestyle=":", marker="")
    axis.errorbar(
        t[selected],
        dm[selected],
        yerr=dm_sem[selected],
        color=color,
        marker=marker,
        fillstyle="full",  # 样式表 markers.fillstyle=none 会让白色面失效，SOSRA 线和误差棒透出
        markerfacecolor="white",
        markeredgecolor=color,
        linestyle="none",
        markersize=3.5,
        markeredgewidth=0.9,
        elinewidth=0.6,
        capsize=0,
        zorder=3,
    )


archives = {
    name: np.load(DATA / f"{name}_chain.npz")
    for name, _, _, _ in ENSEMBLES
}
t = archives["hmc"]["times"][:S]

# APS 双栏全宽，几何按用户 2026-09-10 给的参考图（1350×400 px）量取，全部以英寸手工定位。
# constrained layout 给 24 个等比例小轴留的空隙不均匀，这里不用。
W, H = 7.0, 2.07
fig = plt.figure(figsize=(W, H), layout="none")
C, G, X0, Y_TITLE, G_PAIR = 0.34, 0.035, 0.22, 0.15, 0.08  # 格边长、格距、左缘、标题行、两组间距
Y_BOT, Y_TOP, YLAB = 0.42, 0.15, 0.55  # (b)(c) 下/上边距、y 轴标签留宽


def inch_axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def cell_axes(row, column):
    r = row if row < 2 else row - 1  # 网格行 0,1,3,4 → 0..3
    y = H - Y_TITLE - (r + 1) * C - r * G - (G_PAIR if r >= 2 else 0)
    return inch_axes(X0 + column * (C + G), y, C, C)


A_RIGHT = X0 + 6 * C + 5 * G
Y_GRID_BOTTOM = H - Y_TITLE - 4 * C - 3 * G - G_PAIR
colorbar_axis = inch_axes(X0 + C + G, Y_GRID_BOTTOM - 0.08, 4 * C + 3 * G, 0.05)
PW = (W - 0.14 - A_RIGHT - 2 * YLAB) / 2
ax_b = inch_axes(A_RIGHT + YLAB, Y_BOT, PW, H - Y_BOT - Y_TOP)
ax_c = inch_axes(A_RIGHT + 2 * YLAB + PW, Y_BOT, PW, H - Y_BOT - Y_TOP)

# (a) All twenty-four snapshots share one centered normalization.
snapshot_rows = (
    (0, "hmc", "modelb", "SOSRA", None),
    (1, "hmc", "dm", "DM", "Eq."),
    (3, "gaussian", "modelb", "SOSRA", None),
    (4, "gaussian", "dm", "DM", "Noneq."),
)
# Retain the shared snapshot range. Colorbar extensions mark saturation.
snapshot_vmax = 1.4
snapshot_image = None

for row, ensemble, method, row_label, ensemble_label in snapshot_rows:
    for column, (snapshot_index, physical_time) in enumerate(SNAP_COLS):
        axis = cell_axes(row, column)
        snapshot_image = axis.imshow(
            archives[ensemble][f"snapshots_{method}"][snapshot_index],
            cmap=FIELD_CMAP,
            vmin=-snapshot_vmax,
            vmax=snapshot_vmax,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        if row == 0:
            title = fr"$t={physical_time}$" if column == 0 else fr"${physical_time}$"
            axis.set_title(title, pad=1.0, fontsize=FS)
        if column == 0:
            # 横排后格高约 0.37 in，竖排的 "SOSRA" 超过 8 pt 放不下
            axis.set_ylabel(row_label, fontsize=8, labelpad=1.0)
            if ensemble_label is not None:
                axis.text(
                    0.04,
                    0.05,
                    ensemble_label,
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.5,  # 0.34 in 的格子里放不下更大的 "Noneq."
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 0.5,
                    },
                )
        if row == 0 and column == 0:
            panel_label(axis, "(a)")

colorbar = fig.colorbar(
    snapshot_image,
    cax=colorbar_axis,
    orientation="horizontal",
    ticks=(-snapshot_vmax, 0.0, snapshot_vmax),
    extend="both",
    extendfrac=0.035,
)
colorbar.ax.set_xticklabels(
    (fr"${-snapshot_vmax:.1f}$", "$0$", fr"${snapshot_vmax:.1f}$")
)
colorbar_ticklabels = colorbar.ax.get_xticklabels()
colorbar_ticklabels[0].set_ha("left")
colorbar_ticklabels[-1].set_ha("right")
colorbar.set_label(r"$\phi$", labelpad=-1, fontsize=9)
colorbar.ax.tick_params(
    which="both",
    direction="out",
    top=False,
    bottom=True,
    labelsize=8,
    length=2.0,
    width=0.6,
    pad=1.0,
)
colorbar.outline.set_linewidth(0.7)

# (b) Color identifies the initial ensemble; line/marker identifies dynamics.
for name, label, color, marker in ENSEMBLES:
    values = archives[name]
    truth_and_points(
        ax_b,
        t,
        values["modelb_G"][:S, 0],
        values["modelb_G_sem"][:S, 0],
        values["dm_G"][:S, 0],
        values["dm_G_sem"][:S, 0],
        color,
        marker,
        MARKER_STEPS,
    )

ensemble_handles = [
    Line2D([], [], color=color, marker=marker, fillstyle="full", markerfacecolor="white",
           label=label)
    for _, label, color, marker in ENSEMBLES
]
ax_b.set(
    xscale="log",
    xlim=(9, 2200),
    ylim=(0, 9.5),  # M 配对后 Noneq 从 1 弛豫到 6.5，不到一个量级，对数轴刻度全是次刻度
    xlabel=r"$t$",
    ylabel=r"$G_1(t)$",
)
ax_b.set_xticks((10, 100, 1000))
ax_b.set_yticks((0, 3, 6, 9))
ax_b.yaxis.labelpad = -0.75
ax_b.tick_params(axis="y", which="major", pad=0.8)
ensemble_legend = ax_b.legend(handles=ensemble_handles, loc="lower right", ncol=1)
method_handles_b = [
    Line2D([], [], color="0.35", linestyle=":", marker=""),
    tuple(
        Line2D(
            [], [], color="0.35", linestyle="none", marker=marker,
            markerfacecolor="white", markersize=3.5, markeredgewidth=0.9,
        )
        for marker in ("o", "s")
    ),
]
ax_b.legend(
    method_handles_b,
    ["SOSRA", "DM"],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
    loc="lower right",
    bbox_to_anchor=(1.0, 0.27),
    ncol=1,
    handlelength=1.6,
    handletextpad=0.3,
    columnspacing=0.5,
    borderaxespad=0.15,
)
ax_b.add_artist(ensemble_legend)
ax_b.text(
    0.97,
    0.98,
    r"$L=32,\ \lambda=1$" "\n" r"$\kappa=0.34034$",
    transform=ax_b.transAxes,
    ha="right",
    va="top",
    fontsize=FS_BC,
    linespacing=0.95,
)
panel_label(ax_b, "(b)")

# (c) Same encoding as (b): color = initial ensemble, line = SOSRA, open marker = DM.
# Alternate the two DM marker series on a geometric grid of recorded times.
hmc = archives["hmc"]
drift_slice = slice(1, S)
for i, (name, _, color, marker) in enumerate(ENSEMBLES):
    values = archives[name]
    ax_c.plot(t[drift_slice], values["modelb_mean_abs_drift"][drift_slice],
              color=color, linestyle=":", marker="")
    marker_steps = DRIFT_MARKER_STEPS[i::2]
    ax_c.plot(t[marker_steps], values["dm_mean_abs_drift"][marker_steps],
              color=color, marker=marker, fillstyle="full", markerfacecolor="white",
              markeredgecolor=color, linestyle="none", markersize=3.5, markeredgewidth=0.9,
              zorder=3)
fp32 = np.finfo(np.float32).eps
ax_c.axhline(fp32, color=ZERO, linestyle="--", linewidth=0.8, zorder=0)
ax_c.text(
    0.97,
    fp32,
    r"$\epsilon_{\rm fp32}$",
    transform=ax_c.get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=FS_BC,
    color=ZERO,
)
ax_c.set(
    xscale="log",
    xlim=(9, 2200),
    ylim=(3e-9, 2e-6),
    yscale="log",
    xlabel=r"$t$",
    ylabel=r"$\langle|\Delta M|\rangle$",
)
ax_c.set_xticks((10, 100, 1000))
ax_c.yaxis.labelpad = -0.75
ax_c.tick_params(axis="y", which="major", pad=0.8)
ax_c.legend(
    method_handles_b,
    ["SOSRA", "DM"],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
    loc="lower right",
    handlelength=1.6,
    handletextpad=0.3,
    borderaxespad=0.2,
)
panel_label(ax_c, "(c)")

PAPER.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PAPER, transparent=True)
plt.close(fig)
print(f"saved {PAPER}")

# Supplement: rho_n of the first four diagonal modes on the equilibrium chain.
# Color/marker identifies the mode; method is line versus open marker.
# Restore the paper style after the Fig. 2 panel-specific font settings.
plt.style.use(PLOTS / "cjp_prl_nature.mplstyle")
fig_d, ax_d = plt.subplots()
hmc = archives["hmc"]
mode_styles = list(plt.rcParams["axes.prop_cycle"])
lags = hmc["times"][:RHO_S]
selected_lags = np.unique(np.geomspace(1, RHO_S - 1, 26).astype(int))
for mode in RHO_MODES:
    color = mode_styles[mode]["color"]
    marker = mode_styles[mode]["marker"]
    khat_label = f"{KHAT[mode]:.2f}"
    modelb_rho = hmc["modelb_rho"][:RHO_S, mode]
    modelb_sem = hmc["modelb_rho_sem"][:RHO_S, mode]
    ax_d.fill_between(
        lags,
        modelb_rho - modelb_sem,
        modelb_rho + modelb_sem,
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    ax_d.plot(lags, modelb_rho, color=color, linestyle=":", marker="")
    ax_d.errorbar(
        lags[selected_lags],
        hmc["dm_rho"][:RHO_S, mode][selected_lags],
        yerr=hmc["dm_rho_sem"][:RHO_S, mode][selected_lags],
        color=color,
        marker=marker,
        linestyle="none",
        label=fr"${mode + 1}\;({khat_label})$",
    )

ax_d.axhline(0.0, color=ZERO, linestyle="--", zorder=0)
ax_d.set(
    xscale="log",
    xlim=(9, 1100),
    ylim=(-0.05, 1.02),
    xlabel=r"$\Delta t$",
    ylabel=r"$\rho_n(\Delta t)$",
)
mode_handles, mode_labels = ax_d.get_legend_handles_labels()
# 标题作为无 handle 的首条目，与四行文字列对齐（legend title 会顶到 handle 列）。
mode_legend = ax_d.legend(
    [Line2D([], [], linestyle="none", marker="")] + mode_handles,
    [r"$n\;(\hat k)$"] + mode_labels,
    loc="upper right",
)
method_handles = [
    Line2D([], [], color="0.35", linestyle=":", marker=""),
    Line2D(
        [], [], color="0.35", linestyle="none", marker="o",
    ),
]
ax_d.legend(
    method_handles,
    ["SOSRA", "DM"],
    loc="upper right",
    bbox_to_anchor=(1.0, 0.48),
)
ax_d.add_artist(mode_legend)

fig_d.savefig(RHO_SUPP)
plt.close(fig_d)
print(f"saved {RHO_SUPP}")
