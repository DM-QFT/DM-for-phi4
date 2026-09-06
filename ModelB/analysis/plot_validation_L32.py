"""PRL Fig. 1: zlog DM versus SOSRA integration of Model B at L=32.

The archived NPZ files contain every curve, uncertainty, and snapshot used in
the figure.  This script only changes the publication layout and visual
encoding; it does not rerun either dynamics.
"""

from pathlib import Path
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
    MODE_COLORS,
    MODE_MARKERS,
    REFERENCE,
    ZERO,
)

plt.style.use(PLOTS / "cjp_prl_nature.mplstyle")

DATA = (
    ROOT
    / "results"
    / "dm_vs_modelb"
    / "modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip"
)
# Final vivid single-column layout with the archived initial configurations.
PAPER = (
    ROOT
    / "draft"
    / "figures"
    / "prl"
    / "validation_L32.pdf"
)
RHO_SUPP = PAPER.parent / "rho_L32.pdf"
ENSEMBLES = (
    ("hmc", "Equilibrium", ALT, "o"),
    ("gaussian", "Nonequilibrium", DM, "s"),
)
RHO_MODES = (0, 1, 2, 3)
KHAT = np.sqrt(8.0) * np.sin(np.pi * np.arange(1, 5) / 32)
S = 101  # Plot to t=1000; later samples only increase the rho time origins.
SNAP_COLS = ((0, 0), (1, 10), (3, 50), (4, 100), (6, 500), (7, 1000))


def panel_label(axis, label):
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        zorder=10,
    )


def truth_and_points(axis, t, mb, mb_sem, dm, dm_sem, color, marker, every):
    """Reference line/band and learned open markers for one ensemble."""
    axis.fill_between(
        t,
        mb - mb_sem,
        mb + mb_sem,
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    axis.plot(t, mb, color=color, linestyle="-", marker="")
    selected = slice(2, None, every)
    axis.errorbar(
        t[selected],
        dm[selected],
        yerr=dm_sem[selected],
        color=color,
        marker=marker,
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

fig = plt.figure(figsize=(3.375, 4.05), layout="constrained")
fig.get_layout_engine().set(
    w_pad=0.003,
    h_pad=0.012,
    wspace=0.0,
    hspace=0.035,
)
outer = fig.add_gridspec(
    2,
    1,
    height_ratios=(1.6, 1.0),
    hspace=0.04,
)
snap_grid = outer[0].subgridspec(
    6,
    len(SNAP_COLS),
    height_ratios=(1.0, 1.0, 0.10, 1.0, 1.0, 0.12),
    wspace=0.025,
    hspace=0.025,
)
bottom = outer[1].subgridspec(1, 2, wspace=0.0)
ax_b = fig.add_subplot(bottom[0, 0])
ax_c = fig.add_subplot(bottom[0, 1])
colorbar_axis = fig.add_subplot(snap_grid[5, 1:-1])

# (a) All twenty-four snapshots share one centered normalization.
snapshot_rows = (
    (0, "hmc", "modelb", "SOSRA", None),
    (1, "hmc", "dm", "DM", "Eq."),
    (3, "gaussian", "modelb", "SOSRA", None),
    (4, "gaussian", "dm", "DM", "Noneq."),
)
# Match the relative color occupancy of the ordered-phase snapshots in Fig. 3:
# the median displayed amplitude is about 60% of this range.  Across all 24
# panels, 5.4% of pixels lie outside it; colorbar extensions mark saturation.
snapshot_vmax = 1.4
snapshot_image = None

for row, ensemble, method, row_label, ensemble_label in snapshot_rows:
    for column, (snapshot_index, physical_time) in enumerate(SNAP_COLS):
        axis = fig.add_subplot(snap_grid[row, column])
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
            axis.set_title(title, pad=1.0, fontsize=7.5)
        if column == 0:
            axis.set_ylabel(row_label, fontsize=6.5, labelpad=1.0)
            if ensemble_label is not None:
                axis.text(
                    0.04,
                    0.05,
                    ensemble_label,
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.5,
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
colorbar.set_label(r"$\phi$", labelpad=-1)
colorbar.ax.tick_params(
    which="both",
    direction="out",
    top=False,
    bottom=True,
    labelsize=6,
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
        6,
    )

ensemble_handles = [
    Line2D([], [], color=color, marker=marker, markerfacecolor="white", label=label)
    for _, label, color, marker in ENSEMBLES
]
ax_b.set(
    xlim=(-15, 1015),
    yscale="log",
    xlabel=r"$t$",
    ylabel=r"$G_1(t)$",
)
ax_b.set_xticks((0, 500, 1000))
ax_b.yaxis.labelpad = -0.75
ax_b.tick_params(axis="y", which="major", pad=0.8)
ensemble_legend = ax_b.legend(handles=ensemble_handles, loc="lower right", ncol=1)
method_handles_b = [
    Line2D([], [], color="0.35", linestyle="-", marker=""),
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
    ncol=2,
    fontsize=7.0,
    handlelength=1.4,
    handletextpad=0.4,
    columnspacing=0.9,
    borderaxespad=0.15,
)
ax_b.add_artist(ensemble_legend)
ax_b.text(
    0.97,
    0.82,
    r"$L=32$" "\n" r"$\kappa=0.34034,\ \lambda=1$",
    transform=ax_b.transAxes,
    ha="right",
    va="top",
    fontsize=7.5,
    linespacing=0.95,
)
panel_label(ax_b, "(b)")

# (c) The HMC trajectory cleanly exposes the method-dependent conservation
# floor.  The iid-Gaussian trajectory is already represented in panels (a,b).
hmc = archives["hmc"]
drift_slice = slice(1, S)
ax_c.plot(
    t[drift_slice],
    hmc["modelb_mean_abs_drift"][drift_slice],
    color=REFERENCE,
    linestyle="-",
    marker="",
    label="SOSRA",
)
marker_slice = slice(1, S, 6)
ax_c.plot(
    t[drift_slice],
    hmc["dm_mean_abs_drift"][drift_slice],
    color=DM,
    linewidth=0.8,
    alpha=0.75,
    marker="",
)
ax_c.plot(
    t[marker_slice],
    hmc["dm_mean_abs_drift"][marker_slice],
    color=DM,
    marker="o",
    markerfacecolor="white",
    markeredgecolor=DM,
    linestyle="none",
    label="DM",
)
fp32 = np.finfo(np.float32).eps
ax_c.axhline(fp32, color=ZERO, linestyle="--", linewidth=0.8, zorder=0)
ax_c.text(
    0.97,
    fp32,
    r"$\epsilon_{\rm fp32}$",
    transform=ax_c.get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=6.5,
    color=ZERO,
)
ax_c.set(
    xlim=(-15, 1015),
    ylim=(3e-9, 2e-6),
    yscale="log",
    xlabel=r"$t$",
    ylabel=r"$\langle|\Delta M|\rangle$",
)
ax_c.set_xticks((0, 500, 1000))
ax_c.yaxis.labelpad = -0.75
ax_c.tick_params(axis="y", which="major", pad=0.8)
ax_c.legend(loc="lower right")
panel_label(ax_c, "(c)")

PAPER.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PAPER, transparent=True)
plt.close(fig)
print(f"saved {PAPER}")

# Supplement: rho_n of the first four diagonal modes on the equilibrium chain.
# Color/marker identifies the mode; method is line versus open marker.
fig_d = plt.figure(figsize=(3.375, 2.55), layout="constrained")
ax_d = fig_d.add_subplot()
selected_lags = np.unique(np.geomspace(1, S - 1, 26).astype(int))
for mode in RHO_MODES:
    color = MODE_COLORS[mode]
    marker = MODE_MARKERS[mode]
    khat_label = f"{KHAT[mode]:.2f}".lstrip("0")
    modelb_rho = hmc["modelb_rho"][:S, mode]
    modelb_sem = hmc["modelb_rho_sem"][:S, mode]
    ax_d.fill_between(
        t,
        modelb_rho - modelb_sem,
        modelb_rho + modelb_sem,
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    ax_d.plot(t, modelb_rho, color=color, linestyle="-", marker="")
    ax_d.errorbar(
        t[selected_lags],
        hmc["dm_rho"][:S, mode][selected_lags],
        yerr=hmc["dm_rho_sem"][:S, mode][selected_lags],
        color=color,
        marker=marker,
        markerfacecolor="white",
        markeredgecolor=color,
        linestyle="none",
        markersize=3.5,
        markeredgewidth=0.9,
        elinewidth=0.55,
        capsize=0,
        label=fr"${mode + 1}\;({khat_label})$",
    )

ax_d.axhline(0.0, color=ZERO, linestyle="--", linewidth=0.8, zorder=0)
ax_d.set(
    xscale="log",
    xlim=(9, 1100),
    ylim=(-0.05, 1.02),
    xlabel=r"$\Delta t$",
    ylabel=r"$\rho_n(\Delta t)$",
)
ax_d.yaxis.labelpad = -0.75
ax_d.tick_params(axis="y", which="major", pad=0.8)
mode_handles, mode_labels = ax_d.get_legend_handles_labels()
mode_legend = ax_d.legend(
    mode_handles,
    mode_labels,
    title=r"$n\;(\hat k)$",
    loc="upper right",
    fontsize=7.0,
    title_fontsize=7.0,
    handlelength=0.65,
    handletextpad=0.3,
    labelspacing=0.2,
    borderaxespad=0.15,
)
method_handles = [
    Line2D([], [], color="0.35", linestyle="-", marker=""),
    Line2D(
        [], [], color="0.35", linestyle="none", marker="o",
        markerfacecolor="white", markersize=3.5, markeredgewidth=0.9,
    ),
]
ax_d.legend(
    method_handles,
    ["SOSRA", "DM"],
    loc="upper right",
    bbox_to_anchor=(1.0, 0.60),
    fontsize=7.0,
    handlelength=1.1,
    handletextpad=0.4,
    labelspacing=0.2,
    borderaxespad=0.15,
)
ax_d.add_artist(mode_legend)

fig_d.savefig(RHO_SUPP, transparent=True)
plt.close(fig_d)
print(f"saved {RHO_SUPP}")
