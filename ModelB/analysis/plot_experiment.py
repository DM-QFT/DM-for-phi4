"""End-matter Fig. 5 and Supplement Figs. for the colloid experiment (10 s kernel, 10000 epochs).
Reads saved statistics and trajectories. No dynamics is rerun."""

from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT.parent / "plots"
sys.path.insert(0, str(PLOTS))
from modelb_palette import ALT, DM, GUIDE, REFERENCE  # noqa: E402

plt.style.use(PLOTS / "cjp_prl_nature.mplstyle")
FS_TICK, FS_LAB = 8, 9  # 与 Fig. 2 (b)(c) 相同
plt.rcParams.update({"font.size": FS_LAB, "axes.labelsize": FS_LAB, "xtick.labelsize": FS_TICK,
                     "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK})
RES = ROOT / "experiment" / "results"
OUT = ROOT / "draft" / "figures" / "prl"
RUNS = (("eq", "undriven"), ("driven", "driven"))
DATA_PT = dict(color=REFERENCE, marker="o", ls="none", ms=2.6, mew=0.7, label="data")
KERNEL = dict(color=DM, ls="-", marker="", label="DM")
LANGEVIN = dict(color=ALT, ls="--", marker="", label="Langevin")
STEP = 10.0  # s
FIG_H, ROW_GAP = 3.20, 0.0  # in; gridspec hspace between the two ensemble rows


def load(run):
    d = {n: np.genfromtxt(RES / f"traj_{run}_dt10s_ep10000" / f"{n}.tsv", names=True)
         for n in ("msd", "mean_xy", "mean_xy_models", "drift_u", "folded_u", "depth_vs_time",
                   "depth_true_blocks", "pstay", "fpt", "fpt_vs_t", "tpt")}
    d["summary"] = dict(np.genfromtxt(RES / f"traj_{run}_dt10s_ep10000" / "summary.tsv", dtype=None, encoding=None))
    with h5py.File(ROOT / "experiment" / "data" / "trajectories.h5", "r") as h:
        d["lam"] = float(h[run]["lam"][:].mean())
    return d


def label(ax, s, x=0.03, y=0.95, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=FS_LAB, **{"ha": "left", "va": "top", **kw})


CORNER = {"tl": (0.03, 0.95, "left", "top"), "tr": (0.97, 0.95, "right", "top"),
          "bl": (0.03, 0.05, "left", "bottom"), "br": (0.97, 0.05, "right", "bottom"), "tc": (0.5, 0.95, "center", "top"),
          "r2": (0.97, 0.80, "right", "top"), "l2": (0.03, 0.80, "left", "top"),  # below a top corner letter
          "tr2": (0.94, 0.95, "right", "top")}  # clear of the inward ticks on the right spine


def letters(ax, corners):
    for j, (a, c) in enumerate(zip(ax.ravel(), corners)):
        x, y, ha, va = CORNER[c]
        label(a, f"({'abcdefgh'[j]})", x=x, y=y, ha=ha, va=va, fontweight="bold")


def grid(ncol):
    """Same panel geometry as Fig. 5: its row gap and panel size, figure width scaled by the column count."""
    fig = plt.figure(figsize=(7.0 * ncol / 4, FIG_H))
    gs = fig.add_gridspec(2, ncol, hspace=ROW_GAP)
    return fig, np.array([[fig.add_subplot(gs[i, j]) for j in range(ncol)] for i in range(2)])


def ensemble_labels(ax_col, corners):
    """Ensemble names inside one panel per row, as in Fig. 5 (d,h)."""
    for a, c, (_, title) in zip(ax_col, corners, RUNS):
        x, y, ha, va = CORNER[c]
        label(a, title, x=x, y=y, ha=ha, va=va)


def three(ax, x, true, gen, lang, x_gen=None):
    ax.plot(x, true, **DATA_PT)
    ax.plot(x if x_gen is None else x_gen, gen, **KERNEL)
    ax.plot(x if x_gen is None else x_gen, lang, **LANGEVIN)


def mean_position(ax, d, comp, run):
    m, mm = d["mean_xy"], d["mean_xy_models"]
    t, y, e = m["t_min"][::10], m[f"{comp}_true"][::10], m[f"err_{comp}"][::10]
    ax.fill_between(t, y - e, y + e, color=REFERENCE, alpha=0.15, lw=0)
    # data means every 2 min (undriven) or 1 min (driven), including the final time; the curve is sampled every 10 s
    last_index = len(t) - 1
    ax.plot(t, y, color=REFERENCE, ls="none", marker="o", ms=DATA_PT["ms"], mew=DATA_PT["mew"], mfc="white",
            markevery=[*range(0, last_index, 12 if run == "eq" else 6), last_index], zorder=3, label="data")
    ax.plot(mm["t_min"], mm[f"{comp}_gen"], **KERNEL)
    ax.plot(mm["t_min"], mm[f"{comp}_lang"], **LANGEVIN)
    ax.axvline(10, color=GUIDE, ls=":", lw=0.7)
    ax.set(xlabel=r"$t$ [min]", ylabel=rf"$\langle {comp}\rangle$ [$\mu$m]")


def pdf_dx_end(ax, d, run, K=None):
    """Displacement after K kernel steps (default: whole rollout) in bins of lambda/4 with one bin centred at 0, so every well peak is sampled alike."""
    with h5py.File(ROOT / "experiment" / "runs" / f"traj_{run}_dt10s_ep10000" / "samples" / "rollout_ep10000_R32.h5", "r") as h:
        K = h["r_gen"].shape[0] if K is None else K
        r_gen, r0, origin = h["r_gen"][K - 1], h["r0"][:], int(h.attrs["origin_index"])
    with h5py.File(ROOT / "experiment" / "runs" / f"langevin_{run}" / "samples" / "rollout.h5", "r") as h:
        r_lang = h["r_gen"][2 * K - 1]  # 5 s storage, 10 s kernel
    with h5py.File(ROOT / "experiment" / "data" / "trajectories.h5", "r") as h:
        pos = h[run]["pos"][:, [origin, origin + 250 * K]]
    dx = {"true": pos[:, 1, 0] - pos[:, 0, 0], "gen": (r_gen - r0)[..., 0].ravel(), "lang": (r_lang - r0)[..., 0].ravel()}
    w = d["lam"] / 4
    lo, hi = np.quantile(dx["gen"], [1e-4, 1 - 1e-4])
    edges = w * (np.arange(np.floor(lo / w), np.ceil(hi / w) + 1) - 0.5)
    c = 0.5 * (edges[1:] + edges[:-1])
    three(ax, c, *(np.histogram(dx[s], edges, density=True)[0] for s in ("true", "gen", "lang")))
    ax.set(xlabel=r"$\Delta x$ [$\mu$m]", ylabel=rf"$p(\Delta x,\,{K * STEP / 60:g}\,$min$)$")


def hop_vs_phase(ax, d):
    u = d["drift_u"]
    three(ax, u["u0"], u["P_hop_remainder"], u["P_hop_gen"], u["P_hop_lang"])
    ax.set(yscale="log", xlabel=r"$u_0$", ylabel=r"$P(|\Delta x|>\lambda/2\,|\,u_0)$", xlim=(-0.5, 0.5))


def v_drift(run, u):
    """Eq. (7) of Dieball et al., D(-dU/dx)/kT + v0, with the potential and phase shift of the Langevin reference."""
    with h5py.File(ROOT / "experiment" / "runs" / f"langevin_{run}" / "samples" / "rollout.h5", "r") as h:
        c, U, D, v0, shift = h["potential_u"][:], h["potential_kT"][:], h.attrs["D"], h.attrs["v0"], h.attrs["phase_shift_cycles"]
        lam = float(np.mean(h5py.File(ROOT / "experiment" / "data" / "trajectories.h5", "r")[run]["lam"][:]))
    spline = CubicSpline(np.r_[c, c[0] + 1], np.r_[U, U[0]], bc_type="periodic", extrapolate="periodic")
    return -D * spline(u - shift, 1) / lam + v0


def drift_vs_phase(ax, d, run):
    """Mean one-step displacement per unit time, the finite-lag form of v_drift."""
    u = d["drift_u"]
    three(ax, u["u0"], u["mean_dx_remainder"] / STEP, u["mean_dx_gen"] / STEP, u["mean_dx_lang"] / STEP)
    ug = np.linspace(-0.5, 0.5, 401)
    ax.plot(ug, v_drift(run, ug), color=GUIDE, lw=0.8, ls="--", marker="", label=r"$v_{\rm drift}$")
    ax.axhline(0, color=GUIDE, lw=0.5, ls=":")
    ax.set(xlabel=r"$u_0$", ylabel=r"$\langle\Delta x\,|\,u_0\rangle/\Delta t$ [$\mu$m/s]", xlim=(-0.5, 0.5))


def fpt_cond_mean(ax, d, run):
    f = d["fpt_vs_t"]
    three(ax, f["t_s"], f["cond_mean_s_true"], f["cond_mean_s_gen"], f["cond_mean_s_lang"])
    if run == "driven":
        ax.axhline(d["lam"] / abs(d["summary"]["v_window"]), color=GUIDE, lw=0.7, label=r"$\lambda/\bar v$")
    ax.set(xscale="log", xlabel=r"cut time $t$ [s]", ylabel=r"$\langle\tau\,|\,\tau<t\rangle$ [s]")


# ---------------------------------------------------------------- Fig. 5 (End Matter)
def endmatter(stack_xy, name, ref_pos=None):
    """stack_xy: first column split into <x> (top) and <y> (bottom) sharing the time axis, at the axis positions ref_pos of the plain layout."""
    fig = plt.figure(figsize=(7.0, FIG_H))
    gs = fig.add_gridspec(4, 4, hspace=ROW_GAP)
    ax = np.empty((2, 4), dtype=object)
    ax_y = [None, None]
    for i in range(2):
        rows = gs[2 * i:2 * i + 2, 0]
        if stack_xy:
            ax[i, 0] = fig.add_subplot(gs[2 * i, 0])
            ax_y[i] = fig.add_subplot(gs[2 * i + 1, 0], sharex=ax[i, 0])
        else:
            ax[i, 0] = fig.add_subplot(rows)
        for j in range(1, 4):
            ax[i, j] = fig.add_subplot(gs[2 * i:2 * i + 2, j])
    for i, (run, title) in enumerate(RUNS):
        d = load(run)
        mean_position(ax[i, 0], d, "x", run)
        if stack_xy:
            mean_position(ax_y[i], d, "y", run)
            ax[i, 0].set_xlabel(None)
            ax[i, 0].tick_params(labelbottom=False)
            for a in (ax[i, 0], ax_y[i]):
                a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, steps=[1, 2, 5, 10], integer=True))
                a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
        pdf_dx_end(ax[i, 1], d, run)
        drift_vs_phase(ax[i, 2], d, run)
        fpt_cond_mean(ax[i, 3], d, run)
    letters(ax, ["tl", "tl", "tr", "tl", "bl", "tl", "tl", "tl"])
    # ensemble labels inside (d) and (h): lower right of (d) and upper left of (h) are empty
    label(ax[0, 3], "undriven", x=0.97, y=0.06, ha="right", va="bottom")
    label(ax[1, 3], "driven", x=0.03, y=0.80)  # below the lambda/v line
    h, l = ax[0, 3].get_legend_handles_labels()
    ax[0, 3].legend(h, l, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax[0, 2].legend(handles=[ax[0, 2].lines[-2]], loc="lower left")
    ax[1, 3].legend(loc="lower right", handles=[ax[1, 3].lines[-1]])
    fig.canvas.draw()
    fig.set_layout_engine("none")
    if stack_xy:
        # take every axis position from the plain layout; column 0 is that panel split in two, no gap
        for i in range(2):
            for j in range(4):
                ax[i, j].set_position(ref_pos[i][j])
            b = ref_pos[i][0]
            mid = 0.5 * (b.y0 + b.y1)
            ax[i, 0].set_position([b.x0, mid, b.width, b.y1 - mid])
            ax_y[i].set_position([b.x0, b.y0, b.width, mid - b.y0])
            # keep integer ticks off the shared edge: pad the range instead of dropping the tick
            lo, hi = ax[i, 0].get_ylim()
            t = [v for v in ax[i, 0].get_yticks() if lo <= v <= hi]
            ax[i, 0].set_ylim(min(lo, t[0] - 0.12 * (hi - lo)), hi)
            ax[i, 0].set_yticks(t)
            lo, hi = ax_y[i].get_ylim()
            t = [v for v in ax_y[i].get_yticks() if lo <= v <= hi]
            ax_y[i].set_ylim(lo, max(hi, t[-1] + 0.12 * (hi - lo)))
            ax_y[i].set_yticks(t)
    if name:
        fig.savefig(OUT / name)
    pos = [[ax[i, j].get_position() for j in range(4)] for i in range(2)]
    plt.close(fig)
    return pos


plain = endmatter(False, "experiment_endmatter.pdf")
for i in range(2):
    print("row", i, " ".join(f"[{b.x0:.3f},{b.y0:.3f},{b.x1:.3f},{b.y1:.3f}]" for b in plain[i]),
          f"panel {plain[i][1].height * FIG_H:.2f} in tall")
print(f"gap between rows {(plain[0][1].y0 - plain[1][1].y1) * FIG_H:.2f} in")

# ---------------------------------------------------------------- Fig. S: transport
fig, ax = grid(3)
for i, (run, title) in enumerate(RUNS):
    d = load(run)
    m = d["msd"]
    for j, comp in enumerate("xy"):
        a = ax[i, j]
        a.errorbar(m["tau_s"], m[f"msd_{comp}_true"], m[f"err_{comp}_true"], elinewidth=0.5, **DATA_PT)
        a.plot(m["tau_s"], m[f"msd_{comp}_gen"], **KERNEL)
        a.plot(m["tau_s"], m[f"msd_{comp}_lang"], **LANGEVIN)
        a.plot(m["tau_s"], 2 * d["summary"]["D1_window"] * m["tau_s"], color=GUIDE, lw=0.7, label=r"$2D\,\Delta t$")
        a.set(xscale="log", yscale="log", xlabel=r"$\Delta t$ [s]", ylabel=rf"MSD$_{comp}$ [$\mu$m$^2$]")
    mean_position(ax[i, 2], d, "y", run)
letters(ax, ["tl", "tl", "bl", "tl", "tl", "bl"])
ensemble_labels(ax[:, 0], ["br", "br"])
h, l = ax[0, 0].get_legend_handles_labels()
ax[0, 1].legend([x for x, s in zip(h, l) if s != r"$2D\,\Delta t$"], [s for s in l if s != r"$2D\,\Delta t$"], loc="lower right")
ax[0, 0].legend(handles=[ax[0, 0].lines[-1]], loc="upper left", bbox_to_anchor=(0.0, 0.88))
fig.savefig(OUT / "experiment_transport.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. S: displacement distribution against time
MINUTES = {"eq": (1, 5, 20, 50), "driven": (1, 5, 10, 20)}
fig, ax = grid(4)
for i, (run, title) in enumerate(RUNS):
    d = load(run)
    for j, minutes in enumerate(MINUTES[run]):
        pdf_dx_end(ax[i, j], d, run, K=int(minutes * 60 / STEP))
letters(ax, ["tl"] * 4 + ["tr"] + ["tl"] * 3)
ensemble_labels(ax[:, 3], ["tr2", "tr2"])
ax[1, 0].legend(loc="upper left")
fig.savefig(OUT / "experiment_pdx_time.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. S: potential with milestones and transition-path times, layout of Dieball et al. Fig. 9
LAM = 4.135  # um, nominal period; compare_traj.py places the milestones with the same value
fig, ax = grid(4)
for i, (run, title) in enumerate(RUNS):
    d = load(run)
    u, t = d["folded_u"], d["tpt"]
    a = ax[i, 0]
    # the paper's coordinate: distance from the barrier top along the drive, x = lam (1/2 - u) mod lam; p per um
    x = LAM * (0.5 - u["u"]) % LAM
    o = np.argsort(x)
    a.plot(x[o], -np.log(u["p_window"][o] / LAM), color=GUIDE, lw=0.8)
    three(a, x[o], *(-np.log(u[f"p_{s}"][o] / LAM) for s in ("true", "gen", "lang")))
    legend = a.legend(a.lines[:2], ("0--10 min", f"10--{60 if run == 'eq' else 30} min"),
                      title="data", title_fontsize=7,
                      loc="upper center", bbox_to_anchor=(0.47, 0.92), fontsize=7,
                      handlelength=0, handletextpad=0.6, borderpad=0.2)
    for text in legend.get_texts():
        text.set_usetex(True)
    for xm, name in ((0.5, "A"), (3.0, "B")):
        a.axvline(xm, color=GUIDE, lw=0.7, ls=":")
        a.text(xm + 0.1, 0.05, name, transform=a.get_xaxis_transform(), ha="left", va="bottom", fontsize=FS_LAB)
    a.set(xlabel=r"$x$ [$\mu$m]", ylabel=r"$-\ln p(x)$", xlim=(0, LAM))
    for j, route in enumerate(("all", "well", "barrier")):
        a = ax[i, 1 + j]
        a.plot(t["t_s"], t[f"p_AB_{route}_true"], **{**DATA_PT, "ms": 3.4, "mew": 0.6, "label": r"data $A\to B$"})
        a.plot(t["t_s"], t[f"p_BA_{route}_true"], **{**DATA_PT, "color": GUIDE, "marker": "s", "ms": 3.6, "mew": 0.8, "label": r"data $B\to A$"})
        for s, st in (("gen", KERNEL), ("lang", LANGEVIN)):
            a.plot(t["t_s"], t[f"p_AB_{route}_{s}"], **st)
            a.plot(t["t_s"], t[f"p_BA_{route}_{s}"], **{**st, "alpha": 0.5, "label": None})
        a.set(xlabel="transition time [s]", ylabel="density [1/s]", xlim=(0, 120))
letters(ax, ["tc", "tr", "tr", "tr"] * 2)
ensemble_labels(ax[:, 3], ["r2", "r2"])
ax[0, 1].legend(loc="upper right", bbox_to_anchor=(1.0, 0.88))
fig.savefig(OUT / "experiment_paths.pdf")
plt.close(fig)
print("saved", OUT)
