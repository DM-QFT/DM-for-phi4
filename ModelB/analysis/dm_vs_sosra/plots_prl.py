"""L32 图的 PRL 出版版：复用 run_all.analyze(32)，输出到 dm_vs_sosra/prl/。"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.fft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import core
from core import OUT
import run_all
from run_all import collapse_points, fmt

PLOTS = core.ROOT.parent / "plots"
sys.path.insert(0, str(PLOTS))
from modelb_palette import (  # noqa: E402
    ALT,
    DM,
    GUIDE,
    MODE_COLORS,
    MODE_MARKERS,
    REFERENCE,
    ZERO,
)

# 图上明确区分参考积分器 SOSRA 与物理模型 Model B。
DS_LABEL = {"dm": "DM", "sosra": "SOSRA"}

plt.style.use(PLOTS / "cjp_prl_nature.mplstyle")

PRL = OUT / "prl"
MC, MM = MODE_COLORS, MODE_MARKERS
DS_C = {"dm": DM, "sosra": REFERENCE}
DS_M = {"dm": "s", "sosra": "o"}


def tag(ax, ds, L, loc, params=False, z=None, fontsize=8):
    x, ha = (0.96, "right") if loc == "right" else (0.04, "left")
    text = fr"{DS_LABEL[ds]}, $L={L}$" if ds else fr"$L={L}$"
    if params:
        text += "\n" + r"$\kappa=0.34034$, $\lambda=1$"
    if z is not None:
        text += "\n" + fr"$\tau_n\propto\hat k^{{-{fmt(*z)}}}$"
    ax.text(x, 0.95, text, transform=ax.transAxes, ha=ha, va="top",
            fontsize=fontsize)


def mode_legend(axis, fontsize=6.5, **kw):
    """表头作为无记号的首条目，使 n、k̂ 与下方数字列精确对齐；模条目排最前。"""
    from matplotlib.container import ErrorbarContainer
    handles, labels = axis.get_legend_handles_labels()
    pairs = sorted(zip(handles, labels),
                   key=lambda p: not isinstance(p[0], ErrorbarContainer))
    header = plt.Line2D([], [], ls="none", marker="")
    axis.legend([header] + [p[0] for p in pairs],
                [r"$n$  ($\hat k$)"] + [p[1] for p in pairs], fontsize=fontsize,
                labelspacing=0.2, borderaxespad=0.4, **kw)


def inset_tau_z(axis, res, show_errorbars=True):
    """τ_n–k̂ 内嵌小图：两数据集同画，z 由 τ∝k̂^{-z} 拟合。"""
    # x0=0.13 keeps the rotated tau_n ylabel clear of the parent left spine;
    # y0=0.16 keeps the khat xlabel above the parent bottom spine and ticks;
    # width 0.37 keeps the inset right edge at 0.50, short of the legend.
    ins = axis.inset_axes([0.13, 0.16, 0.37, 0.44])
    khat = res["khat"]
    inset_markers = {"dm": "o", "sosra": "s"}
    for ds in ("sosra", "dm"):
        r = res[ds]
        good = np.isfinite(r["taus"])
        ins.errorbar(khat[good], r["taus"][good],
                     yerr=r["tau_err"][good] if show_errorbars else None,
                     color=DS_C[ds], ls="none", marker=inset_markers[ds], ms=2.4,
                     mfc="none",
                     mec=DS_C[ds], mew=0.7, elinewidth=0.6,
                     capsize=1.2, capthick=0.5, zorder=3,
                     label=DS_LABEL[ds])
        z, _ = r["z"]
        first = int(np.flatnonzero(good)[0])
        grid = np.geomspace(khat[good][0] * 0.9, khat[good][-1] * 1.1, 50)
        ins.plot(grid, r["taus"][first] * (grid / khat[first]) ** (-z),
                 color=DS_C[ds],
                 ls="-" if ds == "sosra" else (0, (3, 2)),
                 lw=0.75, marker="", alpha=0.85,
                 dash_capstyle="butt")
    method_handles = [
        plt.Line2D([], [], ls="-", lw=0.75, marker="",
                   color=REFERENCE, label="SOSRA"),
        plt.Line2D([], [], ls=(0, (3, 2)), lw=0.75, marker="",
                   color=DM, dash_capstyle="butt", label="DM"),
    ]
    # handlelength*fontsize = 13 pt ends the (3 pt, 2 pt) dash pattern on a
    # full dash, so the DM handle is not cut mid-dash (butt caps).
    ins.legend(handles=method_handles, loc="upper right", fontsize=5.9,
               frameon=False, handlelength=2.2, handletextpad=0.35,
               labelspacing=0.15, borderaxespad=0.45)
    ins.set(xscale="log", yscale="log", ylim=(0.6, 450))
    ins.set_xticks([0.3, 0.5, 1.0], ["0.3", "0.5", "1"])
    ins.xaxis.set_minor_formatter(plt.NullFormatter())
    ins.tick_params(labelsize=6.4, length=2.5)
    ins.tick_params(which="minor", length=1.5)
    ins.set_xlabel(r"$\hat k$", fontsize=7.5, labelpad=1)
    ins.set_ylabel(r"$\tau_n$", fontsize=7.5, labelpad=1)
    ins.text(0.05, 0.16, fr"$z_{{\rm DM}}={fmt(*res['dm']['z'])}$",
             transform=ins.transAxes, ha="left", va="bottom", fontsize=6.2,
             color=DM)
    ins.text(0.05, 0.04, fr"$z_{{\rm SOSRA}}={fmt(*res['sosra']['z'])}$",
             transform=ins.transAxes, ha="left", va="bottom", fontsize=6.2,
             color=REFERENCE)


def plot_collapse(L, ds, res, output_dir=PRL, show_errorbars=True,
                  show_modelb_mean=False):
    """Critical collapse with optional error bars and SOSRA mean curves."""
    r = res[ds]
    curves, taus, khat = r["curves"], r["taus"], res["khat"]
    legacy_markers = ("o", "^", "D", "s")
    figure, axis = plt.subplots(figsize=(3.375, 2.55), layout="constrained")
    for mode in sorted(curves):
        if not np.isfinite(taus[mode]):
            continue
        color = MC[mode]
        marker = legacy_markers[mode] if ds == "dm" else MM[mode]

        if show_modelb_mean:
            # Direct SOSRA integration is a same-mode solid curve with capped error
            # bars; the DM is shown as open points with capped error bars.
            modelb = res["sosra"]
            x_modelb, rho_modelb, _, sem_modelb = collapse_points(
                modelb["curves"], modelb["taus"], mode
            )
            valid_modelb = (x_modelb > 0) & (rho_modelb > 0)
            order = np.argsort(x_modelb[valid_modelb])
            xb = x_modelb[valid_modelb][order]
            yb = rho_modelb[valid_modelb][order]
            eb = sem_modelb[valid_modelb][order]
            axis.plot(xb, yb, color=color, linestyle="-", marker="", zorder=2)
            if show_errorbars:
                axis.errorbar(
                    xb,
                    yb,
                    yerr=eb,
                    linestyle="none",
                    marker="",
                    color=color,
                    elinewidth=0.45,
                    capsize=1.5,
                    capthick=0.45,
                    alpha=0.70,
                    zorder=2.2,
                )

        x, rho, _, sem = collapse_points(curves, taus, mode)
        valid = (x > 0) & (rho > 0)
        axis.errorbar(
            x[valid],
            rho[valid],
            yerr=sem[valid] if show_errorbars else None,
            linestyle="none",
            marker=marker,
            color=color,
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=2.6 if ds == "dm" else 3.4,
            markeredgewidth=0.7 if ds == "dm" else 0.9,
            elinewidth=0.45,
            capsize=1.5,
            capthick=0.45,
            zorder=3,
            label=fr"${res['mode_n'][mode]}$  ({khat[mode]:.2f})",
        )
    if ds == "dm":
        fit_grid = np.geomspace(2e-3, 20, 400)
        for fit_ds, linestyle in (("sosra", "-"),
                                  ("dm", (0, (4, 2)))):
            params = res[fit_ds]["fits"]["joint"]["bsum"]["params"]
            axis.plot(
                fit_grid,
                core.f_bfree(fit_grid, *params),
                color=REFERENCE,
                linestyle=linestyle,
                linewidth=0.85,
                alpha=0.72,
                marker="",
                dash_capstyle="butt",
                zorder=4,
                label=f"{DS_LABEL[fit_ds]} fit",
            )
    grid = np.geomspace(2e-3, 20, 200)
    axis.plot(
        grid,
        np.exp(-grid),
        color=GUIDE,
        linestyle=(0, (1, 1.6)) if ds == "dm" else "--",
        linewidth=0.9,
        marker="",
        dash_capstyle="round" if ds == "dm" else "butt",
        zorder=1,
        label=r"$e^{-x}$" if ds == "dm" else "_nolegend_",
    )
    if ds != "dm":
        axis.text(1.8, 0.10, r"$e^{-x}$", color=GUIDE, fontsize=7)
    axis.set(xscale="log", yscale="log", xlim=(2e-3, 20), ylim=(1e-2, 1.6),
             xlabel=r"$x=\Delta t/\tau_n$", ylabel=r"$\rho_n(x)$")
    if ds == "dm":
        axis.xaxis.label.set_size(10)
        axis.yaxis.label.set_size(10)
        # handlelength*fontsize = 16.1 pt ends the (4 pt, 2 pt) dash pattern
        # of the DM fit handle on a full dash (butt caps).
        mode_legend(axis, fontsize=7.0, loc="lower left",
                     bbox_to_anchor=(0.515, 0.02), handlelength=2.3,
                     handletextpad=0.45)
        inset_tau_z(axis, res, show_errorbars=show_errorbars)
        tag(axis, None, L, "right", params=True, fontsize=8.5)
    else:
        mode_legend(axis, loc="lower left", handlelength=1.2)
        tag(axis, ds, L, "right", params=True, z=r["z"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_with_modelb_mean" if show_modelb_mean else ""
    if not show_errorbars:
        suffix += "_no_errorbars"
    figure.savefig(output_dir / f"collapse_{ds}_L{L}{suffix}.pdf",
                   transparent=True)
    plt.close(figure)


def plot_gamma_invx(L, ds, res, output_dir=PRL):
    """单面板：ds="dm" 为正文 Fig.3，"sosra" 进附录。"""
    figure, axis = plt.subplots(figsize=(3.375, 2.55), layout="constrained")
    grid = np.geomspace(0.06, 5.0, 200)
    r = res[ds]
    u, g, err, ids = r["gamma"]
    for mode in np.unique(ids).astype(int):
        m = ids == mode
        axis.errorbar(u[m], g[m], err[m], color=MC[mode], ls="none",
                      marker=MM[mode], ms=2.4, mew=0.6, elinewidth=0.5,
                      capsize=0, zorder=3,
                      label=fr"${res['mode_n'][mode]}$  ({res['khat'][mode]:.2f})")
    lam0_tail = r["lam0_tail"][0]
    axis.plot(grid, lam0_tail + 0.5 * grid, color=GUIDE, ls="--", lw=0.9,
              marker="", label=r"$1/(2x)+\lambda_0$")
    # bsum 拟合的 Γ_eff(u) = λ0 + a·b·u/(u+a)，λ0+a·b=r（和值规则）
    lam0, a, b = r["fits"]["joint"]["bsum"]["params"]
    axis.plot(grid, lam0 + a * b * grid / (grid + a), color=REFERENCE, ls="-.",
              lw=1.0, marker="", label="sum-rule fit")
    axis.axhline(1.0, color=ZERO, ls=":", lw=0.8, marker="")
    axis.set(xscale="log", xlim=(0.06, 5.0), ylim=(0, 1.8),
             yticks=[0.0, 0.5, 1.0, 1.5],
             xlabel=r"$1/x=\tau_n/\Delta t$",
             ylabel=r"$\Gamma_{\mathrm{eff}}=-\mathrm{d}\ln\rho_n/\mathrm{d}x$")
    tag(axis, ds, L, "left")
    axis.text(0.04, 0.85,
              fr"$\lambda_0={lam0_tail:.2f}$, $r={r['r_mean'][0]:.2f}$",
              transform=axis.transAxes, ha="left", va="top", fontsize=7)
    mode_legend(axis, fontsize=6, loc="lower right", handlelength=1.4)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"gamma_invx_{ds}_L{L}.pdf", transparent=True)
    plt.close(figure)


def plot_lambda_m(L, res):
    figure, axis = plt.subplots()
    for ds in ("dm", "sosra"):
        rows = res["bins"][ds]
        axis.errorbar([row["m2"] for row in rows], [row["lam"] for row in rows],
                      [row["dlam"] for row in rows], color=DS_C[ds], ls="none",
                      marker=DS_M[ds], ms=3.6, mew=0.9, elinewidth=0.7,
                      label=DS_LABEL[ds])
    axis.set(xlabel=r"$\langle m^2\rangle_{\mathrm{bin}}$",
             ylabel=r"$\lambda(m)$  $[1/\tau_n]$")
    axis.legend(loc="upper left", fontsize=7.5)
    axis.text(0.96, 0.06, fr"$L={L}$", transform=axis.transAxes, ha="right",
              va="bottom", fontsize=8)
    figure.savefig(PRL / f"lambda_m_L{L}.pdf")
    plt.close(figure)


def plot_tau_z(L, res):
    """τ_n–k̂ 正式版（L64 附录）：两数据集同画，拟合线各锚首个可测模。"""
    figure, axis = plt.subplots()
    khat = res["khat"]
    for ds in ("sosra", "dm"):
        r = res[ds]
        good = np.isfinite(r["taus"])
        axis.errorbar(khat[good], r["taus"][good], r["tau_err"][good],
                      color=DS_C[ds], ls="none", marker=DS_M[ds], ms=3.0,
                      mew=0.8, elinewidth=0.7, zorder=3,
                      label=fr"{DS_LABEL[ds]}: $z={fmt(*r['z'])}$")
        (z, _), taus = r["z"], r["taus"]
        first = int(np.flatnonzero(good)[0])
        grid = np.geomspace(khat[good][0] * 0.9, khat[good][-1] * 1.1, 50)
        axis.plot(grid, taus[first] * (grid / khat[first]) ** (-z),
                  color=DS_C[ds], ls="--", lw=0.7, marker="", alpha=0.7)
    axis.set(xscale="log", yscale="log", xlabel=r"$\hat k$", ylabel=r"$\tau_n$")
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=7)
    # 数据对角贯穿，左下角是唯一整块空白
    axis.text(0.04, 0.05, fr"$L={L}$" + "\n" + r"$\kappa=0.34034$, $\lambda=1$",
              transform=axis.transAxes, ha="left", va="bottom", fontsize=8)
    figure.savefig(PRL / f"tau_z_L{L}.pdf")
    plt.close(figure)


def plot_reverse_nfe_tau_archived(output_dir=PRL):
    """Restyle the archived S1 summary without requesting omitted L64 chains.

    The compact local copy intentionally excludes the three L64 mode caches
    and two reverse-sampling HDF5 files.  These point estimates and error bars
    are recovered from the vector coordinates of the archived publication
    PDF; inverse log-axis mapping reproduces its plotted values and fitted z.
    """
    khat = np.sqrt(8.0) * np.sin(np.pi * np.arange(2, 9) / 64)
    archived = {
        "modelb": {
            "tau": np.array([
                213.82680558, 48.79211398, 15.66396159, 6.38267274,
                3.27658932, 1.88950207, 1.13377234,
            ]),
            "error": np.array([
                7.95593761, 0.87186871, 0.58294967, 0.14138996,
                0.05744915, 0.02425856, 0.01263263,
            ]),
            "z": (3.867, 0.023),
            "color": REFERENCE,
            "marker": "o",
            "label": "Model B",
            "filled": True,
        },
        "ncsnpp": {
            "tau": np.array([
                226.82310261, 46.89532328, 14.44416557, 6.68098681,
                3.34891687, 1.91591475, 1.17961010,
            ]),
            "error": np.array([
                7.60936162, 0.95424740, 0.54961233, 0.18131735,
                0.06170943, 0.02390052, 0.01312677,
            ]),
            "z": (3.855, 0.021),
            "color": DM,
            "marker": "s",
            "label": "NCSN++",
            "filled": False,
        },
        "dpmv3": {
            "tau": np.array([
                221.28093873, 46.78675820, 15.19907754, 6.43380883,
                3.29127852, 1.83625850, 1.17187334,
            ]),
            "error": np.array([
                7.29069746, 0.85275708, 0.63059739, 0.16015295,
                0.05759810, 0.02128833, 0.01505658,
            ]),
            "z": (3.866, 0.020),
            "color": ALT,
            "marker": "^",
            "label": "DPM-v3",
            "filled": False,
        },
    }

    figure, axis = plt.subplots(figsize=(3.375, 2.55), layout="constrained")
    for key in ("modelb", "ncsnpp", "dpmv3"):
        values = archived[key]
        axis.errorbar(
            khat,
            values["tau"],
            yerr=values["error"],
            color=values["color"],
            linestyle="none",
            marker=values["marker"],
            markerfacecolor=(values["color"] if values["filled"] else "none"),
            markeredgecolor=values["color"],
            markersize=4.0,
            markeredgewidth=1.0,
            elinewidth=0.7,
            zorder=3,
            label=fr"{values['label']}, $z={fmt(*values['z'])}$",
        )

    guide_z = archived["modelb"]["z"][0]
    guide_amplitude = np.exp(
        np.mean(
            np.log(archived["modelb"]["tau"])
            + guide_z * np.log(khat)
        )
    )
    guide_grid = np.geomspace(khat[0] * 0.92, khat[-1] * 1.08, 100)
    axis.plot(
        guide_grid,
        guide_amplitude * guide_grid ** (-guide_z),
        color=GUIDE,
        linestyle="--",
        linewidth=0.9,
        marker="",
        zorder=1,
    )
    axis.set(
        xscale="log",
        yscale="log",
        xlim=(khat[0] * 0.88, khat[-1] * 1.12),
        ylim=(0.7, 360),
        xlabel=r"$\hat k$",
        ylabel=r"$\tau_n$",
    )
    axis.legend(loc="upper right", fontsize=6.4)
    axis.text(
        0.04,
        0.05,
        r"$L=64$" "\n" r"$\kappa=0.34034,\ \lambda=1$" "\n" r"$n=2,\ldots,8$",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "reverse_nfe_tau_L64.pdf", transparent=True)
    plt.close(figure)


def plot_reverse_nfe_tau(output_dir=PRL):
    """Common-resolution L64 dispersion for the two samplers and Model B."""
    lattice = 64
    nmode = core.NMODE[lattice]
    mode_n, ky, kx, _, khat, _ = core.mode_layout(lattice)
    mode_n, ky, kx, khat = (
        mode_n[:nmode], ky[:nmode], kx[:nmode], khat[:nmode]
    )

    with np.load(core.CACHE / "L64_dm_t0p5.npz") as archive:
        ncsnpp05 = archive["modes"][:, :, :nmode]
    with np.load(core.CACHE / "L64_dm_t10.npz") as archive:
        ncsnpp10 = archive["modes"][:, :, :nmode]
    with np.load(core.CACHE / "L64_sosra.npz") as archive:
        times = archive["times"]
        model_b = archive["modes"][:, :, :nmode]
    model_b05, model_b10 = (
        core.subgrid(times, [model_b], grid)[0]
        for grid in (core.T05, core.T10)
    )

    with np.load(core.initial_path(lattice)) as archive:
        initial = np.asarray(archive["phi_gen"], dtype=np.float32)

    def extract_dpm(path, grid):
        modes = np.empty(
            (len(grid), len(initial), nmode, 4), dtype=np.complex64
        )
        with h5py.File(path, "r") as handle:
            file_times = np.concatenate((
                [0.0], np.asarray(handle["times"], dtype=np.float64)
            ))
            assert np.allclose(file_times, grid)
            for index in range(len(grid)):
                fields = initial if index == 0 else np.asarray(
                    handle["phi_gen"][index - 1], dtype=np.float32
                )
                fields = fields - fields.mean(axis=(1, 2), keepdims=True)
                modes[index] = scipy.fft.fft2(
                    fields, workers=core.FFT_WORKERS
                )[:, ky, kx]
        return modes

    run05 = (
        core.ROOT
        / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t0.5_"
        "ncsnpp_hmc_noskip"
    )
    run10 = (
        core.ROOT
        / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t10_"
        "ncsnpp_hmc_noskip"
    )
    dpmv305 = extract_dpm(
        run05 / "samples/hmc_L64_N512_t50_dpm_solver_v3_nfe32.h5",
        core.T05,
    )
    dpmv310 = extract_dpm(
        run10 / "samples/hmc_L64_N512_t1000_dpm_solver_v3_nfe32.h5",
        core.T10,
    )

    boot = core.boot_indices(len(initial))
    results = {}
    for name, modes05, modes10 in (
        ("modelb", model_b05, model_b10),
        ("ncsnpp", ncsnpp05, ncsnpp10),
        ("dpmv3", dpmv305, dpmv310),
    ):
        stats05 = run_all.rho_stats(
            core.lag_arrays(modes05, lattice * lattice), boot
        )
        stats10 = run_all.rho_stats(
            core.lag_arrays(modes10, lattice * lattice), boot
        )
        _, tau, tau_boot, _, _ = run_all.dataset_curves(
            stats05, stats10, khat
        )
        error = np.full(nmode, np.nan)
        measurable = np.isfinite(tau)
        error[measurable] = core.boot_sem(tau_boot[:, measurable])
        results[name] = {
            "tau": tau,
            "tau_boot": tau_boot,
            "error": error,
        }

    common = np.logical_and.reduce([
        np.isfinite(result["tau"]) for result in results.values()
    ])
    assert common.sum() >= 2
    for result in results.values():
        masked = np.where(common, result["tau"], np.nan)
        z = core.fit_z(khat, masked)
        z_boot = np.array([
            core.fit_z(khat, np.where(common, sample, np.nan))
            for sample in result["tau_boot"]
        ])
        result["z"] = (z, core.boot_sem(z_boot[np.isfinite(z_boot)]))

    common_n = mode_n[common]
    if np.all(np.diff(common_n) == 1):
        mode_text = fr"$n={common_n[0]},\ldots,{common_n[-1]}$"
    else:
        mode_text = "$n=" + ",".join(map(str, common_n)) + "$"

    colors = {"modelb": "#D55E00", "ncsnpp": "#0072B2", "dpmv3": "#009E73"}
    markers = {"modelb": "s", "ncsnpp": "o", "dpmv3": "D"}
    labels = {
        "modelb": "SOSRA",
        "ncsnpp": "NCSN++ (2000 steps)",
        "dpmv3": "DPM-Solver-v3 (32 NFE)",
    }
    figure, axis = plt.subplots(figsize=(3.375, 2.55))
    grid = np.geomspace(khat[common][0] * 0.96, khat[common][-1] * 1.04, 50)
    for name in ("modelb", "ncsnpp", "dpmv3"):
        tau = results[name]["tau"][common]
        error = results[name]["error"][common]
        zfit = results[name]["z"]
        z, _ = zfit
        amplitude = np.exp(np.mean(np.log(tau) + z * np.log(khat[common])))
        axis.plot(
            grid, amplitude * grid ** (-z), color=colors[name],
            ls="--", lw=0.7, marker="", alpha=0.8,
        )
        axis.errorbar(
            khat[common], tau, error, color=colors[name], ls="none",
            marker=markers[name], ms=3.0, mew=0.8, elinewidth=0.7,
            zorder=3, label=fr"{labels[name]}, $z={fmt(*zfit)}$",
        )
    axis.set(xscale="log", yscale="log", xlabel=r"$\hat k$", ylabel=r"$\tau_n$")
    axis.legend(loc="upper right", fontsize=7.0)
    axis.text(
        0.04, 0.05,
        r"$L=64$" + "\n" + r"$\kappa=0.34034$, $\lambda=1$" + "\n" + mode_text,
        transform=axis.transAxes, ha="left", va="bottom", fontsize=8,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "reverse_nfe_tau_L64.pdf", transparent=True)
    plt.close(figure)
    for name in ("modelb", "ncsnpp", "dpmv3"):
        result = results[name]
        print(
            name, "n", common_n, "tau", result["tau"][common],
            "error", result["error"][common], "z", result["z"],
        )


def plot_health(L, caches, hmc_m, health):
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.4))
    for label, ax in zip("abcd", axes.flat):
        ax.text(0.03, 0.94, f"({label})", transform=ax.transAxes, ha="left",
                va="top", fontsize=9)

    ax = axes[0, 0]
    for name, style in (("dm_t10", "-"), ("sosra", "--")):
        times, modes, _, _ = caches[name]
        power = (modes * modes.conj()).real.astype(np.float64).mean(axis=(1, 3))
        power /= float(L * L)
        for mode in range(power.shape[1]):
            ax.plot(times[1:], power[1:, mode] / power[1:, mode].mean() - 1,
                    color=MC[mode], ls=style, lw=0.7, marker="")
    ax.axhline(0, color="0.6", lw=0.5, marker="")
    ax.set(xlabel=r"$t$", ylabel=r"$G_n(t)/\bar G_n-1$", ylim=(-0.5, 0.5))
    ax.text(0.97, 0.94, "solid DM, dashed Model B", transform=ax.transAxes,
            ha="right", va="top", fontsize=7)

    ax = axes[0, 1]
    for name, ds, style in (("dm_t10", "dm", "-"), ("sosra", "sosra", "--")):
        times, _, _, s = caches[name]
        ax.plot(times[1:], s[1:].mean(axis=1), color=DS_C[ds], ls=style, lw=0.9,
                marker="", label=DS_LABEL[ds])
    ax.set(xlabel=r"$t$", ylabel=r"$s(t)$")
    ax.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.06, 1.0))

    ax = axes[1, 0]
    for name, ds, style in (("dm_t10", "dm", "-"), ("sosra", "sosra", "--")):
        times, _, m, _ = caches[name]
        drift = np.abs(m - m[0]).max(axis=1)
        ax.plot(times[1:], drift[1:], color=DS_C[ds], ls=style, lw=0.9,
                marker="", label=DS_LABEL[ds])
    ax.set(xlabel=r"$t$", ylabel=r"$\max_i|m_i(t)-m_i(0)|$", yscale="log")
    ax.legend(loc="lower right", fontsize=7.5)

    ax = axes[1, 1]
    m0 = caches["dm_t10"][2][0]
    bins = np.linspace(-4 * m0.std(), 4 * m0.std(), 41)
    ax.hist(hmc_m, bins=bins, density=True, histtype="step", color="0.35",
            lw=0.9, label=f"HMC ($N={len(hmc_m)}$)")
    ax.hist(m0, bins=bins, density=True, histtype="step", color=DS_C["dm"],
            lw=0.9, label=f"DM ($N={len(m0)}$)")
    ax.set(xlabel=r"$m=\phi_0/V$", ylabel=r"$P(m)$")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.text(0.03, 0.76, fr"KS $p={health['ks_p']:.2f}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5)

    figure.savefig(PRL / f"health_L{L}.pdf")
    plt.close(figure)


def main():
    PRL.mkdir(parents=True, exist_ok=True)
    res = run_all.analyze(32)
    caches = {}
    for name in ("sosra", "dm_t10"):
        times, modes, m, s = core.load_cache(32, name)
        caches[name] = (times, modes[:, :, :core.NMODE[32]], m, s)
    hmc_m = run_all.load_hmc_m(32)
    for ds in ("dm", "sosra"):
        plot_collapse(32, ds, res)
        plot_gamma_invx(32, ds, res)
    plot_lambda_m(32, res)
    plot_health(32, caches, hmc_m, res["health"])
    plot_tau_z(64, run_all.analyze(64))
    plot_reverse_nfe_tau()
    print(f"saved {PRL}")


if __name__ == "__main__":
    main()
