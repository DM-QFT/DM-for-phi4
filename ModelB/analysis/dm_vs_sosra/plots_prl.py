"""Model-B 论文图的 PRL 出版版，复用既有轨迹和 run_all 的分析结果。"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.fft
from scipy.ndimage import gaussian_filter

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


def tag(ax, ds, L, loc, params=False, z=None, fontsize=None):
    x, ha = (0.96, "right") if loc == "right" else (0.04, "left")
    text = fr"{DS_LABEL[ds]}, $L={L}$" if ds else fr"$L={L}$"
    if params:
        text += "\n" + r"$\kappa=0.34034$" + "\n" + r"$\lambda=1$"
    if z is not None:
        text += "\n" + fr"$\tau_n\propto\hat k^{{-{fmt(*z)}}}$"
    ax.text(x, 0.95, text, transform=ax.transAxes, ha=ha, va="top",
            fontsize=fontsize)


def mode_legend(axis, fontsize=None, **kw):
    """表头作为无记号的首条目，使 n、k̂ 与下方数字列精确对齐；模条目排最前。"""
    from matplotlib.container import ErrorbarContainer
    handles, labels = axis.get_legend_handles_labels()
    pairs = sorted(zip(handles, labels),
                   key=lambda p: not isinstance(p[0], ErrorbarContainer))
    header = plt.Line2D([], [], ls="none", marker="")
    axis.legend([header] + [p[0] for p in pairs],
                [r"$n$  ($\hat k$)"] + [p[1] for p in pairs], fontsize=fontsize,
                **kw)


def inset_tau_z(axis, res, show_errorbars=True):
    """τ_n–k̂ 内嵌小图：两数据集同画，z 由 τ∝k̂^{-z} 拟合。"""
    # x0=0.13 keeps the rotated tau_n ylabel clear of the parent left spine;
    # y0=0.16 keeps the khat xlabel above the parent bottom spine and ticks;
    # width 0.37 keeps the inset right edge at 0.50, short of the legend.
    ins = axis.inset_axes([0.13, 0.16, 0.37, 0.44])
    khat = res["khat"]
    for ds in ("sosra", "dm"):
        r = res[ds]
        good = np.isfinite(r["taus"])
        ins.errorbar(khat[good], r["taus"][good],
                     yerr=r["tau_err"][good] if show_errorbars else None,
                     color=DS_C[ds], ls="none", marker=DS_M[ds], zorder=3,
                     label=DS_LABEL[ds])
        z, _ = r["z"]
        first = int(np.flatnonzero(good)[0])
        grid = np.geomspace(khat[good][0] * 0.9, khat[good][-1] * 1.1, 50)
        # DM 虚线位于标记和 SOSRA 点线之上，保持重合区的线型可辨。
        ins.plot(grid, r["taus"][first] * (grid / khat[first]) ** (-z),
                 color=DS_C[ds], zorder=2 if ds == "sosra" else 4,
                 ls=":" if ds == "sosra" else "--", marker="")
    method_handles = [
        plt.Line2D([], [], ls=":", marker="",
                   color=REFERENCE, label="SOSRA"),
        plt.Line2D([], [], ls="--", marker="", color=DM, label="DM"),
    ]
    # Smaller text keeps labels readable within the single-column inset.
    ins.legend(handles=method_handles, loc="upper right", fontsize=7)
    ins.set(xscale="log", yscale="log", ylim=(0.6, 450))
    ins.set_xticks([0.3, 0.5, 1.0], ["0.3", "0.5", "1"])
    ins.xaxis.set_minor_formatter(plt.NullFormatter())
    ins.tick_params(labelsize=7)
    ins.set_xlabel(r"$\hat k$", fontsize=8)
    ins.set_ylabel(r"$\tau_n$", fontsize=8)
    ins.text(0.05, 0.16, fr"$z_{{\rm DM}}={fmt(*res['dm']['z'])}$",
             transform=ins.transAxes, ha="left", va="bottom", fontsize=7,
             color=DM)
    ins.text(0.05, 0.04, fr"$z_{{\rm SOSRA}}={fmt(*res['sosra']['z'])}$",
             transform=ins.transAxes, ha="left", va="bottom", fontsize=7,
             color=REFERENCE)


def plot_collapse(L, ds, res, output_dir=PRL, show_errorbars=True,
                  show_modelb_mean=False):
    """Critical collapse with optional error bars and SOSRA mean curves."""
    r = res[ds]
    curves, taus, khat = r["curves"], r["taus"], res["khat"]
    mode_styles = list(plt.rcParams["axes.prop_cycle"])
    figure, axis = plt.subplots()
    for mode in sorted(curves):
        if not np.isfinite(taus[mode]):
            continue
        color = mode_styles[mode]["color"]
        marker = mode_styles[mode]["marker"]

        if show_modelb_mean:
            # SOSRA uses same-mode dotted curves with capped error bars.
            # DM uses open points with capped error bars.
            modelb = res["sosra"]
            x_modelb, rho_modelb, _, sem_modelb = collapse_points(
                modelb["curves"], modelb["taus"], mode
            )
            valid_modelb = (x_modelb > 0) & (rho_modelb > 0)
            order = np.argsort(x_modelb[valid_modelb])
            xb = x_modelb[valid_modelb][order]
            yb = rho_modelb[valid_modelb][order]
            eb = sem_modelb[valid_modelb][order]
            axis.plot(xb, yb, color=color, linestyle=":", marker="", zorder=2)
            if show_errorbars:
                axis.errorbar(
                    xb,
                    yb,
                    yerr=eb,
                    linestyle="none",
                    marker="",
                    color=color,
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
            zorder=3,
            label=fr"${res['mode_n'][mode]}$  ({khat[mode]:.2f})",
        )
    if ds == "dm":
        fit_grid = np.geomspace(2e-3, 20, 400)
        # 两条拟合在 x<3 几乎重合，DM fit 用 DM 蓝色并压在最上，否则同色虚线看不出来
        for fit_ds, linestyle in (("sosra", ":"), ("dm", "--")):
            params = res[fit_ds]["fits"]["joint"]["bsum"]["params"]
            axis.plot(
                fit_grid,
                core.f_bfree(fit_grid, *params),
                color=DS_C[fit_ds],
                linestyle=linestyle,
                marker="",
                zorder=4 if fit_ds == "sosra" else 5,
                label=f"{DS_LABEL[fit_ds]} fit",
            )
    grid = np.geomspace(2e-3, 20, 200)
    axis.plot(
        grid,
        np.exp(-grid),
        color=GUIDE,
        linestyle="-." if ds == "dm" else "--",
        marker="",
        zorder=6,
        label=r"$e^{-x}$" if ds == "dm" else "_nolegend_",
    )
    if ds != "dm":
        axis.text(1.8, 0.10, r"$e^{-x}$", color=GUIDE)
    axis.set(xscale="log", yscale="log", xlim=(2e-3, 20), ylim=(1e-2, 1.6),
             xlabel=r"$x=\Delta t/\tau_n$", ylabel=r"$\rho_n(x)$")
    if ds == "dm":
        # Eight legend entries must fit beside the inset without covering data.
        mode_legend(axis, fontsize=8, loc="lower left",
                    bbox_to_anchor=(0.49, 0.01))
        inset_tau_z(axis, res, show_errorbars=show_errorbars)
        tag(axis, None, L, "right", params=True)
    else:
        mode_legend(axis, loc="lower left", handlelength=1.2)
        tag(axis, ds, L, "right", params=True, z=r["z"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_with_modelb_mean" if show_modelb_mean else ""
    if not show_errorbars:
        suffix += "_no_errorbars"
    figure.savefig(output_dir / f"collapse_{ds}_L{L}{suffix}.pdf")
    plt.close(figure)


def plot_gamma_invx(L, ds, res, output_dir=PRL):
    """Effective decay rate against inverse rescaled lag."""
    figure, axis = plt.subplots()
    mode_styles = list(plt.rcParams["axes.prop_cycle"])
    grid = np.geomspace(0.06, 50.0, 300)
    r = res[ds]
    u, g, err, ids = r["gamma"]
    for mode in np.unique(ids).astype(int):
        m = ids == mode
        axis.errorbar(u[m], g[m], err[m], **mode_styles[mode], ls="none",
                      zorder=3,
                      label=fr"${res['mode_n'][mode]}$  ({res['khat'][mode]:.2f})")
    # Separate legends keep the style's font size without covering the data.
    mode_legend(axis, loc="upper left", bbox_to_anchor=(0, 1.04))
    axis.add_artist(axis.get_legend())
    lam0_tail = r["lam0_tail"][0]
    # Neutral reference curves remain distinct from the colored mode data.
    tail_line, = axis.plot(grid, lam0_tail + 0.5 * grid, color=GUIDE, ls="--",
                           marker="", label=r"$1/(2x)+\lambda_0$")
    # bsum 拟合的 Γ_eff(u) = λ0 + a·b·u/(u+a)，λ0+a·b=r（和值规则）
    lam0, a, b = r["fits"]["joint"]["bsum"]["params"]
    fit_line, = axis.plot(grid, lam0 + a * b * grid / (grid + a),
                          color=REFERENCE, ls="-.", marker="",
                          label="sum-rule fit")
    axis.axhline(1.0, color=ZERO, ls=":", marker="")
    # 短时极限 r=τ_n q_n²/G_n 由静态量定，标成水平参考线
    r_mean = r["r_mean"][0]
    axis.plot([0.45, 50.0], [r_mean, r_mean], color=ZERO, ls=":", marker="")
    axis.text(45.0, r_mean + 0.03, r"$r$", ha="right", va="bottom")
    # 长时极限 λ0（尾部拟合值，与文字框和虚线一致）标在左半段
    axis.plot([0.06, 1.0], [lam0_tail, lam0_tail], color=ZERO, ls=":", marker="")
    axis.text(0.95, lam0_tail + 0.03, r"$\lambda_0$", ha="right", va="bottom")
    axis.set(xscale="log", xlim=(0.06, 50.0), ylim=(0, 2.0),
             yticks=[0.0, 0.5, 1.0, 1.5, 2.0],
             xlabel=r"$1/x=\tau_n/\Delta t$",
             ylabel=r"$\Gamma_{\mathrm{eff}}=-\mathrm{d}\ln\rho_n/\mathrm{d}x$")
    axis.text(0.97, 0.25,
              fr"{DS_LABEL[ds]}, $L={L}$" + "\n"
              + fr"$\lambda_0={lam0_tail:.2f}$" + "\n"
              + fr"$r={r['r_mean'][0]:.2f}$",
              transform=axis.transAxes, ha="right", va="bottom")
    axis.legend(handles=[tail_line, fit_line], loc="lower right")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"gamma_invx_{ds}_L{L}.pdf")
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
            "label": "Model B",
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
            "label": "EM (2000 steps)",
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
            "label": "DPM-v3 (32 NFE)",
        },
    }

    figure, axis = plt.subplots()
    for key in ("modelb", "ncsnpp", "dpmv3"):
        values = archived[key]
        points = axis.errorbar(
            khat,
            values["tau"],
            yerr=values["error"],
            linestyle="none",
            zorder=3,
            label=values["label"] + "\n" + fr"$z={fmt(*values['z'])}$",
        )
        if key == "modelb":
            guide_color = points.lines[0].get_color()

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
        color=guide_color,
        linestyle="--",
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
    axis.legend(loc="upper right")
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
    figure.savefig(output_dir / "reverse_nfe_tau_L64.pdf")
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

    labels = {
        "modelb": "SOSRA",
        "ncsnpp": "EM (2000 steps)",
        "dpmv3": "DPM-v3 (32 NFE)",
    }
    figure, axis = plt.subplots()
    grid = np.geomspace(khat[common][0] * 0.96, khat[common][-1] * 1.04, 50)
    handles = []
    for name in ("modelb", "ncsnpp", "dpmv3"):
        tau = results[name]["tau"][common]
        error = results[name]["error"][common]
        zfit = results[name]["z"]
        z, _ = zfit
        amplitude = np.exp(np.mean(np.log(tau) + z * np.log(khat[common])))
        points = axis.errorbar(
            khat[common], tau, error, ls="none",
            zorder=3, label=labels[name] + "\n" + fr"$z={fmt(*zfit)}$",
        )
        axis.plot(
            grid, amplitude * grid ** (-z),
            color=points.lines[0].get_color(),
            ls=":" if name == "modelb" else "--", marker="",
            zorder=2.5 if name == "modelb" else 2,
        )
        handles.append(points)
    axis.set(xscale="log", yscale="log", xlabel=r"$\hat k$", ylabel=r"$\tau_n$")
    axis.legend(handles, [points.get_label() for points in handles], loc="upper right")
    axis.text(
        0.04, 0.05,
        r"$L=64$" + "\n" + r"$\kappa=0.34034$, $\lambda=1$" + "\n" + mode_text,
        transform=axis.transAxes, ha="left", va="bottom",
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "reverse_nfe_tau_L64.pdf")
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


def plot_conserved_transport_preview(sigma=2):
    """Zoomed fields and the recorded total integrated SOSRA current."""
    from matplotlib.patches import Rectangle

    output_dir = core.ROOT / "results/coarsening"
    with np.load(output_dir / "conserved_transport.npz") as data:
        phi = data["phi"][:, :, [0, -1]].astype(np.float64)
        jx, jy = (data[key][:, :, -1] for key in ("jx", "jy"))
        t0, t1 = data["times"][[0, -1]]

    # Apply the same periodic filter to fields and link currents. It commutes
    # with the discrete divergence and preserves the continuity equation.
    if sigma:
        phi = gaussian_filter(phi, (sigma, sigma, 0), mode="wrap")
        jx, jy = (gaussian_filter(j, sigma, mode="wrap") for j in (jx, jy))
    delta = phi[:, :, 1] - phi[:, :, 0]

    # The box encloses the small domain. Interpolate link components to sites
    # for the displayed streamlines of the integrated current.
    x0, x1, y0, y1 = 30, 48, 16, 36
    jx_sites = (jx + np.roll(jx, 1, axis=0)) / 2
    jy_sites = (jy + np.roll(jy, 1, axis=1)) / 2
    sites_x, sites_y = np.arange(16, 80), np.arange(64)
    u = jx_sites[np.ix_(sites_x, sites_y)].T
    v = jy_sites[np.ix_(sites_x, sites_y)].T
    speed = np.hypot(u, v)
    # Stop curves in weak-current regions before they form small loops.
    weak = speed < 1.5
    grid = np.arange(phi.shape[0]) + 0.5
    text_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5}

    with plt.style.context(PLOTS / "cjp_prl_nature.mplstyle"):
        # Match the draft's 510 pt text width to retain the style's font sizes.
        figure = plt.figure(figsize=(510 / 72.27, 2.3))
        # Keep PDF glyphs clear of the canvas edge without cropping the output.
        figure.get_layout_engine().set(rect=(0.01, 0.02, 0.98, 0.96))
        layout = figure.add_gridspec(1, 6, width_ratios=(1, 1, 0.045, 0.15, 1, 0.045))
        axes = [figure.add_subplot(layout[0, index]) for index in (0, 1, 4)]
        color_axes = [figure.add_subplot(layout[0, index]) for index in (2, 5)]
        for axis in color_axes:
            axis.set_box_aspect(1 / 0.045)
        for index, axis in enumerate(axes):
            field = phi[:, :, index] if index < 2 else delta
            im = axis.imshow(
                field.T, origin="lower", extent=(0, 128, 0, 128),
                vmin=-2.2, vmax=2.2,
                interpolation="nearest", rasterized=True,
            )
            if index == 0:
                field_image = im
            axis.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False,
                edgecolor=REFERENCE, linewidth=plt.rcParams["lines.linewidth"],
            ))
            axis.text(x0 + 1, y1 - 1, r"$\Omega$", ha="left", va="top", bbox=text_box)
            axis.text(0.02, 0.98, f"({chr(97 + index)})", transform=axis.transAxes,
                      ha="left", va="top", bbox=text_box)
            label = fr"$t={int((t0, t1)[index])}$" if index < 2 else r"$\Delta\phi,\ \mathbf{J}$"
            axis.text(0.97, 0.98, label, transform=axis.transAxes,
                      ha="right", va="top", bbox=text_box)
            axis.set(xlim=(16, 80), ylim=(0, 64), xticks=(16, 48, 80),
                     yticks=(0, 32, 64), xlabel=r"$x$")
            if index:
                axis.tick_params(labelleft=False)
        axes[0].set_ylabel(r"$y$")
        axes[1].contour(grid, grid, phi[:, :, 0].T, levels=[0],
                        colors=[REFERENCE], linestyles="dotted")
        axes[2].streamplot(
            sites_x + 0.5, sites_y + 0.5,
            np.ma.masked_where(weak, u), np.ma.masked_where(weak, v),
            color=REFERENCE, density=1.6, linewidth=0.6, arrowsize=0.55,
            minlength=0.18, maxlength=2.5,
        )
        figure.colorbar(field_image, cax=color_axes[0], ticks=(-2, 0, 2))
        figure.colorbar(im, cax=color_axes[1], ticks=(-2, 0, 2))
        for axis, label in zip(color_axes, (r"$\phi$", r"$\Delta\phi$")):
            axis.set_xlabel(label)
            axis.xaxis.set_label_position("top")
        name = "conserved_transport_smoothed" if sigma else "conserved_transport_preview"
        for suffix in ("pdf", "png"):
            figure.savefig(output_dir / f"{name}.{suffix}")
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
