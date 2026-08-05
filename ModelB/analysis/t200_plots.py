"""Figures for the ep10000 chains through t=200; reads only t200_data.py output."""

from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = Path(os.environ.get(
    "MODELB_T200_RUN",
    ROOT / "runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_corrgauss_uniformM_noskip",
)).expanduser().resolve()
DATA = RUN / "t200_ep10000_dm_vs_modelb"
CHAIN_OUTPUT = RUN / "t200_ep10000_analysis"
LABELS = {"hmc": "HMC initial", "gaussian": "Gaussian initial"}
STYLES = {"modelb": ("Model B", "#1f77b4", "o"), "dm": ("DM", "#d62728", "s")}
CHAIN_STYLES = {"hmc": ("#1f77b4", "o"), "gaussian": ("#d62728", "s")}
SELECTED = (1, 2, 4, 8, 16)


def save(figure, directory, name):
    figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def grid_axes():
    return plt.subplots(4, 4, figsize=(13, 11), sharex=True,
                        constrained_layout=True)


def decorate(axis, mode, khat, times, ylabel, xlabel):
    axis.grid(alpha=0.18)
    axis.set_xticks(np.linspace(0, times[-1], 5))
    axis.set_title(fr"$n={mode + 1}$, $\hat k={khat[mode]:.3f}$")
    axis.set_ylabel(ylabel)
    if mode // 4 == 3:
        axis.set_xlabel(xlabel)


def plot_lag_average(key, data):
    times, khat = data["times"], data["khat"]
    figure, axes = grid_axes()
    for mode, axis in enumerate(axes.flat):
        lower, upper = np.inf, -np.inf
        for chain, (label, color, marker) in STYLES.items():
            rho = data[f"{chain}_rho"][1:, mode]
            error = data[f"{chain}_rho_sem"][1:, mode]
            axis.errorbar(times[1:], rho, yerr=error, color=color, marker=marker,
                          ms=2.8, lw=1.15, elinewidth=0.7, capsize=1.8, label=label)
            lower, upper = min(lower, (rho - error).min()), max(upper, (rho + error).max())
        span = max(upper - lower, 0.02)
        axis.set_ylim(min(lower - 0.1 * span, -0.005), max(upper + 0.1 * span, 0.005))
        axis.set_xlim(times[1] / 2, times[-1] + times[1] / 2)
        axis.axhline(0, color="black", lw=0.5)
        decorate(axis, mode, khat, times, r"$\rho_n(\Delta t)$", r"$\Delta t$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2,
                  title=LABELS[key] + ": lag average")
    save(figure, DATA, f"{key}_dm_vs_modelb_diagonal_lag_average")


def plot_power(key, data):
    times, khat = data["times"], data["khat"]
    figure, axes = grid_axes()
    for mode, axis in enumerate(axes.flat):
        for chain, (label, color, marker) in STYLES.items():
            axis.errorbar(times, data[f"{chain}_G"][:, mode],
                          yerr=data[f"{chain}_G_sem"][:, mode], color=color,
                          marker=marker, ms=2.8, lw=1.15, elinewidth=0.7,
                          capsize=1.8, label=label)
        axis.set_xlim(-3, 203)
        decorate(axis, mode, khat, times, r"$G_n(t)$", r"$t$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2,
                  title=LABELS[key] + ": equal-time diagonal power")
    save(figure, DATA, f"{key}_dm_vs_modelb_diagonal_power")


def plot_observables(key, data):
    times = data["times"]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    quantities = (
        ("phi2", r"$\langle V^{-1}\sum_x\phi_x^2\rangle$"),
        ("phi4", r"$\langle V^{-1}\sum_x\phi_x^4\rangle$"),
        ("variance", r"$\langle V^{-1}\sum_x(\phi_x-M)^2\rangle$"),
    )
    for axis, (quantity, ylabel) in zip(axes.flat[:3], quantities):
        for chain, (label, color, marker) in STYLES.items():
            axis.errorbar(times, data[f"{chain}_{quantity}"],
                          yerr=data[f"{chain}_{quantity}_sem"], color=color,
                          marker=marker, ms=3, lw=1.2, elinewidth=0.75,
                          capsize=2, label=label)
        axis.set(xlabel=r"$t$", ylabel=ylabel, xlim=(-3, 203))
        axis.set_xticks((0, 50, 100, 150, 200))
        axis.grid(alpha=0.2)

    axis = axes.flat[3]
    for chain, (label, color, marker) in STYLES.items():
        axis.errorbar(times[1:], data[f"{chain}_mean_abs_drift"][1:],
                      yerr=data[f"{chain}_mean_abs_drift_sem"][1:], color=color,
                      marker=marker, ms=3, lw=1.2, elinewidth=0.75, capsize=2,
                      label=label)
    axis.set_yscale("log")
    axis.set(xlabel=r"$t$", ylabel=r"$\langle|M(t)-M(0)|\rangle$", xlim=(-3, 203))
    axis.set_xticks((0, 50, 100, 150, 200))
    axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2,
                  title=LABELS[key])
    save(figure, DATA, f"{key}_dm_vs_modelb_observables")


def plot_snapshots(key, data):
    snapshot_times = data["snapshot_times"]
    fields = np.concatenate([data[f"snapshots_{c}"] for c in STYLES], axis=0)
    limit = float(np.abs(fields).max())
    norm = Normalize(-limit, limit)
    figure, axes = plt.subplots(2, len(snapshot_times), figsize=(16, 5.1),
                                constrained_layout=True)
    image = None
    for row, (chain, (label, _, _)) in enumerate(STYLES.items()):
        for column, time in enumerate(snapshot_times):
            image = axes[row, column].imshow(
                data[f"snapshots_{chain}"][column], origin="lower",
                cmap="RdBu_r", norm=norm, interpolation="nearest", rasterized=True,
            )
            axes[row, column].set_xticks(())
            axes[row, column].set_yticks(())
            if row == 0:
                axes[row, column].set_title(fr"$t={time}$")
        axes[row, 0].set_ylabel(label)
    figure.colorbar(image, ax=axes, shrink=0.92, pad=0.008,
                    label=fr"$\phi(x,y)$, shared range $[-{limit:.2f},{limit:.2f}]$")
    figure.suptitle(LABELS[key] + ", trajectory index 0")
    save(figure, DATA, f"{key}_dm_vs_modelb_spatial_index0000")


def plot_chain_lag_average(results):
    khat = results["hmc"]["khat"]
    figure, axes = grid_axes()
    for mode, axis in enumerate(axes.flat):
        lower, upper = np.inf, -np.inf
        for key, (color, marker) in CHAIN_STYLES.items():
            times = results[key]["times"]
            rho = results[key]["dm_rho"][1:, mode]
            error = results[key]["dm_rho_sem"][1:, mode]
            axis.errorbar(times[1:], rho, yerr=error, color=color, marker=marker,
                          ms=2.8, lw=1.15, elinewidth=0.7, capsize=1.8,
                          label=LABELS[key])
            lower, upper = min(lower, (rho - error).min()), max(upper, (rho + error).max())
        span = max(upper - lower, 0.02)
        axis.set_ylim(min(lower - 0.1 * span, -0.005), max(upper + 0.1 * span, 0.005))
        axis.set_xlim(5, 205)
        axis.axhline(0, color="black", lw=0.5)
        decorate(axis, mode, khat, times, r"$\rho_n(\Delta t)$", r"$\Delta t$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2)
    save(figure, CHAIN_OUTPUT, "hmc_gaussian_diagonal_lag_average_all_modes")


def plot_chain_fixed_origin(results):
    khat = results["hmc"]["khat"]
    figure, axes = plt.subplots(2, len(SELECTED), figsize=(16, 6.6), sharex=True,
                                constrained_layout=True)
    for row, (key, (color, marker)) in enumerate(CHAIN_STYLES.items()):
        times = results[key]["times"]
        for column, n in enumerate(SELECTED):
            axis = axes[row, column]
            axis.errorbar(times, results[key]["dm_fixed_rho"][:, n - 1],
                          yerr=results[key]["dm_fixed_rho_sem"][:, n - 1],
                          color=color, marker=marker, ms=2.8, lw=1.15,
                          elinewidth=0.7, capsize=1.8)
            axis.axhline(0, color="black", lw=0.5)
            axis.grid(alpha=0.18)
            axis.set(xlim=(-3, 203), ylim=(-0.13, 1.04))
            axis.set_xticks((0, 50, 100, 150, 200))
            axis.set_title(fr"$n={n}$, $\hat k={khat[n - 1]:.3f}$")
            if row == 1:
                axis.set_xlabel(r"$t$")
        axes[row, 0].set_ylabel(LABELS[key] + r" $\rho_n(t,0)$")
    save(figure, CHAIN_OUTPUT, "hmc_gaussian_diagonal_fixed_origin_selected_modes")


def plot_chain_heatmaps(results):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    image = None
    for axis, (key, data) in zip(axes, results.items()):
        image = axis.pcolormesh(
            data["times"], data["mode_n"], data["dm_fixed_rho"].T,
            shading="nearest", cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=1), rasterized=True,
        )
        axis.set(xlabel=r"$t$", ylabel=r"diagonal mode $n$", title=LABELS[key])
        axis.set_yticks((1, 2, 4, 8, 12, 16))
    figure.colorbar(image, ax=axes, label=r"$\rho_n(t,0)$")
    save(figure, CHAIN_OUTPUT, "hmc_gaussian_diagonal_fixed_origin_heatmaps")


def main():
    CHAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in LABELS:
        with np.load(DATA / f"{key}_t200.npz") as archive:
            results[key] = {name: archive[name] for name in archive.files}
        plot_lag_average(key, results[key])
        plot_power(key, results[key])
        plot_observables(key, results[key])
        plot_snapshots(key, results[key])
    plot_chain_lag_average(results)
    plot_chain_fixed_origin(results)
    plot_chain_heatmaps(results)
    print(f"saved {DATA} and {CHAIN_OUTPUT}")


if __name__ == "__main__":
    main()
