"""Plot time-origin averaged correlations for the mixed-L extrapolations."""

from pathlib import Path
import csv

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagonal_modes import (
    diagonal_fft,
    mode_layout,
    paired_jackknife_sem,
    power,
    statistics,
)


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/multiL_ep10000_extrapolation"
OLD = ROOT / "runs/multiL_ep10000_reverse_analysis"
OUTPUT = RUN / "analysis"
N = 512
LS = (32, 64)
COLORS = {
    "modelb_t10": "black", "modelb_t0p5": "black",
    "t10": "#1f77b4", "t0p5": "#d62728",
}
LABELS = {
    "modelb_t10": "Model B", "modelb_t0p5": "Model B",
    "t10": r"mixed-L, $\delta t=10$",
    "t0p5": r"mixed-L, $\delta t=0.5$",
}


def number_tag(value):
    return f"{value:g}".replace(".", "p")


def initial_path(lattice):
    if lattice == 32:
        return RUN / "hmc_L32_N512_t0.npz"
    return (
        ROOT
        / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip"
        / "samples/hmc_L64_N512_seed2026080201/hmc_ep10000_L64_t0_N512.npz"
    )


def dm_t10_path(lattice, time):
    if time == 0:
        return initial_path(lattice)
    if lattice == 32:
        seed = 2026081100 + time // 10
        directory = RUN / "t10/L32"
    elif time <= 200:
        seed = 2026080800 + time // 10
        directory = OLD / "samples/t10/mixed_L64_chain_t200"
    else:
        seed = 2026081201 + (time - 210) // 10
        directory = RUN / "t10/L64_t210_t1000"
    return directory / (
        f"chain_ep10000_t{time}_via_iterated_t10dm512_steps2000_log_seed{seed}.npz"
    )


def dm_t0p5_path(lattice, index):
    if index == 0:
        return initial_path(lattice)
    time = 0.5 * index
    seed = (2026082100 if lattice == 32 else 2026082200) + index
    return RUN / f"t0p5/L{lattice}" / (
        f"chain_ep10000_t{number_tag(time)}_via_iterated_t0p5dm512_"
        f"steps2000_log_seed{seed}.npz"
    )


def modelb_path(lattice, time):
    if time == 0:
        return initial_path(lattice)
    seed = 2026085101 if lattice == 32 else 2026085201
    return (
        RUN / f"modelb_reference/L{lattice}"
        / f"modelB_L{lattice}_Ne512_hmc_t{number_tag(time)}_"
          f"chain_dt0.002_seed{seed}.jld2"
    )


def load_fields(path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive["phi_gen"][:N], dtype=np.float32)
    with h5py.File(path, "r") as handle:
        fields = np.asarray(handle["cfgs"][:N], dtype=np.float32)
    return fields.swapaxes(1, 2)


def load_modes(paths, ky, kx, label):
    result = []
    for i, path in enumerate(paths):
        if i % 20 == 0:
            print(f"{label}: {i}/{len(paths) - 1}", flush=True)
        result.append(diagonal_fft(load_fields(path), ky, kx))
    return np.stack(result)


def lag_statistics(modes, volume):
    powers = power(modes, volume)
    nt, nensemble, nmodes = powers.shape
    rho = np.empty((nt, nmodes))
    sem = np.empty_like(rho)
    leaves = []
    for lag in range(nt):
        count = nt - lag
        left = modes[:count]
        right = modes[lag:]
        cross_values = np.mean(
            (left * np.conjugate(right)).real, axis=(0, 3)
        ) / volume
        values = statistics(
            cross_values,
            powers[:count].mean(axis=0),
            powers[lag:].mean(axis=0),
        )
        rho[lag], sem[lag] = values[2], values[3]
        leaves.append(values[4])
    return rho, sem, leaves


def analyze_lattice(lattice):
    mode_n, ky, kx, _, khat, _ = mode_layout(lattice)
    t10_times = np.arange(0, 1001, 10)
    t05_times = np.arange(0, 50.5, 0.5)
    paths = {
        "t10": [dm_t10_path(lattice, int(t)) for t in t10_times],
        "t0p5": [dm_t0p5_path(lattice, i) for i in range(len(t05_times))],
        "modelb_t10": [modelb_path(lattice, int(t)) for t in t10_times],
        "modelb_t0p5": [modelb_path(lattice, t) for t in t05_times],
    }
    times = {
        "t10": t10_times, "t0p5": t05_times,
        "modelb_t10": t10_times, "modelb_t0p5": t05_times,
    }
    result = {"mode_n": mode_n, "khat": khat, "times": times}
    for name in ("t10", "t0p5", "modelb_t10", "modelb_t0p5"):
        modes = load_modes(paths[name], ky, kx, f"L{lattice} {name}")
        result[name] = lag_statistics(modes, lattice * lattice)
    return result


def mode_grid(data):
    ncols = 4
    nrows = len(data["mode_n"]) // ncols
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(13, 2.55 * nrows + 0.7),
        sharex=True, constrained_layout=True,
    )
    return figure, axes.reshape(-1), ncols


def set_correlation_ylim(axis, curves):
    lower = min(np.min(values - errors) for values, errors in curves)
    upper = max(np.max(values + errors) for values, errors in curves)
    span = max(upper - lower, 0.02)
    axis.set_ylim(min(lower - 0.08 * span, -0.005),
                  max(upper + 0.08 * span, 0.005))


def plot_short(results):
    for lattice in LS:
        data = results[lattice]
        figure, axes, ncols = mode_grid(data)
        for mode, axis in enumerate(axes):
            curves = []
            for name, marker, indices in (
                ("modelb_t0p5", "o", slice(1, None)),
                ("t10", "s", slice(1, 6)),
                ("t0p5", "x", slice(1, None)),
            ):
                times = data["times"][name][indices]
                rho, sem, _ = data[name]
                axis.errorbar(
                    times, rho[indices, mode], yerr=sem[indices, mode],
                    color=COLORS[name], marker=marker,
                    ms=2.4 if name != "t10" else 3.4,
                    lw=0.8, elinewidth=0.35,
                    capsize=0 if name != "t10" else 1.5,
                    label=LABELS[name],
                )
                curves.append((rho[indices, mode], sem[indices, mode]))
            set_correlation_ylim(axis, curves)
            axis.set_title(
                fr"$n={data['mode_n'][mode]}$, $\hat k={data['khat'][mode]:.3f}$"
            )
            axis.set_xlim(0, 50)
            axis.axhline(0, color="black", lw=0.5)
            axis.grid(alpha=0.2)
            if mode // ncols == len(axes) // ncols - 1:
                axis.set_xlabel(r"$\Delta t$")
            if mode % ncols == 0:
                axis.set_ylabel(r"$\rho_n(\Delta t)$")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="outside upper center", ncol=3,
            title=fr"L={lattice}; every $\delta t=0.5$ point is shown",
        )
        figure.savefig(
            OUTPUT / f"rho_L{lattice}_dt0p5_vs_dt10_short_all_modes.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_short_difference(results):
    for lattice in LS:
        data = results[lattice]
        figure, axes, ncols = mode_grid(data)
        modelb_rho, _, modelb_leaves = data["modelb_t0p5"]
        for mode, axis in enumerate(axes):
            for name, marker in (("t10", "s"), ("t0p5", "x")):
                rho, _, leaves = data[name]
                if name == "t10":
                    indices = np.arange(1, 6)
                    modelb_indices = np.arange(20, 101, 20)
                else:
                    indices = np.arange(1, 101)
                    modelb_indices = indices
                times = data["times"][name][indices]
                difference = rho[indices, mode] - modelb_rho[modelb_indices, mode]
                error = np.array([
                    paired_jackknife_sem(leaves[i], modelb_leaves[j])[mode]
                    for i, j in zip(indices, modelb_indices)
                ])
                axis.errorbar(
                    times, difference, yerr=error, color=COLORS[name],
                    marker=marker, ms=2.5 if name == "t0p5" else 3.5,
                    lw=0.8, elinewidth=0.35,
                    capsize=0 if name == "t0p5" else 1.7,
                    label=LABELS[name],
                )
            axis.axhline(0, color="black", lw=0.6)
            axis.set_title(fr"$n={data['mode_n'][mode]}$")
            axis.set_xlim(0, 50)
            axis.grid(alpha=0.2)
            if mode // ncols == len(axes) // ncols - 1:
                axis.set_xlabel(r"$\Delta t$")
            if mode % ncols == 0:
                axis.set_ylabel(r"$\rho_n^{\rm DM}-\rho_n^{\rm Model\ B}$")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=2,
                      title=f"L={lattice}")
        figure.savefig(
            OUTPUT / f"rho_L{lattice}_short_difference_all_modes.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_long(results):
    for lattice in LS:
        data = results[lattice]
        figure, axes, ncols = mode_grid(data)
        for mode, axis in enumerate(axes):
            curves = []
            for name, marker in (("modelb_t10", "o"), ("t10", "x")):
                times = data["times"][name][1:]
                rho, sem, _ = data[name]
                axis.errorbar(
                    times, rho[1:, mode], yerr=sem[1:, mode],
                    color=COLORS[name], marker=marker,
                    ms=2.3,
                    lw=1, elinewidth=0.35, capsize=0, label=LABELS[name],
                )
                curves.append((rho[1:, mode], sem[1:, mode]))
            set_correlation_ylim(axis, curves)
            axis.axhline(0, color="black", lw=0.5)
            axis.set_title(fr"$n={data['mode_n'][mode]}$")
            axis.set_xlim(0, 1000)
            axis.grid(alpha=0.2)
            if mode // ncols == len(axes) // ncols - 1:
                axis.set_xlabel(r"$\Delta t$")
            if mode % ncols == 0:
                axis.set_ylabel(r"$\rho_n(\Delta t)$")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=2,
                      title=f"L={lattice}")
        figure.savefig(
            OUTPUT / f"rho_L{lattice}_dt10_to_t1000_all_modes.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)


def write_tables(results):
    with (OUTPUT / "rho_time_origin_average.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("L", "chain", "lag", "mode_n", "khat", "rho", "rho_sem", "n_time_origins"))
        for lattice in LS:
            data = results[lattice]
            for name in ("modelb_t10", "t10", "modelb_t0p5", "t0p5"):
                times = data["times"][name]
                rho, sem, _ = data[name]
                for i, lag in enumerate(times):
                    for mode, n in enumerate(data["mode_n"]):
                        writer.writerow((lattice, name, lag, n, data["khat"][mode], rho[i, mode], sem[i, mode], len(times) - i))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {lattice: analyze_lattice(lattice) for lattice in LS}
    write_tables(results)
    plot_short(results)
    plot_short_difference(results)
    plot_long(results)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
