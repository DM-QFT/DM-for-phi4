"""Compare ep10000 single-L and mixed-L reverse samples with Model B."""

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
    moments,
    paired_jackknife_sem,
    power,
    statistics,
)


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/multiL_ep10000_reverse_analysis"
N = 512
LS = (32, 64)
TIMES = np.arange(0, 201, 10)

TRAINING = ROOT / "training_data"
L64_BASE = (
    ROOT
    / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip"
    / "samples/hmc_L64_N512_seed2026080201"
)
SOURCE = {
    8: TRAINING / "modelB_L8_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080601.jld2",
    16: TRAINING / "modelB_L16_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080501.jld2",
    32: TRAINING / "modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080101.jld2",
    64: L64_BASE / "canonical_L64_N512_k0p340340_l1_seed2026080201.jld2",
}
TARGET = {
    (0.5, 8): TRAINING / "modelB_L8_Ne51200_hmc_l1_kappa_0.340340_t0p5_dt0.002_seed2026080602.jld2",
    (0.5, 16): TRAINING / "modelB_L16_Ne51200_hmc_l1_kappa_0.340340_t0p5_dt0.002_seed2026080502.jld2",
    (0.5, 32): TRAINING / "modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t0p5_dt0.002_seed2026080102.jld2",
    (0.5, 64): OUT / "reference/modelB_L64_Ne512_hmc_t0p5_dt0.002_seed2026080701.jld2",
    (10.0, 8): TRAINING / "modelB_L8_Ne51200_hmc_l1_kappa_0.340340_t10_dt0.002_seed2026080602.jld2",
    (10.0, 16): TRAINING / "modelB_L16_Ne51200_hmc_l1_kappa_0.340340_t10_dt0.002_seed2026080502.jld2",
    (10.0, 32): TRAINING / "modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t10_dt0.002_seed2026080102.jld2",
    (10.0, 64): L64_BASE / "physical_reference_t200/modelB_L64_Ne512_hmc_t10_chain_dt0.002_seed2026080401.jld2",
}
SAMPLE_LABELS = {
    "single": "single-L32",
    "mixed": "mixed-L8/16/32",
}
COLORS = {"modelb": "black", "single": "#1f77b4", "mixed": "#d62728"}
MARKERS = {"modelb": "o", "single": "s", "mixed": "^"}


def sample_path(horizon, model, lattice):
    tag = "t0p5" if horizon == 0.5 else "t10"
    return OUT / f"samples/{tag}/{model}/L{lattice}.npz"


def load_fields(path, lattice=None):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive["phi_gen"], dtype=np.float32)
    start = 0 if lattice == 64 else 51200 - N
    with h5py.File(path, "r") as handle:
        fields = np.asarray(handle["cfgs"][start:start + N], dtype=np.float32)
    if "physical_reference_t200" in path.parts:
        fields = fields.swapaxes(1, 2)
    return fields


def mode_statistics(initial, endpoint, ky, kx):
    volume = initial.shape[1] ** 2
    mode0 = diagonal_fft(initial, ky, kx)
    mode1 = diagonal_fft(endpoint, ky, kx)
    p0 = power(mode0, volume)
    p1 = power(mode1, volume)
    cross_values = np.mean(
        (mode0 * np.conjugate(mode1)).real, axis=-1
    ) / volume
    return statistics(cross_values, p0, p1), p1


def mean_and_sem(values):
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def endpoint_observables(initial, endpoint):
    mean0 = moments(initial)[0]
    mean, phi2, phi4, variance = moments(endpoint)
    return {
        "phi2": phi2,
        "phi4": phi4,
        "variance": variance,
        "abs_M_drift": np.abs(mean - mean0),
    }


def analyze_one_step():
    diagonal_rows = []
    observable_rows = []
    plot_data = {}

    for horizon in (0.5, 10.0):
        plot_data[horizon] = {}
        for lattice in LS:
            initial = load_fields(SOURCE[lattice], lattice)
            exact = load_fields(TARGET[horizon, lattice], lattice)
            mode_n, ky, kx, _, khat, _ = mode_layout(lattice)
            exact_stats, exact_power = mode_statistics(initial, exact, ky, kx)
            exact_observables = endpoint_observables(initial, exact)
            fields = {
                "modelb": exact,
                "single": load_fields(sample_path(horizon, "single", lattice)),
                "mixed": load_fields(sample_path(horizon, "mixed", lattice)),
            }
            plot_data[horizon][lattice] = {"mode_n": mode_n, "khat": khat}

            for model, endpoint in fields.items():
                if model == "modelb":
                    values, endpoint_power = exact_stats, exact_power
                else:
                    values, endpoint_power = mode_statistics(initial, endpoint, ky, kx)
                _, _, rho, rho_sem, leaves = values
                rho_difference = rho - exact_stats[2]
                difference_sem = (
                    np.zeros_like(rho)
                    if model == "modelb"
                    else paired_jackknife_sem(leaves, exact_stats[4])
                )
                plot_data[horizon][lattice][model] = (rho, rho_sem, rho_difference, difference_sem)
                power_mean = endpoint_power.mean(axis=0)
                power_sem = endpoint_power.std(axis=0, ddof=1) / np.sqrt(N)
                exact_power_mean = exact_power.mean(axis=0)
                for i, n in enumerate(mode_n):
                    diagonal_rows.append((
                        horizon, lattice, int(n), khat[i], model, rho[i], rho_sem[i],
                        rho_difference[i], difference_sem[i],
                        rho_difference[i] / difference_sem[i] if difference_sem[i] else 0.0,
                        power_mean[i], power_sem[i], power_mean[i] / exact_power_mean[i],
                    ))

                observables = endpoint_observables(initial, endpoint)
                for quantity, trajectory_values in observables.items():
                    mean, sem = mean_and_sem(trajectory_values)
                    exact_values = exact_observables[quantity]
                    difference, difference_sem = mean_and_sem(trajectory_values - exact_values)
                    if model == "modelb":
                        difference = difference_sem = 0.0
                    observable_rows.append((
                        horizon, lattice, model, quantity, mean, sem,
                        difference, difference_sem,
                        difference / difference_sem if difference_sem else 0.0,
                    ))

    with (OUT / "one_step_diagonal.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "horizon", "L", "mode_n", "khat", "model", "rho", "rho_sem",
            "rho_minus_modelb", "paired_sem", "paired_z", "G_endpoint", "G_sem",
            "G_over_modelb",
        ))
        writer.writerows(diagonal_rows)
    with (OUT / "one_step_observables.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "horizon", "L", "model", "quantity", "mean", "sem",
            "minus_modelb", "paired_sem", "paired_z",
        ))
        writer.writerows(observable_rows)

    for horizon in (0.5, 10.0):
        figure, axes = plt.subplots(
            2, len(LS), figsize=(8, 6.6), constrained_layout=True
        )
        for column, lattice in enumerate(LS):
            data = plot_data[horizon][lattice]
            x = data["khat"]
            for model in ("modelb", "single", "mixed"):
                rho, error, difference, difference_error = data[model]
                label = "Model B" if model == "modelb" else SAMPLE_LABELS[model]
                axes[0, column].errorbar(
                    x, rho, yerr=error, color=COLORS[model], marker=MARKERS[model],
                    ms=3, lw=1, elinewidth=0.6, capsize=1.5, label=label,
                )
                if model != "modelb":
                    axes[1, column].errorbar(
                        x, difference, yerr=difference_error, color=COLORS[model],
                        marker=MARKERS[model], ms=3, lw=1, elinewidth=0.6,
                        capsize=1.5, label=label,
                    )
            axes[0, column].set_title(f"L={lattice}")
            axes[0, column].set_ylabel(r"$\rho_n(\Delta t)$")
            axes[1, column].set_ylabel(r"$\rho_n^{\rm DM}-\rho_n^{\rm Model\ B}$")
            axes[1, column].set_xlabel(r"$\hat k$")
            axes[1, column].axhline(0, color="black", lw=0.6)
            for axis in axes[:, column]:
                axis.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=3)
        tag = "t0p5" if horizon == 0.5 else "t10"
        figure.savefig(OUT / f"one_step_{tag}_diagonal.pdf", bbox_inches="tight")
        plt.close(figure)


def chain_path(model, time):
    if time == 0:
        return L64_BASE / "hmc_ep10000_L64_t0_N512.npz"
    if model == "modelb":
        return L64_BASE / "physical_reference_t200" / (
            f"modelB_L64_Ne512_hmc_t{time}_chain_dt0.002_seed2026080401.jld2"
        )
    directory = (
        L64_BASE / "dm_chain_t200"
        if model == "single"
        else OUT / "samples/t10/mixed_L64_chain_t200"
    )
    first_seed = 2026080301 if model == "single" else 2026080801
    return directory / (
        f"chain_ep10000_t{time}_via_iterated_t10dm512_steps2000_log_"
        f"seed{first_seed + time // 10 - 1}.npz"
    )


def analyze_chain():
    chains = ("modelb", "single", "mixed")
    mode_n, ky, kx, _, khat, _ = mode_layout(64)
    fields = {
        model: [load_fields(chain_path(model, int(time)), 64) for time in TIMES]
        for model in chains
    }
    modes = {
        model: np.stack([diagonal_fft(value, ky, kx) for value in fields[model]])
        for model in chains
    }
    powers = {model: power(modes[model], 64 * 64) for model in chains}
    rho = {model: np.empty((len(TIMES), len(mode_n))) for model in chains}
    rho_sem = {model: np.empty_like(rho[model]) for model in chains}
    leaves = {model: [] for model in chains}

    for lag_index in range(len(TIMES)):
        count = len(TIMES) - lag_index
        for model in chains:
            cross_sum = np.zeros((N, len(mode_n)))
            start_power = np.zeros_like(cross_sum)
            end_power = np.zeros_like(cross_sum)
            for start in range(count):
                left = modes[model][start]
                right = modes[model][start + lag_index]
                cross_sum += np.mean((left * np.conjugate(right)).real, axis=-1) / (64 * 64)
                start_power += powers[model][start]
                end_power += powers[model][start + lag_index]
            values = statistics(cross_sum / count, start_power / count, end_power / count)
            rho[model][lag_index], rho_sem[model][lag_index] = values[2], values[3]
            leaves[model].append(values[4])

    with (OUT / "L64_chain_diagonal_lags.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "lag", "n_time_origins", "mode_n", "khat", "rho_modelb", "modelb_sem",
            "rho_single", "single_sem", "single_minus_modelb", "single_paired_sem",
            "single_z", "rho_mixed", "mixed_sem", "mixed_minus_modelb",
            "mixed_paired_sem", "mixed_z",
        ))
        for lag_index, lag in enumerate(TIMES):
            for i, n in enumerate(mode_n):
                row = [lag, len(TIMES) - lag_index, int(n), khat[i], rho["modelb"][lag_index, i], rho_sem["modelb"][lag_index, i]]
                for model in ("single", "mixed"):
                    difference = rho[model][lag_index, i] - rho["modelb"][lag_index, i]
                    error = paired_jackknife_sem(
                        leaves[model][lag_index], leaves["modelb"][lag_index]
                    )[i]
                    row.extend((rho[model][lag_index, i], rho_sem[model][lag_index, i], difference, error, difference / error if error else 0.0))
                writer.writerow(row)

    quantities = {model: {name: [] for name in ("phi2", "phi4", "variance", "abs_M_drift")} for model in chains}
    initial_mean = moments(fields["modelb"][0])[0]
    for model in chains:
        for endpoint in fields[model]:
            mean, phi2, phi4, variance = moments(endpoint)
            for name, value in zip(
                ("phi2", "phi4", "variance", "abs_M_drift"),
                (phi2, phi4, variance, np.abs(mean - initial_mean)),
            ):
                quantities[model][name].append(value)
        for name in quantities[model]:
            quantities[model][name] = np.stack(quantities[model][name])

    with (OUT / "L64_chain_observables.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "time", "quantity", "modelb", "modelb_sem", "single", "single_sem",
            "single_minus_modelb", "single_paired_sem", "mixed", "mixed_sem",
            "mixed_minus_modelb", "mixed_paired_sem",
        ))
        for time_index, time in enumerate(TIMES):
            for quantity in quantities["modelb"]:
                exact = quantities["modelb"][quantity][time_index]
                row = [time, quantity, *mean_and_sem(exact)]
                for model in ("single", "mixed"):
                    value = quantities[model][quantity][time_index]
                    row.extend((*mean_and_sem(value), *mean_and_sem(value - exact)))
                writer.writerow(row)

    with (OUT / "L64_chain_diagonal_power.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "time", "mode_n", "khat", "G_modelb", "modelb_sem", "G_single",
            "single_sem", "single_over_modelb", "G_mixed", "mixed_sem", "mixed_over_modelb",
        ))
        for time_index, time in enumerate(TIMES):
            for i, n in enumerate(mode_n):
                exact_mean, exact_sem = mean_and_sem(powers["modelb"][time_index, :, i])
                row = [time, int(n), khat[i], exact_mean, exact_sem]
                for model in ("single", "mixed"):
                    mean, sem = mean_and_sem(powers[model][time_index, :, i])
                    row.extend((mean, sem, mean / exact_mean))
                writer.writerow(row)

    figure, axes = plt.subplots(4, 4, figsize=(13, 11), sharex=True, constrained_layout=True)
    for i, axis in enumerate(axes.flat):
        for model in chains:
            label = "Model B" if model == "modelb" else SAMPLE_LABELS[model]
            axis.errorbar(
                TIMES[1:], rho[model][1:, i], yerr=rho_sem[model][1:, i],
                color=COLORS[model], marker=MARKERS[model], ms=2.5, lw=1,
                elinewidth=0.5, capsize=1.2, label=label,
            )
        axis.set_title(fr"$n={mode_n[i]}$, $\hat k={khat[i]:.3f}$")
        axis.set_ylabel(r"$\rho_n(\Delta t)$")
        axis.grid(alpha=0.2)
        if i >= 12:
            axis.set_xlabel(r"$\Delta t$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, title="L64, time-origin average")
    figure.savefig(OUT / "L64_chain_diagonal_lag_average.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 7.6), constrained_layout=True)
    labels = {
        "phi2": r"$\langle\phi^2\rangle$",
        "phi4": r"$\langle\phi^4\rangle$",
        "variance": r"$\langle(\phi-M)^2\rangle$",
        "abs_M_drift": r"$\langle|M(t)-M(0)|\rangle$",
    }
    for axis, quantity in zip(axes.flat, labels):
        for model in chains:
            values = quantities[model][quantity]
            mean = values.mean(axis=1)
            sem = values.std(axis=1, ddof=1) / np.sqrt(N)
            label = "Model B" if model == "modelb" else SAMPLE_LABELS[model]
            axis.errorbar(TIMES, mean, yerr=sem, color=COLORS[model], marker=MARKERS[model], ms=2.8, lw=1, capsize=1.4, label=label)
        if quantity == "abs_M_drift":
            axis.set_yscale("log")
        axis.set(xlabel=r"$t$", ylabel=labels[quantity])
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3, title="L64")
    figure.savefig(OUT / "L64_chain_observables.pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    analyze_one_step()
    analyze_chain()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
