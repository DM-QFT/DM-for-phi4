"""Diagonal-mode and observable tables for the ep10000 chains through t=200.

Writes tables and one npz per initial condition; makes no figures.
"""

from pathlib import Path
import csv

import h5py
import numpy as np

from diagonal_modes import (
    cross,
    diagonal_fft,
    mean_sem,
    mode_layout,
    moments,
    paired_jackknife_sem,
    power,
    statistics,
)

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_corrgauss_uniformM_noskip"
OUTPUT = RUN / "t200_ep10000_dm_vs_modelb"
TIMES = np.arange(0, 201, 10, dtype=np.int32)
SNAPSHOT_TIMES = (0, 10, 20, 50, 100, 150, 200)
VOLUME = 32 * 32
CHAINS = ("modelb", "dm")
KEYS = ("phi2", "phi4", "variance", "mean_abs_drift")
INITIALS = {
    "hmc": (
        RUN / "samples/hmc_chain_ep10000_N5120_seed20260901",
        "hmc_ep10000_t0_N5120.npz",
        20260901,
        20261101,
    ),
    "gaussian": (
        RUN / "samples/gaussian_chain_ep10000_N5120_seed20261001",
        "gaussian_ep10000_t0_N5120_seed20260721.npz",
        20261001,
        20261201,
    ),
}


def dm_path(directory, initial_name, first_seed, time):
    if time == 0:
        return directory / initial_name
    return directory / (
        f"chain_ep10000_t{time}_via_iterated_t10dm5120_"
        f"steps2000_log_seed{first_seed + time // 10 - 1}.npz"
    )


def modelb_path(directory, initial_key, seed, time):
    return directory / "physical_reference_t200" / (
        f"modelB_L32_Ne5120_{initial_key}_t{time}_chain_dt0.002_seed{seed}.jld2"
    )


def load_fields(path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive["phi_gen"], dtype=np.float32)
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["cfgs"], dtype=np.float32)


def lag_correlations(modes):
    """Time-origin averaged rho per lag, plus the jackknife leaves."""
    powers = {chain: power(modes[chain], VOLUME) for chain in CHAINS}
    nt, nensemble, nmodes = powers["dm"].shape
    result = {}
    for chain in CHAINS:
        result[f"{chain}_G"] = np.mean(powers[chain], axis=1)
        result[f"{chain}_G_sem"] = (
            np.std(powers[chain], axis=1, ddof=1) / np.sqrt(nensemble)
        )
        for name in ("rho", "rho_sem"):
            result[f"{chain}_{name}"] = np.empty((nt, nmodes))
    result["rho_difference"] = np.empty((nt, nmodes))
    result["rho_difference_sem"] = np.empty((nt, nmodes))
    result["dm_fixed_rho"] = np.empty((nt, nmodes))
    result["dm_fixed_rho_sem"] = np.empty((nt, nmodes))

    for lag in range(nt):
        leaves = {}
        for chain in CHAINS:
            origin_cross = np.zeros((nensemble, nmodes), dtype=np.float64)
            origin_start = np.zeros_like(origin_cross)
            origin_endpoint = np.zeros_like(origin_cross)
            for start in range(nt - lag):
                origin_cross += cross(modes[chain], start, start + lag, VOLUME)
                origin_start += powers[chain][start]
                origin_endpoint += powers[chain][start + lag]
            count = nt - lag
            values = statistics(
                origin_cross / count, origin_start / count,
                origin_endpoint / count,
            )
            result[f"{chain}_rho"][lag] = values[2]
            result[f"{chain}_rho_sem"][lag] = values[3]
            leaves[chain] = values[4]
        result["rho_difference"][lag] = (
            result["dm_rho"][lag] - result["modelb_rho"][lag]
        )
        result["rho_difference_sem"][lag] = paired_jackknife_sem(
            leaves["dm"], leaves["modelb"]
        )
        values = statistics(
            cross(modes["dm"], 0, lag, VOLUME), powers["dm"][0],
            powers["dm"][lag],
        )
        result["dm_fixed_rho"][lag] = values[2]
        result["dm_fixed_rho_sem"][lag] = values[3]
    return result


def analyze(initial_key, ky, kx):
    directory, initial_name, dm_seed, modelb_seed = INITIALS[initial_key]
    values = {
        chain: {key: [] for key in ("mean", "phi2", "phi4", "variance")}
        for chain in CHAINS
    }
    modes = {chain: [] for chain in CHAINS}
    snapshots = {chain: [] for chain in CHAINS}

    for time in TIMES:
        print(f"{initial_key}: t={time}", flush=True)
        dm_fields = load_fields(dm_path(directory, initial_name, dm_seed, int(time)))
        modelb_fields = dm_fields if time == 0 else load_fields(
            modelb_path(directory, initial_key, modelb_seed, int(time))
        )
        for chain, fields in (("modelb", modelb_fields), ("dm", dm_fields)):
            mean, phi2, phi4, variance = moments(fields)
            for key, array in zip(
                ("mean", "phi2", "phi4", "variance"),
                (mean, phi2, phi4, variance),
            ):
                values[chain][key].append(array)
            modes[chain].append(diagonal_fft(fields, ky, kx))
            if time in SNAPSHOT_TIMES:
                snapshots[chain].append(fields[0].copy())

    for chain in CHAINS:
        for key in values[chain]:
            values[chain][key] = np.stack(values[chain][key])
        mean = values[chain]["mean"]
        values[chain]["mean_abs_drift"] = np.abs(mean - mean[0])

    result = lag_correlations({c: np.stack(modes[c]) for c in CHAINS})
    for key in KEYS:
        for chain in CHAINS:
            result[f"{chain}_{key}"], result[f"{chain}_{key}_sem"] = mean_sem(
                values[chain][key]
            )
        difference = values["dm"][key] - values["modelb"][key]
        result[f"{key}_difference"] = np.mean(difference, axis=1)
        result[f"{key}_difference_sem"] = (
            np.std(difference, axis=1, ddof=1) / np.sqrt(difference.shape[1])
        )
    for chain in CHAINS:
        result[f"{chain}_rms_drift"] = np.sqrt(np.mean(
            (values[chain]["mean"] - values[chain]["mean"][0]) ** 2, axis=1
        ))
        result[f"snapshots_{chain}"] = np.stack(snapshots[chain])
    return result


def write_tables(initial_key, result, mode_n, khat):
    with (OUTPUT / f"{initial_key}_observables.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "time", "quantity", "modelb", "modelb_sem", "dm", "dm_sem",
            "dm_minus_modelb", "paired_sem",
        ))
        for i, time in enumerate(TIMES):
            for key in KEYS:
                writer.writerow((
                    time, key, result[f"modelb_{key}"][i],
                    result[f"modelb_{key}_sem"][i], result[f"dm_{key}"][i],
                    result[f"dm_{key}_sem"][i], result[f"{key}_difference"][i],
                    result[f"{key}_difference_sem"][i],
                ))

    with (OUTPUT / f"{initial_key}_diagonal_lags.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "lag", "n_time_origins", "mode_n", "khat", "rho_modelb",
            "rho_modelb_sem", "rho_dm", "rho_dm_sem", "dm_minus_modelb",
            "paired_sem", "paired_z",
        ))
        for lag_index, lag in enumerate(TIMES):
            for mode, n in enumerate(mode_n):
                difference = result["rho_difference"][lag_index, mode]
                error = result["rho_difference_sem"][lag_index, mode]
                writer.writerow((
                    lag, len(TIMES) - lag_index, n, khat[mode],
                    result["modelb_rho"][lag_index, mode],
                    result["modelb_rho_sem"][lag_index, mode],
                    result["dm_rho"][lag_index, mode],
                    result["dm_rho_sem"][lag_index, mode], difference, error,
                    difference / error if error else 0.0,
                ))

    with (OUTPUT / f"{initial_key}_diagonal_power.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow((
            "time", "mode_n", "khat", "G_modelb", "G_modelb_sem",
            "G_dm", "G_dm_sem", "G_dm_over_modelb",
        ))
        for time_index, time in enumerate(TIMES):
            for mode, n in enumerate(mode_n):
                writer.writerow((
                    time, n, khat[mode], result["modelb_G"][time_index, mode],
                    result["modelb_G_sem"][time_index, mode],
                    result["dm_G"][time_index, mode],
                    result["dm_G_sem"][time_index, mode],
                    result["dm_G"][time_index, mode]
                    / result["modelb_G"][time_index, mode],
                ))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mode_n, ky, kx, _, khat, _ = mode_layout(32)
    for initial_key in INITIALS:
        result = analyze(initial_key, ky, kx)
        write_tables(initial_key, result, mode_n, khat)
        np.savez_compressed(
            OUTPUT / f"{initial_key}_t200.npz", times=TIMES, mode_n=mode_n,
            khat=khat, snapshot_times=np.array(SNAPSHOT_TIMES), **result,
        )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
