"""Diagonal-mode and observable tables for the configured ep10000 chains.

Writes tables and one npz per initial condition; makes no figures.
"""

from pathlib import Path
import csv
import os

import numpy as np

from diagonal_modes import (
    cross,
    diagonal_fft,
    mean_sem,
    mode_layout,
    moments,
    paired_jackknife_sem,
    load_fields,
    load_series,
    series_index,
    power,
    statistics,
)

ROOT = Path(__file__).resolve().parent.parent
RUN = Path(os.environ.get(
    "MODELB_CHAIN_RUN",
    ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip",
)).expanduser().resolve()
OUTPUT = ROOT / "results" / "dm_vs_modelb" / RUN.name
TMAX = int(os.environ.get("MODELB_CHAIN_TMAX", "2000"))
TIMES = np.arange(0, TMAX + 1, 10, dtype=np.int32)
SNAPSHOT_TIMES = (0, 10, 20, 50, 100, 200, 500, 1000)
N_SAMPLES = int(os.environ.get("MODELB_CHAIN_N", "1024"))
# Model B 参照全部集中在 results/modelb_reference/，按 <run 标签>_<初态>_L<L>_N<N> 命名
REFERENCES = ROOT / "results" / "modelb_reference"
REFERENCE_TAG = "corrgauss" if "corrgauss" in RUN.name else "hmc_noskip"
CHAINS = ("modelb", "dm")
KEYS = ("phi2", "phi4", "variance", "mean_abs_drift")
# t=0 初态与 Model B 参照沿用 corrgauss run 的（zlog 链就是从它们出发采的）
T0 = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip/samples"
HMC_SUFFIX = "" if "zlog" in RUN.name else "_seed20260901"
GAUSSIAN_SUFFIX = "" if "zlog" in RUN.name else "_seed20261001"
# 初态名 -> (链目录, 链文件, t=0 初态, Model B 参照, 格点尺寸)
INITIALS = {
    "hmc": (
        RUN / f"samples/hmc_chain_ep10000_N{N_SAMPLES}{HMC_SUFFIX}", "chain.h5",
        str(T0 / f"hmc_chain_ep10000_N{N_SAMPLES}_seed20260901"
             / f"hmc_ep10000_t0_N{N_SAMPLES}.npz"),
        f"{REFERENCE_TAG}_hmc_L32_N{N_SAMPLES}.h5", 32,
    ),
    "gaussian": (
        RUN / f"samples/gaussian_chain_ep10000_N{N_SAMPLES}{GAUSSIAN_SUFFIX}", "chain.h5",
        str(T0 / f"gaussian_chain_ep10000_N{N_SAMPLES}_seed20261001"
             / f"gaussian_ep10000_t0_N{N_SAMPLES}_seed20260721.npz"),
        f"{REFERENCE_TAG}_gaussian_L32_N{N_SAMPLES}.h5", 32,
    ),
}


def lag_correlations(modes, volume):
    """Time-origin averaged rho per lag, plus the jackknife leaves."""
    powers = {chain: power(modes[chain], volume) for chain in CHAINS}
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
                origin_cross += cross(modes[chain], start, start + lag, volume)
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
            cross(modes["dm"], 0, lag, volume), powers["dm"][0],
            powers["dm"][lag],
        )
        result["dm_fixed_rho"][lag] = values[2]
        result["dm_fixed_rho_sem"][lag] = values[3]
    return result


def analyze(initial_key, ky, kx, volume):
    directory, chain_name, initial_name, reference_name, _ = INITIALS[initial_key]
    values = {
        chain: {key: [] for key in ("mean", "phi2", "phi4", "variance")}
        for chain in CHAINS
    }
    modes = {chain: [] for chain in CHAINS}
    snapshots = {chain: [] for chain in CHAINS}

    dm_times, dm_data = load_series(directory / chain_name)
    modelb_times, modelb_data = load_series(REFERENCES / reference_name)
    initial = load_fields(directory / initial_name)

    for time in TIMES:
        print(f"{initial_key}: t={time}", flush=True)
        dm_fields = initial if time == 0 else np.asarray(
            dm_data[series_index(dm_times, time)], dtype=np.float32)
        modelb_fields = initial if time == 0 else np.asarray(
            modelb_data[series_index(modelb_times, time)], dtype=np.float32)
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

    result = lag_correlations({c: np.stack(modes[c]) for c in CHAINS}, volume)
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
    for initial_key, (directory, chain_name, initial_name, reference_name,
                      lattice) in INITIALS.items():
        # 不是每个 run 都有全部初态（hmc_noskip 只有 HMC 链）
        missing = [p for p in (directory / chain_name, directory / initial_name,
                               REFERENCES / reference_name) if not p.exists()]
        if missing:
            print(f"跳过 {initial_key}: 缺 {missing[0].name}", flush=True)
            continue
        mode_n, ky, kx, _, khat, _ = mode_layout(lattice)
        result = analyze(initial_key, ky, kx, lattice * lattice)
        write_tables(initial_key, result, mode_n, khat)
        np.savez_compressed(
            OUTPUT / f"{initial_key}_chain.npz", times=TIMES, mode_n=mode_n,
            khat=khat, snapshot_times=np.array(SNAPSHOT_TIMES), **result,
        )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
