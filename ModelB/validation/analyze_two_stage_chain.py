"""Compare the L32 two-stage DM chain with fixed-solver references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import fft

try:
    from .compare_endpoint_timecorr import (
        centered,
        mean_sem,
        paired_jackknife_sem,
        radial_layout,
        ratio_with_jackknife,
        read_rows,
        scalar_summary,
        shell_mean,
    )
except ImportError:
    from compare_endpoint_timecorr import (
        centered,
        mean_sem,
        paired_jackknife_sem,
        radial_layout,
        ratio_with_jackknife,
        read_rows,
        scalar_summary,
        shell_mean,
    )


MODEL_B_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = (
    MODEL_B_DIR
    / "runs/modelBcond_L32_k0.28_l0.022_t10_ncsnpp_all_data"
)
SAMPLE_DIR = RUN_DIR / "samples"
REFERENCE_DIR = RUN_DIR / "physical_reference"
L32_DATA_DIR = MODEL_B_DIR / "runs/L32_Ne10240_dt0p002_hmc_t1_t10_t100"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dm10",
        type=Path,
        default=SAMPLE_DIR
        / "chain_ep2257_t10_from_hmc5120_steps2000_log_seed1234.npz",
    )
    parser.add_argument(
        "--dm20",
        type=Path,
        default=SAMPLE_DIR
        / "chain_ep2257_t20_via_t10dm5120_steps2000_log_seed5678.npz",
    )
    parser.add_argument(
        "--hmc",
        type=Path,
        default=MODEL_B_DIR.parent
        / "2Dphi4/trainingdata/cfgs_wolff_fahmc_k=0.28_l=0.022_32^2.jld2",
    )
    parser.add_argument(
        "--fixed10",
        type=Path,
        default=L32_DATA_DIR
        / "modelB_L32_Ne10240_hmc_kappa_0.28_t10_dt0.002.jld2",
    )
    parser.add_argument(
        "--fixed20",
        type=Path,
        default=REFERENCE_DIR
        / "modelB_L32_Ne5120_true_t10_to_t20_dt0.002_seed20260725.jld2",
    )
    parser.add_argument(
        "--fixed20_from_dm10",
        type=Path,
        default=REFERENCE_DIR
        / "modelB_L32_Ne5120_dm_t10_to_phys_t20_dt0.002_seed20260726.jld2",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=RUN_DIR / "validation/chain_ep2257_two_stage_5120",
    )
    parser.add_argument("--fft_batch", type=int, default=128)
    parser.add_argument("--fft_workers", type=int, default=8)
    return parser


def load_dm(path):
    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["phi_gen"], dtype=np.float32),
            np.asarray(archive["cond_indices"], dtype=np.int64),
        )


def load_cfgs(path, rows=None):
    with h5py.File(path, "r") as handle:
        dataset = handle["cfgs"]
        if rows is None:
            return np.asarray(dataset[:], dtype=np.float32)
        return read_rows(dataset, rows)


def value_summary(values):
    mean, sem = mean_sem(values)
    values = np.asarray(values, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "mean": float(mean),
        "bias": float(mean),
        "sem": float(sem),
        "rms": float(np.sqrt(np.mean(values**2))),
        "mean_abs": float(np.mean(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(np.max(absolute)),
    }


def paired_summary(dm_values, physical_values):
    dm_mean, dm_sem = mean_sem(dm_values)
    physical_mean, physical_sem = mean_sem(physical_values)
    difference, difference_sem = mean_sem(
        np.asarray(dm_values, dtype=np.float64)
        - np.asarray(physical_values, dtype=np.float64)
    )
    return {
        "dm": {"mean": float(dm_mean), "sem": float(dm_sem)},
        "physical": {
            "mean": float(physical_mean),
            "sem": float(physical_sem),
        },
        "dm_minus_physical": float(difference),
        "paired_sem": float(difference_sem),
        "z": float(difference / difference_sem),
        "relative_difference": float(difference / physical_mean),
    }


def spatial_moment(fields, power):
    return np.mean(
        np.asarray(fields, dtype=np.float64) ** power,
        axis=(1, 2),
        dtype=np.float64,
    )


def field_statistics(fields, layout, fft_batch, fft_workers):
    shell_ids, pixels, starts, counts, k, khat = layout
    del shell_ids, k, khat
    nensemble, length, _ = fields.shape
    volume = length * length
    means = np.mean(fields, axis=(1, 2), dtype=np.float64)
    raw_g = np.einsum(
        "nij,nij->n", fields, fields, dtype=np.float64, optimize=True
    ) / volume
    transforms = np.empty(fields.shape, dtype=np.complex64)
    spectra = np.empty((nensemble, len(counts)), dtype=np.float64)
    for start in range(0, nensemble, fft_batch):
        stop = min(start + fft_batch, nensemble)
        batch, _ = centered(fields[start:stop])
        batch_fft = fft.fft2(
            batch, axes=(-2, -1), workers=fft_workers
        )
        transforms[start:stop] = batch_fft
        power = (batch_fft.real**2 + batch_fft.imag**2) / volume
        spectra[start:stop] = shell_mean(power, pixels, starts, counts)
    return {
        "mean": means,
        "raw_g": raw_g,
        "connected_g": raw_g - means**2,
        "fft": transforms,
        "spectrum": spectra,
    }


def scalar_pair(fields, stats, start, endpoint, kind):
    length = fields[start].shape[-1]
    cross = np.einsum(
        "nij,nij->n",
        fields[endpoint],
        fields[start],
        dtype=np.float64,
        optimize=True,
    ) / (length * length)
    if kind == "connected":
        cross = cross - stats[endpoint]["mean"] * stats[start]["mean"]
        g_endpoint = stats[endpoint]["connected_g"]
        g_start = stats[start]["connected_g"]
    else:
        g_endpoint = stats[endpoint]["raw_g"]
        g_start = stats[start]["raw_g"]
    return cross, g_endpoint, g_start


def compare_scalar_pairs(fields, stats, dm_pair, physical_pair, kind):
    dm_values = scalar_pair(fields, stats, *dm_pair, kind)
    physical_values = scalar_pair(fields, stats, *physical_pair, kind)
    dm_summary, dm_leave = scalar_summary(*dm_values)
    physical_summary, physical_leave = scalar_summary(*physical_values)
    rho_difference = dm_summary["rho"] - physical_summary["rho"]
    rho_sem = float(paired_jackknife_sem(dm_leave, physical_leave))
    c_difference, c_sem = mean_sem(dm_values[0] - physical_values[0])
    return {
        "dm": dm_summary,
        "physical": physical_summary,
        "C_dm_minus_physical": float(c_difference),
        "C_difference_paired_sem": float(c_sem),
        "C_difference_z": float(c_difference / c_sem),
        "rho_dm_minus_physical": rho_difference,
        "rho_difference_paired_jackknife_sem": rho_sem,
        "rho_difference_z": rho_difference / rho_sem,
    }


def momentum_pair(stats, start, endpoint, layout):
    _, pixels, starts, counts, _, _ = layout
    length = stats[start]["fft"].shape[-1]
    fft_start = stats[start]["fft"]
    fft_endpoint = stats[endpoint]["fft"]
    cross = (
        fft_endpoint.real * fft_start.real
        + fft_endpoint.imag * fft_start.imag
    ) / (length * length)
    cross = shell_mean(cross, pixels, starts, counts)
    power_endpoint = stats[endpoint]["spectrum"]
    power_start = stats[start]["spectrum"]
    rho, rho_sem, leave_rho = ratio_with_jackknife(
        cross, power_endpoint, power_start
    )
    return cross, power_endpoint, power_start, rho, rho_sem, leave_rho


def compare_momentum_pairs(stats, dm_pair, physical_pair, layout):
    dm = momentum_pair(stats, *dm_pair, layout)
    physical = momentum_pair(stats, *physical_pair, layout)
    result = {}
    for key, dm_values, physical_values in (
        ("F", dm[0], physical[0]),
        ("G_endpoint", dm[1], physical[1]),
        ("G_start", dm[2], physical[2]),
    ):
        dm_mean, dm_sem = mean_sem(dm_values)
        physical_mean, physical_sem = mean_sem(physical_values)
        difference, difference_sem = mean_sem(dm_values - physical_values)
        result[key] = (
            dm_mean,
            dm_sem,
            physical_mean,
            physical_sem,
            difference,
            difference_sem,
        )
    rho_difference = dm[3] - physical[3]
    rho_difference_sem = paired_jackknife_sem(dm[5], physical[5])
    result["rho"] = (
        dm[3],
        dm[4],
        physical[3],
        physical[4],
        rho_difference,
        rho_difference_sem,
    )
    return result


def write_structure_table(path, comparisons, stats, layout):
    shell_ids, _, _, counts, k, khat = layout
    header = (
        "comparison", "shell", "mode_count", "k", "khat",
        "S_dm", "S_dm_sem", "S_physical", "S_physical_sem",
        "delta_S", "delta_S_paired_sem", "z",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for name, dm_name, physical_name in comparisons:
            dm = stats[dm_name]["spectrum"]
            physical = stats[physical_name]["spectrum"]
            dm_mean, dm_sem = mean_sem(dm)
            physical_mean, physical_sem = mean_sem(physical)
            difference, difference_sem = mean_sem(dm - physical)
            for i in range(len(shell_ids)):
                writer.writerow((
                    name,
                    int(shell_ids[i]),
                    int(counts[i]),
                    f"{k[i]:.10e}",
                    f"{khat[i]:.10e}",
                    f"{dm_mean[i]:.10e}",
                    f"{dm_sem[i]:.10e}",
                    f"{physical_mean[i]:.10e}",
                    f"{physical_sem[i]:.10e}",
                    f"{difference[i]:.10e}",
                    f"{difference_sem[i]:.10e}",
                    f"{difference[i] / difference_sem[i]:.10e}",
                ))


def write_momentum_table(path, comparisons, stats, layout):
    shell_ids, _, _, counts, k, khat = layout
    quantity_columns = (
        "dm", "dm_sem", "physical", "physical_sem",
        "dm_minus_physical", "paired_sem",
    )
    header = ["comparison", "shell", "mode_count", "k", "khat"]
    for quantity in ("G_start", "G_endpoint", "F", "rho"):
        header.extend(f"{quantity}_{column}" for column in quantity_columns)
    header.append("rho_difference_z")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for name, dm_pair, physical_pair in comparisons:
            values = compare_momentum_pairs(
                stats, dm_pair, physical_pair, layout
            )
            for i in range(len(shell_ids)):
                row = [
                    name, int(shell_ids[i]), int(counts[i]),
                    f"{k[i]:.10e}", f"{khat[i]:.10e}",
                ]
                for quantity in ("G_start", "G_endpoint", "F", "rho"):
                    row.extend(f"{array[i]:.10e}" for array in values[quantity])
                row.append(
                    f"{values['rho'][4][i] / values['rho'][5][i]:.10e}"
                )
                writer.writerow(row)


def main(argv=None):
    args = build_parser().parse_args(argv)
    dm10, cond_indices = load_dm(args.dm10)
    dm20, dm20_indices = load_dm(args.dm20)
    nensemble = len(dm10)
    if not np.array_equal(dm20_indices, np.arange(nensemble)):
        raise ValueError("dm20 must follow the row order of dm10")

    fields = {
        "hmc": load_cfgs(args.hmc, cond_indices),
        "dm10": dm10,
        "dm20": dm20,
        "fixed10": load_cfgs(args.fixed10, cond_indices),
        "fixed20": load_cfgs(args.fixed20),
        "fixed20_from_dm10": load_cfgs(args.fixed20_from_dm10),
    }
    if any(field.shape != dm10.shape for field in fields.values()):
        raise ValueError("all fields must have the same (N,L,L) shape")

    length = dm10.shape[-1]
    layout = radial_layout(length)
    stats = {}
    for name, field in fields.items():
        print(f"FFT {name}", flush=True)
        stats[name] = field_statistics(
            field, layout, args.fft_batch, args.fft_workers
        )

    conservation = {
        "dm": {
            "delta_M_01": value_summary(
                stats["dm10"]["mean"] - stats["hmc"]["mean"]
            ),
            "delta_M_12": value_summary(
                stats["dm20"]["mean"] - stats["dm10"]["mean"]
            ),
            "delta_M_02": value_summary(
                stats["dm20"]["mean"] - stats["hmc"]["mean"]
            ),
        },
        "fixed_chain": {
            "delta_M_01": value_summary(
                stats["fixed10"]["mean"] - stats["hmc"]["mean"]
            ),
            "delta_M_12": value_summary(
                stats["fixed20"]["mean"] - stats["fixed10"]["mean"]
            ),
            "delta_M_02": value_summary(
                stats["fixed20"]["mean"] - stats["hmc"]["mean"]
            ),
        },
        "fixed_second_kernel_from_dm10": {
            "delta_M_12": value_summary(
                stats["fixed20_from_dm10"]["mean"]
                - stats["dm10"]["mean"]
            )
        },
    }

    one_time_moments = {}
    moment_values = {}
    for name, field in fields.items():
        moment_values[name] = {
            "phi2": spatial_moment(field, 2),
            "phi4": spatial_moment(field, 4),
        }
        one_time_moments[name] = {
            key: {
                "mean": float(mean_sem(values)[0]),
                "sem": float(mean_sem(values)[1]),
            }
            for key, values in moment_values[name].items()
        }

    endpoint_comparisons = (
        ("t10_dm_vs_fixed", "dm10", "fixed10"),
        ("t20_dm_vs_fixed", "dm20", "fixed20"),
        (
            "second_kernel_dm20_vs_fixed_from_dm10",
            "dm20",
            "fixed20_from_dm10",
        ),
    )
    paired_moments = {
        label: {
            observable: paired_summary(
                moment_values[dm_name][observable],
                moment_values[physical_name][observable],
            )
            for observable in ("phi2", "phi4")
        }
        for label, dm_name, physical_name in endpoint_comparisons
    }

    time_comparisons = (
        ("0_10", ("hmc", "dm10"), ("hmc", "fixed10")),
        ("0_20", ("hmc", "dm20"), ("hmc", "fixed20")),
        (
            "10_20_chain",
            ("dm10", "dm20"),
            ("fixed10", "fixed20"),
        ),
        (
            "10_20_same_dm10",
            ("dm10", "dm20"),
            ("dm10", "fixed20_from_dm10"),
        ),
    )
    scalar_correlations = {
        label: {
            kind: compare_scalar_pairs(
                fields, stats, dm_pair, physical_pair, kind
            )
            for kind in ("raw", "connected")
        }
        for label, dm_pair, physical_pair in time_comparisons
    }

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    structure_path = output_prefix.with_name(
        output_prefix.name + "_structure.tsv"
    )
    momentum_path = output_prefix.with_name(
        output_prefix.name + "_timecorr_momentum.tsv"
    )
    write_structure_table(
        structure_path, endpoint_comparisons, stats, layout
    )
    write_momentum_table(momentum_path, time_comparisons, stats, layout)

    document = {
        "inputs": {
            "dm10": str(args.dm10.resolve()),
            "dm20": str(args.dm20.resolve()),
            "hmc": str(args.hmc.resolve()),
            "fixed10": str(args.fixed10.resolve()),
            "fixed20": str(args.fixed20.resolve()),
            "fixed20_from_dm10": str(args.fixed20_from_dm10.resolve()),
        },
        "nensemble": nensemble,
        "L": length,
        "definitions": {
            "M": "spatial mean(phi)",
            "phi2_phi4": "ensemble mean of per-field spatial moments",
            "connected": "the spatial mean is removed separately per field",
            "rho": "<F>/sqrt(<G_endpoint><G_start>)",
            "momentum_shells": "0 < |n| <= L/2; corners and k=0 excluded",
            "errors": "SEM; rho differences use paired delete-one jackknife",
        },
        "conservation": conservation,
        "one_time_moments": one_time_moments,
        "paired_moment_differences": paired_moments,
        "scalar_time_correlations": scalar_correlations,
        "tables": {
            "structure": str(structure_path),
            "momentum_time_correlations": str(momentum_path),
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    print(f"saved {json_path}")
    print(f"saved {structure_path}")
    print(f"saved {momentum_path}")


if __name__ == "__main__":
    main()
