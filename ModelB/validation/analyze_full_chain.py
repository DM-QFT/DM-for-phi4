"""Analyze the paired L32 DM and physical chains at t=0,10,...,100.

The script writes tables and one JSON summary; it deliberately makes no
plots.  Every delete-one error treats a trajectory (not a time origin or a
Fourier mode) as the statistical unit.
"""

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
TIMES = tuple(range(0, 101, 10))
DM_NAMES = {
    10: "chain_ep2257_t10_from_hmc5120_steps2000_log_seed1234.npz",
    20: "chain_ep2257_t20_via_t10dm5120_steps2000_log_seed5678.npz",
    **{
        time: (
            f"chain_ep2257_t{time}_via_iterated_t10dm5120_"
            f"steps2000_log_seed{seed}.npz"
        )
        for time, seed in zip(range(30, 101, 10), range(9012, 9020))
    },
}
PHYSICAL_NAMES = {
    20: "modelB_L32_Ne5120_true_t10_to_t20_dt0.002_seed20260725.jld2",
    **{
        time: (
            f"modelB_L32_Ne5120_true_t{time}_chain_"
            "dt0.002_seed20260727.jld2"
        )
        for time in range(30, 101, 10)
    },
}


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    parser.add_argument(
        "--hmc",
        type=Path,
        default=MODEL_B_DIR.parent
        / "2Dphi4/trainingdata/cfgs_wolff_fahmc_k=0.28_l=0.022_32^2.jld2",
    )
    parser.add_argument(
        "--true10",
        type=Path,
        default=MODEL_B_DIR
        / "runs/L32_Ne10240_dt0p002_hmc_t1_t10_t100"
        / "modelB_L32_Ne10240_hmc_kappa_0.28_t10_dt0.002.jld2",
    )
    parser.add_argument(
        "--original_true100",
        type=Path,
        default=MODEL_B_DIR
        / "runs/L32_Ne10240_dt0p002_hmc_t1_t10_t100"
        / "modelB_L32_Ne10240_hmc_kappa_0.28_t100_dt0.002.jld2",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        help="Default: RUN_DIR/validation/chain_ep2257_full_t0_t100_5120",
    )
    parser.add_argument("--fft_batch", type=positive_int, default=128)
    parser.add_argument("--fft_workers", type=int, default=8)
    return parser


def input_paths(args):
    sample_dir = args.run_dir / "samples"
    physical_dir = args.run_dir / "physical_reference"
    dm = {time: sample_dir / name for time, name in DM_NAMES.items()}
    physical = {
        10: args.true10,
        **{
            time: physical_dir / name
            for time, name in PHYSICAL_NAMES.items()
        },
    }
    required = [args.hmc, args.original_true100, *dm.values(), *physical.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "full-chain inputs are not ready:\n" + "\n".join(missing)
        )
    return dm, physical


def load_dm(path):
    with np.load(path, allow_pickle=False) as archive:
        fields = np.asarray(archive["phi_gen"], dtype=np.float32)
        key = "cond_indices" if "cond_indices" in archive.files else "index"
        indices = np.asarray(archive[key], dtype=np.int64)
    return fields, indices


def load_cfgs(path, rows=None):
    with h5py.File(path, "r") as handle:
        dataset = handle["cfgs"]
        fields = (
            np.asarray(dataset[:], dtype=np.float32)
            if rows is None
            else read_rows(dataset, np.asarray(rows, dtype=np.int64))
        )
        indices = (
            np.asarray(handle["cond_indices_0based"][:], dtype=np.int64)
            if "cond_indices_0based" in handle
            else None
        )
    return fields, indices


def prepare_field(fields, layout, fft_batch, fft_workers, keep_fft=True):
    """Compute per-trajectory observables and center ``fields`` in place."""
    fields = np.asarray(fields, dtype=np.float32)
    nensemble, length, length_y = fields.shape
    if length != length_y:
        raise ValueError(f"fields must be square, got {fields.shape}")
    volume = length * length
    mean = np.mean(fields, axis=(1, 2), dtype=np.float64)
    phi2 = np.mean(fields.astype(np.float64) ** 2, axis=(1, 2))
    phi4 = np.mean(fields.astype(np.float64) ** 4, axis=(1, 2))
    fields -= mean[:, None, None]
    connected_g = np.einsum(
        "nij,nij->n", fields, fields, dtype=np.float64, optimize=True
    ) / volume

    _, pixels, starts, counts, _, _ = layout
    transforms = (
        np.empty(fields.shape, dtype=np.complex64) if keep_fft else None
    )
    spectrum = np.empty((nensemble, len(counts)), dtype=np.float64)
    for start in range(0, nensemble, fft_batch):
        stop = min(start + fft_batch, nensemble)
        transformed = fft.fft2(
            fields[start:stop], axes=(-2, -1), workers=fft_workers
        )
        power = (transformed.real**2 + transformed.imag**2) / volume
        spectrum[start:stop] = shell_mean(
            power, pixels, starts, counts
        )
        if keep_fft:
            transforms[start:stop] = transformed
    return {
        "field": fields,
        "mean": mean,
        "phi2": phi2,
        "phi4": phi4,
        "connected_g": connected_g,
        "raw_g": connected_g + mean**2,
        "fft": transforms,
        "spectrum": spectrum,
    }


def value_summary(values):
    values = np.asarray(values, dtype=np.float64)
    mean, sem = mean_sem(values)
    absolute = np.abs(values)
    return {
        "mean": float(mean),
        "sem": float(sem),
        "rms": float(np.sqrt(np.mean(values**2))),
        "mean_abs": float(np.mean(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(np.max(absolute)),
    }


def divide_or_none(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.ndim or denominator.ndim:
        result = np.full(np.broadcast_shapes(
            numerator.shape, denominator.shape
        ), np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator != 0)
        return result
    if denominator == 0:
        return None
    return float(numerator / denominator)


def scalar_or_array(value):
    value = np.asarray(value, dtype=np.float64)
    return float(value) if value.ndim == 0 else value


def finite_float_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def paired_mean_summary(dm_values, physical_values):
    dm_mean, dm_sem = mean_sem(dm_values)
    physical_mean, physical_sem = mean_sem(physical_values)
    difference, difference_sem = mean_sem(
        np.asarray(dm_values, dtype=np.float64)
        - np.asarray(physical_values, dtype=np.float64)
    )
    return {
        "dm": scalar_or_array(dm_mean),
        "dm_sem": scalar_or_array(dm_sem),
        "physical": scalar_or_array(physical_mean),
        "physical_sem": scalar_or_array(physical_sem),
        "dm_minus_physical": scalar_or_array(difference),
        "paired_sem": scalar_or_array(difference_sem),
        "z": divide_or_none(difference, difference_sem),
        "relative_difference": divide_or_none(difference, physical_mean),
    }


def descriptive_reference_summary(values, reference):
    mean, sem = mean_sem(values)
    reference_mean, reference_sem = mean_sem(reference)
    difference = mean - reference_mean
    return {
        "value": scalar_or_array(mean),
        "value_sem": scalar_or_array(sem),
        "reference": scalar_or_array(reference_mean),
        "reference_sem": scalar_or_array(reference_sem),
        "value_minus_reference": scalar_or_array(difference),
        "relative_difference": divide_or_none(difference, reference_mean),
    }


def scalar_components(start, endpoint, kind):
    volume = start["field"].shape[-1] ** 2
    cross = np.einsum(
        "nij,nij->n",
        endpoint["field"],
        start["field"],
        dtype=np.float64,
        optimize=True,
    ) / volume
    if kind == "raw":
        cross = cross + endpoint["mean"] * start["mean"]
        return cross, endpoint["raw_g"], start["raw_g"]
    return cross, endpoint["connected_g"], start["connected_g"]


def scalar_comparison(dm_values, physical_values):
    dm_summary, dm_leave = scalar_summary(*dm_values)
    physical_summary, physical_leave = scalar_summary(*physical_values)
    c_difference, c_sem = mean_sem(dm_values[0] - physical_values[0])
    rho_difference = dm_summary["rho"] - physical_summary["rho"]
    rho_sem = paired_jackknife_sem(dm_leave, physical_leave)
    return {
        "dm": dm_summary,
        "physical": physical_summary,
        "C_dm_minus_physical": float(c_difference),
        "C_difference_paired_jackknife_sem": float(c_sem),
        "C_difference_z": divide_or_none(c_difference, c_sem),
        "rho_dm_minus_physical": float(rho_difference),
        "rho_difference_paired_jackknife_sem": float(rho_sem),
        "rho_difference_z": divide_or_none(rho_difference, rho_sem),
    }


def scalar_pair(stats, start_time, endpoint_time, kind):
    return scalar_components(stats[start_time], stats[endpoint_time], kind)


def scalar_lag(stats, lag, kind):
    origins = TIMES[: len(TIMES) - lag // 10]
    pieces = [scalar_pair(stats, time, time + lag, kind) for time in origins]
    return tuple(np.mean([piece[i] for piece in pieces], axis=0) for i in range(3))


def momentum_components(start, endpoint, layout):
    _, pixels, starts, counts, _, _ = layout
    volume = start["fft"].shape[-1] ** 2
    cross_grid = (
        endpoint["fft"].real * start["fft"].real
        + endpoint["fft"].imag * start["fft"].imag
    ) / volume
    cross = shell_mean(cross_grid, pixels, starts, counts)
    return cross, endpoint["spectrum"], start["spectrum"]


def momentum_pair(stats, start_time, endpoint_time, layout):
    return momentum_components(
        stats[start_time], stats[endpoint_time], layout
    )


def momentum_lag(stats, lag, layout):
    origins = TIMES[: len(TIMES) - lag // 10]
    pieces = [
        momentum_pair(stats, time, time + lag, layout)
        for time in origins
    ]
    return tuple(np.mean([piece[i] for piece in pieces], axis=0) for i in range(3))


def correlation_array_comparison(dm_values, physical_values):
    result = {}
    for name, dm_array, physical_array in (
        ("F", dm_values[0], physical_values[0]),
        ("G_endpoint", dm_values[1], physical_values[1]),
        ("G_start", dm_values[2], physical_values[2]),
    ):
        dm_mean, dm_sem = mean_sem(dm_array)
        physical_mean, physical_sem = mean_sem(physical_array)
        difference, difference_sem = mean_sem(dm_array - physical_array)
        result[name] = (
            dm_mean, dm_sem, physical_mean, physical_sem,
            difference, difference_sem,
        )
    dm_rho, dm_rho_sem, dm_leave = ratio_with_jackknife(*dm_values)
    physical_rho, physical_rho_sem, physical_leave = ratio_with_jackknife(
        *physical_values
    )
    difference = dm_rho - physical_rho
    difference_sem = paired_jackknife_sem(dm_leave, physical_leave)
    result["rho"] = (
        dm_rho, dm_rho_sem, physical_rho, physical_rho_sem,
        difference, difference_sem,
    )
    return result


def format_value(value):
    if value is None:
        return "nan"
    return f"{float(value):.10e}"


def write_one_time(path, stats, original, cond_indices):
    columns = (
        "comparison", "time", "observable", "dm", "dm_sem",
        "physical", "physical_sem", "difference", "difference_sem",
        "z", "relative_difference",
    )
    rows = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for time in TIMES:
            for observable in ("phi2", "phi4"):
                summary = paired_mean_summary(
                    stats["dm"][time][observable],
                    stats["physical"][time][observable],
                )
                row = {
                    "comparison": "dm_vs_physical",
                    "time": time,
                    "observable": observable,
                    **summary,
                }
                rows.append(row)
                writer.writerow((
                    row["comparison"], time, observable,
                    *[format_value(summary[key]) for key in (
                        "dm", "dm_sem", "physical", "physical_sem",
                        "dm_minus_physical", "paired_sem", "z",
                        "relative_difference",
                    )],
                ))
        for observable in ("phi2", "phi4"):
            paired = paired_mean_summary(
                stats["physical"][100][observable],
                original[observable][cond_indices],
            )
            writer.writerow((
                "physical_chain_vs_original_true100_paired", 100, observable,
                *[format_value(paired[key]) for key in (
                    "dm", "dm_sem", "physical", "physical_sem",
                    "dm_minus_physical", "paired_sem", "z",
                    "relative_difference",
                )],
            ))
            full = descriptive_reference_summary(
                stats["physical"][100][observable], original[observable]
            )
            writer.writerow((
                "physical_chain_vs_original_true100_full_descriptive",
                100, observable,
                *[format_value(full[key]) for key in (
                    "value", "value_sem", "reference", "reference_sem",
                    "value_minus_reference",
                )],
                "nan", "nan", format_value(full["relative_difference"]),
            ))
    return rows


def write_structure(path, stats, original, cond_indices, layout):
    shell_ids, _, _, counts, k, khat = layout
    columns = (
        "comparison", "time", "shell", "mode_count", "k", "khat",
        "S_dm", "S_dm_sem", "S_physical", "S_physical_sem",
        "delta_S", "delta_S_sem", "z", "relative_difference",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for time in TIMES:
            summary = paired_mean_summary(
                stats["dm"][time]["spectrum"],
                stats["physical"][time]["spectrum"],
            )
            for i, shell in enumerate(shell_ids):
                writer.writerow((
                    "dm_vs_physical", time, int(shell), int(counts[i]),
                    format_value(k[i]), format_value(khat[i]),
                    *[format_value(summary[key][i] if isinstance(
                        summary[key], np.ndarray
                    ) else summary[key]) for key in (
                        "dm", "dm_sem", "physical", "physical_sem",
                        "dm_minus_physical", "paired_sem", "z",
                        "relative_difference",
                    )],
                ))

        paired = paired_mean_summary(
            stats["physical"][100]["spectrum"],
            original["spectrum"][cond_indices],
        )
        full = descriptive_reference_summary(
            stats["physical"][100]["spectrum"], original["spectrum"]
        )
        for label, summary, keys, inferential in (
            (
                "physical_chain_vs_original_true100_paired", paired,
                ("dm", "dm_sem", "physical", "physical_sem",
                 "dm_minus_physical", "paired_sem", "z",
                 "relative_difference"), True,
            ),
            (
                "physical_chain_vs_original_true100_full_descriptive", full,
                ("value", "value_sem", "reference", "reference_sem",
                 "value_minus_reference"), False,
            ),
        ):
            for i, shell in enumerate(shell_ids):
                values = [format_value(summary[key][i]) for key in keys]
                if not inferential:
                    values.extend(("nan", "nan"))
                    values.append(format_value(summary["relative_difference"][i]))
                writer.writerow((
                    label, 100, int(shell), int(counts[i]),
                    format_value(k[i]), format_value(khat[i]),
                    *values,
                ))


def scalar_row(label, start, endpoint, kind, result):
    dm = result["dm"]
    physical = result["physical"]
    return (
        label, start, endpoint,
        endpoint - start if isinstance(start, int) else None,
        kind,
        dm["C"], dm["C_sem"], physical["C"], physical["C_sem"],
        result["C_dm_minus_physical"],
        result["C_difference_paired_jackknife_sem"],
        result["C_difference_z"],
        dm["rho"], dm["rho_jackknife_sem"],
        physical["rho"], physical["rho_jackknife_sem"],
        result["rho_dm_minus_physical"],
        result["rho_difference_paired_jackknife_sem"],
        result["rho_difference_z"],
    )


def write_scalar_tables(pair_path, lag_path, stats):
    columns = (
        "aggregation", "start", "endpoint", "lag", "kind",
        "C_dm", "C_dm_sem", "C_physical", "C_physical_sem",
        "C_difference", "C_difference_paired_jackknife_sem", "C_z",
        "rho_dm", "rho_dm_jackknife_sem", "rho_physical",
        "rho_physical_jackknife_sem", "rho_difference",
        "rho_difference_paired_jackknife_sem", "rho_z",
    )
    pair_results = []
    pair_by_lag = {lag: [] for lag in TIMES[1:]}
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for i, start in enumerate(TIMES[:-1]):
            for endpoint in TIMES[i + 1:]:
                for kind in ("raw", "connected"):
                    result = scalar_comparison(
                        scalar_pair(stats["dm"], start, endpoint, kind),
                        scalar_pair(stats["physical"], start, endpoint, kind),
                    )
                    writer.writerow([
                        format_value(value) if index >= 5 else value
                        for index, value in enumerate(scalar_row(
                            "specific_pair", start, endpoint, kind, result
                        ))
                    ])
                    record = {
                        "start": start, "endpoint": endpoint,
                        "lag": endpoint - start, "kind": kind, **result,
                    }
                    pair_results.append(record)
                    pair_by_lag[endpoint - start].append(record)

    lag_results = []
    lag_columns = columns + (
        "n_time_origins", "C_dm_origin_sd", "C_physical_origin_sd",
        "rho_dm_origin_sd", "rho_physical_origin_sd",
    )
    with lag_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(lag_columns)
        for lag in TIMES[1:]:
            for kind in ("raw", "connected"):
                result = scalar_comparison(
                    scalar_lag(stats["dm"], lag, kind),
                    scalar_lag(stats["physical"], lag, kind),
                )
                origin = [
                    row for row in pair_by_lag[lag] if row["kind"] == kind
                ]
                spreads = (
                    np.std([row["dm"]["C"] for row in origin]),
                    np.std([row["physical"]["C"] for row in origin]),
                    np.std([row["dm"]["rho"] for row in origin]),
                    np.std([row["physical"]["rho"] for row in origin]),
                )
                base = scalar_row(
                    "lag_average", "all", "all", kind, result
                )
                base = base[:3] + (lag,) + base[4:]
                writer.writerow([
                    format_value(value) if index >= 5 else value
                    for index, value in enumerate(base)
                ] + [len(origin), *[format_value(value) for value in spreads]])
                lag_results.append({
                    "lag": lag, "kind": kind,
                    "n_time_origins": len(origin),
                    "C_dm_origin_sd": float(spreads[0]),
                    "C_physical_origin_sd": float(spreads[1]),
                    "rho_dm_origin_sd": float(spreads[2]),
                    "rho_physical_origin_sd": float(spreads[3]),
                    **result,
                })
    return pair_results, lag_results


def write_momentum_table(path, stats, layout, lag_averaged):
    shell_ids, _, _, counts, k, khat = layout
    quantity_keys = (
        "dm", "dm_sem", "physical", "physical_sem",
        "difference", "difference_sem",
    )
    columns = [
        "aggregation", "start", "endpoint", "lag", "n_time_origins",
        "shell", "mode_count", "k", "khat",
    ]
    for quantity in ("G_start", "G_endpoint", "F", "rho"):
        columns.extend(f"{quantity}_{key}" for key in quantity_keys)
    columns.append("rho_difference_z")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        if lag_averaged:
            pairs = [(None, None, lag) for lag in TIMES[1:]]
        else:
            pairs = [
                (start, endpoint, endpoint - start)
                for i, start in enumerate(TIMES[:-1])
                for endpoint in TIMES[i + 1:]
            ]
        for start, endpoint, lag in pairs:
            if lag_averaged:
                dm_values = momentum_lag(stats["dm"], lag, layout)
                physical_values = momentum_lag(
                    stats["physical"], lag, layout
                )
                n_origins = len(TIMES) - lag // 10
                label, start_text, endpoint_text = "lag_average", "all", "all"
            else:
                dm_values = momentum_pair(
                    stats["dm"], start, endpoint, layout
                )
                physical_values = momentum_pair(
                    stats["physical"], start, endpoint, layout
                )
                n_origins = 1
                label, start_text, endpoint_text = (
                    "specific_pair", start, endpoint
                )
            result = correlation_array_comparison(dm_values, physical_values)
            for shell_index, shell in enumerate(shell_ids):
                row = [
                    label, start_text, endpoint_text, lag, n_origins,
                    int(shell), int(counts[shell_index]),
                    format_value(k[shell_index]),
                    format_value(khat[shell_index]),
                ]
                for quantity in ("G_start", "G_endpoint", "F", "rho"):
                    row.extend(
                        format_value(values[shell_index])
                        for values in result[quantity]
                    )
                row.append(format_value(
                    result["rho"][4][shell_index]
                    / result["rho"][5][shell_index]
                ))
                writer.writerow(row)


def conservation_rows(stats):
    rows = []
    spans = [
        ("segment", start, endpoint)
        for start, endpoint in zip(TIMES[:-1], TIMES[1:])
    ] + [("cumulative", 0, endpoint) for endpoint in TIMES[1:]]
    for span, start, endpoint in spans:
        dm = stats["dm"][endpoint]["mean"] - stats["dm"][start]["mean"]
        physical = (
            stats["physical"][endpoint]["mean"]
            - stats["physical"][start]["mean"]
        )
        rows.extend((
            {"span": span, "start": start, "endpoint": endpoint,
             "chain": "dm", **value_summary(dm)},
            {"span": span, "start": start, "endpoint": endpoint,
             "chain": "physical", **value_summary(physical)},
            {"span": span, "start": start, "endpoint": endpoint,
             "chain": "dm_minus_physical", **value_summary(dm - physical)},
        ))
    return rows


def write_conservation(path, rows):
    columns = (
        "span", "start", "endpoint", "chain", "mean", "sem", "rms",
        "mean_abs", "p95_abs", "p99_abs", "max_abs",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([
                row[key] if key in ("span", "start", "endpoint", "chain")
                else format_value(row[key])
                for key in columns
            ])


def accumulation_rows(stats, layout):
    counts = layout[3]
    rows = []
    for time in TIMES:
        phi2 = paired_mean_summary(
            stats["dm"][time]["phi2"], stats["physical"][time]["phi2"]
        )
        phi4 = paired_mean_summary(
            stats["dm"][time]["phi4"], stats["physical"][time]["phi4"]
        )
        physical_s = mean_sem(stats["physical"][time]["spectrum"])[0]
        delta_s, delta_s_sem = mean_sem(
            stats["dm"][time]["spectrum"]
            - stats["physical"][time]["spectrum"]
        )
        weighted_relative_l2 = np.sqrt(
            np.sum(counts * delta_s**2) / np.sum(counts * physical_s**2)
        )
        relative_s = delta_s / physical_s
        structure_z = np.zeros_like(delta_s)
        np.divide(
            delta_s, delta_s_sem, out=structure_z,
            where=delta_s_sem != 0,
        )
        max_structure_z = np.max(np.abs(structure_z))
        dm_delta_m = stats["dm"][time]["mean"] - stats["dm"][0]["mean"]
        physical_delta_m = (
            stats["physical"][time]["mean"]
            - stats["physical"][0]["mean"]
        )
        row = {
            "time": time,
            "phi2_relative_difference": phi2["relative_difference"],
            "phi2_z": phi2["z"],
            "phi4_relative_difference": phi4["relative_difference"],
            "phi4_z": phi4["z"],
            "structure_weighted_relative_l2": float(weighted_relative_l2),
            "structure_mean_abs_relative": float(np.mean(np.abs(relative_s))),
            "structure_max_abs_relative": float(np.max(np.abs(relative_s))),
            "structure_max_abs_z": float(max_structure_z),
            "dm_cumulative_delta_M_rms": float(np.sqrt(np.mean(dm_delta_m**2))),
            "physical_cumulative_delta_M_rms": float(
                np.sqrt(np.mean(physical_delta_m**2))
            ),
            "delta_M_chain_difference_rms": float(
                np.sqrt(np.mean((dm_delta_m - physical_delta_m) ** 2))
            ),
        }
        if time == 0:
            row.update({
                "rho_raw_0_t_difference": None,
                "rho_raw_0_t_z": None,
                "rho_connected_0_t_difference": None,
                "rho_connected_0_t_z": None,
            })
        else:
            for kind in ("raw", "connected"):
                result = scalar_comparison(
                    scalar_pair(stats["dm"], 0, time, kind),
                    scalar_pair(stats["physical"], 0, time, kind),
                )
                row[f"rho_{kind}_0_t_difference"] = result[
                    "rho_dm_minus_physical"
                ]
                row[f"rho_{kind}_0_t_z"] = result["rho_difference_z"]
        rows.append(row)
    return rows


def write_accumulation(path, rows):
    columns = tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([
                row[key] if key == "time" else format_value(row[key])
                for key in columns
            ])


def static_summary(stats, original, cond_indices, layout):
    result = {"paired_first_5120": {}, "full_10240_reference": {}}
    for observable in ("phi2", "phi4"):
        result["paired_first_5120"][observable] = paired_mean_summary(
            stats["physical"][100][observable],
            original[observable][cond_indices],
        )
        result["full_10240_reference"][observable] = (
            descriptive_reference_summary(
            stats["physical"][100][observable], original[observable]
            )
        )
    counts = layout[3]
    chain_s = mean_sem(stats["physical"][100]["spectrum"])[0]
    original_s = mean_sem(original["spectrum"])[0]
    paired_delta = mean_sem(
        stats["physical"][100]["spectrum"]
        - original["spectrum"][cond_indices]
    )[0]
    paired_reference = mean_sem(original["spectrum"][cond_indices])[0]
    paired_relative = paired_delta / paired_reference
    result["paired_first_5120"]["structure_weighted_relative_l2"] = float(
        np.sqrt(np.sum(counts * paired_delta**2) / np.sum(
            counts * mean_sem(original["spectrum"][cond_indices])[0] ** 2
        ))
    )
    result["paired_first_5120"]["structure_mean_abs_relative"] = float(
        np.mean(np.abs(paired_relative))
    )
    result["paired_first_5120"]["structure_max_abs_relative"] = float(
        np.max(np.abs(paired_relative))
    )
    full_relative = (chain_s - original_s) / original_s
    result["full_10240_reference"]["structure_weighted_relative_l2"] = float(
        np.sqrt(np.sum(counts * (chain_s - original_s) ** 2) / np.sum(
            counts * original_s**2
        ))
    )
    result["full_10240_reference"]["structure_mean_abs_relative"] = float(
        np.mean(np.abs(full_relative))
    )
    result["full_10240_reference"]["structure_max_abs_relative"] = float(
        np.max(np.abs(full_relative))
    )
    return result


def write_t100_scalar_cross_validation(path, stats, original):
    endpoints = {
        "dm": stats["dm"][100],
        "new_physical": stats["physical"][100],
        "original_true100": original,
    }
    pairs = (
        ("dm", "new_physical"),
        ("dm", "original_true100"),
        ("new_physical", "original_true100"),
    )
    columns = (
        "kind", "record", "left", "right", "C", "C_sem", "rho",
        "rho_jackknife_sem", "C_difference",
        "C_difference_paired_jackknife_sem", "C_z", "rho_difference",
        "rho_difference_paired_jackknife_sem", "rho_z",
    )
    document = {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for kind in ("raw", "connected"):
            values = {
                name: scalar_components(stats["dm"][0], endpoint, kind)
                for name, endpoint in endpoints.items()
            }
            summaries = {
                name: scalar_summary(*item)[0] for name, item in values.items()
            }
            comparisons = {
                f"{left}_minus_{right}": scalar_comparison(
                    values[left], values[right]
                )
                for left, right in pairs
            }
            document[kind] = {
                "endpoints": summaries,
                "pairwise": comparisons,
            }
            for name, summary in summaries.items():
                writer.writerow((
                    kind, "endpoint", name, "", summary["C"],
                    summary["C_sem"], summary["rho"],
                    summary["rho_jackknife_sem"],
                    *("nan",) * 6,
                ))
            for left, right in pairs:
                result = comparisons[f"{left}_minus_{right}"]
                writer.writerow((
                    kind, "pairwise_difference", left, right,
                    *("nan",) * 4,
                    result["C_dm_minus_physical"],
                    result["C_difference_paired_jackknife_sem"],
                    format_value(result["C_difference_z"]),
                    result["rho_dm_minus_physical"],
                    result["rho_difference_paired_jackknife_sem"],
                    format_value(result["rho_difference_z"]),
                ))
    return document


def momentum_endpoint_summary(values):
    cross, endpoint_power, start_power = values
    f_mean, f_sem = mean_sem(cross)
    endpoint_mean, endpoint_sem = mean_sem(endpoint_power)
    start_mean, start_sem = mean_sem(start_power)
    rho, rho_sem, leave = ratio_with_jackknife(*values)
    return ({
        "G_start": start_mean,
        "G_start_sem": start_sem,
        "G_endpoint": endpoint_mean,
        "G_endpoint_sem": endpoint_sem,
        "F": f_mean,
        "F_sem": f_sem,
        "rho": rho,
        "rho_jackknife_sem": rho_sem,
    }, leave)


def momentum_pairwise_summary(left, right, left_summary, right_summary,
                              left_leave, right_leave):
    delta_g, delta_g_sem = mean_sem(left[1] - right[1])
    delta_f, delta_f_sem = mean_sem(left[0] - right[0])
    delta_rho = left_summary["rho"] - right_summary["rho"]
    delta_rho_sem = paired_jackknife_sem(left_leave, right_leave)
    return {
        "G_endpoint_difference": delta_g,
        "G_endpoint_difference_paired_sem": delta_g_sem,
        "F_difference": delta_f,
        "F_difference_paired_sem": delta_f_sem,
        "rho_difference": delta_rho,
        "rho_difference_paired_jackknife_sem": delta_rho_sem,
        "rho_difference_z": divide_or_none(delta_rho, delta_rho_sem),
    }


def write_t100_momentum_cross_validation(path, stats, original, layout):
    shell_ids, _, _, counts, k, khat = layout
    endpoints = {
        "dm": stats["dm"][100],
        "new_physical": stats["physical"][100],
        "original_true100": original,
    }
    pairs = (
        ("dm", "new_physical"),
        ("dm", "original_true100"),
        ("new_physical", "original_true100"),
    )
    values = {
        name: momentum_components(stats["dm"][0], endpoint, layout)
        for name, endpoint in endpoints.items()
    }
    summaries = {}
    leaves = {}
    for name, item in values.items():
        summaries[name], leaves[name] = momentum_endpoint_summary(item)
    comparisons = {
        f"{left}_minus_{right}": momentum_pairwise_summary(
            values[left], values[right], summaries[left], summaries[right],
            leaves[left], leaves[right],
        )
        for left, right in pairs
    }

    columns = (
        "record", "left", "right", "shell", "mode_count", "k", "khat",
        "G_start", "G_start_sem", "G_endpoint", "G_endpoint_sem", "F",
        "F_sem", "rho", "rho_jackknife_sem", "G_endpoint_difference",
        "G_endpoint_difference_paired_sem", "F_difference",
        "F_difference_paired_sem", "rho_difference",
        "rho_difference_paired_jackknife_sem", "rho_z",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for name, summary in summaries.items():
            for i, shell in enumerate(shell_ids):
                writer.writerow((
                    "endpoint", name, "", int(shell), int(counts[i]),
                    k[i], khat[i],
                    *[summary[key][i] for key in (
                        "G_start", "G_start_sem", "G_endpoint",
                        "G_endpoint_sem", "F", "F_sem", "rho",
                        "rho_jackknife_sem",
                    )],
                    *("nan",) * 7,
                ))
        for left, right in pairs:
            comparison = comparisons[f"{left}_minus_{right}"]
            for i, shell in enumerate(shell_ids):
                writer.writerow((
                    "pairwise_difference", left, right, int(shell),
                    int(counts[i]), k[i], khat[i], *("nan",) * 8,
                    *[comparison[key][i] for key in (
                        "G_endpoint_difference",
                        "G_endpoint_difference_paired_sem",
                        "F_difference", "F_difference_paired_sem",
                        "rho_difference",
                        "rho_difference_paired_jackknife_sem",
                        "rho_difference_z",
                    )],
                ))

    shell1 = {
        "endpoints": {
            name: {
                key: finite_float_or_none(summary[key][0])
                for key in ("rho", "rho_jackknife_sem", "F", "F_sem")
            }
            for name, summary in summaries.items()
        },
        "pairwise": {
            name: {
                key: finite_float_or_none(comparison[key][0])
                for key in (
                    "rho_difference",
                    "rho_difference_paired_jackknife_sem",
                    "rho_difference_z",
                )
            }
            for name, comparison in comparisons.items()
        },
    }
    maximum = {}
    for name, comparison in comparisons.items():
        z = comparison["rho_difference_z"]
        index = int(np.nanargmax(np.abs(z))) if np.isfinite(z).any() else 0
        maximum[name] = {
            "shell": int(shell_ids[index]),
            "rho_difference": finite_float_or_none(
                comparison["rho_difference"][index]
            ),
            "rho_difference_z": finite_float_or_none(z[index]),
        }
    return {"shell1": shell1, "max_abs_rho_difference_z": maximum}


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.run_dir = args.run_dir.expanduser().resolve()
    args.hmc = args.hmc.expanduser().resolve()
    args.true10 = args.true10.expanduser().resolve()
    args.original_true100 = args.original_true100.expanduser().resolve()
    dm_paths, physical_paths = input_paths(args)

    prefix = (
        args.output_prefix.expanduser().resolve()
        if args.output_prefix
        else args.run_dir / "validation/chain_ep2257_full_t0_t100_5120"
    )
    paths = {
        "json": prefix.with_suffix(".json"),
        "one_time": prefix.with_name(prefix.name + "_one_time.tsv"),
        "structure": prefix.with_name(prefix.name + "_structure.tsv"),
        "conservation": prefix.with_name(prefix.name + "_conservation.tsv"),
        "scalar_pairs": prefix.with_name(prefix.name + "_scalar_pairs.tsv"),
        "scalar_lag": prefix.with_name(prefix.name + "_scalar_lag.tsv"),
        "momentum_pairs": prefix.with_name(prefix.name + "_momentum_pairs.tsv"),
        "momentum_lag": prefix.with_name(prefix.name + "_momentum_lag.tsv"),
        "accumulation": prefix.with_name(prefix.name + "_accumulation.tsv"),
        "t100_scalar": prefix.with_name(
            prefix.name + "_t100_timecorr_scalar.tsv"
        ),
        "t100_momentum": prefix.with_name(
            prefix.name + "_t100_timecorr_momentum.tsv"
        ),
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite:\n" + "\n".join(existing))
    prefix.parent.mkdir(parents=True, exist_ok=True)

    dm_fields = {}
    dm_indices = {}
    for time, path in dm_paths.items():
        dm_fields[time], dm_indices[time] = load_dm(path)
    cond_indices = dm_indices[10]
    nensemble, length, length_y = dm_fields[10].shape
    expected_order = np.arange(nensemble, dtype=np.int64)
    if length != length_y or not np.array_equal(cond_indices, expected_order):
        raise ValueError("the full chain expects LxL fields and HMC rows 0:5120")
    for time in TIMES[1:]:
        if (dm_fields[time].shape != dm_fields[10].shape
                or not np.array_equal(dm_indices[time], expected_order)):
            raise ValueError(f"DM t={time} breaks the paired row order")

    initial, _ = load_cfgs(args.hmc, cond_indices)
    physical_fields = {0: initial}
    dm_fields[0] = initial
    physical_fields[10], _ = load_cfgs(args.true10, cond_indices)
    for time in TIMES[2:]:
        physical_fields[time], indices = load_cfgs(physical_paths[time])
        if not np.array_equal(indices, expected_order):
            raise ValueError(f"physical t={time} breaks the paired row order")
    for chain, fields in (("dm", dm_fields), ("physical", physical_fields)):
        for time in TIMES:
            if fields[time].shape != (nensemble, length, length):
                raise ValueError(f"{chain} t={time} has shape {fields[time].shape}")

    layout = radial_layout(length)
    stats = {"dm": {}, "physical": {}}
    print("FFT common t=0", flush=True)
    initial_stats = prepare_field(
        initial, layout, args.fft_batch, args.fft_workers
    )
    stats["dm"][0] = initial_stats
    stats["physical"][0] = initial_stats
    for chain, fields in (("dm", dm_fields), ("physical", physical_fields)):
        for time in TIMES[1:]:
            print(f"FFT {chain} t={time}", flush=True)
            stats[chain][time] = prepare_field(
                fields[time], layout, args.fft_batch, args.fft_workers
            )

    print("FFT original true t=100 static reference", flush=True)
    original_fields, _ = load_cfgs(args.original_true100)
    original = prepare_field(
        original_fields, layout, args.fft_batch, args.fft_workers
    )
    original_paired = {
        key: value[cond_indices]
        for key, value in original.items()
        if isinstance(value, np.ndarray)
    }

    one_time = write_one_time(
        paths["one_time"], stats, original, cond_indices
    )
    write_structure(paths["structure"], stats, original, cond_indices, layout)
    conservation = conservation_rows(stats)
    write_conservation(paths["conservation"], conservation)
    scalar_pairs, scalar_lag_results = write_scalar_tables(
        paths["scalar_pairs"], paths["scalar_lag"], stats
    )
    print("momentum correlations: 55 specific pairs", flush=True)
    write_momentum_table(
        paths["momentum_pairs"], stats, layout, lag_averaged=False
    )
    print("momentum correlations: 10 lag averages", flush=True)
    write_momentum_table(
        paths["momentum_lag"], stats, layout, lag_averaged=True
    )
    accumulation = accumulation_rows(stats, layout)
    write_accumulation(paths["accumulation"], accumulation)
    t100_scalar = write_t100_scalar_cross_validation(
        paths["t100_scalar"], stats, original_paired
    )
    t100_momentum = write_t100_momentum_cross_validation(
        paths["t100_momentum"], stats, original_paired, layout
    )

    document = {
        "inputs": {
            "hmc_t0": str(args.hmc),
            "dm": {str(time): str(path.resolve()) for time, path in dm_paths.items()},
            "physical": {
                str(time): str(path.resolve())
                for time, path in physical_paths.items()
            },
            "original_true100_static_reference": str(args.original_true100),
        },
        "nensemble": nensemble,
        "L": length,
        "times": TIMES,
        "npairs": len(TIMES) * (len(TIMES) - 1) // 2,
        "definitions": {
            "M": "per-trajectory spatial mean(phi)",
            "connected": "subtract each field's own spatial mean",
            "S(k)": "radial shell mean of |FFT(phi-mean(phi))|^2/V",
            "rho": "<F>/sqrt(<G_endpoint><G_start>)",
            "lag_average": (
                "average all time origins within each trajectory first, then "
                "ensemble average and delete one trajectory"
            ),
            "paired_errors": (
                "delete-one-trajectory jackknife; for linear means this equals "
                "the SEM of paired per-trajectory differences"
            ),
            "static_true100": (
                "paired_first_5120 is the inferential comparison; the full "
                "10240 reference shares the first 5120 initial trajectories "
                "and is reported descriptively without a difference SEM or z"
            ),
            "t100_timecorr_cross_validation": (
                "DM, the new physical branch, and the original true t=100 "
                "endpoint are compared on the same first 5120 HMC rows with "
                "paired delete-one-trajectory jackknife errors"
            ),
            "momentum_shells": "0 < |n| <= L/2; corners and k=0 excluded",
        },
        "one_time_dm_vs_physical": one_time,
        "conservation": conservation,
        "scalar_specific_pairs": scalar_pairs,
        "scalar_lag_average": scalar_lag_results,
        "accumulation": accumulation,
        "static_true100_cross_validation": static_summary(
            stats, original, cond_indices, layout
        ),
        "t100_timecorr_cross_validation": {
            "scalar": t100_scalar,
            "momentum": t100_momentum,
        },
        "tables": {key: str(path) for key, path in paths.items() if key != "json"},
    }
    with paths["json"].open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for path in paths.values():
        print(f"saved {path}")


if __name__ == "__main__":
    main()
