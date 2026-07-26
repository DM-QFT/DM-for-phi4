"""Compare generated and true Model-B endpoint time correlations.

The generated endpoint, true endpoint, and initial HMC field are paired by
``cond_indices``.  Scalar correlations and radially averaged momentum-space
correlations use the ratio of ensemble means and delete-one-trajectory
jackknife errors, matching ``analysis/analyze_dynamics.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import fft


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def read_rows(dataset, indices):
    """Read arbitrary HDF5 rows and restore their requested order."""
    order = np.argsort(indices)
    values = dataset[indices[order].tolist()]
    return np.asarray(values[np.argsort(order)], dtype=np.float32)


def centered(fields):
    fields = np.asarray(fields, dtype=np.float32).copy()
    spatial_mean = np.mean(fields, axis=(1, 2), dtype=np.float64)
    fields -= spatial_mean[:, None, None]
    return fields, spatial_mean


def mean_sem(values):
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values, axis=0, dtype=np.float64)
    sem = np.std(values, axis=0, ddof=1, dtype=np.float64) / np.sqrt(
        values.shape[0]
    )
    return mean, sem


def ratio_with_jackknife(cross, power_t, power_0):
    """Ratio of means and its paired delete-one-trajectory jackknife SEM."""
    cross = np.asarray(cross, dtype=np.float64)
    power_t = np.asarray(power_t, dtype=np.float64)
    power_0 = np.asarray(power_0, dtype=np.float64)
    n = cross.shape[0]

    rho = np.mean(cross, axis=0) / np.sqrt(
        np.mean(power_t, axis=0) * np.mean(power_0, axis=0)
    )
    leave_cross = (np.sum(cross, axis=0)[None, ...] - cross) / (n - 1)
    leave_power_t = (
        np.sum(power_t, axis=0)[None, ...] - power_t
    ) / (n - 1)
    leave_power_0 = (
        np.sum(power_0, axis=0)[None, ...] - power_0
    ) / (n - 1)
    leave_rho = leave_cross / np.sqrt(leave_power_t * leave_power_0)
    leave_mean = np.mean(leave_rho, axis=0)
    jackknife_sem = np.sqrt(
        (n - 1) / n
        * np.sum((leave_rho - leave_mean) ** 2, axis=0, dtype=np.float64)
    )
    return rho, jackknife_sem, leave_rho


def paired_jackknife_sem(leave_gen, leave_true):
    leave_difference = leave_gen - leave_true
    n = leave_difference.shape[0]
    leave_mean = np.mean(leave_difference, axis=0)
    return np.sqrt(
        (n - 1) / n
        * np.sum(
            (leave_difference - leave_mean) ** 2,
            axis=0,
            dtype=np.float64,
        )
    )


def radial_layout(length):
    """Integer-radius shells over 0 < |n| <= L/2, excluding corners."""
    frequency = np.fft.fftfreq(length) * length
    ky, kx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.hypot(kx, ky)
    shell = np.floor(radius + 0.5).astype(np.int32)
    valid = (radius > 0.0) & (radius <= length / 2.0)

    flat_indices = np.flatnonzero(valid)
    shell_at_index = shell.ravel()[flat_indices]
    order = np.argsort(shell_at_index, kind="stable")
    pixels = flat_indices[order]
    shell_ids, starts, counts = np.unique(
        shell_at_index[order], return_index=True, return_counts=True
    )

    khat_grid = np.sqrt(
        4.0 * np.sin(np.pi * kx / length) ** 2
        + 4.0 * np.sin(np.pi * ky / length) ** 2
    )
    k = np.array([
        np.mean(radius.ravel()[pixels[start:start + count]])
        for start, count in zip(starts, counts)
    ])
    khat = np.array([
        np.mean(khat_grid.ravel()[pixels[start:start + count]])
        for start, count in zip(starts, counts)
    ])
    return shell_ids, pixels, starts, counts, k, khat


def shell_mean(values, pixels, starts, counts):
    grouped = values.reshape(values.shape[0], -1)[:, pixels]
    return np.add.reduceat(
        grouped, starts, axis=1, dtype=np.float64
    ) / counts[None, :]


def scalar_summary(cross, power_t, power_0):
    c_mean, c_sem = mean_sem(cross)
    gt_mean, gt_sem = mean_sem(power_t)
    g0_mean, g0_sem = mean_sem(power_0)
    rho, rho_sem, leave_rho = ratio_with_jackknife(
        cross, power_t, power_0
    )
    return ({
        "C": float(c_mean),
        "C_sem": float(c_sem),
        "G0": float(g0_mean),
        "G0_sem": float(g0_sem),
        "G_endpoint": float(gt_mean),
        "G_endpoint_sem": float(gt_sem),
        "rho": float(rho),
        "rho_jackknife_sem": float(rho_sem),
    }, leave_rho)


def resolve_metadata_path(value, metadata_path):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = metadata_path.parent / path
    return path.resolve()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--fft_batch", type=positive_int, default=64)
    parser.add_argument("--fft_workers", type=int, default=8)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Default: the run's validation directory",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    samples_path = args.samples.expanduser().resolve()
    metadata_path = samples_path.with_suffix(".json")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    cond_path = resolve_metadata_path(
        metadata.get("cond_path", metadata["source_path"]), metadata_path
    )
    target_path = resolve_metadata_path(metadata["target_path"], metadata_path)

    with np.load(samples_path, allow_pickle=False) as archive:
        phi_gen = np.asarray(archive["phi_gen"], dtype=np.float32)
        cond_indices = np.asarray(archive["cond_indices"], dtype=np.int64)
        n_per_cond = int(archive["n_per_cond"])
    if n_per_cond != 1:
        raise ValueError("this endpoint comparison expects n_per_cond=1")

    nensemble, length, length_y = phi_gen.shape
    if nensemble != len(cond_indices) or length != length_y:
        raise ValueError("phi_gen and cond_indices have incompatible shapes")
    volume = length * length

    shell_ids, pixels, starts, counts, k, khat = radial_layout(length)
    nshell = len(shell_ids)
    scalar = {
        name: np.empty(nensemble, dtype=np.float64)
        for name in (
            "C_gen", "C_true", "G0", "G_gen", "G_true",
            "mean0", "mean_gen", "mean_true",
        )
    }
    momentum = {
        name: np.empty((nensemble, nshell), dtype=np.float64)
        for name in ("G0", "G_gen", "G_true", "F_gen", "F_true")
    }

    with h5py.File(cond_path, "r") as cond_file, h5py.File(
        target_path, "r"
    ) as target_file:
        cond_data = cond_file["cfgs"]
        target_data = target_file["cfgs"]
        for start in range(0, nensemble, args.fft_batch):
            stop = min(start + args.fft_batch, nensemble)
            rows = cond_indices[start:stop]
            field0, mean0 = centered(read_rows(cond_data, rows))
            field_gen, mean_gen = centered(phi_gen[start:stop])
            field_true, mean_true = centered(read_rows(target_data, rows))

            scalar["mean0"][start:stop] = mean0
            scalar["mean_gen"][start:stop] = mean_gen
            scalar["mean_true"][start:stop] = mean_true
            scalar["G0"][start:stop] = np.einsum(
                "nij,nij->n", field0, field0, dtype=np.float64,
                optimize=True,
            ) / volume
            scalar["G_gen"][start:stop] = np.einsum(
                "nij,nij->n", field_gen, field_gen, dtype=np.float64,
                optimize=True,
            ) / volume
            scalar["G_true"][start:stop] = np.einsum(
                "nij,nij->n", field_true, field_true, dtype=np.float64,
                optimize=True,
            ) / volume
            scalar["C_gen"][start:stop] = np.einsum(
                "nij,nij->n", field_gen, field0, dtype=np.float64,
                optimize=True,
            ) / volume
            scalar["C_true"][start:stop] = np.einsum(
                "nij,nij->n", field_true, field0, dtype=np.float64,
                optimize=True,
            ) / volume

            fft0 = fft.fft2(
                field0, axes=(-2, -1), workers=args.fft_workers
            )
            fft_gen = fft.fft2(
                field_gen, axes=(-2, -1), workers=args.fft_workers
            )
            fft_true = fft.fft2(
                field_true, axes=(-2, -1), workers=args.fft_workers
            )
            power0 = (fft0.real**2 + fft0.imag**2) / volume
            power_gen = (fft_gen.real**2 + fft_gen.imag**2) / volume
            power_true = (fft_true.real**2 + fft_true.imag**2) / volume
            cross_gen = (
                fft_gen.real * fft0.real + fft_gen.imag * fft0.imag
            ) / volume
            cross_true = (
                fft_true.real * fft0.real + fft_true.imag * fft0.imag
            ) / volume
            momentum["G0"][start:stop] = shell_mean(
                power0, pixels, starts, counts
            )
            momentum["G_gen"][start:stop] = shell_mean(
                power_gen, pixels, starts, counts
            )
            momentum["G_true"][start:stop] = shell_mean(
                power_true, pixels, starts, counts
            )
            momentum["F_gen"][start:stop] = shell_mean(
                cross_gen, pixels, starts, counts
            )
            momentum["F_true"][start:stop] = shell_mean(
                cross_true, pixels, starts, counts
            )
            print(f"FFT {stop}/{nensemble}", flush=True)

    raw_g0 = scalar["G0"] + scalar["mean0"] ** 2
    raw_g_gen = scalar["G_gen"] + scalar["mean_gen"] ** 2
    raw_g_true = scalar["G_true"] + scalar["mean_true"] ** 2
    raw_c_gen = scalar["C_gen"] + scalar["mean_gen"] * scalar["mean0"]
    raw_c_true = scalar["C_true"] + scalar["mean_true"] * scalar["mean0"]

    scalar_output = {}
    for kind, c_gen, c_true, g_gen, g_true, g0 in (
        (
            "connected", scalar["C_gen"], scalar["C_true"],
            scalar["G_gen"], scalar["G_true"], scalar["G0"],
        ),
        ("raw", raw_c_gen, raw_c_true, raw_g_gen, raw_g_true, raw_g0),
    ):
        gen_summary, gen_leave = scalar_summary(c_gen, g_gen, g0)
        true_summary, true_leave = scalar_summary(c_true, g_true, g0)
        difference = gen_summary["rho"] - true_summary["rho"]
        difference_sem = float(paired_jackknife_sem(gen_leave, true_leave))
        scalar_output[kind] = {
            "generated": gen_summary,
            "true": true_summary,
            "rho_gen_minus_true": difference,
            "rho_difference_paired_jackknife_sem": difference_sem,
            "rho_difference_z": difference / difference_sem,
        }

    g0_mean, g0_sem = mean_sem(momentum["G0"])
    gg_mean, gg_sem = mean_sem(momentum["G_gen"])
    gt_mean, gt_sem = mean_sem(momentum["G_true"])
    fg_mean, fg_sem = mean_sem(momentum["F_gen"])
    ft_mean, ft_sem = mean_sem(momentum["F_true"])
    rho_gen, rho_gen_sem, rho_gen_leave = ratio_with_jackknife(
        momentum["F_gen"], momentum["G_gen"], momentum["G0"]
    )
    rho_true, rho_true_sem, rho_true_leave = ratio_with_jackknife(
        momentum["F_true"], momentum["G_true"], momentum["G0"]
    )
    rho_difference = rho_gen - rho_true
    rho_difference_sem = paired_jackknife_sem(rho_gen_leave, rho_true_leave)
    delta_g_mean, delta_g_sem = mean_sem(
        momentum["G_gen"] - momentum["G_true"]
    )
    delta_f_mean, delta_f_sem = mean_sem(
        momentum["F_gen"] - momentum["F_true"]
    )

    columns = (
        "shell", "mode_count", "k", "khat", "G0", "G0_sem",
        "G_endpoint_gen", "G_endpoint_gen_sem",
        "G_endpoint_true", "G_endpoint_true_sem",
        "delta_G_endpoint", "delta_G_endpoint_paired_sem",
        "F_gen", "F_gen_sem", "F_true", "F_true_sem",
        "delta_F", "delta_F_paired_sem",
        "rho_gen", "rho_gen_jackknife_sem",
        "rho_true", "rho_true_jackknife_sem",
        "rho_gen_minus_true", "rho_difference_paired_jackknife_sem",
        "rho_difference_z",
    )
    table = np.column_stack((
        shell_ids, counts, k, khat, g0_mean, g0_sem,
        gg_mean, gg_sem, gt_mean, gt_sem, delta_g_mean, delta_g_sem,
        fg_mean, fg_sem, ft_mean, ft_sem, delta_f_mean, delta_f_sem,
        rho_gen, rho_gen_sem, rho_true, rho_true_sem,
        rho_difference, rho_difference_sem,
        rho_difference / rho_difference_sem,
    ))

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else samples_path.parent.parent / "validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = samples_path.stem
    scalar_path = output_dir / f"{base}_timecorr_scalar.json"
    momentum_path = output_dir / f"{base}_timecorr_momentum.dat"
    scalar_document = {
        "samples": str(samples_path),
        "condition_source": str(cond_path),
        "true_target": str(target_path),
        "nensemble": nensemble,
        "L": length,
        "time": metadata.get("horizon"),
        "normalization": (
            "rho=<F>/sqrt(<G_endpoint><G0>); errors are paired "
            "delete-one-trajectory jackknife SEM"
        ),
        "radial_domain": "0 < |n| <= L/2; Fourier-grid corners excluded",
        **scalar_output,
    }
    with scalar_path.open("w", encoding="utf-8") as handle:
        json.dump(scalar_document, handle, indent=2)
        handle.write("\n")
    np.savetxt(
        momentum_path,
        table,
        fmt=["%d", "%d"] + ["%.10e"] * (len(columns) - 2),
        header=(
            "  ".join(columns)
            + "\nG=|FFT(delta_phi)|^2/V; "
            "F=Re[FFT(delta_phi_t) conj(FFT(delta_phi_0))]/V"
        ),
    )
    print(f"saved {scalar_path}")
    print(f"saved {momentum_path}")


if __name__ == "__main__":
    main()
