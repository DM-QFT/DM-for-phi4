"""All-shell propagator comparison for the Conv2d/Conv3d diagnostic."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT3D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT3D))
from _layout import to_NLLL  # noqa: E402


def load_hmc(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return to_NLLL(np.asarray(handle["cfgs"], dtype=np.float64))


def load_dm(path: Path) -> np.ndarray:
    return to_NLLL(np.load(path)).astype(np.float64)


def shell_propagator(fields: np.ndarray):
    if fields.ndim != 4 or len(set(fields.shape[1:])) != 1:
        raise ValueError(f"expected (N,L,L,L), got {fields.shape}")
    count, L = fields.shape[:2]
    spatial_axes = (1, 2, 3)
    centered = fields - fields.mean(axis=spatial_axes, keepdims=True)
    mode_values = np.abs(np.fft.fftn(centered, axes=spatial_axes)) ** 2 / L**3

    n = np.fft.fftfreq(L) * L
    component = 2.0 * np.sin(np.pi * n / L)
    grids = np.meshgrid(component, component, component, indexing="ij")
    p2 = sum(grid * grid for grid in grids)
    keys = np.round(p2, decimals=12)
    unique = np.unique(keys)

    rows = []
    flat_modes = mode_values.reshape(count, -1)
    flat_keys = keys.ravel()
    for key in unique:
        if key <= 1.0e-12:
            continue
        selected = flat_keys == key
        per_configuration = flat_modes[:, selected].mean(axis=1)
        rows.append(
            (
                np.sqrt(key),
                int(selected.sum()),
                float(per_configuration.mean()),
                float(per_configuration.std(ddof=1) / np.sqrt(count)),
            )
        )
    return np.asarray(rows), np.abs(fields.mean(axis=spatial_axes))


def weighted_rms(ratio: np.ndarray, degeneracy: np.ndarray, mask=None) -> float:
    if mask is None:
        mask = np.ones_like(ratio, dtype=bool)
    return float(
        np.sqrt(
            np.sum(degeneracy[mask] * (ratio[mask] - 1.0) ** 2)
            / np.sum(degeneracy[mask])
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmc", type=Path, required=True)
    sample_group = parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument(
        "--dm", type=Path, help="generic diffusion-model sample file"
    )
    sample_group.add_argument(
        "--conv2d", type=Path, help="legacy alias for a Conv2d-fold sample file"
    )
    parser.add_argument("--dm_label", help="plot/table label for --dm")
    parser.add_argument(
        "--native",
        type=Path,
        help="optional native-Conv3d sample file for a three-way comparison",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tag", default="ep0095")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_path = args.dm if args.dm is not None else args.conv2d
    sample_label = args.dm_label or (
        "DM" if args.dm is not None else "Conv2d-fold"
    )
    ensembles = {
        "HMC": load_hmc(args.hmc),
        sample_label: load_dm(sample_path),
    }
    if args.native is not None:
        ensembles["native Conv3d"] = load_dm(args.native)
    results = {name: shell_propagator(fields) for name, fields in ensembles.items()}
    reference = results["HMC"][0]
    p, degeneracy, g_ref, e_ref = reference.T

    summary = []
    for name, (table, abs_m) in results.items():
        if not np.array_equal(table[:, :2], reference[:, :2]):
            raise RuntimeError(f"momentum shells do not match for {name}")
        ratio = table[:, 2] / g_ref
        summary.append(
            {
                "ensemble": name,
                "n_samples": len(ensembles[name]),
                "all_rms_percent": 100.0 * weighted_rms(ratio, degeneracy),
                "ir_rms_percent": 100.0
                * weighted_rms(ratio, degeneracy, p <= 2.0),
                "mean_ratio_weighted": float(
                    np.sum(degeneracy * ratio) / np.sum(degeneracy)
                ),
                "first_shell_ratio": float(ratio[0]),
                "mean_abs_m": float(abs_m.mean()),
            }
        )

    with (args.out_dir / f"summary_{args.tag}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    with (args.out_dir / f"shells_{args.tag}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["ensemble", "p_hat", "degeneracy", "G", "SEM", "ratio_to_HMC"]
        )
        for name, (table, _) in results.items():
            for row, ref in zip(table, g_ref, strict=True):
                writer.writerow([name, *row, row[2] / ref])

    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": (2.5, 1.0), "hspace": 0.08},
    )
    for index, (name, (table, _)) in enumerate(results.items()):
        ax.errorbar(
            table[:, 0],
            table[:, 2],
            yerr=table[:, 3],
            fmt="o-" if index == 0 else "s--",
            ms=3.5,
            lw=1.2,
            capsize=2,
            label=name,
        )
        if index:
            ratio_ax.plot(table[:, 0], table[:, 2] / g_ref, "s--", ms=3.5)
    ax.set_yscale("log")
    ax.set_ylabel(r"$G(\hat p)$")
    ax.set_title(f"3D L=8 direct-score comparison ({args.tag}, p=0 excluded)")
    ax.grid(alpha=0.3)
    ax.legend()
    ratio_ax.axhline(1.0, color="black", lw=1)
    ratio_ax.set_xlabel(r"$|\hat p|$")
    ratio_ax.set_ylabel(r"$G/G_{\rm HMC}$")
    ratio_ax.grid(alpha=0.3)
    figure = args.out_dir / f"propagator_{args.tag}.png"
    fig.savefig(figure, dpi=170, bbox_inches="tight")
    fig.savefig(figure.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for row in summary:
        print(row)
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
