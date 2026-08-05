"""Fit diagonal propagators before and after one 2x2 average block spin."""

from pathlib import Path
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "trainingdata/cfgs_wolff_fahmc_k=0.28_l=0.022_128^2.jld2"
OUT = ROOT / "results/k0.28_L128_average_block_spin"
N_BOOT = 500


def average_block_2x2(cfgs):
    n_cfg, length, _ = cfgs.shape
    return cfgs.reshape(n_cfg, length // 2, 2, length // 2, 2).mean(axis=(2, 4))


def diagonal_modes(cfgs):
    """Return p_hat^2 and per-configuration G(p) for all (n,n), n=1,...,L/2."""
    _, length, _ = cfgs.shape
    n = np.arange(1, length // 2 + 1)
    fft = np.fft.rfft2(cfgs, axes=(-2, -1))
    modes = fft[:, n, n]
    g_per_cfg = (modes.real**2 + modes.imag**2) / length**2
    p2 = 8.0 * np.sin(np.pi * n / length) ** 2
    return n, p2, g_per_cfg


def propagator(p2, z, m2):
    return 1.0 / (z * p2 + m2)


def fit_propagator(p2, g_per_cfg, rng, n_fit=None):
    """Weighted fit to diagonal modes; bootstrap configurations."""
    x = p2 if n_fit is None else p2[:n_fit]
    samples = g_per_cfg if n_fit is None else g_per_cfg[:, :n_fit]
    y = samples.mean(axis=0)
    sigma = samples.std(axis=0, ddof=1) / np.sqrt(samples.shape[0])

    slope, intercept = np.polyfit(x, 1.0 / y, 1)
    initial = np.maximum([slope, intercept], np.finfo(float).eps)
    params, _ = curve_fit(
        propagator, x, y, p0=initial, sigma=sigma,
        absolute_sigma=True, bounds=(0.0, np.inf), maxfev=10000,
    )

    boot = np.empty((N_BOOT, 2))
    n_cfg = samples.shape[0]
    for b in range(N_BOOT):
        idx = rng.integers(0, n_cfg, n_cfg)
        yb = samples[idx].mean(axis=0)
        boot[b], _ = curve_fit(
            propagator, x, yb, p0=params, sigma=sigma,
            absolute_sigma=True, bounds=(0.0, np.inf), maxfev=10000,
        )

    residual = (y - propagator(x, *params)) / sigma
    mass_boot = np.sqrt(boot[:, 1])
    return {
        "z": float(params[0]),
        "z_error": float(boot[:, 0].std(ddof=1)),
        "m2": float(params[1]),
        "m2_error": float(boot[:, 1].std(ddof=1)),
        "mass_lattice_units": float(np.sqrt(params[1])),
        "mass_lattice_units_error": float(mass_boot.std(ddof=1)),
        "Z": float(params[0]),
        "Z_error": float(boot[:, 0].std(ddof=1)),
        "xi_lattice_units": float(np.sqrt(params[0] / params[1])),
        "xi_lattice_units_error": float(
            np.sqrt(boot[:, 0] / boot[:, 1]).std(ddof=1)
        ),
        "chi2_per_dof": float(np.sum(residual**2) / (len(x) - 2)),
    }


def save_blocked(cfgs, metadata):
    path = OUT / "cfgs_average_block_2a.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("cfgs", data=cfgs)
        f.create_dataset("N", data=cfgs.shape[1])
        f.create_dataset("n_samples", data=cfgs.shape[0])
        f.create_dataset("kappa", data=metadata["kappa"])
        f.create_dataset("lambda", data=metadata["lambda"])
        f.create_dataset("block_factor", data=2)
        f.create_dataset("lattice_spacing_in_a", data=2.0)
        f.attrs["blocking"] = "non-overlapping 2x2 arithmetic mean"
        f.attrs["source"] = str(SOURCE)
    return path


def save_modes(path, n, p2, g_per_cfg):
    g = g_per_cfg.mean(axis=0)
    error = g_per_cfg.std(axis=0, ddof=1) / np.sqrt(g_per_cfg.shape[0])
    np.savetxt(
        path, np.column_stack((n, p2, g, error)),
        header="n\tp_hat_squared\tG\tG_standard_error", delimiter="\t",
        fmt=["%d", "%.10e", "%.10e", "%.10e"],
    )
    return g, error


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(SOURCE, "r") as f:
        cfgs = np.asarray(f["cfgs"])
        metadata = {
            "kappa": float(f["κ"][()]),
            "lambda": float(f["λ"][()]),
        }

    blocked = average_block_2x2(cfgs)
    blocked_path = save_blocked(blocked, metadata)

    rng = np.random.default_rng(20260731)
    n_a, p2_a, gp_a = diagonal_modes(cfgs)
    fit_a = fit_propagator(p2_a, gp_a, rng)
    fit_a_low5 = fit_propagator(p2_a, gp_a, rng, n_fit=5)
    del cfgs

    n_2a, p2_2a, gp_2a = diagonal_modes(blocked)
    fit_2a = fit_propagator(p2_2a, gp_2a, rng)
    fit_2a_low5 = fit_propagator(p2_2a, gp_2a, rng, n_fit=5)
    del blocked

    g_a, e_a = save_modes(OUT / "diagonal_modes_a.tsv", n_a, p2_a, gp_a)
    g_2a, e_2a = save_modes(OUT / "diagonal_modes_2a.tsv", n_2a, p2_2a, gp_2a)

    fit_a["xi_in_a"] = fit_a["xi_lattice_units"]
    fit_a["xi_in_a_error"] = fit_a["xi_lattice_units_error"]
    fit_2a["xi_in_a"] = 2.0 * fit_2a["xi_lattice_units"]
    fit_2a["xi_in_a_error"] = 2.0 * fit_2a["xi_lattice_units_error"]
    fit_a_low5["xi_in_a"] = fit_a_low5["xi_lattice_units"]
    fit_a_low5["xi_in_a_error"] = fit_a_low5["xi_lattice_units_error"]
    fit_2a_low5["xi_in_a"] = 2.0 * fit_2a_low5["xi_lattice_units"]
    fit_2a_low5["xi_in_a_error"] = 2.0 * fit_2a_low5["xi_lattice_units_error"]
    summary = {
        "source": str(SOURCE),
        "blocked_configurations": str(blocked_path),
        "n_configurations": int(gp_a.shape[0]),
        "blocking": "non-overlapping 2x2 arithmetic mean",
        "fit_model": "G(p) = 1 / (z * p_hat_squared + m2)",
        "diagonal_modes": {
            "a": "(n,n), n=1,...,64; all nonzero diagonal modes fitted",
            "2a": "(n,n), n=1,...,32; all nonzero diagonal modes fitted",
        },
        "momentum": "p_hat_squared = 8 sin(pi*n/L)^2 in each lattice's units",
        "bootstrap_samples": N_BOOT,
        "a": {"L": 128, "spacing_in_a": 1, **fit_a},
        "2a": {"L": 64, "spacing_in_a": 2, **fit_2a},
    }
    (OUT / "fit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    summary_low5 = {
        **{k: v for k, v in summary.items() if k not in ("diagonal_modes", "a", "2a")},
        "diagonal_modes": "(n,n), n=1,...,5; n=0 excluded",
        "a": {"L": 128, "spacing_in_a": 1, **fit_a_low5},
        "2a": {"L": 64, "spacing_in_a": 2, **fit_2a_low5},
    }
    (OUT / "fit_summary_first5.json").write_text(
        json.dumps(summary_low5, indent=2) + "\n", encoding="utf-8"
    )

    for stem, fits, fit_label in [
        ("diagonal_propagator_fit", (fit_a, fit_2a), "fit to all diagonal modes"),
        (
            "diagonal_propagator_fit_first5",
            (fit_a_low5, fit_2a_low5),
            "fit to n=1,...,5; extrapolated to 8",
        ),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for panel, (ax, label, p2, g, error, fit) in enumerate([
            (axes[0], r"$a,\ 128^2$", p2_a, g_a, e_a, fits[0]),
            (axes[1], r"$2a,\ 64^2$", p2_2a, g_2a, e_2a, fits[1]),
        ]):
            ax.errorbar(
                p2, g, yerr=error, fmt=".", ms=5, capsize=2,
                label="diagonal modes",
            )
            xf = np.linspace(0.0, 8.0, 500)
            ax.plot(xf, propagator(xf, fit["z"], fit["m2"]), label=fit_label)
            annotation = (
                rf"$m={fit['mass_lattice_units']:.3f}"
                rf"\pm{fit['mass_lattice_units_error']:.3f}$"
            )
            if panel == 0:
                annotation += (
                    "\n"
                    rf"$2m={2 * fit['mass_lattice_units']:.3f}"
                    rf"\pm{2 * fit['mass_lattice_units_error']:.3f}$"
                )
            annotation += (
                "\n"
                rf"$Z={fit['Z']:.3f}\pm{fit['Z_error']:.3f}$"
            )
            ax.text(
                0.97, 0.72,
                annotation,
                transform=ax.transAxes, ha="right", va="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )
            ax.set(xlabel=r"$\hat p^2$", ylabel=r"$G(\hat p)$", title=label)
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"{stem}.png", dpi=180)
        fig.savefig(OUT / f"{stem}.pdf")
        plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("\nFirst five nonzero diagonal modes:")
    print(json.dumps(summary_low5, indent=2))


if __name__ == "__main__":
    main()
