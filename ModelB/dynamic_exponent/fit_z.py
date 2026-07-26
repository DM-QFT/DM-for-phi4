#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.fft import fft, ifft, next_fast_len

NBOOT = 1000
RNG = np.random.default_rng(20260726)


def read_run(path):
    with h5py.File(path) as f:
        raw = f["modes"][()]
        modes = raw["re"] + 1j * raw["im"]
        modes = modes.transpose(2, 1, 0)
        run = {
            "path": str(path),
            "L": int(f["L"][()]),
            "n": int(f["Nensemble"][()]),
            "kappa": float(f["kappa"][()]),
            "lambda": float(f["lambda"][()]),
            "times": f["times"][()].astype(float),
            "modes": modes,
        }
        assert modes.shape == (len(run["times"]), 2, run["n"])
        return run


def autocovariance(modes):
    nt, _, n = modes.shape
    modes = modes - modes.mean(axis=(0, 2), keepdims=True)
    traces = modes.transpose(2, 1, 0)
    nf = next_fast_len(2 * nt - 1)
    spectrum = fft(traces, nf, axis=-1, workers=8)
    nlag = nt // 2 + 1
    cov = ifft(spectrum * spectrum.conj(), axis=-1, workers=8).real[..., :nlag]
    cov /= np.arange(nt, nt - nlag, -1)
    return cov.mean(axis=1)


def relaxation(run):
    cov = autocovariance(run["modes"])
    n = len(cov)
    total = cov.sum(axis=0)
    corr = total / total[0]
    leave = (total[None, :] - cov) / (n - 1)
    leave /= leave[:, :1]
    sem = np.sqrt((n - 1) * np.mean((leave - leave.mean(axis=0)) ** 2, axis=0))

    high = np.flatnonzero(corr <= 0.8)
    assert len(high), "correlator does not reach 0.8"
    start = high[0]
    low = np.flatnonzero(corr[start:] <= 0.2)
    assert len(low), "correlator does not reach 0.2; increase tmax"
    stop = start + low[0]
    indices = np.arange(start, stop + 1)
    if len(indices) > 200:
        indices = np.unique(np.rint(np.linspace(start, stop, 200)).astype(int))
    assert len(indices) >= 8 and np.all(corr[indices] > 0)

    t = run["times"][: len(corr)][indices] - run["times"][0]
    y = np.log(corr[indices])
    sigma = np.maximum(sem[indices] / corr[indices], 1e-6)
    weight = 1 / sigma**2
    denominator = np.sum(weight * t**2)
    tau = -1 / (np.sum(weight * t * y) / denominator)

    weights = RNG.multinomial(n, np.full(n, 1 / n), size=NBOOT)
    boot_corr = (weights @ cov[:, indices]) / (weights @ cov[:, 0])[:, None]
    valid = np.all(boot_corr > 0, axis=1)
    slopes = np.sum(weight * t * np.log(boot_corr[valid]), axis=1) / denominator
    boot = -1 / slopes[slopes < 0]
    assert len(boot) > 0.8 * NBOOT
    return {**run, "tau": tau, "boot": boot, "tmin": t[0], "tmax": t[-1]}


def fit_z(results):
    x = np.log([r["L"] for r in results])
    y = np.log([r["tau"] for r in results])
    xc = x - x.mean()
    z = np.dot(xc, y) / np.dot(xc, xc)
    nb = min(len(r["boot"]) for r in results)
    yb = np.log(np.column_stack([r["boot"][:nb] for r in results]))
    zb = yb @ xc / np.dot(xc, xc)
    drop = np.nan
    if len(results) >= 4:
        xd = x[1:] - x[1:].mean()
        drop = np.dot(xd, y[1:]) / np.dot(xd, xd)
    drop_two = np.nan
    if len(results) >= 5:
        xd = x[2:] - x[2:].mean()
        drop_two = np.dot(xd, y[2:]) / np.dot(xd, xd)
    return z, zb, drop, drop_two


def main(args):
    assert len(args) >= 4
    output = Path(args[0])
    results = sorted((relaxation(read_run(Path(p))) for p in args[1:]), key=lambda x: x["L"])
    assert len({r["L"] for r in results}) == len(results) >= 3
    assert np.ptp([r["kappa"] for r in results]) < 1e-10
    assert np.ptp([r["lambda"] for r in results]) < 1e-10
    output.mkdir(parents=True, exist_ok=True)

    table = np.array([
        [
            r["L"], r["n"], r["kappa"], r["lambda"], r["tau"],
            np.std(r["boot"], ddof=1), *np.quantile(r["boot"], [0.16, 0.84]),
            r["tmin"], r["tmax"],
        ]
        for r in results
    ])
    np.savetxt(
        output / "relaxation.tsv", table, delimiter="\t",
        header="L\tN\tkappa\tlambda\ttau\tstd\tq16\tq84\tfit_tmin\tfit_tmax",
    )
    z, zboot, drop, drop_two = fit_z(results)
    summary = {
        "z": z,
        "z_std": np.std(zboot, ddof=1),
        "z_16": np.quantile(zboot, 0.16),
        "z_84": np.quantile(zboot, 0.84),
        "z_drop_Lmin": drop if np.isfinite(drop) else None,
        "z_drop_two_Lmin": drop_two if np.isfinite(drop_two) else None,
        "kappa": results[0]["kappa"],
        "lambda": results[0]["lambda"],
        "lengths": [r["L"] for r in results],
    }
    with (output / "z.json").open("w") as f:
        json.dump(summary, f, indent=2)
    for r in results:
        print(f"L={r['L']:3d}  tau={r['tau']:.6g} ± {np.std(r['boot'], ddof=1):.3g}")
    print(
        f"z={z:.6f} [{summary['z_16']:.6f}, {summary['z_84']:.6f}]"
        f"  drop={drop:.6f},{drop_two:.6f}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
