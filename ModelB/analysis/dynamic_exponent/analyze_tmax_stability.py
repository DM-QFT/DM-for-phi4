#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
from diagonal_modes import cov_per_ens, tau_1e

path = Path(sys.argv[1])
output = Path(sys.argv[2])
RNG = np.random.default_rng(20260729)
NBOOT = 400



def estimate(x, sample_dt):
    cov = cov_per_ens(x)
    c = cov.sum(axis=0)
    c /= c[0]
    t = np.arange(c.size) * sample_dt
    tau = tau_1e(t, c)
    assert np.isfinite(tau)
    boot = np.empty(NBOOT)
    p = np.full(cov.shape[0], 1 / cov.shape[0])
    for b in range(NBOOT):
        cb = RNG.multinomial(cov.shape[0], p) @ cov
        boot[b] = tau_1e(t, cb / cb[0])
    valid = np.isfinite(boot)
    return tau, boot[valid].std(ddof=1), c[-1], valid.sum()


with h5py.File(path) as f:
    L = int(f["L"][()])
    n_ens = int(f["Nensemble"][()])
    times = f["times"][()].astype(float)
    sample_dt = times[1] - times[0]
    ds = f["fields"]
    assert ds.shape == (times.size, n_ens, L, L)
    modes = {
        "axial": np.empty((times.size, 2, n_ens), np.complex64),
        "diag": np.empty((times.size, 2, n_ens), np.complex64),
    }
    phase = np.exp(-2j * np.pi * np.arange(L) / L).astype(np.complex64)
    for i in range(0, times.size, 64):
        j = min(i + 64, times.size)
        u = ds[i:j]
        modes["axial"][i:j, 0] = np.einsum(
            "tex,x->te", u.sum(axis=2), phase, optimize=True
        ) / L**2
        modes["axial"][i:j, 1] = np.einsum(
            "tey,y->te", u.sum(axis=3), phase, optimize=True
        ) / L**2
        modes["diag"][i:j, 0] = np.einsum(
            "teyx,y,x->te", u, phase, phase, optimize=True
        ) / L**2
        modes["diag"][i:j, 1] = np.einsum(
            "teyx,y,x->te", u, phase.conj(), phase, optimize=True
        ) / L**2

duration = int(round(times[-1]))
prefixes = [t for t in (2000, 2500, 3000, 4000, 5000, 6000) if t <= duration]
windows = [("prefix", 0, t) for t in prefixes]
if duration >= 6000:
    windows += [("first_half", 0, 3000), ("second_half", 3000, 6000)]

rows = []
for cls, x in modes.items():
    for label, t0, t1 in windows:
        i0 = round(t0 / sample_dt)
        i1 = round(t1 / sample_dt) + 1
        assert np.isclose(times[i0], t0) and np.isclose(times[i1 - 1], t1)
        tau, err, c_half, n_valid = estimate(x[i0:i1], sample_dt)
        rows.append((cls, label, t0, t1, tau, err, c_half, n_valid))
        print(
            f"{cls:5s} {label:11s} [{t0:4d},{t1:4d}] "
            f"tau={tau:8.3f} ± {err:.3f}  C(T/2)={c_half:+.4f} "
            f"boot={n_valid}/{NBOOT}"
        )

with output.open("w") as f:
    f.write(f"# source={path}\tNboot={NBOOT}\tsample_dt={sample_dt:g}\n")
    f.write(
        "# class\twindow\tt_start\tt_end\ttau\ttau_err\t"
        "C_half_duration\tNboot_valid\n"
    )
    for row in rows:
        f.write(
            f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t"
            f"{row[4]:.6f}\t{row[5]:.6f}\t{row[6]:.8f}\t{row[7]}\n"
        )
