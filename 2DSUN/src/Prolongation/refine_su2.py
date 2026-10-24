"""Forward scans and network validation in README_SU2_L16_to_L32.md.

The quaternion, loop and increment formulas follow train_su2_dm.py and
sample_su2_em.py. The check and scan commands use NumPy on CPU. The sample
command imports the existing CUDA sampler only when validating the network.
"""
import argparse
from contextlib import ExitStack
import csv
import math
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.special import ive

ROOT = Path(__file__).resolve().parents[2]
BETA_C = 0.1408904756906944
BETA_F = 2.0
DELTA_MAX = 1.0
TAUS = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2, 2.5, 3, 4, 6, 8, 10])
RT = ((1, 1), (1, 2), (2, 2), (4, 4))
LOOPS = ("W11", "W12", "W22", "W44")
NAMES = ("fine", "uniform", "interior", "relaxed")
EDGES = np.linspace(-1, 1, 41)
ANGLE_EDGES = np.linspace(0, np.pi, len(EDGES))
HOLONOMY_NAMES = ("theta_x", "theta_y", "theta_x_centered_second", "theta_y_centered_second",
                  "Polyakov_x_bar_second", "Polyakov_y_bar_second",
                  "holonomy_commutator", "holonomy_commutator_second")
BLOCK = 4
NBOOT = 1000


def roll(a, mu, k):
    return np.roll(a, -k, axis=a.ndim - 3 + mu)


def qmul(p, q):
    # U = q0 I + i q.sigma has the negative cross product convention.
    p0, p1, p2, p3 = np.moveaxis(p, -1, 0)
    q0, q1, q2, q3 = np.moveaxis(q, -1, 0)
    return np.stack((p0*q0 - p1*q1 - p2*q2 - p3*q3,
                     p0*q1 + p1*q0 - p2*q3 + p3*q2,
                     p0*q2 + p2*q0 - p3*q1 + p1*q3,
                     p0*q3 + p3*q0 - p1*q2 + p2*q1), axis=-1)


def conj(q):
    return q * np.array([1, -1, -1, -1])


def qexp(xi):
    th = np.linalg.norm(xi, axis=-1, keepdims=True)
    return np.concatenate((np.cos(th/2), np.sinc(th/(2*np.pi))*xi/2), axis=-1)


def normalize(u):
    norm = np.linalg.norm(u, axis=-1, keepdims=True)
    err = np.max(np.abs(norm - 1))
    if not np.isfinite(err) or err > 1e-7:
        raise ValueError(f"Invalid SU(2) norm, maximum error {err}")
    return u / norm, float(err)


def haar(shape, rng):
    q = rng.normal(size=(*shape, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def to_matrix(u):
    a = u.transpose(0, 3, 2, 1, 4)
    return np.stack((np.stack((a[..., 0]+1j*a[..., 3], a[..., 2]+1j*a[..., 1]), -1),
                     np.stack((-a[..., 2]+1j*a[..., 1], a[..., 0]-1j*a[..., 3]), -1)), -2)


def from_matrix(m):
    q = np.stack((m[..., 0, 0].real, m[..., 0, 1].imag,
                  m[..., 0, 1].real, m[..., 0, 0].imag), -1).transpose(0, 3, 2, 1, 4)
    q, err = normalize(q)
    matrix_err = np.max(np.abs(to_matrix(q) - m))
    if not np.isfinite(matrix_err) or matrix_err > 1e-10:
        raise ValueError(f"Input matrices violate SU(2), maximum error {matrix_err}")
    return q, max(err, float(matrix_err))


def load_cfgs(path, indices, lattice, beta):
    with h5py.File(path) as f:
        if f.attrs["L"] != lattice or not np.isclose(f.attrs["beta"], beta, rtol=0, atol=1e-14):
            raise ValueError(f"Unexpected L or beta in {path}")
        if f["cfgs"].shape[1:] != (lattice, lattice, 2, 2, 2):
            raise ValueError(f"Unexpected matrix axes in {path}")
        return from_matrix(f["cfgs"][indices])


def line(u, mu, n):
    m = u[:, mu]
    length = 1
    # P_{2n}(x) = P_n(x) P_n(x+n), preserving the order of the links.
    while 2*length <= n:
        m = qmul(m, roll(m, mu, length))
        length *= 2
    for k in range(length, n):
        m = qmul(m, roll(u[:, mu], mu, k))
    return m


def loop_quaternion(u, r, t):
    a, b = line(u, 0, r), line(u, 1, t)
    return qmul(qmul(qmul(a, roll(b, 0, r)), conj(roll(a, 1, t))), conj(b))


def plaquette(u):
    return loop_quaternion(u, 1, 1)


def wilson(u, r, t):
    return loop_quaternion(u, r, t)[..., 0]


def gauge_transform(u, omega):
    return np.stack([qmul(qmul(omega, u[:, mu]), conj(roll(omega, mu, 1)))
                     for mu in range(2)], axis=1)


def fourth_root(p):
    s = np.linalg.norm(p[..., 1:], axis=-1, keepdims=True)
    if np.any((p[..., :1] < 0) & (s < 5e-15)):
        raise ValueError("Ambiguous principal root at minus identity")
    angle = np.arctan2(s, p[..., :1]) / 4
    scale = np.divide(np.sin(angle), s, out=np.full_like(s, 0.25), where=s > 0)
    q = np.concatenate((np.cos(angle), scale*p[..., 1:]), axis=-1)
    reconstruction = np.max(np.abs(qmul(qmul(q, q), qmul(q, q)) - p))
    if reconstruction > 1e-10:
        raise ValueError(f"Fourth root reconstruction error {reconstruction}")
    return q, int(np.count_nonzero((p[..., :1] < 0) & (s < 1e-6))), float(reconstruction)


def prolong(u):
    q, near_minus, root_err = fourth_root(plaquette(u))
    n, _, l, _, _ = u.shape
    w = np.zeros((n, 2, 2*l, 2*l, 4))
    w[..., 0] = 1
    w[:, 0, 1::2, ::2] = u[:, 0]
    w[:, 0, ::2, 1::2] = conj(q)
    w[:, 0, 1::2, 1::2] = qmul(conj(q), u[:, 0])
    w[:, 1, ::2, 1::2] = u[:, 1]
    w[:, 1, 1::2, 1::2] = qmul(qmul(q, q), u[:, 1])
    w, unit_err = normalize(w)
    return w, (near_minus, root_err, unit_err)


def staple(u, mu):
    # Same two staples as su2_2d_heatbath.jl, with periodic array shifts.
    nu = 1 - mu
    a, b = u[:, mu], u[:, nu]
    forward = qmul(qmul(roll(b, mu, 1), conj(roll(a, nu, 1))), conj(b))
    backward = qmul(qmul(conj(roll(roll(b, mu, 1), nu, -1)),
                         conj(roll(a, nu, -1))), roll(b, nu, -1))
    return forward + backward


def local_delta(old, proposal, st, beta=BETA_F):
    return -beta * qmul(proposal - old, st)[..., 0]


def metropolis(u, sweeps, interior, rng, beta=BETA_F):
    u = u.copy()
    l = u.shape[2]
    x, y = np.indices((l, l))
    accepted = attempted = 0
    max_err = 0.0
    for _ in range(sweeps):
        for mu in range(2):
            for parity in range(2):
                mask = (x+y) % 2 == parity
                if interior:
                    mask &= (y if mu == 0 else x) % 2 == 1
                st = staple(u, mu)[:, mask]
                old = u[:, mu, mask]
                proposal = qmul(qexp(rng.normal(size=(*old.shape[:-1], 3))/np.sqrt(beta)), old)
                proposal, err = normalize(proposal)
                max_err = max(max_err, err)
                ds = local_delta(old, proposal, st, beta)
                take = np.log(rng.random(ds.shape)) < -ds
                u[:, mu, mask] = np.where(take[..., None], proposal, old)
                accepted += int(take.sum())
                attempted += take.size
    return u, accepted/attempted, max_err


def heat_kernel_step(shape, delta, rng):
    # Same Gaussian rejection kernel as _heat_kernel_draw in train_su2_dm.py.
    # The sinc factor is the square root of the exponential-map Jacobian.
    out = np.empty((math.prod(shape), 3))
    pending = np.arange(len(out))
    while len(pending):
        xi = rng.normal(scale=np.sqrt(delta), size=(len(pending), 3))
        theta = np.linalg.norm(xi, axis=1)
        accepted = rng.random(len(pending)) < np.sinc(theta/(2*np.pi))
        out[pending[accepted]] = xi[accepted]
        pending = pending[~accepted]
    return out.reshape(*shape, 3)


def forward(u, tau, rng, delta_max=DELTA_MAX):
    if tau == 0:
        return u, 0.0
    k = max(1, math.ceil(tau/delta_max))
    err = 0.0
    for _ in range(k):
        u = qmul(qexp(heat_kernel_step(u.shape[:-1], tau/k, rng)), u)
        u, drift = normalize(u)
        err = max(err, drift)
    return u, err


def winding_lines(u):
    # x loops start at x=0. y loops start at y=0. Both index-zero loops share (0,0).
    px, py = u[:, 0, 0], u[:, 1, :, 0]
    for x in range(1, u.shape[2]):
        px = qmul(px, u[:, 0, x])
    for y in range(1, u.shape[3]):
        py = qmul(py, u[:, 1, :, y])
    return np.stack((px, py), axis=1)


def measure(u):
    loops = np.stack([wilson(u, r, t) for r, t in RT], axis=1)
    # One representative per transverse line avoids counting repeated origins.
    polyakov = winding_lines(u)[..., 0]
    means = loops.mean(axis=(2, 3))
    seconds = (loops**2).mean(axis=(2, 3))
    labels = list(LOOPS) + [f"{s}_second" for s in LOOPS]
    pieces = [means, seconds, polyakov.mean(axis=2), (polyakov**2).mean(axis=2)]
    labels += ["Polyakov_x", "Polyakov_y", "Polyakov_x_second", "Polyakov_y_second"]
    for k in (0, 2):
        for power in (1, 2):
            for x in range(2):
                for y in range(2):
                    pieces.append((loops[:, k, x::2, y::2]**power).mean(axis=(1, 2))[:, None])
                    labels.append(f"{LOOPS[k]}_power{power}_parity{x}{y}")
    for i in range(4):
        for j in range(i+1, 4):
            pieces.append((loops[:, i]*loops[:, j]).mean(axis=(1, 2))[:, None])
            labels.append(f"{LOOPS[i]}_{LOOPS[j]}_product")
    for i in range(4):
        for j in range(i, 4):
            pieces.append((means[:, i]*means[:, j])[:, None])
            labels.append(f"{LOOPS[i]}_{LOOPS[j]}_config_product")
    hist = []
    for a in [loops[:, k].reshape(len(u), -1) for k in range(4)] + [polyakov[:, k] for k in range(2)]:
        bins = np.clip(np.searchsorted(EDGES, a, side="right") - 1, 0, len(EDGES)-2)
        offset = np.arange(len(u))[:, None]*(len(EDGES)-1)
        counts = np.bincount((bins+offset).ravel(), minlength=len(u)*(len(EDGES)-1))
        hist.append(counts.reshape(len(u), -1)/a.shape[1])
    return {"loops": loops, "polyakov": polyakov, "features": np.concatenate(pieces, axis=1),
            "histogram": np.stack(hist, axis=1)}, labels


def measure_holonomy(u):
    omega = winding_lines(u)
    p = omega[..., 0]
    angle = np.arccos(np.clip(p, -1, 1))
    x, y = omega[:, 0, 0], omega[:, 1, 0]
    c = qmul(qmul(qmul(x, y), conj(x)), conj(y))[..., 0]
    values = np.column_stack((angle.mean(axis=2), ((angle-np.pi/2)**2).mean(axis=2),
                              p.mean(axis=2)**2, c, c**2))
    hist = []
    for a, edges in ((angle[:, 0], ANGLE_EDGES), (angle[:, 1], ANGLE_EDGES), (c[:, None], EDGES)):
        bins = np.clip(np.searchsorted(edges, a, side="right")-1, 0, len(edges)-2)
        offset = np.arange(len(a))[:, None]*(len(edges)-1)
        h = np.bincount((bins+offset).ravel(), minlength=len(a)*(len(edges)-1))
        hist.append(h.reshape(len(a), -1)/a.shape[1])
    return values, np.stack(hist, axis=1), p


def blocks(values):
    return values.reshape(len(values)//BLOCK, BLOCK, *values.shape[1:]).mean(axis=1)


def bootstrap_weights(n, rng):
    return rng.multinomial(n, np.full(n, 1/n), size=NBOOT)/n


def statistics(values, labels, weights):
    grouped = blocks(values)
    mean = grouped.mean(axis=0)
    boot = weights @ grouped
    # Derived variances distinguish a local loop from its spatial average.
    out = [mean]
    sampled = [boot]
    names = list(labels)
    for i in range(4):
        out.append(np.atleast_1d(mean[i+4]-mean[i]**2))
        sampled.append((boot[:, i+4]-boot[:, i]**2)[:, None])
        names.append(f"{labels[i]}_site_variance")
    for i in range(4):
        for j in range(i, 4):
            k = labels.index(f"{labels[i]}_{labels[j]}_config_product")
            correction = len(values)/(len(values)-1)
            out.append(np.atleast_1d(correction*(mean[k]-mean[i]*mean[j])))
            sampled.append((correction*(boot[:, k]-boot[:, i]*boot[:, j]))[:, None])
            names.append(f"{labels[i]}_{labels[j]}_config_covariance")
            if i != j:
                k = labels.index(f"{labels[i]}_{labels[j]}_product")
                out.append(np.atleast_1d(mean[k]-mean[i]*mean[j]))
                sampled.append((boot[:, k]-boot[:, i]*boot[:, j])[:, None])
                names.append(f"{labels[i]}_{labels[j]}_site_covariance")
    out, sampled = np.concatenate(out), np.concatenate(sampled, axis=1)
    return names, out, sampled.std(axis=0, ddof=1), np.quantile(sampled, [0.025, 0.975], axis=0)


def compare(reference, candidate, rng):
    # Exchange whole MC blocks. Links and noise replicas are never resampling units.
    a, b = blocks(reference["features"]), blocks(candidate["features"])
    ca = blocks(reference["histogram"].cumsum(axis=-1)[..., :-1])
    cb = blocks(candidate["histogram"].cumsum(axis=-1)[..., :-1])
    n = len(a)
    pool = np.concatenate((np.concatenate((a, ca.reshape(n, -1)), axis=1),
                           np.concatenate((b, cb.reshape(n, -1)), axis=1)))
    scale = np.maximum(pool.std(axis=0, ddof=1)*np.sqrt(2/n), 1e-15)
    observed = np.abs(pool[n:].mean(axis=0)-pool[:n].mean(axis=0))
    # Pooled scales stay fixed under permutations, including nearly fixed features.
    ranks = rng.random((999, 2*n)).argsort(axis=1).argsort(axis=1)
    weights = np.where(ranks < n, 1/n, -1/n)
    permuted = np.abs(weights @ pool)
    p_global = (1 + np.count_nonzero((permuted/scale).max(axis=1) >= (observed/scale).max()))/1000
    cdf_distance = np.abs(cb.mean(axis=0)-ca.mean(axis=0)).max(axis=1)
    cdf_permuted = permuted[:, a.shape[1]:].reshape(999, *ca.shape[1:]).max(axis=2)
    cdf_p = (1 + (cdf_permuted >= cdf_distance).sum(axis=0))/1000
    return p_global, cdf_distance, cdf_p


def correlation(x, y, weights):
    x, y = x.reshape(len(x), -1), y.reshape(len(y), -1)
    moments = np.stack((x.mean(axis=1), y.mean(axis=1), (x*x).mean(axis=1),
                        (y*y).mean(axis=1), (x*y).mean(axis=1)), axis=1)
    def rho(m):
        varx = np.maximum(m[..., 2]-m[..., 0]**2, 1e-30)
        vary = np.maximum(m[..., 3]-m[..., 1]**2, 1e-30)
        return np.clip((m[..., 4]-m[..., 0]*m[..., 1])/np.sqrt(varx*vary), -1, 1)
    boot = rho(weights @ blocks(moments))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(rho(moments.mean(axis=0))), boot.std(ddof=1), lo, hi


def chain_statistics(values):
    x = values-values.mean()
    n = len(x)
    transform = np.fft.rfft(x, n=2*n)
    cov = np.fft.irfft(transform*transform.conj(), n=2*n)[:min(201, n//2)]
    cov /= n-np.arange(len(cov))
    rho = cov/cov[0]
    stop = next((k for k in range(1, len(rho)) if rho[k] <= 0), len(rho))
    tau = 0.5 + rho[1:stop].sum()
    out = [n, values.mean(), values.std(ddof=1)/np.sqrt(n), tau,
           0.5+rho[1:21].sum(), rho[1], np.max(np.abs(rho[1:21]))]
    for size in (1, 4, 16, 64):
        grouped = values[:n//size*size].reshape(-1, size).mean(axis=1)
        out.append(grouped.std(ddof=1)/np.sqrt(len(grouped)))
    return out


def csv_write(path, header, rows):
    with open(path, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def scan(args):
    started = time.perf_counter()
    taus = np.asarray(args.tau)
    if taus[0] != 0 or not np.isfinite(taus).all() or np.any(np.diff(taus) <= 0):
        raise ValueError("Sigma squared must start at zero and increase strictly")
    if args.n_cfgs % BLOCK or args.n_cfgs < 64:
        raise ValueError("Use at least 64 configurations and a multiple of four")
    with h5py.File(args.fine) as f, h5py.File(args.coarse) as c:
        lattice, beta_f = int(f.attrs["L"]), float(f.attrs["beta"])
        coarse_lattice, beta_c = int(c.attrs["L"]), float(c.attrs["beta"])
    tension_f = -np.log(ive(2, beta_f)/ive(1, beta_f))
    tension_c = -np.log(ive(2, beta_c)/ive(1, beta_c))
    if lattice != 2*coarse_lattice or not np.isclose(tension_c, 4*tension_f, rtol=1e-9, atol=0):
        raise ValueError("Coarse and fine ensembles must match physical size at refinement factor two")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a completed scan.
    with h5py.File(args.out, "x") as out:
        out.attrs.update(beta_c=beta_c, beta_f=beta_f, L_coarse=coarse_lattice, L=lattice,
                         seed=args.seed, device="cpu", delta_max=DELTA_MAX,
                         n_cfgs=args.n_cfgs, block_size=BLOCK, bootstrap=NBOOT,
                         coarse=str(args.coarse.resolve()), fine=str(args.fine.resolve()),
                         proposal_width=1/np.sqrt(beta_f), numpy_version=np.__version__)
        out["tau"] = taus
        out["Sigma2"] = taus
        out["Sigma"] = np.sqrt(taus)
        out["histogram_edges"] = EDGES
        ids = {}
        for name, path in (("coarse", args.coarse), ("fine", args.fine)):
            with h5py.File(path) as f:
                total = len(f["cfgs"])
                if total < args.n_cfgs:
                    raise ValueError(f"Insufficient configurations in {path}")
                stride = total//args.n_cfgs
                ids[name] = np.arange(args.n_cfgs)*stride + stride//2
                source = out.create_group(f"source_{name}")
                source["parent_id"] = ids[name]
                for key, value in f.attrs.items():
                    source.attrs[key] = value
                source.attrs["stride_saved_configs"] = stride
        fine, fine_err = load_cfgs(args.fine, ids["fine"], lattice, beta_f)
        coarse, coarse_err = load_cfgs(args.coarse, ids["coarse"], coarse_lattice, beta_c)
        out.attrs.update(fine_input_error=fine_err, coarse_input_error=coarse_err)
        coarse_data, labels = measure(coarse)
        out["source_coarse/cfgs"] = to_matrix(coarse)
        out["source_coarse/loops"] = coarse_data["loops"]
        diagnostic_rows = []
        # Use saved plaquettes for both full chains and loops for the selected parents.
        chain = coarse_data["features"][:, list(range(4))+list(range(8, 12))]
        out["source_coarse/chain_observables"] = chain
        chain_labels = list(LOOPS)+["Polyakov_x", "Polyakov_y", "Polyakov_x_second", "Polyakov_y_second"]
        for j, label in enumerate(chain_labels):
            diagnostic_rows.append(["coarse_selected", label, *chain_statistics(chain[:, j])])
        for name, path in (("coarse", args.coarse), ("fine", args.fine)):
            with h5py.File(path) as f:
                diagnostic_rows.append([name+"_full", "W11", *chain_statistics(f["plaq"][:])])
        uniform, root_info = prolong(coarse)
        out.attrs.update(near_minus_identity=root_info[0], root_reconstruction_error=root_info[1],
                         prolong_norm_error=root_info[2])
        timing = []
        ts = time.perf_counter()
        interior, acc_interior, err_interior = metropolis(uniform, 2, True, np.random.default_rng([args.seed, 10]), beta_f)
        timing.append(["interior_updates", time.perf_counter()-ts])
        ts = time.perf_counter()
        relaxed, acc_relaxed, err_relaxed = metropolis(interior, 1, False, np.random.default_rng([args.seed, 11]), beta_f)
        timing.append(["full_update", time.perf_counter()-ts])
        out.attrs.update(interior_acceptance=acc_interior, full_acceptance=acc_relaxed,
                         interior_norm_error=err_interior, full_norm_error=err_relaxed)
        print(f"Metropolis acceptance interior {acc_interior:.6f} full {acc_relaxed:.6f}", flush=True)
        initials = [fine, uniform, interior, relaxed]
        for index in range(1, 4):
            ts = time.perf_counter()
            initials[index], err = normalize(gauge_transform(initials[index], haar(
                (args.n_cfgs, lattice, lattice), np.random.default_rng([args.seed, index, 100]))))
            out.attrs[f"{NAMES[index]}_gauge_norm_error"] = err
            timing.append([f"{NAMES[index]}_gauge", time.perf_counter()-ts])
        weights = bootstrap_weights(args.n_cfgs//BLOCK, np.random.default_rng([args.seed, 200]))
        out["bootstrap_weights"] = weights
        statistics_rows, comparison_rows, retention_rows, decay_rows = [], [], [], []
        references = []
        for index, (name, initial) in enumerate(zip(NAMES, initials)):
            initial_data = None
            for ti, tau in enumerate(taus):
                stage_started = time.perf_counter()
                ts = time.perf_counter()
                noisy, drift = forward(initial, tau, np.random.default_rng([args.seed, index, ti, 300]))
                noise_seconds = time.perf_counter()-ts
                ts = time.perf_counter()
                data, labels = measure(noisy)
                measure_seconds = time.perf_counter()-ts
                if ti == 0:
                    initial_data = data
                    if name == "fine":
                        for j, label in enumerate(labels[:12]):
                            diagnostic_rows.append(["fine_selected", label, *chain_statistics(data["features"][:, j])])
                if name == "fine":
                    references.append(data)
                group = out.create_group(f"{name}/tau_{tau:g}")
                group.attrs.update(tau=tau, Sigma2=tau, Sigma=np.sqrt(tau), beta=beta_f, L=lattice, seed=args.seed,
                                   noise_stream=[args.seed, index, ti, 300], norm_error=drift,
                                   noise_seconds=noise_seconds, measure_seconds=measure_seconds,
                                   increment_count=0 if tau == 0 else math.ceil(tau/DELTA_MAX))
                group["parent_id"] = ids["fine" if name == "fine" else "coarse"]
                ts = time.perf_counter()
                group.create_dataset("cfgs", data=to_matrix(noisy), chunks=(1, lattice, lattice, 2, 2, 2))
                for key, values in data.items():
                    group[key] = values
                group.attrs["write_seconds"] = time.perf_counter()-ts
                names, means, se, ci = statistics(data["features"], labels, weights)
                group["statistic_names"] = np.asarray(names, dtype=h5py.string_dtype())
                group["statistic_mean"] = means
                group["statistic_se"] = se
                group["statistic_ci95"] = ci
                for j, label in enumerate(names):
                    statistics_rows.append([name, tau, label, means[j], se[j], ci[0, j], ci[1, j],
                                            abs(means[j])/se[j] if se[j] > 0 else np.nan])
                # Conditional heat-kernel identity, paired by the original configuration.
                for j, (r, t) in enumerate(RT):
                    factor = np.exp(-3*2*(r+t)*tau/8)
                    difference = data["features"][:, j]-factor*initial_data["features"][:, j]
                    err = (weights @ blocks(difference)).std(ddof=1)
                    residual = difference.mean()
                    decay_rows.append([name, tau, LOOPS[j], factor, residual, err,
                                       residual/err if err > 0 else 0])
                # Matching physical areas. The fine reference uses the same anchors.
                for j, coarse_j in ((2, 0), (3, 2)):
                    y = data["loops"][:, j, ::2, ::2]
                    x = (initial_data["loops"][:, j, ::2, ::2] if name == "fine"
                         else coarse_data["loops"][:, coarse_j])
                    for kind in ("site", "configuration"):
                        xx, yy = (x, y) if kind == "site" else (x.mean(axis=(1, 2)), y.mean(axis=(1, 2)))
                        retention_rows.append([name, tau, LOOPS[j], kind, *correlation(xx, yy, weights)])
                if name != "fine":
                    p, distance, p_cdf = compare(references[ti], data, np.random.default_rng([args.seed, index, ti, 400]))
                    _, refmean, refse, _ = statistics(references[ti]["features"], labels, weights)
                    group.attrs["comparison_global_p"] = p
                    for j, label in enumerate(names):
                        error = np.hypot(se[j], refse[j])
                        comparison_rows.append([name, tau, label, means[j]-refmean[j], error,
                                                (means[j]-refmean[j])/error if error > 0 else np.nan, p,
                                                np.nan, np.nan])
                    for j, label in enumerate((*LOOPS, "Polyakov_x", "Polyakov_y")):
                        comparison_rows.append([name, tau, label+"_cdf", np.nan, np.nan, np.nan,
                                                p, distance[j], p_cdf[j]])
                timing.append([f"{name}_Sigma2_{tau:g}", time.perf_counter()-stage_started])
                print(f"{name:8s} Sigma^2 {tau:4g} P {means[0]:.7f} +/- {se[0]:.7f} "
                      f"noise {noise_seconds:.2f}s measure {measure_seconds:.2f}s", flush=True)
        out["feature_names"] = np.asarray(labels, dtype=h5py.string_dtype())
        out.attrs["elapsed_seconds"] = time.perf_counter()-started
        timing.append(["total_scan", out.attrs["elapsed_seconds"]])
    prefix = args.out.with_suffix("")
    csv_write(prefix.with_name(prefix.name+"_statistics.csv"),
              ["ensemble", "Sigma2", "observable", "mean", "se", "ci_low", "ci_high", "snr"], statistics_rows)
    csv_write(prefix.with_name(prefix.name+"_comparison.csv"),
              ["ensemble", "Sigma2", "observable", "difference", "se", "z", "global_p", "cdf_distance", "cdf_p"], comparison_rows)
    csv_write(prefix.with_name(prefix.name+"_retention.csv"),
              ["ensemble", "Sigma2", "fine_loop", "averaging", "correlation", "se", "ci_low", "ci_high"], retention_rows)
    csv_write(prefix.with_name(prefix.name+"_heat_kernel.csv"),
              ["ensemble", "Sigma2", "loop", "attenuation", "residual", "se", "z"], decay_rows)
    csv_write(prefix.with_name(prefix.name+"_chains.csv"),
              ["source", "observable", "n", "mean", "iid_se", "tau_first_nonpositive", "tau_window20", "rho1",
               "max_abs_rho20", "se_block1", "se_block4", "se_block16", "se_block64"], diagnostic_rows)
    csv_write(prefix.with_name(prefix.name+"_timing.csv"), ["stage", "seconds"], timing)
    plot(args.out)
    print(f"Saved {args.out} and comparison tables and figures", flush=True)


def measure_square_scan(path):
    sizes = (1, 2, 4, 8, 16, 32)
    rows = []
    with h5py.File(path, "r+") as f:
        output = f.create_group("wilson_squares")
        output["sizes"] = sizes
        weights = f["bootstrap_weights"][:]
        for name in NAMES:
            for variance in f["tau"][:]:
                started = time.perf_counter()
                key = f"{name}/tau_{variance:g}"
                source = f[key]
                values = np.empty((len(source["cfgs"]), len(sizes)))
                values[:, :3] = source["features"][:, [0, 2, 3]]
                for begin in range(0, len(values), 16):
                    end = min(begin+16, len(values))
                    u, _ = from_matrix(source["cfgs"][begin:end])
                    for j, size in enumerate(sizes[3:], 3):
                        values[begin:end, j] = wilson(u, size, size).mean(axis=(1, 2))
                grouped = blocks(values)
                boot = weights @ grouped
                mean, se = grouped.mean(axis=0), boot.std(axis=0, ddof=1)
                ci = np.quantile(boot, [0.025, 0.975], axis=0)
                group = output.create_group(key)
                group["per_config"] = values
                group["mean"], group["se"], group["ci95"] = mean, se, ci
                for j, size in enumerate(sizes[3:], 3):
                    rows.append([name, variance, f"W{size}{size}", mean[j], se[j], ci[0, j], ci[1, j],
                                 abs(mean[j])/se[j]])
                print(f"Square loops {name:8s} Sigma^2 {variance:4g} "
                      f"W32 {mean[-1]:.7f} +/- {se[-1]:.7f} "
                      f"seconds {time.perf_counter()-started:.2f}", flush=True)
    prefix = path.with_suffix("")
    with prefix.with_name(prefix.name+"_statistics.csv").open("a", newline="") as stream:
        csv.writer(stream).writerows(rows)


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.style.use(ROOT.parent / "plots" / "cjp.mplstyles")
    colors = ("#315fce", "#ca3300", "#ec9800", "#00982a")
    markers = ("o", "^", "D", "s")
    prefix = path.with_suffix("")
    with h5py.File(path) as f, PdfPages(prefix.with_suffix(".pdf")) as pdf:
        taus = f["tau"][:]
        ticks = [x for x in (0, 0.05, 0.2, 0.8, 3, 10) if x <= taus[-1]]
        snapshots = (0, len(taus)//2, len(taus)-1)
        stats = {}
        for name in NAMES:
            stats[name] = []
            for tau in taus:
                group = f[f"{name}/tau_{tau:g}"]
                names = group["statistic_names"].asstr()[:]
                stats[name].append(dict(zip(names, zip(group["statistic_mean"][:], group["statistic_se"][:]))))

        mean_rt, mean_loops, mean_stats = RT, LOOPS, stats
        if "wilson_squares" in f:
            sizes = f["wilson_squares/sizes"][:]
            mean_rt = [(size, size) for size in sizes]
            mean_loops = [f"W{size}{size}" for size in sizes]
            mean_stats = {}
            for name in NAMES:
                mean_stats[name] = []
                for tau in taus:
                    group = f[f"wilson_squares/{name}/tau_{tau:g}"]
                    mean_stats[name].append(dict(zip(mean_loops, zip(group["mean"][:], group["se"][:]))))

        def curves(ax, observable, ylabel, source=stats):
            for name, color, marker in zip(NAMES, colors, markers):
                values = np.array([row[observable] for row in source[name]])
                ax.errorbar(taus, values[:, 0], yerr=values[:, 1], color=color, marker=marker,
                            label=name, markersize=3)
            ax.set_xscale("symlog", linthresh=0.03)
            ax.set_xlim(-0.004, taus[-1]*1.2)
            ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
            ax.set_xlabel(r"$\Sigma^2$")
            ax.set_ylabel(ylabel)

        for page, (suffix, title, ylabel) in enumerate((
                ("", rf"Forward diffusion    $L={f.attrs['L']}$, $\beta={f.attrs['beta_f']:g}$", r"$\langle W\rangle$"),
                ("_second", "Second moments of individual loops", r"$\langle W^2\rangle$"),
                ("_site_variance", "Variance of individual loops", r"$\mathrm{var}(W)$"),
                ("_config_covariance", "Variance of spatially averaged loops", r"$\mathrm{var}(\overline{W})$"))):
            rt, loops, source = (mean_rt, mean_loops, mean_stats) if page == 0 else (RT, LOOPS, stats)
            fig, axes = plt.subplots(2, len(loops)//2, figsize=(7.0, 4.8))
            for j, (ax, loop) in enumerate(zip(axes.flat, loops)):
                key = f"{loop}_{loop}{suffix}" if suffix == "_config_covariance" else loop+suffix
                curves(ax, key, ylabel, source)
                ax.set_title(rf"${rt[j][0]}\times{rt[j][1]}$")
                if page == 0 and j == len(loops)-1:
                    ax.axhline(0, color="0.5", linewidth=0.6)
            axes.flat[0].legend(ncol=2, fontsize=7)
            fig.suptitle(title + rf"    $N={f.attrs['n_cfgs']}$    Block errors")
            pdf.savefig(fig)
            if page == 0:
                fig.savefig(prefix.with_suffix(".png"), dpi=180)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for j, (ax, loop) in enumerate(zip(axes.flat, LOOPS)):
            reference = np.array([row[loop] for row in stats["fine"]])
            for name, color, marker in zip(NAMES[1:], colors[1:], markers[1:]):
                values = np.array([row[loop] for row in stats[name]])
                ax.errorbar(taus, values[:, 0]-reference[:, 0],
                            yerr=np.hypot(values[:, 1], reference[:, 1]),
                            color=color, marker=marker, label=name, markersize=3)
            ax.axhline(0, color="0.5", linewidth=0.7)
            ax.set_xscale("symlog", linthresh=0.03)
            ax.set_xlim(-0.004, taus[-1]*1.2)
            ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
            ax.set_xlabel(r"$\Sigma^2$")
            ax.set_ylabel(r"$\langle W\rangle-\langle W\rangle_{\mathrm{fine}}$")
            ax.set_title(rf"${RT[j][0]}\times{RT[j][1]}$")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle("Absolute differences from the fine ensemble at the same noise scale")
        pdf.savefig(fig)
        plt.close(fig)

        for kind in ("site", "config"):
            fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
            for ax, (i, j) in zip(axes.flat, ((i, j) for i in range(4) for j in range(i+1, 4))):
                curves(ax, f"{LOOPS[i]}_{LOOPS[j]}_{kind}_covariance", "Covariance")
                ax.set_title(rf"${RT[i][0]}\times{RT[i][1]}$ and ${RT[j][0]}\times{RT[j][1]}$")
                ax.axhline(0, color="0.5", linewidth=0.6)
            axes.flat[0].legend(ncol=2, fontsize=6)
            fig.suptitle("Covariances of " + ("individual loops at the same anchor" if kind == "site" else "spatially averaged loops"))
            pdf.savefig(fig)
            plt.close(fig)

        # Fixed ensemble colors and anchor line styles expose the block pattern over time.
        for power in (1, 2):
            fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
            for row, loop in enumerate(("W11", "W22")):
                for col, (name, color) in enumerate(zip(NAMES[1:], colors[1:])):
                    ax = axes[row, col]
                    ref = np.array([r[loop if power == 1 else loop+"_second"] for r in stats["fine"]])
                    ax.errorbar(taus, ref[:, 0], yerr=ref[:, 1], color=colors[0], marker="o", label="fine all anchors", markersize=2)
                    for parity, style, marker in zip(("00", "01", "10", "11"), ("-", "--", "-.", ":"), ("o", "^", "s", "D")):
                        values = np.array([r[f"{loop}_power{power}_parity{parity}"] for r in stats[name]])
                        ax.errorbar(taus, values[:, 0], yerr=values[:, 1], color=color, linestyle=style,
                                    marker=marker, markersize=2, label=parity, linewidth=0.8)
                    ax.set_xscale("symlog", linthresh=0.03)
                    ax.set_xlim(-0.004, taus[-1]*1.2)
                    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
                    ax.set_xlabel(r"$\Sigma^2$")
                    ax.set_ylabel(r"$\langle W\rangle$" if power == 1 else r"$\langle W^2\rangle$")
                    ax.set_title(rf"{name}    ${loop[1]}\times{loop[2]}$")
            axes[0, 0].legend(ncol=2, fontsize=5.5)
            fig.suptitle("Four anchor parities as a function of noise scale")
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
        parity = ("00", "01", "10", "11")
        for row, loop in enumerate(("W11", "W22")):
            for col, ti in enumerate(snapshots):
                ax = axes[row, col]
                for name, color, marker in zip(NAMES, colors, markers):
                    values = np.array([stats[name][ti][f"{loop}_power1_parity{p}"] for p in parity])
                    ax.errorbar(np.arange(4), values[:, 0], yerr=values[:, 1], color=color, marker=marker,
                                label=name, markersize=3)
                ax.set_xticks(range(4), parity)
                ax.set_xlabel("Anchor parity")
                ax.set_ylabel(r"$\langle W\rangle$")
                ax.set_title(rf"${loop[1]}\times{loop[2]}$    $\Sigma^2={taus[ti]:g}$")
        axes[0, 0].legend(ncol=2, fontsize=6)
        fig.suptitle("Even and odd anchors retain their original block coordinates")
        pdf.savefig(fig)
        plt.close(fig)

        with open(prefix.with_name(prefix.name+"_retention.csv")) as stream:
            retention = list(csv.DictReader(stream))
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for row, kind in enumerate(("site", "configuration")):
            for col, loop in enumerate(("W22", "W44")):
                ax = axes[row, col]
                for name, color, marker in zip(NAMES, colors, markers):
                    selected = [r for r in retention if r["ensemble"] == name and r["averaging"] == kind and r["fine_loop"] == loop]
                    ax.errorbar(taus, [float(r["correlation"]) for r in selected],
                                yerr=[float(r["se"]) for r in selected], color=color, marker=marker,
                                label=name, markersize=3)
                ax.axhline(0, color="0.5", linewidth=0.6)
                ax.set_ylim(-0.3, 1.05)
                ax.set_xscale("symlog", linthresh=0.03)
                ax.set_xlim(-0.004, taus[-1]*1.2)
                ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
                ax.set_xlabel(r"$\Sigma^2$")
                ax.set_ylabel("Correlation")
                ax.set_title(rf"${loop[1]}\times{loop[2]}$    {kind}")
        axes[0, 0].legend(ncol=2, fontsize=7)
        fig.suptitle("Input retention at matching physical area    Even anchors")
        pdf.savefig(fig)
        plt.close(fig)

        for obs, title in ((range(4), "Distributions of individual Wilson loops"),
                           (range(4, 6), "Distributions of winding Polyakov loops")):
            fig, axes = plt.subplots(len(obs), 3, figsize=(7.0, 2.0*len(obs)), squeeze=False)
            for row, j in enumerate(obs):
                for col, ti in enumerate(snapshots):
                    ax = axes[row, col]
                    for name, color in zip(NAMES, colors):
                        hist = f[f"{name}/tau_{taus[ti]:g}/histogram"][:, j].mean(axis=0)
                        ax.stairs(hist/np.diff(EDGES), EDGES, color=color, label=name)
                    label = (*LOOPS, "Polyakov x", "Polyakov y")[j]
                    ax.set_title(rf"{label}    $\Sigma^2={taus[ti]:g}$")
                    ax.set_xlabel(r"$\mathrm{Re\ Tr}\,U/2$")
                    ax.set_ylabel("Density")
            axes[0, 0].legend(fontsize=6)
            fig.suptitle(title)
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for col, mu in enumerate(("x", "y")):
            curves(axes[0, col], f"Polyakov_{mu}", r"$\langle P_\mu\rangle$")
            curves(axes[1, col], f"Polyakov_{mu}_second", r"$\langle P_\mu^2\rangle$")
            axes[0, col].axhline(0, color="0.5", linestyle="--", linewidth=0.7)
            axes[1, col].axhline(0.25, color="0.5", linestyle="--", linewidth=0.7)
            axes[0, col].set_title(f"Polyakov {mu}")
        axes[0, 0].legend(ncol=2, fontsize=7)
        fig.suptitle("Polyakov moments    Dashed lines show the Haar values")
        pdf.savefig(fig)
        plt.close(fig)

        with open(prefix.with_name(prefix.name+"_comparison.csv")) as stream:
            comparison = list(csv.DictReader(stream))
        fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
        for ax, loop in zip(axes.flat, (*LOOPS, "Polyakov_x", "Polyakov_y")):
            for name, color, marker in zip(NAMES[1:], colors[1:], markers[1:]):
                rows = [r for r in comparison if r["ensemble"] == name and r["observable"] == loop+"_cdf"]
                ax.plot(taus, [float(r["cdf_distance"]) for r in rows], color=color, marker=marker, label=name, markersize=3)
            ax.set_xscale("symlog", linthresh=0.03)
            ax.set_xlim(-0.004, taus[-1]*1.2)
            ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
            ax.set_xlabel(r"$\Sigma^2$")
            ax.set_ylabel(r"$\max_w |F(w)-F_{\mathrm{fine}}(w)|$")
            ax.set_title(loop.replace("_", " "))
        axes.flat[0].legend(fontsize=6)
        fig.suptitle("Distribution differences at equal noise scale    40 histogram bins")
        pdf.savefig(fig)
        plt.close(fig)


def check():
    rng = np.random.default_rng(38491)
    u = haar((3, 2, 4, 4), rng)
    recovered, _ = from_matrix(to_matrix(u))
    np.testing.assert_allclose(recovered, u, atol=1e-14)
    v = haar(u.shape[:-1], rng)
    np.testing.assert_allclose(to_matrix(qmul(u, v)), to_matrix(u) @ to_matrix(v), atol=1e-14)
    bad = to_matrix(u)
    bad[0, 0, 0, 0, 1, 0] += 0.1
    try:
        from_matrix(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid matrices must fail")
    # Explicit path crossing both periodic boundaries, in complex matrices.
    m = to_matrix(u)
    product = np.eye(2, dtype=complex)
    x = y = 3
    for mu, count, sign in ((0, 2, 1), (1, 3, 1), (0, 2, -1), (1, 3, -1)):
        for _ in range(count):
            if sign < 0:
                x, y = (x-1) % 4 if mu == 0 else x, (y-1) % 4 if mu == 1 else y
            link = m[0, y, x, mu]
            product = product @ (link if sign > 0 else link.conj().T)
            if sign > 0:
                x, y = (x+1) % 4 if mu == 0 else x, (y+1) % 4 if mu == 1 else y
    np.testing.assert_allclose(product.trace().real/2, wilson(u, 2, 3)[0, 3, 3], atol=1e-14)
    w, _ = prolong(u)
    for mu in range(2):
        np.testing.assert_allclose(line(w, mu, 2)[:, ::2, ::2], u[:, mu], atol=1e-14)
    q, _, _ = fourth_root(plaquette(u))
    for x in range(2):
        for y in range(2):
            np.testing.assert_allclose(plaquette(w)[:, x::2, y::2], q, atol=1e-14)
    omega = haar((3, 4, 4), rng)
    lifted = omega.repeat(2, axis=1).repeat(2, axis=2)
    transformed, _ = prolong(gauge_transform(u, omega))
    np.testing.assert_allclose(transformed, gauge_transform(w, lifted), atol=1e-14)
    np.testing.assert_allclose(fourth_root(np.array([1., 0, 0, 0]))[0], [1, 0, 0, 0])
    near = np.array([-np.sqrt(1-1e-16), 1e-8, 0, 0])
    assert fourth_root(near)[1] == 1
    try:
        fourth_root(np.array([-1., 0, 0, 0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Minus identity must fail")
    for mu in range(2):
        proposed = qmul(qexp(rng.normal(size=(3, 3))), u[:, mu, 3, 0])
        ds = local_delta(u[:, mu, 3, 0], proposed, staple(u, mu)[:, 3, 0])
        changed = u.copy()
        changed[:, mu, 3, 0] = proposed
        full = -BETA_F*(wilson(changed, 1, 1)-wilson(u, 1, 1)).sum(axis=(1, 2))
        np.testing.assert_allclose(ds, full, atol=1e-13)
        # A whole checkerboard substep has additive local action differences.
        mask = np.indices((4, 4)).sum(axis=0) % 2 == 0
        proposed = qmul(qexp(rng.normal(size=(*u[:, mu, mask].shape[:-1], 3))), u[:, mu, mask])
        ds = local_delta(u[:, mu, mask], proposed, staple(u, mu)[:, mask]).sum(axis=1)
        changed = u.copy()
        changed[:, mu, mask] = proposed
        full = -BETA_F*(wilson(changed, 1, 1)-wilson(u, 1, 1)).sum(axis=(1, 2))
        np.testing.assert_allclose(ds, full, atol=1e-13)
    # Rotate the same proposals, and share the acceptance uniforms, in both gauges.
    a = u.copy()
    b = gauge_transform(a, omega)
    for mu in range(2):
        for parity in range(2):
            mask = np.indices((4, 4)).sum(axis=0) % 2 == parity
            xi = rng.normal(size=(*a[:, mu, mask].shape[:-1], 3))/np.sqrt(BETA_F)
            algebra = np.concatenate((np.zeros_like(xi[..., :1]), xi), axis=-1)
            rotated = qmul(qmul(omega[:, mask], algebra), conj(omega[:, mask]))[..., 1:]
            pa = qmul(qexp(xi), a[:, mu, mask])
            pb = qmul(qexp(rotated), b[:, mu, mask])
            da = local_delta(a[:, mu, mask], pa, staple(a, mu)[:, mask])
            db = local_delta(b[:, mu, mask], pb, staple(b, mu)[:, mask])
            np.testing.assert_allclose(da, db, atol=1e-13)
            threshold = np.log(rng.random(da.shape))
            np.testing.assert_array_equal(threshold < -da, threshold < -db)
            a[:, mu, mask] = np.where((threshold < -da)[..., None], pa, a[:, mu, mask])
            b[:, mu, mask] = np.where((threshold < -db)[..., None], pb, b[:, mu, mask])
    np.testing.assert_allclose(b, gauge_transform(a, omega), atol=1e-13)
    wi, rate, _ = metropolis(w, 2, True, rng)
    assert 0 < rate < 1
    np.testing.assert_array_equal(wi[:, 0, :, ::2], w[:, 0, :, ::2])
    np.testing.assert_array_equal(wi[:, 1, ::2, :], w[:, 1, ::2, :])
    for r, t in ((1, 1), (1, 2), (2, 2)):
        np.testing.assert_allclose(wilson(wi, 2*r, 2*t)[:, ::2, ::2], wilson(u, r, t), atol=2e-14)
    wg = gauge_transform(wi, haar((3, 8, 8), rng))
    for r, t in RT:
        np.testing.assert_allclose(wilson(wg, r, t), wilson(wi, r, t), atol=2e-14)
    np.testing.assert_allclose(measure(wg)[0]["polyakov"], measure(wi)[0]["polyakov"], atol=2e-14)
    assert forward(u, 0, rng)[0] is u
    for delta in (0.02, 0.8, 1.0):
        scalar = qexp(heat_kernel_step((200000,), delta, rng))[:, 0]
        z = (scalar.mean() - np.exp(-3*delta/8))/(scalar.std(ddof=1)/np.sqrt(len(scalar)))
        assert abs(z) < 5, (delta, z)
        print(f"Heat kernel increment delta {delta:g} fundamental-character residual {z:.3f} SE", flush=True)
    identity = np.zeros((128, 2, 16, 16, 4))
    identity[..., 0] = 1
    for cap in (DELTA_MAX, DELTA_MAX/2):
        noisy, _ = forward(identity, 1.6, rng, cap)
        for r, t in RT:
            values = wilson(noisy, r, t).mean(axis=(1, 2))
            exact = np.exp(-3*2*(r+t)*1.6/8)
            z = (values.mean()-exact)/(values.std(ddof=1)/np.sqrt(len(values)))
            assert abs(z) < 5, (cap, r, t, z)
            print(f"Heat kernel loop {r}x{t} cap {cap:g} residual {z:.3f} SE", flush=True)
    print("CPU checks passed for matrices, periodic paths, roots, skeleton, flux, gauge covariance, action and heat kernel", flush=True)


def save_network_group(out, name, u, weights, attributes):
    # Network arithmetic is float32. Correct only its small accumulated norm drift.
    norm = np.linalg.norm(u, axis=-1, keepdims=True)
    err = float(np.max(np.abs(norm-1)))
    if not np.isfinite(err) or err > 1e-3:
        raise ValueError(f"Invalid network output norm error {err}")
    u = u/norm
    data, labels = measure(u)
    names, mean, se, ci = statistics(data["features"], labels, weights)
    group = out.create_group(name)
    group.attrs.update(**attributes, norm_error_before_projection=err)
    group.create_dataset("cfgs", data=to_matrix(u), chunks=(1, 32, 32, 2, 2, 2))
    for key, values in data.items():
        group[key] = values
    group["statistic_names"] = np.asarray(names, dtype=h5py.string_dtype())
    group["statistic_mean"] = mean
    group["statistic_se"] = se
    group["statistic_ci95"] = ci
    out.flush()
    print(f"Saved {name} P {mean[0]:.8f} +/- {se[0]:.8f} norm drift {err:.2e}", flush=True)
    return u


def validate_network(args):
    import sys
    sys.path.insert(0, str(ROOT / "src/SU2"))
    import torch
    from torch_ema import ExponentialMovingAverage
    import train_su2_dm as train
    import sample_su2_em as sampler

    if args.n_cfgs < 64 or args.n_cfgs % BLOCK:
        raise ValueError("Use at least 64 configurations and a multiple of four")
    counts = np.asarray(args.steps, dtype=int)
    if np.any(counts <= 0) or np.any(counts > sampler.N_STEPS) or np.any(np.diff(counts) <= 0):
        raise ValueError("Remaining step counts must increase and lie within the full grid")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ck["L"] != 32 or ck["beta"] != BETA_F:
        raise ValueError("Network validation requires an L32 beta=2 checkpoint")
    sigma = float(ck["sigma"])
    full_times = sampler.em_time_grid(eps=args.eps)
    t_starts = np.array([0.]+[full_times[-(int(n)+1)].item() for n in counts])
    taus = np.expm1(2*t_starts*np.log(sigma))/(2*np.log(sigma))
    model = train.LCNN(ck["ch"], ck["nlayers"], sigma)
    ema = ExponentialMovingAverage(model.parameters(), decay=train.EMA_DECAY)
    ema.load_state_dict(ck["ema"])
    ema.copy_to(model.parameters())
    model.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.forward) as source, h5py.File(args.out, "x") as out, torch.no_grad():
        if source.attrs["L"] != 32 or source.attrs["beta_f"] != BETA_F:
            raise ValueError("Forward input has the wrong lattice or coupling")
        if Path(source.attrs["fine"]).resolve() != args.fine.resolve():
            raise ValueError("Forward parent indices must refer to the supplied fine ensemble")
        with h5py.File(args.fine) as f:
            stride = len(f["cfgs"])//args.n_cfgs
        if stride < 2:
            raise ValueError("Need disjoint fine input and reference ensembles")
        if not 0 <= args.parent_offset < stride//2:
            raise ValueError("Parent offset must lie in the first half of the sampling stride")
        reference_id = np.arange(args.n_cfgs)*stride + args.parent_offset
        if args.parent_offset == 0 and args.n_cfgs <= source.attrs["n_cfgs"]:
            parent_id = source["source_fine/parent_id"][:args.n_cfgs]
            clean, _ = from_matrix(source["fine/tau_0/cfgs"][:args.n_cfgs])
        else:
            # Larger validation runs draw additional independent fine parents.
            parent_id = reference_id + stride//2
            clean, _ = load_cfgs(args.fine, parent_id, 32, BETA_F)
        if np.intersect1d(parent_id, reference_id).size:
            raise ValueError("The equilibrium reference must use disjoint parent configurations")
        reference, _ = load_cfgs(args.fine, reference_id, 32, BETA_F)
        weights = bootstrap_weights(args.n_cfgs//BLOCK, np.random.default_rng([args.seed, 500]))
        out.attrs.update(checkpoint=str(args.checkpoint.resolve()), forward=str(args.forward.resolve()),
                         fine=str(args.fine.resolve()), epoch=ck["epoch"], L=ck["L"], beta=ck["beta"],
                         sigma=sigma, ch=ck["ch"], nlayers=ck["nlayers"], ema=True,
                         device=torch.cuda.get_device_name(), dtype="float32", eps=args.eps,
                         seed=args.seed, n_cfgs=args.n_cfgs, block_size=BLOCK, bootstrap=NBOOT,
                         parent_offset=args.parent_offset,
                         schedule="suffix_of_full_linear_grid", full_steps=sampler.N_STEPS,
                         final_noise=0, delta_max=train.DELTA_MAX,
                         torch_version=str(torch.__version__), numpy_version=np.__version__)
        out["tau"] = taus
        out["t_start"] = t_starts
        out["steps"] = np.r_[0, counts]
        out["full_times"] = full_times.cpu().numpy()
        out["parent_id"] = parent_id
        out["reference_parent_id"] = reference_id
        out["bootstrap_weights"] = weights
        out["histogram_edges"] = EDGES
        out["feature_names"] = np.asarray(measure(clean[:1])[1], dtype=h5py.string_dtype())
        save_network_group(out, "reference", reference, weights, {"kind": "reference", "tau": 0., "n_steps": 0})
        save_network_group(out, "input", clean, weights, {"kind": "input", "tau": 0., "n_steps": 0})
        initials = {"fine": clean}
        if args.coarse is not None:
            preparation_start = time.perf_counter()
            with h5py.File(args.coarse) as f:
                coarse_stride = len(f["cfgs"])//args.n_cfgs
                coarse_attributes = dict(f.attrs)
            if coarse_stride < 1:
                raise ValueError("Need one distinct coarse parent per output")
            coarse_id = np.arange(args.n_cfgs)*coarse_stride + coarse_stride//2
            coarse, coarse_error = load_cfgs(args.coarse, coarse_id, 16, BETA_C)
            source_coarse = out.create_group("source_coarse")
            source_coarse.attrs.update(coarse_attributes)
            source_coarse.attrs.update(L=16, beta=BETA_C, stride_saved_configs=coarse_stride)
            source_coarse["parent_id"], source_coarse["cfgs"] = coarse_id, to_matrix(coarse)
            coarse_data, _ = measure(coarse)
            for key, values in coarse_data.items():
                source_coarse[key] = values
            uniform, root_info = prolong(coarse)
            interior, acc_i, err_i = metropolis(uniform, 2, True, np.random.default_rng([args.seed, 10]))
            relaxed, acc_r, err_r = metropolis(interior, 1, False, np.random.default_rng([args.seed, 11]))
            out.attrs.update(coarse=str(args.coarse.resolve()), beta_c=BETA_C, L_coarse=16,
                             coarse_input_error=coarse_error, coarse_stride=coarse_stride,
                             near_minus_identity=root_info[0], root_reconstruction_error=root_info[1],
                             prolong_norm_error=root_info[2], interior_acceptance=acc_i, full_acceptance=acc_r,
                             interior_norm_error=err_i, full_norm_error=err_r, proposal_width=1/np.sqrt(BETA_F))
            for index, (kind, initial) in enumerate(zip(NAMES[1:], (uniform, interior, relaxed)), start=1):
                if kind not in args.initializations:
                    continue
                initial, error = normalize(gauge_transform(initial, haar(
                    (args.n_cfgs, 32, 32), np.random.default_rng([args.seed, index, 100]))))
                out.attrs[f"{kind}_gauge_norm_error"] = error
                initials[kind] = save_network_group(out, f"{kind}_input", initial, weights,
                                                    {"kind": kind, "stage": "initial", "tau": 0., "n_steps": 0})
            out.attrs["preparation_seconds"] = time.perf_counter()-preparation_start
            print(f"Refinement acceptance interior {acc_i:.6f} full {acc_r:.6f}", flush=True)
        print(f"Validate epoch {ck['epoch']} EMA sigma {sigma:g} N {args.n_cfgs} eps {args.eps:g}", flush=True)
        print(f"Remaining steps {out['steps'][:].tolist()}. Derived tau {taus.tolist()}. Haar steps {sampler.N_STEPS}", flush=True)
        # Compile once at the production shape. Per-case seeds are set after warmup.
        u = torch.tensor(clean, dtype=torch.float32, device="cuda")
        time_value, time_step, noise = (torch.tensor(x, device="cuda") for x in (0.1, 0.001, 1.0))
        print("Warm up the existing compiled EM step", flush=True)
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            u.copy_(sampler.em_step(model, u, time_value, time_step, noise))
        torch.cuda.synchronize()
        del u
        started = time.perf_counter()
        # Haar keeps the original 2000-step setting. The remaining cases start at t_star.
        cases = [("haar_steps2000", "haar", 1., taus[-1] if counts[-1] == sampler.N_STEPS else (sigma**2-1)/(2*np.log(sigma)),
                  sampler.N_STEPS, args.seed+1000001)]
        cases += [(f"fine_steps{steps}", "fine", t_starts[ti], taus[ti], int(steps), args.seed+10000+int(steps))
                  for ti, steps in enumerate(counts, start=1)]
        for kind in tuple(initials)[1:]:
            index = NAMES.index(kind)
            cases += [(f"{kind}_steps{steps}", kind, t_starts[ti], taus[ti], int(steps),
                       args.seed+10000+int(steps)+100000*index) for ti, steps in enumerate(counts, start=1)]
        clean_gpu = {kind: torch.tensor(initial, dtype=torch.float32, device="cuda") for kind, initial in initials.items()}
        for name, kind, t_start, tau, planned_steps, seed in cases:
            if kind != "haar":
                forward_seed = seed+10000
                torch.manual_seed(forward_seed)
                k = max(1, math.ceil(tau/train.DELTA_MAX))
                delta = torch.full((args.n_cfgs, 1, 1, 1), float(tau/k), device="cuda")
                u_start = clean_gpu[kind].clone()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(k):
                    noise_field = train.heat_kernel_step(u_start.shape[:-1], delta)
                    u_start = train.qmul(train.qexp(noise_field), u_start)
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter()-start
                forward_name = f"forward_steps{planned_steps}" if kind == "fine" else f"{kind}_forward_steps{planned_steps}"
                initial = save_network_group(out, forward_name, u_start.cpu().numpy().astype(np.float64), weights,
                                             {"kind": "forward", "tau": tau, "t_start": t_start, "n_steps": 0,
                                              "initialization": kind,
                                              "reverse_steps": planned_steps, "seed": forward_seed,
                                              "forward_increments": k, "increment_variance": tau/k,
                                              "sampling_seconds": forward_seconds})
                u_start = torch.tensor(initial, dtype=torch.float32, device="cuda")
            else:
                u_start = None
            if args.coarse is not None and kind == "fine" and "fine" not in args.initializations:
                # Keep the noisy-fine control even when its reverse branch is omitted.
                del u_start
                continue
            print(f"Sample {name}. Configurations {args.n_cfgs}. Reverse steps {planned_steps}", flush=True)
            torch.manual_seed(seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            u = sampler.em_sample(model, args.n_cfgs, ck["L"], U_start=u_start,
                                  t_start=float(t_start), eps=args.eps)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            times = sampler.em_time_grid(float(t_start), eps=args.eps)
            steps = len(times)-1
            if steps != planned_steps or not torch.equal(times, full_times[-(steps+1):]):
                raise AssertionError("Partial reverse grid must be an exact suffix of the full grid")
            max_variance = (train.diffusion_coeff(times[:-1], sigma)**2*(times[:-1]-times[1:])).max().item()
            q = u.cpu().numpy().astype(np.float64)
            save_network_group(out, name, q, weights,
                               {"kind": kind, "tau": tau, "t_start": t_start, "n_steps": steps,
                                "eps": args.eps, "seed": seed, "sampling_seconds": seconds,
                                "max_increment_variance": max_variance, "final_noise": 0})
            out[name]["times"] = times.cpu().numpy()
            print(f"Sampling {seconds:.2f}s max increment {max_variance:.6f}", flush=True)
            del u, u_start
        out.attrs["sampling_and_measurement_seconds"] = time.perf_counter()-started
    if args.coarse is None:
        analyze_network(args.out)
        analyze_holonomy(args.out)
        plot_network(args.out)
    else:
        analyze_refinement(args.out)
        plot_refinement(args.out)
    print(f"Network validation saved to {args.out}", flush=True)


def analyze_network(path):
    summary_rows, comparison_rows, paired_rows, retention_rows = [], [], [], []
    with h5py.File(path, "r+") as f:
        weights = f["bootstrap_weights"][:]
        labels = f["feature_names"].asstr()[:].tolist()
        reference = {key: f[f"reference/{key}"][:] for key in ("features", "histogram")}
        clean = f["input/features"][:]
        clean_loops = f["input/loops"][:]
        for name in ("reference", "input", "haar_steps2000", *[f"fine_steps{steps}" for steps in f["steps"][1:]]):
            group = f[name]
            kind, tau, steps = group.attrs["kind"], group.attrs["tau"], group.attrs["n_steps"]
            names = group["statistic_names"].asstr()[:]
            mean, se, ci = (group[key][:] for key in ("statistic_mean", "statistic_se", "statistic_ci95"))
            for j, label in enumerate(names):
                summary_rows.append([name, kind, tau, steps, label, mean[j], se[j], ci[0, j], ci[1, j]])
            if kind == "reference":
                continue
            data = {key: group[key][:] for key in ("features", "histogram")}
            stream = [int(f.attrs["seed"]), 600, int(round(tau*10000)), int(steps)]
            p, distance, cdf_p = compare(reference, data, np.random.default_rng(stream))
            group.attrs["comparison_global_p"] = p
            refmean, refse = f["reference/statistic_mean"][:], f["reference/statistic_se"][:]
            for j, label in enumerate(names):
                error = np.hypot(se[j], refse[j])
                comparison_rows.append([name, kind, tau, steps, label, mean[j]-refmean[j], error,
                                        (mean[j]-refmean[j])/error if error else np.nan, p, np.nan, np.nan])
            for j, label in enumerate((*LOOPS, "Polyakov_x", "Polyakov_y")):
                comparison_rows.append([name, kind, tau, steps, label+"_cdf", np.nan, np.nan, np.nan,
                                        p, distance[j], cdf_p[j]])
            if kind != "fine":
                continue
            difference = data["features"]-clean
            paired_se = (weights @ blocks(difference)).std(axis=0, ddof=1)
            for j, label in enumerate(labels):
                paired_rows.append([name, "input", tau, label, difference[:, j].mean(), paired_se[j],
                                    difference[:, j].mean()/paired_se[j]])
            loops = group["loops"][:]
            for j in (2, 3):
                x, y = clean_loops[:, j, ::2, ::2], loops[:, j, ::2, ::2]
                for averaging in ("site", "configuration"):
                    xx, yy = (x, y) if averaging == "site" else (x.mean(axis=(1, 2)), y.mean(axis=(1, 2)))
                    retention_rows.append(["reverse", tau, steps, LOOPS[j], averaging, *correlation(xx, yy, weights)])
        for tau, steps in zip(f["tau"][:], f["steps"][:]):
            noisy_loops = clean_loops if steps == 0 else f[f"forward_steps{steps}/loops"][:]
            for j in (2, 3):
                x, y = clean_loops[:, j, ::2, ::2], noisy_loops[:, j, ::2, ::2]
                for averaging in ("site", "configuration"):
                    xx, yy = (x, y) if averaging == "site" else (x.mean(axis=(1, 2)), y.mean(axis=(1, 2)))
                    retention_rows.append(["forward", tau, steps, LOOPS[j], averaging, *correlation(xx, yy, weights)])
    prefix = path.with_suffix("")
    csv_write(prefix.with_name(prefix.name+"_statistics.csv"),
              ["ensemble", "kind", "tau", "steps", "observable", "mean", "se", "ci_low", "ci_high"], summary_rows)
    csv_write(prefix.with_name(prefix.name+"_comparison.csv"),
              ["ensemble", "kind", "tau", "steps", "observable", "difference", "se", "z", "global_p", "cdf_distance", "cdf_p"], comparison_rows)
    csv_write(prefix.with_name(prefix.name+"_paired.csv"),
              ["a", "b", "tau", "observable", "difference", "se", "z"], paired_rows)
    csv_write(prefix.with_name(prefix.name+"_retention.csv"),
              ["stage", "tau", "steps", "loop", "averaging", "correlation", "se", "ci_low", "ci_high"], retention_rows)


def analyze_holonomy(path):
    summary, comparisons, retention = [], [], []
    names = list(HOLONOMY_NAMES)
    angle_edges = ANGLE_EDGES
    with h5py.File(path, "r+") as f:
        weights = f["bootstrap_weights"][:]
        f["holonomy_feature_names"] = np.asarray(names, dtype=h5py.string_dtype())
        f["holonomy_angle_edges"] = angle_edges
        counts = f["steps"][1:]
        ensembles = ("reference", "input", "haar_steps2000",
                     *[f"fine_steps{n}" for n in counts], *[f"forward_steps{n}" for n in counts])
        for name in ensembles:
            group = f[name]
            u, _ = from_matrix(group["cfgs"][:])
            values, hist, p = measure_holonomy(u)
            if not np.allclose(p, group["polyakov"][:], rtol=0, atol=1e-12):
                raise AssertionError("Holonomy traces must reproduce the saved Polyakov loops")
            mean = values.mean(axis=0)
            boot = weights @ blocks(values)
            se, ci = boot.std(axis=0, ddof=1), np.quantile(boot, [0.025, 0.975], axis=0)
            group["holonomy_features"] = values
            group["holonomy_histogram"] = hist
            group["holonomy_mean"], group["holonomy_se"], group["holonomy_ci95"] = mean, se, ci
            tau = group.attrs["tau"]
            steps = group.attrs.get("reverse_steps", group.attrs["n_steps"])
            for j, observable in enumerate(names):
                summary.append([name, group.attrs["kind"], tau, steps, observable,
                                mean[j], se[j], ci[0, j], ci[1, j]])
            print(f"Holonomy {name} commutator {mean[-2]:.6f} +/- {se[-2]:.6f}", flush=True)
            del u
        reference = {key: f[f"reference/holonomy_{key}"][:] for key in ("features", "histogram")}
        clean = f["input/polyakov"][:]
        for name in ensembles[1:]:
            group = f[name]
            kind, tau = group.attrs["kind"], group.attrs["tau"]
            steps = int(group.attrs.get("reverse_steps", group.attrs["n_steps"]))
            if kind != "forward":
                data = {key: group[f"holonomy_{key}"][:] for key in ("features", "histogram")}
                rng = np.random.default_rng([int(f.attrs["seed"]), 700, steps, int(kind == "haar")])
                global_p, distance, cdf_p = compare(reference, data, rng)
                group.attrs["holonomy_global_p"] = global_p
                diff = group["holonomy_mean"][:]-f["reference/holonomy_mean"][:]
                se = np.hypot(group["holonomy_se"][:], f["reference/holonomy_se"][:])
                for j, observable in enumerate(names):
                    comparisons.append([name, kind, tau, steps, observable, diff[j], se[j], diff[j]/se[j],
                                        global_p, np.nan, np.nan])
                for j, observable in enumerate(("theta_x", "theta_y", "commutator")):
                    comparisons.append([name, kind, tau, steps, observable+"_cdf", np.nan, np.nan, np.nan,
                                        global_p, distance[j], cdf_p[j]])
            if kind in ("input", "fine", "forward"):
                p = group["polyakov"][:]
                for mu in range(2):
                    for averaging in ("line", "configuration"):
                        a, b = clean[:, mu], p[:, mu]
                        if averaging == "configuration":
                            a, b = a.mean(axis=1), b.mean(axis=1)
                        retention.append([kind, tau, steps, "xy"[mu], averaging, *correlation(a, b, weights)])
        f.flush()
    prefix = path.with_suffix("")
    csv_write(prefix.with_name(prefix.name+"_holonomy_statistics.csv"),
              ["ensemble", "kind", "tau", "steps", "observable", "mean", "se", "ci_low", "ci_high"], summary)
    csv_write(prefix.with_name(prefix.name+"_holonomy_comparison.csv"),
              ["ensemble", "kind", "tau", "steps", "observable", "difference", "se", "z", "global_p", "cdf_distance", "cdf_p"], comparisons)
    csv_write(prefix.with_name(prefix.name+"_holonomy_retention.csv"),
              ["kind", "tau", "steps", "direction", "averaging", "correlation", "se", "ci_low", "ci_high"], retention)


def plot_network(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.style.use(ROOT.parent / "plots/cjp.mplstyles")
    prefix = path.with_suffix("")
    with h5py.File(path) as f, PdfPages(prefix.with_suffix(".pdf")) as pdf:
        taus, steps = f["tau"][:], f["steps"][:]
        ticks = [x for x in (0, 0.05, 0.4, 3) if x < taus[-1]]+[taus[-1]]
        groups = ["input"]+[f"fine_steps{n}" for n in steps[1:]]
        stats = {}
        for name in ("reference", "haar_steps2000", *groups):
            group = f[name]
            stats[name] = dict(zip(group["statistic_names"].asstr()[:],
                                  zip(group["statistic_mean"][:], group["statistic_se"][:])))
            stats[name].update(zip(f["holonomy_feature_names"].asstr()[:],
                                   zip(group["holonomy_mean"][:], group["holonomy_se"][:])))

        def time_axis(ax):
            ax.set_xscale("symlog", linthresh=0.03)
            ax.set_xlim(-0.004, taus[-1]*1.2)
            ax.set_xticks(ticks, [f"{x:.3g}" for x in ticks])
            ax.set_xlabel(r"Initial noise variance $\Sigma_\star^2$")

        def curves(ax, key, ylabel, difference=False):
            ref, refse = stats["reference"][key]
            hm, he = stats["haar_steps2000"][key]
            values = np.array([stats[name][key] for name in groups])
            ax.errorbar(taus, values[:, 0]-(ref if difference else 0),
                        yerr=np.hypot(values[:, 1], refse) if difference else values[:, 1],
                        color="#00982a", marker="s", label="Fine start, matched grid", markersize=3)
            ax.axhline(0 if difference else ref, color="#315fce", linestyle="--", label="Fine equilibrium")
            if not difference:
                ax.axhspan(ref-refse, ref+refse, color="#315fce", alpha=0.12)
            else:
                hm, he = hm-ref, np.hypot(he, refse)
            ax.axhline(hm, color="#ca3300", linestyle=":", label="Haar 2000 steps")
            ax.axhspan(hm-he, hm+he, color="#ca3300", alpha=0.08)
            time_axis(ax)
            ax.set_ylabel(ylabel)

        pages = (("", "Reverse diffusion on the full sampler's time grid", r"$\langle W\rangle$", False),
                 ("", "Differences from independent fine equilibrium fields", r"$\langle W\rangle-\langle W\rangle_{\rm fine}$", True),
                 ("_second", "Second moments of individual loops", r"$\langle W^2\rangle$", False),
                 ("_site_variance", "Variance of individual loops", r"$\mathrm{var}(W)$", False),
                 ("_config_covariance", "Variance of spatially averaged loops", r"$\mathrm{var}(\overline{W})$", False))
        for page, (suffix, title, ylabel, difference) in enumerate(pages):
            fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
            for ax, loop, (r, t) in zip(axes.flat, LOOPS, RT):
                key = f"{loop}_{loop}{suffix}" if suffix == "_config_covariance" else loop+suffix
                curves(ax, key, ylabel, difference)
                ax.set_title(rf"${r}\times{t}$")
            axes.flat[0].legend(fontsize=6)
            fig.suptitle(title+rf"    $N_{{\rm cfg}}={f.attrs['n_cfgs']}$")
            pdf.savefig(fig)
            if page == 0:
                fig.savefig(prefix.with_suffix(".png"), dpi=180)
            plt.close(fig)

        with open(prefix.with_name(prefix.name+"_paired.csv")) as stream:
            paired = list(csv.DictReader(stream))
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for ax, loop, (r, t) in zip(axes.flat, LOOPS, RT):
            rows = [next(row for row in paired if row["a"] == name and row["observable"] == loop) for name in groups[1:]]
            ax.errorbar(taus[1:], [float(row["difference"]) for row in rows],
                        yerr=[float(row["se"]) for row in rows], color="#00982a", marker="s", markersize=3)
            ax.axhline(0, color="0.4", linewidth=0.7)
            time_axis(ax)
            ax.set_ylabel(r"$\langle W_{\rm out}-W_{\rm input}\rangle$")
            ax.set_title(rf"${r}\times{t}$")
        fig.suptitle("Paired change from the clean parent fields    Block bootstrap errors")
        pdf.savefig(fig)
        plt.close(fig)

        for kind in ("site", "config"):
            fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
            for ax, (i, j) in zip(axes.flat, ((i, j) for i in range(4) for j in range(i+1, 4))):
                curves(ax, f"{LOOPS[i]}_{LOOPS[j]}_{kind}_covariance", "Covariance")
                ax.set_title(rf"${RT[i][0]}\times{RT[i][1]}$ and ${RT[j][0]}\times{RT[j][1]}$")
            axes.flat[0].legend(fontsize=5.5)
            fig.suptitle("Covariances of "+("individual loops at the same anchor" if kind == "site" else "spatially averaged loops"))
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for row, loop in enumerate(("W11", "W22")):
            for col, power in enumerate((1, 2)):
                ax = axes[row, col]
                ref, se = stats["reference"][loop if power == 1 else loop+"_second"]
                ax.axhline(ref, color="#315fce", linestyle="--", label="Fine equilibrium")
                ax.axhspan(ref-se, ref+se, color="#315fce", alpha=0.12)
                for parity, style, marker in zip(("00", "01", "10", "11"), ("-", "--", "-.", ":"), ("o", "^", "s", "D")):
                    values = np.array([stats[name][f"{loop}_power{power}_parity{parity}"] for name in groups])
                    ax.errorbar(taus, values[:, 0], yerr=values[:, 1], linestyle=style,
                                marker=marker, markersize=2, label=parity, linewidth=0.8)
                time_axis(ax)
                ax.set_title(rf"${loop[1]}\times{loop[2]}$")
                ax.set_ylabel(r"$\langle W\rangle$" if power == 1 else r"$\langle W^2\rangle$")
        axes[0, 0].legend(ncol=2, fontsize=6)
        fig.suptitle("Four anchor parities after reverse diffusion")
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for col, mu in enumerate(("x", "y")):
            curves(axes[0, col], f"Polyakov_{mu}", r"$\langle P_\mu\rangle$")
            curves(axes[1, col], f"Polyakov_{mu}_second", r"$\langle P_\mu^2\rangle$")
            axes[0, col].set_title(f"Polyakov {mu}")
        axes[0, 0].legend(fontsize=6)
        fig.suptitle("Polyakov moments after reverse diffusion")
        pdf.savefig(fig)
        plt.close(fig)

        with open(prefix.with_name(prefix.name+"_retention.csv")) as stream:
            retained = list(csv.DictReader(stream))
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for row, averaging in enumerate(("site", "configuration")):
            for col, loop in enumerate(("W22", "W44")):
                ax = axes[row, col]
                for stage, color, label in (("forward", "#315fce", "Forward only"), ("reverse", "#00982a", "After reverse diffusion")):
                    rows = [r for r in retained if r["stage"] == stage and r["loop"] == loop and r["averaging"] == averaging]
                    ax.errorbar([float(r["tau"]) for r in rows], [float(r["correlation"]) for r in rows],
                                yerr=[float(r["se"]) for r in rows], color=color, label=label, markersize=3)
                ax.axhline(0, color="0.4", linewidth=0.7)
                ax.set_ylim(-0.3, 1.05)
                time_axis(ax)
                ax.set_ylabel("Correlation with clean input")
                ax.set_title(rf"${loop[1]}\times{loop[2]}$    {averaging}")
        axes[0, 0].legend(fontsize=6)
        fig.suptitle("Input information before and after reverse diffusion    Even anchors")
        pdf.savefig(fig)
        plt.close(fig)

        snapshots = (steps[1], steps[1:][len(steps[1:])//2], steps[-1])
        for observations, title in ((range(4), "Wilson loop distributions"), (range(4, 6), "Polyakov loop distributions")):
            fig, axes = plt.subplots(len(observations), 3, figsize=(7.0, 2.0*len(observations)), squeeze=False)
            for row, j in enumerate(observations):
                for col, step in enumerate(snapshots):
                    ax = axes[row, col]
                    for name, label, color in (("reference", "Fine equilibrium", "#315fce"),
                                                ("haar_steps2000", "Haar 2000 steps", "#ca3300"),
                                                (f"fine_steps{step}", "Fine start", "#00982a")):
                        hist = f[f"{name}/histogram"][:, j].mean(axis=0)
                        ax.stairs(hist/np.diff(EDGES), EDGES, color=color, label=label)
                    ax.set_title(rf"{(*LOOPS, 'Polyakov x', 'Polyakov y')[j]}    $N={step}$")
                    ax.set_xlabel(r"$\mathrm{Re\ Tr}\,U/2$")
                    ax.set_ylabel("Density")
            axes[0, 0].legend(fontsize=5.5)
            fig.suptitle(title+"    N is the remaining number of reverse steps")
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        p = [f[name].attrs["comparison_global_p"] for name in groups[1:]]
        ax.plot(taus[1:], p, color="#00982a", marker="s", label="Fine start, matched grid")
        ax.axhline(f["haar_steps2000"].attrs["comparison_global_p"], color="#ca3300", linestyle=":", label="Haar 2000 steps")
        ax.axhline(0.05, color="0.4", linestyle="--", label=r"$p=0.05$")
        ax.set_yscale("log")
        ax.set_ylim(0.0008, 1.3)
        time_axis(ax)
        ax.set_ylabel("Global permutation p")
        ax.legend(fontsize=7)
        fig.suptitle("Comparison with independent fine equilibrium fields    999 block permutations")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.plot(steps, taus, marker="o")
        ax.set_xlabel("Remaining steps on the full 2000-step grid")
        ax.set_ylabel(r"$\Sigma_\star^2$")
        ax.set_title(r"$t_\star=\epsilon+(1-\epsilon)N/2000$    $\sigma=14$")
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for ax, key, label, title in zip(axes.flat,
                ("theta_x_centered_second", "theta_y_centered_second", "holonomy_commutator", "holonomy_commutator_second"),
                (r"$\langle(\theta_x-\pi/2)^2\rangle$", r"$\langle(\theta_y-\pi/2)^2\rangle$", r"$\langle C\rangle$", r"$\langle C^2\rangle$"),
                ("Holonomy x", "Holonomy y", "Commutator at the shared base point", "Commutator second moment")):
            curves(ax, key, label)
            ax.set_title(title)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(r"Holonomy eigenangles and $C=\mathrm{Tr}(\Omega_x\Omega_y\Omega_x^\dagger\Omega_y^\dagger)/2$")
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.5))
        angle_edges = f["holonomy_angle_edges"][:]
        for j, (edges, label) in enumerate(((angle_edges, r"$\theta_x$"), (angle_edges, r"$\theta_y$"), (EDGES, r"$C$"))):
            for col, n in enumerate(snapshots):
                ax = axes[j, col]
                for name, legend, color in (("reference", "Fine equilibrium", "#315fce"),
                                            ("haar_steps2000", "Haar 2000 steps", "#ca3300"),
                                            (f"fine_steps{n}", "Fine start", "#00982a")):
                    h = f[f"{name}/holonomy_histogram"][:, j].mean(axis=0)
                    ax.stairs(h/np.diff(edges), edges, color=color, label=legend)
                ax.set_xlabel(label)
                ax.set_ylabel("Density")
                ax.set_title(rf"$N={n}$")
                if j < 2:
                    ax.set_xticks([0, np.pi/2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
        axes[0, 0].legend(fontsize=5.5)
        fig.suptitle("Holonomy angle and commutator distributions")
        pdf.savefig(fig)
        plt.close(fig)

        with open(prefix.with_name(prefix.name+"_holonomy_retention.csv")) as stream:
            holonomy_retention = list(csv.DictReader(stream))
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for row, averaging in enumerate(("line", "configuration")):
            for col, mu in enumerate("xy"):
                ax = axes[row, col]
                for kind, color, label in (("forward", "#315fce", "Forward only"), ("fine", "#00982a", "After reverse diffusion")):
                    rows = [r for r in holonomy_retention if r["kind"] in ("input", kind)
                            and r["direction"] == mu and r["averaging"] == averaging]
                    ax.errorbar([float(r["tau"]) for r in rows], [float(r["correlation"]) for r in rows],
                                yerr=[float(r["se"]) for r in rows], color=color, label=label, markersize=3)
                time_axis(ax)
                ax.axhline(0, color="0.4", linewidth=0.7)
                ax.set_ylim(-0.3, 1.05)
                ax.set_ylabel("Correlation with clean input")
                ax.set_title(f"Holonomy {mu}    {averaging}")
        axes[0, 0].legend(fontsize=6)
        fig.suptitle("Holonomy trace retention before and after reverse diffusion")
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
        for ax, mu in zip(axes, "xy"):
            curves(ax, f"Polyakov_{mu}_bar_second", rf"$\langle\overline{{P}}_{mu}^2\rangle$")
            ax.set_title(f"Holonomy {mu}")
        axes[0].legend(fontsize=6)
        fig.suptitle("Second moments of the spatially averaged winding traces")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.plot(taus[1:], [f[name].attrs["holonomy_global_p"] for name in groups[1:]],
                color="#00982a", marker="s", label="Fine start, matched grid")
        ax.axhline(f["haar_steps2000"].attrs["holonomy_global_p"], color="#ca3300", linestyle=":", label="Haar 2000 steps")
        ax.axhline(0.05, color="0.4", linestyle="--", label=r"$p=0.05$")
        time_axis(ax)
        ax.set_yscale("log")
        ax.set_ylim(0.0008, 1.3)
        ax.set_ylabel("Holonomy permutation p")
        ax.legend(fontsize=7)
        fig.suptitle("Holonomy comparison with independent fine equilibrium fields")
        pdf.savefig(fig)
        plt.close(fig)


def analyze_refinement(path):
    summary, comparisons, retention, decay = [], [], [], []
    with h5py.File(path, "r+") as f:
        counts = f["steps"][1:]
        weights = f["bootstrap_weights"][:]
        initial = ["input"]+[f"{kind}_input" for kind in NAMES[1:]]
        stages = [("reference", "reference", "reference"), ("haar_steps2000", "reverse", "haar")]
        stages += [(name, "initial", kind) for name, kind in zip(initial, NAMES) if name in f]
        pairs = [(name, "reference", "initial") for name in initial if name in f]
        # Keep the original permutation streams when adding a shorter starting time.
        for ni, n in enumerate(counts[::-1]):
            noisy = [f"forward_steps{n}"]+[f"{kind}_forward_steps{n}" for kind in NAMES[1:]]
            final = [f"{kind}_steps{n}" for kind in NAMES]
            stages += [(name, stage, kind) for stage, names in (("forward", noisy), ("reverse", final))
                       for name, kind in zip(names, NAMES) if name in f]
            pairs += [(name, noisy[0], "forward") for name in noisy[1:] if name in f]
            if ni == 0:
                pairs.append(("haar_steps2000", "reference", "reverse"))
            pairs += [(name, "reference", "reverse") for name in final if name in f]
            pairs += [(name, final[0], "reverse") for name in final[1:] if name in f and final[0] in f]
            pairs += [(name, "haar_steps2000", "reverse") for name in final if name in f]
        f["holonomy_feature_names"] = np.asarray(HOLONOMY_NAMES, dtype=h5py.string_dtype())
        f["holonomy_angle_edges"] = ANGLE_EDGES
        for name, stage, kind in stages:
            group = f[name]
            group.attrs.update(stage=stage, initialization=kind)
            if "holonomy_features" in group:
                values = group["holonomy_features"][:]
            else:
                u, _ = from_matrix(group["cfgs"][:])
                values, hist, p = measure_holonomy(u)
                np.testing.assert_allclose(p, group["polyakov"][:], rtol=0, atol=1e-12)
                group["holonomy_features"], group["holonomy_histogram"] = values, hist
                del u, hist, p
            boot = weights @ blocks(values)
            mean, se = values.mean(axis=0), boot.std(axis=0, ddof=1)
            ci = np.quantile(boot, [0.025, 0.975], axis=0)
            group["holonomy_mean"], group["holonomy_se"], group["holonomy_ci95"] = mean, se, ci
            for prefix, labels in (("statistic", group["statistic_names"].asstr()[:]), ("holonomy", HOLONOMY_NAMES)):
                m, e, c = (group[f"{prefix}_{key}"][:] for key in ("mean", "se", "ci95"))
                for j, label in enumerate(labels):
                    summary.append([name, stage, kind, group.attrs["tau"], label, m[j], e[j], c[0, j], c[1, j]])
        for i, (name, refname, stage) in enumerate(pairs):
            for domain, (prefix, statprefix, cdf_labels) in enumerate((
                    ("", "statistic", (*LOOPS, "Polyakov_x", "Polyakov_y")),
                    ("holonomy_", "holonomy", ("theta_x", "theta_y", "commutator")))):
                a, b = f[refname], f[name]
                reference = {key: a[prefix+key][:] for key in ("features", "histogram")}
                candidate = {key: b[prefix+key][:] for key in ("features", "histogram")}
                rng = np.random.default_rng([int(f.attrs["seed"]), 900, i, domain])
                p, distance, cdf_p = compare(reference, candidate, rng)
                if refname == "reference" or stage == "forward":
                    b.attrs["holonomy_global_p" if domain else "comparison_global_p"] = p
                if refname == "haar_steps2000":
                    b.attrs["holonomy_haar_p" if domain else "comparison_haar_p"] = p
                labels = HOLONOMY_NAMES if domain else b["statistic_names"].asstr()[:]
                difference = b[f"{statprefix}_mean"][:]-a[f"{statprefix}_mean"][:]
                se = np.hypot(a[f"{statprefix}_se"][:], b[f"{statprefix}_se"][:])
                for j, label in enumerate(labels):
                    comparisons.append([name, refname, stage, label, difference[j], se[j],
                                        difference[j]/se[j], p, np.nan, np.nan])
                for j, label in enumerate(cdf_labels):
                    comparisons.append([name, refname, stage, label+"_cdf", np.nan, np.nan, np.nan,
                                        p, distance[j], cdf_p[j]])
        if "holonomy_features" in f["source_coarse"]:
            coarse_h = f["source_coarse/holonomy_features"][:]
        else:
            coarse, _ = from_matrix(f["source_coarse/cfgs"][:])
            coarse_h, _, _ = measure_holonomy(coarse)
            f["source_coarse/holonomy_features"] = coarse_h
        indices = [0, 1, 2, 3, 8, 9]
        lengths = np.array([4, 6, 8, 16, 32, 32])
        for n, tau in zip(counts, f["tau"][1:]):
            noisy = [f"forward_steps{n}"]+[f"{kind}_forward_steps{n}" for kind in NAMES[1:]]
            final = [f"{kind}_steps{n}" for kind in NAMES]
            for ki, kind in enumerate(NAMES):
                if initial[ki] not in f:
                    continue
                start, forward_group = f[initial[ki]], f[noisy[ki]]
                difference = forward_group["features"][:, indices]-start["features"][:, indices]*np.exp(-3*lengths*tau/8)
                se = (weights @ blocks(difference)).std(axis=0, ddof=1)
                for j, label in enumerate((*LOOPS, "Polyakov_x", "Polyakov_y")):
                    decay.append([kind, tau, label, difference[:, j].mean(), se[j], difference[:, j].mean()/se[j]])
                for stage, groupname in (("initial", initial[ki]), ("forward", noisy[ki]), ("reverse", final[ki])):
                    if groupname not in f:
                        continue
                    group = f[groupname]
                    for j, coarse_j in ((2, 0), (3, 2)):
                        x = (f["input/loops"][:, j, ::2, ::2] if ki == 0 else f["source_coarse/loops"][:, coarse_j])
                        y = group["loops"][:, j, ::2, ::2]
                        for averaging in ("site", "configuration"):
                            a, b = (x, y) if averaging == "site" else (x.mean(axis=(1, 2)), y.mean(axis=(1, 2)))
                            retention.append([kind, stage, LOOPS[j], averaging, *correlation(a, b, weights), tau, n])
                    for mu in range(2):
                        x = f["input/polyakov"][:, mu, ::2] if ki == 0 else f["source_coarse/polyakov"][:, mu]
                        y = group["polyakov"][:, mu, ::2]
                        for averaging in ("line", "configuration"):
                            a, b = (x, y) if averaging == "line" else (x.mean(axis=1), y.mean(axis=1))
                            retention.append([kind, stage, f"Polyakov_{'xy'[mu]}", averaging, *correlation(a, b, weights), tau, n])
                    x = f["input/holonomy_features"][:, 6] if ki == 0 else coarse_h[:, 6]
                    retention.append([kind, stage, "commutator", "configuration",
                                      *correlation(x, group["holonomy_features"][:, 6], weights), tau, n])
            print(f"Refinement statistics complete at tau {tau:.8f}. Steps {n}", flush=True)
    prefix = path.with_suffix("")
    csv_write(prefix.with_name(prefix.name+"_statistics.csv"),
              ["ensemble", "stage", "initialization", "tau", "observable", "mean", "se", "ci_low", "ci_high"], summary)
    csv_write(prefix.with_name(prefix.name+"_comparison.csv"),
              ["ensemble", "reference", "stage", "observable", "difference", "se", "z", "global_p", "cdf_distance", "cdf_p"], comparisons)
    csv_write(prefix.with_name(prefix.name+"_retention.csv"),
              ["initialization", "stage", "observable", "averaging", "correlation", "se", "ci_low", "ci_high", "tau_start", "reverse_steps"], retention)
    csv_write(prefix.with_name(prefix.name+"_heat_kernel.csv"),
              ["initialization", "tau", "observable", "residual", "se", "z"], decay)


def plot_refinement(path, steps=None):
    """Plot stored results at the selected reverse step counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.style.use(ROOT.parent / "plots/cjp.mplstyles")
    prefix = path.with_suffix("")
    with h5py.File(path) as f, PdfPages(prefix.with_suffix(".pdf")) as pdf:
        wilson_data = f.get("wilson23", f)
        loop_names = tuple(wilson_data["feature_names"].asstr()[:4])
        loop_sizes = [(int(name[1]), int(name[2])) for name in loop_names]
        counts, taus = f["steps"][1:], f["tau"][1:]
        if steps is None and "wilson23" in f:
            steps = wilson_data["steps"][:]
        if steps is not None:
            selected = np.isin(counts, steps)
            counts, taus = counts[selected], taus[selected]
        multiple_times = len(counts) > 1
        has_fine_start = any(f"fine_steps{n}" in f for n in counts)
        control = "reference"
        control_label = "Fine equilibrium"
        final = [f"fine_steps{n}" for n in counts]+["haar_steps2000"]
        final += [f"{kind}_steps{n}" for kind in NAMES[1:] for n in counts]
        final = [name for name in final if name in f]
        kinds = (*NAMES, "haar")
        kind_labels = dict(zip(kinds, ("Fine start", "Uniform", "Interior", "Relaxed", "Haar 2000")))
        kind_colors = dict(zip(kinds, ("#00982a", "#7a45b8", "#008b9c", "#d19100", "#ca3300")))
        labels, colors, markers, line_styles = [], [], [], []
        for name in final:
            group = f[name]
            kind = group.attrs["initialization"]
            label = kind_labels[kind]
            if multiple_times and kind != "haar":
                label += "\n"+rf"$\Sigma^2={group.attrs['tau']:.2f}$"
            time_index = list(counts).index(group.attrs["n_steps"]) if kind != "haar" else len(counts)-1
            marker, line_style = (("o", "-"), ("s", "--"), ("^", ":"))[(len(counts)-1-time_index) % 3]
            labels.append(label)
            colors.append(kind_colors[kind])
            markers.append(marker)
            line_styles.append(line_style)
        initial = ["input"]+[f"{kind}_input" for kind in NAMES[1:]]
        noisy = [f"forward_steps{n}" for n in counts]
        noisy += [f"{kind}_forward_steps{n}" for kind in NAMES[1:] for n in counts]
        stats = {}
        for name in ("reference", control, *final, *initial, *noisy):
            if name not in f:
                continue
            group = wilson_data[name]
            stats[name] = dict(zip(group["statistic_names"].asstr()[:],
                                   zip(group["statistic_mean"][:], group["statistic_se"][:])))
            plaquette_mean, plaquette_se = stats[name]["W11"]
            beta = float(f.attrs["beta"])
            stats[name]["action_density"] = (beta*(1-plaquette_mean), beta*plaquette_se)
        heading = f"EMA epoch {f.attrs['epoch']}    "+rf"$N_{{\rm cfg}}={f.attrs['n_cfgs']}$    "
        if multiple_times:
            heading += ", ".join(str(n) for n in counts)+" reverse steps"
        else:
            heading += rf"$\Sigma_\star^2={taus[0]:.5f}$    {counts[0]} reverse steps"

        def points(ax, key, ylabel):
            ref, refse = stats[control][key]
            for i, (name, color, marker) in enumerate(zip(final, colors, markers)):
                mean, se = stats[name][key]
                ax.errorbar(i, mean, yerr=se,
                            color=color, marker=marker, linestyle="none", capsize=2)
            ax.axhline(ref, color="#315fce", linestyle="--", label=control_label)
            ax.axhspan(ref-refse, ref+refse, color="#315fce", alpha=0.12)
            ax.set_xticks(range(len(final)), labels, rotation=0 if multiple_times else 15)
            if multiple_times:
                ax.tick_params(axis="x", labelsize=5.5 if len(final) > 5 else 7)
            ax.set_ylabel(ylabel)

        pages = (("", "Final Wilson loops", r"$\langle W\rangle$"),
                 ("_second", "Second moments of individual loops", r"$\langle W^2\rangle$"),
                 ("_site_variance", "Variance of individual loops", r"$\mathrm{var}(W)$"),
                 ("_config_covariance", "Variance of spatially averaged loops", r"$\mathrm{var}(\overline{W})$"))
        for page, (suffix, title, ylabel) in enumerate(pages):
            fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
            for ax, loop, (r, t) in zip(axes.flat, loop_names, loop_sizes):
                key = f"{loop}_{loop}{suffix}" if suffix == "_config_covariance" else loop+suffix
                points(ax, key, ylabel)
                ax.set_title(rf"${r}\times{t}$")
            axes[0, 0].legend(fontsize=6)
            fig.suptitle(title+"\n"+heading)
            pdf.savefig(fig)
            if page == 0:
                fig.savefig(prefix.with_suffix(".png"), dpi=180)
            plt.close(fig)
            if page == 0:
                fig, ax = plt.subplots(figsize=(7.0, 3.0))
                points(ax, "action_density", r"$\langle S/V\rangle$")
                ax.legend(fontsize=7)
                fig.suptitle(r"Wilson action density  $S/V=\beta(1-\overline{P})$"+"\n"+heading)
                pdf.savefig(fig)
                plt.close(fig)
                if "string_tension_mean" in wilson_data[control]:
                    fig, ax = plt.subplots(figsize=(7.0, 3.6))
                    areas = np.prod(loop_sizes, axis=1)
                    for name, label, color, marker, style in zip(
                            [control, *final], [control_label, *labels], ["#315fce", *colors],
                            ["D", *markers], ["-", *line_styles]):
                        group = wilson_data[name]
                        ax.errorbar(areas, group["string_tension_mean"][:],
                                    yerr=group["string_tension_se"][:], color=color,
                                    marker=marker, linestyle=style, label=label)
                    exact = float(wilson_data.attrs["string_tension_exact"])
                    ax.axhline(exact, color="0.3", linestyle=":",
                               label=rf"Exact $\sigma a^2={exact:.5f}$")
                    ax.set_xticks(areas, [rf"${r}\times{t}$" for r, t in loop_sizes])
                    ax.set_xlabel("Wilson-loop size")
                    ax.set_ylabel(r"$\sigma_{\rm eff}a^2$")
                    ax.legend(fontsize=7, ncol=2, loc="upper left")
                    fig.suptitle(r"Effective string tension  $\sigma_{\rm eff}a^2=-\ln\langle W\rangle/(RT)$"
                                 +"\n"+heading)
                    pdf.savefig(fig)
                    plt.close(fig)

        for suffix, title in (("_config_covariance", "Covariance of spatially averaged loops"),
                              ("_site_covariance", "Covariance of loops at the same site")):
            fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.5))
            pairs = [(a, b) for i, a in enumerate(loop_names) for b in loop_names[i+1:]]
            for ax, (a, b) in zip(axes.flat, pairs):
                points(ax, a+"_"+b+suffix, "Covariance")
                ax.set_title(a+" and "+b)
            axes[0, 0].legend(fontsize=6)
            fig.suptitle(title)
            pdf.savefig(fig)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for ax, (loop, power) in zip(axes.flat, (("W11", 1), ("W11", 2), ("W22", 1), ("W22", 2))):
            keys = [f"{loop}_power{power}_parity{x}{y}" for x in range(2) for y in range(2)]
            for name, label, color, marker, line_style in zip(final, labels, colors, markers, line_styles):
                values = np.array([stats[name][key] for key in keys])
                ref = np.array([stats[control][key] for key in keys])
                ax.errorbar(np.arange(4), values[:, 0]-ref[:, 0], yerr=np.hypot(values[:, 1], ref[:, 1]),
                            color=color, marker=marker, linestyle=line_style, label=label, markersize=3)
            ax.axhline(0, color="0.4", linewidth=0.7)
            ax.set_xticks(range(4), ["00", "01", "10", "11"])
            ax.set_xlabel("Anchor parity")
            ax.set_ylabel("Difference from "+control_label)
            ax.set_title(f"{loop} moment {power}")
        axes[0, 0].legend(fontsize=5.5, ncol=2)
        fig.suptitle("Residual block patterns after reverse diffusion")
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for j, ax in enumerate(axes.flat):
            for name, label, color, line_style in zip([control, *final], [control_label, *labels],
                                                    ["#315fce", *colors], ["-", *line_styles]):
                hist = wilson_data[f"{name}/histogram"][:, j].mean(axis=0)
                ax.stairs(hist/np.diff(EDGES), EDGES, color=color, linestyle=line_style, label=label)
            ax.set_xlabel(loop_names[j])
            ax.set_ylabel("Density")
        axes[0, 0].legend(fontsize=5.5, ncol=2)
        fig.suptitle("Final Wilson-loop distributions")
        pdf.savefig(fig)
        plt.close(fig)

        stage_cases = [(name, f[name].attrs["initialization"], int(f[name].attrs["n_steps"]), label, color, marker, style)
                       for name, label, color, marker, style in zip(final, labels, colors, markers, line_styles)
                       if name != "haar_steps2000"]
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        for ax, key in zip(axes.flat, loop_names):
            for name, kind, n, label, color, marker, style in stage_cases:
                start = "input" if kind == "fine" else f"{kind}_input"
                forward_name = f"forward_steps{n}" if kind == "fine" else f"{kind}_forward_steps{n}"
                values = np.array([stats[group][key] for group in (start, forward_name, name)])
                ax.errorbar(range(3), values[:, 0], yerr=values[:, 1], color=color, marker=marker,
                            linestyle=style, label=label)
            ax.set_xticks(range(3), ["Initial", "Noisy", "Reverse"])
            ax.set_ylabel(r"$\langle W\rangle$")
            ax.set_title(key)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle("Observable means through the three stages\n"+heading)
        pdf.savefig(fig)
        plt.close(fig)

        with open(prefix.with_name(prefix.name+"_retention.csv")) as stream:
            rows = list(csv.DictReader(stream))
        retained_loops = tuple(loop for loop, (r, t) in zip(loop_names, loop_sizes) if r % 2 == t % 2 == 0)
        fig, axes = plt.subplots(1 if len(retained_loops) == 1 else 2, 2,
                                 figsize=(7.0, 2.8 if len(retained_loops) == 1 else 4.8))
        axes = axes.reshape(2, len(retained_loops))
        for row, averaging in enumerate(("site", "configuration")):
            for col, observable in enumerate(retained_loops):
                ax = axes[row, col]
                for _, kind, n, label, color, marker, style in stage_cases:
                    data = [next(r for r in rows if r["initialization"] == kind and r["stage"] == stage
                                 and int(r.get("reverse_steps", n)) == n
                                 and r["observable"] == observable and r["averaging"] == averaging)
                            for stage in ("initial", "forward", "reverse")]
                    ax.errorbar(range(3), [float(r["correlation"]) for r in data],
                                yerr=[float(r["se"]) for r in data], color=color, marker=marker,
                                linestyle=style, label=label)
                ax.axhline(0, color="0.4", linewidth=0.7)
                ax.set_ylim(-0.3, 1.05)
                ax.set_xticks(range(3), ["Initial", "Noisy", "Reverse"])
                ax.set_ylabel("Correlation with parent")
                ax.set_title(observable+"    "+averaging)
        axes[0, 0].legend(fontsize=6)
        subtitle = "Fine starts use their original fine fields as parents" if has_fine_start else "Correlation with the original coarse fields"
        fig.suptitle("Matching-area loop retention\n"+subtitle)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
        selected_noisy = [f"{kind}_forward_steps{n}" for kind in NAMES[1:] for n in counts if f"{kind}_forward_steps{n}" in f]
        n_refined = len(selected_noisy)
        for row, key in enumerate(("comparison_global_p", "holonomy_global_p")):
            source = f if row else wilson_data
            for col, (names, names_labels, names_colors) in enumerate(((selected_noisy, labels[-n_refined:], colors[-n_refined:]), (final, labels, colors))):
                ax = axes[row, col]
                comparison_key = "holonomy_haar_p" if row else "comparison_haar_p"
                for i, (name, color) in enumerate(zip(names, names_colors)):
                    offset = .07 if col == 1 else 0
                    if offset:
                        if name != "haar_steps2000":
                            ax.plot(i-offset, source[name].attrs[comparison_key], marker="o", color=color,
                                    label="vs Haar 2000" if name == next(x for x in final if x != "haar_steps2000") else None)
                        ax.plot(i+offset, source[name].attrs[key], marker="s", color=color,
                                label="vs Fine equilibrium" if i == 0 else None)
                    else:
                        ax.plot(i, source[name].attrs[key], marker="o", color=color)
                if col == 1:
                    ax.legend(fontsize=6)
                ax.axhline(.05, color="0.4", linestyle="--")
                ax.set_yscale("log")
                ax.set_ylim(.0008, 1.3)
                ax.set_xticks(range(len(names)), names_labels, rotation=0 if multiple_times else 15)
                if multiple_times:
                    ax.tick_params(axis="x", labelsize=5.5 if len(final) > 5 else 7)
                ax.set_ylabel("Holonomy p" if row else "Loop-feature p")
                ax.set_title("Noisy fields vs noisy fine control" if col == 0 else "Final fields vs both references")
        fig.suptitle("Block-permutation comparisons at the selected starting times" if multiple_times else
                     "Block-permutation comparisons at the selected start time")
        pdf.savefig(fig)
        plt.close(fig)


def merge_refinement(paths, output):
    """Pool the node ensembles and recompute uncertainties before plotting."""
    with ExitStack() as stack:
        sources = [stack.enter_context(h5py.File(path)) for path in paths]
        first = sources[0]
        batches, n = len(sources), int(first.attrs["n_cfgs"])
        total = batches*n
        common = ("checkpoint", "fine", "forward", "epoch", "L", "beta", "sigma", "ch", "nlayers",
                  "ema", "dtype", "eps", "block_size", "bootstrap", "schedule", "full_steps", "final_noise", "delta_max")
        grids = ("steps", "tau", "t_start", "full_times", "histogram_edges", "feature_names")
        groups = [name for name in first if isinstance(first[name], h5py.Group)]
        for rank, source in enumerate(sources):
            assert source.attrs["n_cfgs"] == n and source.attrs["parent_offset"] == rank
            assert set(groups) == {name for name in source if isinstance(source[name], h5py.Group)}
            for key in common:
                assert source.attrs[key] == first.attrs[key], key
            for key in grids:
                np.testing.assert_array_equal(source[key][:], first[key][:])
        assert len({int(f.attrs["seed"]) for f in sources}) == batches
        assert len({int(f["source_coarse"].attrs["seed"]) for f in sources}) == batches
        # Interleave fine fields by original MC index. Keep each independent coarse chain contiguous.
        parents = np.stack([f["parent_id"][:] for f in sources], axis=1).ravel()
        reference_ids = np.stack([f["reference_parent_id"][:] for f in sources], axis=1).ravel()
        assert np.all(np.diff(parents) > 0) and np.all(np.diff(reference_ids) > 0)
        assert not np.intersect1d(parents, reference_ids).size
        weights = bootstrap_weights(total//BLOCK, np.random.default_rng([20261002, 500]))
        out = stack.enter_context(h5py.File(output, "x"))
        out.attrs.update({key: first.attrs[key] for key in common})
        out.attrs.update(n_cfgs=total, n_batches=batches, cfgs_per_batch=n, seed=20261002)
        for key in grids:
            first.copy(key, out)
        out["parent_id"], out["reference_parent_id"] = parents, reference_ids
        out["fine_batch_id"] = np.tile(np.arange(batches), n)
        out["coarse_batch_id"] = np.repeat(np.arange(batches), n)
        out["batch_seeds"] = [f.attrs["seed"] for f in sources]
        out["bootstrap_weights"] = weights
        labels = first["feature_names"].asstr()[:].tolist()
        for name in groups:
            original = first[name]
            group = out.create_group(name)
            attributes = ("kind", "stage", "initialization", "tau", "t_start", "n_steps", "eps", "final_noise",
                          "reverse_steps", "forward_increments", "increment_variance", "max_increment_variance",
                          "L", "beta", "n_therm", "n_skip", "n_or", "stride_saved_configs")
            for key in attributes:
                if key in original.attrs:
                    assert all(f[name].attrs[key] == original.attrs[key] for f in sources), (name, key)
                    group.attrs[key] = original.attrs[key]
            if "seed" in original.attrs:
                group["batch_seeds"] = [f[name].attrs["seed"] for f in sources]
            if "norm_error_before_projection" in original.attrs:
                group.attrs["norm_error_before_projection"] = max(f[name].attrs["norm_error_before_projection"] for f in sources)
            if "times" in original:
                for f in sources:
                    np.testing.assert_array_equal(f[name]["times"][:], original["times"][:])
                original.copy("times", group)
            fine_order = name in ("input", "reference") or name.startswith(("fine_steps", "forward_steps"))
            group.attrs["configuration_order"] = "fine_parent_index" if fine_order else "batch_then_parent"
            for key in ("cfgs", "parent_id", "loops", "polyakov", "features", "histogram", "holonomy_features", "holonomy_histogram"):
                if key not in original:
                    continue
                shape = (total, *original[key].shape[1:])
                data = group.create_dataset(key, shape=shape, dtype=original[key].dtype,
                                            chunks=(1, *shape[1:]) if key == "cfgs" else None)
                for rank, source in enumerate(sources):
                    assert source[name][key].shape == original[key].shape and len(source[name][key]) == n
                    for begin in range(0, n, 128):
                        end = min(begin+128, n)
                        dest = slice(rank+begin*batches, rank+end*batches, batches) if fine_order else slice(rank*n+begin, rank*n+end)
                        data[dest] = source[name][key][begin:end]
            if name != "source_coarse":
                names, mean, se, ci = statistics(group["features"][:], labels, weights)
                group["statistic_names"] = np.asarray(names, dtype=h5py.string_dtype())
                group["statistic_mean"], group["statistic_se"], group["statistic_ci95"] = mean, se, ci
            out.flush()
            print(f"Merged {name}. Configurations {total}", flush=True)
    analyze_refinement(output)
    plot_refinement(output)
    print(f"Merged comparison saved to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "scan", "sample"))
    parser.add_argument("--coarse", type=Path)
    parser.add_argument("--initializations", choices=NAMES, nargs="+", default=list(NAMES[1:]))
    parser.add_argument("--fine", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-cfgs", type=int, default=256)
    parser.add_argument("--parent-offset", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--Sigma2", "--tau", dest="tau", metavar="SIGMA2", type=float, nargs="+", default=TAUS.tolist(),
                        help="Cumulative noise variance Sigma squared")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--forward", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 100, 200, 400, 600, 800, 1000, 1200, 1500, 2000])
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()
    if args.action == "check":
        check()
    elif args.action == "scan":
        if any(path is None for path in (args.coarse, args.fine, args.out)):
            parser.error("scan requires --coarse, --fine and --out")
        scan(args)
    else:
        if args.device != "cuda" or any(path is None for path in (args.fine, args.forward, args.checkpoint, args.out)):
            parser.error("sample requires --device cuda, --fine, --forward, --checkpoint and --out")
        validate_network(args)


if __name__ == "__main__":
    main()
