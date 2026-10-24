"""A <-> 2A Brownian reaction-diffusion in a periodic box, structured initial states, h5 io of (X_t, X_{t+DT_PAIR}) pairs."""
import numpy as np
import h5py
from scipy.spatial import cKDTree
from config import R, D, LAM, GAM, DT_PAIR, DT_SIM


def step(x, L, dt, rng, lam, gam):
    # Brownian kick, then every pair closer than R merges with prob gam dt, then every particle branches with prob lam dt
    N = x.shape[0]
    x = (x + np.sqrt(2 * D * dt) * rng.standard_normal((N, 2))) % L
    x[x >= L] -= L   # float mod can return exactly L, cKDTree boxsize needs [0, L)
    alive = np.ones(N, bool)
    if gam > 0:
        pairs = cKDTree(x, boxsize=L).query_pairs(R, output_type="ndarray")
        sel = pairs[rng.random(len(pairs)) < gam * dt]
        for a, b in sel[rng.permutation(len(sel))]:
            if alive[a] and alive[b]:
                alive[a if rng.random() < 0.5 else b] = False
    born = alive & (rng.random(N) < lam * dt)
    nb = int(born.sum())
    rho, th = R * np.sqrt(rng.random(nb)), 2 * np.pi * rng.random(nb)
    child = (x[born] + np.stack([rho * np.cos(th), rho * np.sin(th)], -1)) % L
    child[child >= L] -= L
    return np.concatenate([x[alive], child]), N - int(alive.sum()), nb


def evolve(x, L, t, dt, rng, lam, gam):
    n_merge = n_branch = 0
    for _ in range(int(round(t / dt))):
        x, m, b = step(x, L, dt, rng, lam, gam)
        n_merge += m
        n_branch += b
    return x, n_merge, n_branch


def initial_state(L, kind, rng):
    # Poisson point process of mean density 1 times a profile. 0 uniform, 1 sinusoidal modulation, 2 Gaussian clusters and voids, 3 one particle
    if kind == 0:
        return rng.random((rng.poisson(L * L), 2)) * L
    if kind == 3:
        return rng.random((1, 2)) * L
    if kind == 1:
        A, phi = rng.uniform(0, 0.7), rng.uniform(0, 2 * np.pi)
        n = rng.integers(-3, 4, size=2)
        while not n.any():
            n = rng.integers(-3, 4, size=2)
        x = rng.random((rng.poisson((1 + A) * L * L), 2)) * L
        return x[rng.random(len(x)) * (1 + A) < 1 + A * np.cos(2 * np.pi * x @ n / L + phi)]
    n_c = rng.integers(1, 5)
    c = rng.random((n_c, 2)) * L
    a = np.where(rng.random(n_c) < 0.5, rng.uniform(2, 4, n_c), rng.uniform(-0.8, -0.5, n_c))   # density 3-5x or 0.2-0.5x at the centre
    w = rng.uniform(1.5, 3, n_c)
    fmax = 1 + a[a > 0].sum()
    x = rng.random((rng.poisson(fmax * L * L), 2)) * L
    d = x[:, None, :] - c[None]
    d -= L * np.round(d / L)
    f = np.maximum(1 + (a * np.exp(-(d ** 2).sum(-1) / (2 * w ** 2))).sum(1), 0)
    return x[rng.random(len(x)) * fmax < f]


def trajectory(args):
    # K consecutive snapshots DT_PAIR apart from an initial state of the given kind
    L, seed, lam, gam, K, kind = args
    rng = np.random.default_rng(seed)
    snaps = [initial_state(L, kind, rng)]
    for _ in range(K):
        snaps.append(evolve(snaps[-1], L, DT_PAIR, DT_SIM, rng, lam, gam)[0])
    return snaps


def future(args):
    x0, L, seed, lam, gam = args
    return evolve(x0, L, DT_PAIR, DT_SIM, np.random.default_rng(seed), lam, gam)[0]


def write_pairs(path, x0s, x10s, L, traj, k, init):
    with h5py.File(path, "w") as f:
        f["x0"] = np.concatenate(x0s).astype(np.float32)
        f["x10"] = np.concatenate(x10s).astype(np.float32)
        f["off0"] = np.cumsum([0] + [len(a) for a in x0s])
        f["off10"] = np.cumsum([0] + [len(a) for a in x10s])
        f["L"] = np.full(len(x0s), float(L))
        f["traj"], f["k"], f["init"] = np.asarray(traj), np.asarray(k), np.asarray(init)
        f.attrs.update({"R": R, "D": D, "LAM": LAM, "GAM": GAM, "DT_PAIR": DT_PAIR, "DT_SIM": DT_SIM})
    n0 = np.array([len(a) for a in x0s])
    n10 = np.array([len(a) for a in x10s])
    print(f"{path}: {len(x0s)} pairs  N0 {n0.mean():.1f}  N10 {n10.mean():.1f} +- {n10.std():.1f}  min {n10.min()} max {n10.max()}")


def read_pairs(path):
    with h5py.File(path, "r") as f:
        x0, x10, off0, off10 = f["x0"][:], f["x10"][:], f["off0"][:], f["off10"][:]
        L, traj, k, init = f["L"][:], f["traj"][:], f["k"][:], f["init"][:]
    x0s = [x0[off0[e]:off0[e + 1]] for e in range(len(L))]
    x10s = [x10[off10[e]:off10[e + 1]] for e in range(len(L))]
    return x0s, x10s, L, traj, k, init
