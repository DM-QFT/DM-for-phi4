"""Periodic point-set statistics in numpy: pair correlations, structure factor, initial-to-final density propagator, box-count Fano factor."""
import numpy as np
from scipy.spatial import cKDTree


def min_image(d, L):
    return d - L * np.round(d / L)


def pair_correlation(xs, L, rho_max=5.0, n_bins=50):
    # g(rho) of one set, averaged over events. normalisation: uniform density gives 1
    edges = np.linspace(0, rho_max, n_bins + 1)
    counts, norm = np.zeros(n_bins), 0.0
    for x in xs:
        p = cKDTree(x, boxsize=L).query_pairs(rho_max, output_type="ndarray")
        counts += np.histogram(np.linalg.norm(min_image(x[p[:, 1]] - x[p[:, 0]], L), axis=1), edges)[0]
        norm += len(x) * (len(x) - 1) / 2 / L ** 2
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return 0.5 * (edges[1:] + edges[:-1]), counts / (norm * shell)


def cross_correlation(x0s, x1s, L, rho_max=5.0, n_bins=50):
    # density of set 1 at distance rho from a particle of set 0, relative to the mean density. memory of X0 at particle level
    edges = np.linspace(0, rho_max, n_bins + 1)
    counts, norm = np.zeros(n_bins), 0.0
    for x0, x1 in zip(x0s, x1s):
        pairs = cKDTree(x0, boxsize=L).query_ball_tree(cKDTree(x1, boxsize=L), rho_max)
        i = np.repeat(np.arange(len(x0)), [len(p) for p in pairs])
        j = np.concatenate([np.asarray(p, dtype=np.int64) for p in pairs])
        counts += np.histogram(np.linalg.norm(min_image(x1[j] - x0[i], L), axis=1), edges)[0]
        norm += len(x0) * len(x1) / L ** 2
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return 0.5 * (edges[1:] + edges[:-1]), counts / (norm * shell)


def modes(L, n_max):
    # all nonzero integer wave vectors with |n_x|,|n_y| <= n_max, grouped by |n|^2
    n = np.array([(a, b) for a in range(-n_max, n_max + 1) for b in range(-n_max, n_max + 1) if (a, b) != (0, 0)])
    n2 = (n ** 2).sum(1)
    return 2 * np.pi * n / L, n2


def rho_k(x, k):
    return np.exp(1j * x.astype(np.float64) @ k.T).sum(0)   # (n_modes,)


def structure_factor(xs, L, n_max=8):
    # S(k) = <|rho_k|^2>/N on shells of equal |n|^2. Poisson gives 1
    k, n2 = modes(L, n_max)
    s = np.mean([np.abs(rho_k(x, k)) ** 2 / len(x) for x in xs], axis=0)
    shells = np.unique(n2)
    return 2 * np.pi * np.sqrt(shells) / L, np.array([s[n2 == q].mean() for q in shells])


def propagator(x0s, x1s, L, n_max=8):
    # G(k) = Re<rho_k(X0)^* rho_k(X1)> / <|rho_k(X0)|^2>, fraction of the initial density mode that persists. independent sets give 0
    k, n2 = modes(L, n_max)
    num = np.mean([np.real(np.conj(rho_k(x0, k)) * rho_k(x1, k)) for x0, x1 in zip(x0s, x1s)], axis=0)
    den = np.mean([np.abs(rho_k(x0, k)) ** 2 for x0 in x0s], axis=0)
    shells = np.unique(n2)
    return 2 * np.pi * np.sqrt(shells) / L, np.array([num[n2 == q].sum() / den[n2 == q].sum() for q in shells])


def fano(xs, L, cells=(1, 2, 4, 8)):
    # variance over mean of the particle count in square cells of side ell. Poisson gives 1
    out = []
    for ell in cells:
        m = int(round(L / ell))
        c = np.concatenate([np.bincount((np.floor(x[:, 0] / ell).astype(int) % m) * m + np.floor(x[:, 1] / ell).astype(int) % m, minlength=m * m) for x in xs])
        out.append(c.var() / c.mean())
    return np.array(cells, dtype=float), np.array(out)
