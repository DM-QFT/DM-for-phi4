"""Single free particle: generated displacement over DT_PAIR versus the same Heun sampler driven by the exact wrapped-Gaussian score,
and versus the analytic wrapped Gaussian. Usage: python analysis/brownian/single_particle.py <run> <step> <K>"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
import modelb_palette as pal
from config import D, DT_PAIR, SIGMA_MIN, SIGMA_MAX
from simulate import read_pairs

RUN, STEP, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
L_LIST = [16, 24, 32]
OUT = os.path.join(PROJECT_DIR, "results", "brownian")
plt.style.use(os.path.join(PROJECT_DIR, "cjp_prl_nature.mplstyle"))
s = 2 * D * DT_PAIR     # variance per component of the free displacement


def displacement(x0s, x1s, L):
    d = np.concatenate([x1 - x0 for x0, x1 in zip(x0s, x1s)]).astype(np.float64)
    return (d - L * np.round(d / L)).ravel()


def exact_eps(z, x0, sigma, L):
    # eps = -sigma * score of the wrapped Gaussian with variance sigma^2 + s per component, images |n| <= 3
    var = sigma ** 2 + s
    dz = z - x0
    sh = np.stack([dz + n * L for n in range(-3, 4)])
    w = np.exp(-sh ** 2 / (2 * var))
    return sigma * (w * sh).sum(0) / (var * w.sum(0))


def exact_sampler(x0, L, rng):
    # the sampler of sample/sample_particle.py with the exact score, so integrator error is separated from network error
    sigmas = np.geomspace(SIGMA_MAX, SIGMA_MIN, K + 1)
    z = rng.random(x0.shape) * L
    for hi, lo in zip(sigmas[:-1], sigmas[1:]):
        dv = hi * hi - lo * lo
        xi = np.sqrt(dv) * rng.standard_normal(z.shape)
        e1 = exact_eps(z, x0, hi, L)
        e2 = exact_eps((z - dv / hi * e1 + xi) % L, x0, lo, L)
        z = (z - dv / 2 * (e1 / hi + e2 / lo) + xi) % L
    return z


os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
grid = np.linspace(-4, 4, 400)
analytic = sum(np.exp(-(grid + n * 16) ** 2 / (2 * s)) for n in range(-3, 4)) / np.sqrt(2 * np.pi * s)
fig, ax = plt.subplots()
ax.plot(grid, analytic, color=pal.REFERENCE, marker="", label="wrapped Gaussian")
rows = []
for L in L_LIST:
    x0s, x10s, _, _, _, _ = read_pairs(f"{PROJECT_DIR}/trainingdata/single_L{L}_test.h5")
    x0g, gen, _, _, _, _ = read_pairs(f"{PROJECT_DIR}/runs/{RUN}/samples/single_L{L}_test_step{STEP}_K{K}.h5")
    x0 = np.concatenate(x0s).astype(np.float64)
    ref = exact_sampler(np.concatenate([x0] * 8), L, rng)      # 8 reference samples per test event
    d_true, d_gen, d_ref = displacement(x0s, x10s, L), displacement(x0g, gen, L), ((ref - np.concatenate([x0] * 8) + L / 2) % L - L / 2).ravel()
    ks = ks_2samp(d_ref, d_gen)
    rows.append((L, len(d_gen) // 2, d_true.var(), d_ref.var(), d_gen.var(), ks.statistic, ks.pvalue))
    print(f"L={L}: var simulation {d_true.var():.3f}  exact-score sampler {d_ref.var():.3f}  network {d_gen.var():.3f}  (analytic {s:.3f})   KS(network, exact sampler) D {ks.statistic:.3f} p {ks.pvalue:.2f}")
    h, edges = np.histogram(d_gen, bins=np.linspace(-4, 4, 41), density=True)
    ax.plot(0.5 * (edges[1:] + edges[:-1]), h, linestyle="none", label=f"$L={L}$")
    if L == L_LIST[0]:
        h, edges = np.histogram(d_ref, bins=np.linspace(-4, 4, 41), density=True)
        ax.plot(0.5 * (edges[1:] + edges[:-1]), h, color=pal.GUIDE, linestyle="--", marker="", label=f"exact score, $K={K}$")
ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel(r"$p(\Delta x)$")
ax.set_xlim(-4, 4)
ax.legend()
fig.savefig(f"{OUT}/single_particle_{RUN}_step{STEP}_K{K}.pdf")
np.savetxt(f"{OUT}/single_particle_{RUN}_step{STEP}_K{K}.tsv", np.array(rows), delimiter="\t", fmt="%.6g",
           header="L n_events var_sim var_exact_sampler var_net ks_D ks_p")
