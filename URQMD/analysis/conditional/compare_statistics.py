"""Generated versus true futures at fixed L: g(rho) (collapse check), cross correlation g01 with X0, S(k), density propagator G(k).
Linear-response references for uniform X0: G = exp(-(lam + D k^2) dt), g01 = 1 + exp(-lam dt) Gauss(rho; 2 D dt).
Usage: python analysis/conditional/compare_statistics.py <run> <step> <L> <split> <K>"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
import modelb_palette as pal
from config import D, LAM, DT_PAIR
from simulate import read_pairs
from observables import pair_correlation, cross_correlation, structure_factor, propagator

RUN, STEP, L, SPLIT, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
OUT = os.path.join(PROJECT_DIR, "results", "conditional")
plt.style.use(os.path.join(PROJECT_DIR, "cjp_prl_nature.mplstyle"))

tag = json.load(open(f"{PROJECT_DIR}/runs/{RUN}/meta.json"))["tag"]
lam = LAM if tag == "reaction" else 0.0
x0s, x10s, _, _, _, _ = read_pairs(f"{PROJECT_DIR}/trainingdata/{tag}_L{L}_{SPLIT}.h5")
_, gens, _, _, _, _ = read_pairs(f"{PROJECT_DIR}/runs/{RUN}/samples/{tag}_L{L}_{SPLIT}_step{STEP}_K{K}.h5")
name = f"{tag}_L{L}_{SPLIT}_{RUN}_step{STEP}_K{K}"
os.makedirs(OUT, exist_ok=True)


def collapsed(xs):
    # fraction of particles whose nearest neighbour is closer than 0.1 r
    return np.mean([(cKDTree(x, boxsize=L).query(x, k=2)[0][:, 1] < 0.1).mean() for x in xs])


stats = {}
for lab, xs in (("true", x10s), ("gen", gens)):
    stats[lab] = {"g": pair_correlation(xs, L), "g01": cross_correlation(x0s, xs, L), "S": structure_factor(xs, L), "G": propagator(x0s, xs, L)}
    print(f"{lab}: collapsed fraction {collapsed(xs):.4f}  G(k_min) {stats[lab]['G'][1][0]:.3f}  g01(0) {stats[lab]['g01'][1][0]:.3f}")
with open(f"{OUT}/{name}.tsv", "w") as f:
    for key in ("g", "g01", "S", "G"):
        f.write(f"# {key}: x true gen\n")
        np.savetxt(f, np.column_stack([stats["true"][key][0], stats["true"][key][1], stats["gen"][key][1]]), fmt="%.6g", delimiter="\t")

s = 2 * D * DT_PAIR
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.06))
for ax, key, xl, yl, lab in zip(axes.ravel(), ("g", "g01", "S", "G"), (r"$\rho$", r"$\rho$", r"$k$", r"$k$"),
                                (r"$g(\rho)$", r"$g_{01}(\rho)$", r"$S(k)$", r"$G(k)$"), "abcd"):
    x, yt = stats["true"][key]
    ax.plot(x, yt, color=pal.REFERENCE, marker="", label="simulation")
    ax.plot(x, stats["gen"][key][1], color=pal.DM, linestyle="none", marker="o", markersize=3, label="diffusion model")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.text(0.04, 0.92, f"({lab})", transform=ax.transAxes)
rho, k = stats["true"]["g01"][0], stats["true"]["G"][0]
axes[0, 1].plot(rho, 1 + np.exp(-lam * DT_PAIR) * np.exp(-rho ** 2 / (2 * s)) / (2 * np.pi * s), color=pal.GUIDE, linestyle=":", marker="", label="linear response")
axes[1, 1].plot(k, np.exp(-(lam + D * k ** 2) * DT_PAIR), color=pal.GUIDE, linestyle=":", marker="", label="linear response")
axes[1, 1].axhline(0, color=pal.ZERO, linewidth=0.8, zorder=0)
axes[0, 1].legend(loc="upper right")
fig.savefig(f"{OUT}/{name}.pdf")
