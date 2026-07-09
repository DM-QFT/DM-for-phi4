"""
Histograms comparing HMC vs DM (log / linear schedule) for the magnetization
distribution P(M) and the bootstrap sampling distribution of
chi = V(<M^2> - <|M|>^2).

Usage:
    python plot_chi_hist.py --ep 0019
"""

import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

COL = {"HMC": "#2a78d6", "DM log": "#1baf7a", "DM linear": "#eda100"}
INK = "#333333"


def chi_conn(m, V):
    return V * (np.mean(m ** 2) - np.mean(np.abs(m)) ** 2)


def bootstrap_chi(M, V, n_boot=2000, block=1, seed=1):
    """Bootstrap replicas of chi; block>1 resamples blocks (autocorrelated data)."""
    rng = np.random.default_rng(seed)
    n = len(M) // block * block
    blocks = M[:n].reshape(-1, block)
    n_blk = blocks.shape[0]
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_blk, size=n_blk)
        reps[b] = chi_conn(blocks[idx].ravel(), V)
    return reps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=float, default=0.1)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--ep", type=str, default="0019")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    V = args.L ** 2

    run_dir = f"runs/yukawa_L{args.L}_g{args.g}_ncsnpp"
    with h5py.File(f"../samples/yukawa_g{args.g}_L{args.L}_1000000.jld2", "r") as f:
        M = {"HMC": np.array(f["configs"]).mean(axis=(1, 2))}
    for sched in ["log", "linear"]:
        s = np.load(f"{run_dir}/data/samples_em_{sched}_steps{args.steps}_{args.ep}.npy")
        M[f"DM {sched}"] = s.mean(axis=(0, 1))

    boot = {
        "HMC": bootstrap_chi(M["HMC"], V, block=500),
        "DM log": bootstrap_chi(M["DM log"], V),
        "DM linear": bootstrap_chi(M["DM linear"], V),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(colors=INK, labelsize=9)

    # Panel A: P(M)
    lim = max(np.abs(m).max() for m in M.values())
    bins = np.linspace(-lim, lim, 61)
    for name, m in M.items():
        ax1.hist(m, bins=bins, density=True, histtype="step",
                 linewidth=1.8, color=COL[name], label=name)
    ax1.set_xlabel("M = (1/V) Σ φ(x)", color=INK)
    ax1.set_ylabel("P(M)", color=INK)
    ax1.set_title("Magnetization distribution\n(χ = V·connected variance of this)",
                  fontsize=10, color=INK)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK)

    # Panel B: bootstrap distribution of chi
    lo = min(r.min() for r in boot.values())
    hi = max(r.max() for r in boot.values())
    bins2 = np.linspace(lo, hi, 51)
    for name, reps in boot.items():
        c = chi_conn(M[name], V)
        e = reps.std()
        ax2.hist(reps, bins=bins2, density=True, histtype="step",
                 linewidth=1.8, color=COL[name],
                 label=f"{name}: χ = {c:.3f}({e * 1e3:02.0f})")
    ax2.set_xlabel("χ = V(⟨M²⟩ − ⟨|M|⟩²)", color=INK)
    ax2.set_ylabel("bootstrap density", color=INK)
    ax2.set_title("Sampling distribution of χ̂ (bootstrap)", fontsize=10, color=INK)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.suptitle(f"Yukawa g={args.g}, L={args.L} — HMC (N={len(M['HMC'])}) vs "
                 f"DM epoch {args.ep} (N={len(M['DM log'])}, EM {args.steps} steps)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    out = f"{run_dir}/data/chi_hist_ep{args.ep}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
