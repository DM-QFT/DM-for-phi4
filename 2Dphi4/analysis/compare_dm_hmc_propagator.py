"""
Compare DIAGONAL momentum-space propagator G(k) of DM samples vs HMC.

Diagonal modes: (n, n) for n = 1, ..., L/2
  k² = 8 sin²(π n / L)
  G(k) = <|φ̃(n,n)|²> / V

Usage (from 2Dphi4/):
    python analysis/compare_dm_hmc_propagator.py \
        --k 0.28 --L 128 --run_suffix _sigma320_opt \
        --epochs 0001,0010,0095,0104,0509,0977,1072,2477 \
        --method em
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJ = Path(__file__).resolve().parent.parent  # 2Dphi4/


def diagonal_propagator(cfgs, n_boot=200, seed=0):
    """cfgs: (N, L, L). Returns k_lat, G_mean, G_err on diagonal modes (n,n)."""
    N, L, _ = cfgs.shape
    V = L * L
    phi = cfgs - cfgs.mean()
    fk = np.fft.fft2(phi, axes=(1, 2))
    pk = (fk * fk.conj()).real / V                # (N, L, L)

    n_diag = np.arange(1, L // 2 + 1)             # skip n=0
    g_per_cfg = pk[:, n_diag, n_diag]             # (N, L/2)

    k2 = 8.0 * np.sin(np.pi * n_diag / L) ** 2    # diagonal: 2 × 4sin²(πn/L)
    k_lat = np.sqrt(k2)

    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(n_diag)))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = g_per_cfg[idx].mean(axis=0)
    return k_lat, boots.mean(axis=0), boots.std(axis=0)


def fit_xi(k2, G, n_low=5):
    """1/G = a k² + b → ξ = √(a/b)."""
    if len(k2) < n_low:
        return float("nan")
    a, b = np.polyfit(k2[:n_low], 1.0 / G[:n_low], 1)
    if a <= 0 or b <= 0:
        return float("nan")
    return 1.0 / np.sqrt(b / a)


def load_dm(path):
    return np.load(path).transpose(2, 0, 1).astype(np.float64)   # (L,L,N) → (N,L,L)


def load_hmc(path):
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"]).astype(np.float64)
    sa = int(np.argmax(cfgs.shape))
    if sa != 0:
        cfgs = np.moveaxis(cfgs, sa, 0)
    return cfgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--run_suffix", default="_sigma640")
    p.add_argument("--epochs", default="0001,0010,0097,0102,0962,1070,5013,10018")
    p.add_argument("--method", choices=["em", "ode"], default="em")
    p.add_argument("--sde_steps", type=int, default=2000)
    p.add_argument("--ode_steps", type=int, default=400)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    steps = args.sde_steps if args.method == "em" else args.ode_steps
    eps = [e.strip() for e in args.epochs.split(",") if e.strip()]

    run_dir = PROJ / f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_ncsnpp{args.run_suffix}"
    data_dir = run_dir / "data"
    hmc_path = PROJ / f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"
    out = args.out or str(run_dir / f"diagonal_propagator_{args.method}.pdf")

    print(f"HMC : {hmc_path}")
    print(f"DM  : {data_dir} ({args.method}/{steps})")
    print(f"eps : {eps}")
    print()

    print(f"{'source':>20}  {'N':>5}  {'ξ_fit':>7}  {'G(k_min)':>16}")

    hmc = load_hmc(str(hmc_path))
    k_lat, G_hmc, Ge_hmc = diagonal_propagator(hmc)
    xi_hmc = fit_xi(k_lat ** 2, G_hmc)
    print(f"{'HMC':>20}  {hmc.shape[0]:>5}  {xi_hmc:7.2f}  {G_hmc[0]:8.4f}±{Ge_hmc[0]:6.4f}")
    del hmc

    dm = {}
    for ep in eps:
        path = data_dir / f"samples_{args.method}_steps{steps}_epoch={ep}.npy"
        if not path.is_file():
            print(f"  [skip] {path.name}")
            continue
        x = load_dm(path)
        _, G, Ge = diagonal_propagator(x)
        xi = fit_xi(k_lat ** 2, G)
        dm[ep] = (G, Ge, xi)
        print(f"{'DM ep=' + ep:>20}  {x.shape[0]:>5}  {xi:7.2f}  {G[0]:8.4f}±{Ge[0]:6.4f}")
        del x

    plot_compare(k_lat, G_hmc, Ge_hmc, xi_hmc, dm, args, out)
    print(f"\nsaved: {out}")


def plot_compare(k_lat, G_hmc, Ge_hmc, xi_hmc, dm, args, outpath):
    eps_sorted = sorted(dm, key=int)
    cmap = mpl.colormaps.get_cmap("viridis")

    def draw(ax):
        ax.errorbar(k_lat, G_hmc, yerr=Ge_hmc, fmt="o-", ms=4, color="black",
                    label=f"HMC  (ξ={xi_hmc:.1f})", capsize=2, zorder=10)
        for i, ep in enumerate(eps_sorted):
            G, Ge, xi = dm[ep]
            c = cmap(i / max(1, len(eps_sorted) - 1))
            ax.errorbar(k_lat, G, yerr=Ge, fmt="s-", ms=3, color=c, lw=1.2,
                        label=f"ep={ep}  (ξ={xi:.1f})", capsize=1.5, alpha=0.85)
        ax.set_xlabel(r"$|\hat k|$  (diagonal, $n_x=n_y$)")
        ax.set_ylabel(r"$G(k)$")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    draw(axes[0])
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_title("log-log")

    draw(axes[1])
    axes[1].set_xscale("log")
    axes[1].set_title("log-x, linear-y")

    fig.suptitle(f"DM ({args.method}) vs HMC | κ={args.k}, L={args.L}{args.run_suffix}",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    if outpath.endswith(".pdf"):
        plt.savefig(outpath.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
