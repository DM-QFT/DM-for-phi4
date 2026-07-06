"""
Compare DIAGONAL momentum-space propagator G(k) of 3D DM samples vs HMC.

Diagonal modes: (n, n, n) for n = 1, ..., L/2
  |k̂| = sqrt(3) · 2·sin(π n / L)
  G(k) = <|φ̃(n,n,n)|²> / V    (V = L^3)

Mirrors 2Dphi4/analysis/compare_dm_hmc_propagator.py but for 3D L^3 cfgs and
HMC files named cfgs_wolff_fahmc_k={k}_l={l}_{L}^3.jld2.

Usage (from 3Dphi4/):
    python analysis/compare_dm_hmc_propagator_3d.py \\
        --k 0.1923 --L 64 --run_suffix _sigma2760 \\
        --epochs 0001,0010,0095,0509,0977,1072,2477,10000 \\
        --method em
"""
import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).resolve().parent.parent  # 3Dphi4/
sys.path.append(str(Path(__file__).resolve().parent))
from compare_L64_crossL_3d import (  # noqa: E402
    diagonal_propagator_3d, load_dm_3d, load_hmc_3d,
)


def fit_xi_3d(k2, G, n_low=5):
    """1/G = a k² + b → ξ = √(a/b). Returns NaN if degenerate."""
    if len(k2) < n_low:
        return float("nan")
    a, b = np.polyfit(k2[:n_low], 1.0 / G[:n_low], 1)
    if a <= 0 or b <= 0:
        return float("nan")
    return 1.0 / np.sqrt(b / a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=0.1923)
    p.add_argument("--l", type=float, default=0.9)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--run_suffix", default="_sigma2760")
    p.add_argument("--epochs", default="0001,0010,0095,0509,0977,1072,2477,10000")
    p.add_argument("--method", choices=["em", "ode"], default="em")
    p.add_argument("--sde_steps", type=int, default=1000)
    p.add_argument("--ode_steps", type=int, default=400)
    p.add_argument("--max_diagonality", type=float, default=0.3334,
                   help="Radial-bin selector (Julia CorrelationUtils convention): "
                        "0.3334 keeps exactly the 8 cubic-symmetric diagonal "
                        "vertices (d=1/3), 0.51 gives a wider radial average.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    steps = args.sde_steps if args.method == "em" else args.ode_steps
    eps = [e.strip() for e in args.epochs.split(",") if e.strip()]

    run_dir = PROJ / f"runs/phi4_3d_L{args.L}_k{args.k}_l{args.l}_ncsnpp{args.run_suffix}"
    data_dir = run_dir / "data"
    hmc_path = PROJ / f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2"
    out = args.out or str(run_dir / f"diagonal_propagator_3d_{args.method}.pdf")
    mma_dir = run_dir / "mathematica_data"
    mma_dir.mkdir(parents=True, exist_ok=True)

    print(f"HMC : {hmc_path}")
    print(f"DM  : {data_dir} ({args.method}/{steps})")
    print(f"eps : {eps}")
    print(f"mma : {mma_dir}")
    print()

    print(f"{'source':>20}  {'N':>5}  {'ξ_fit':>7}  {'G(k_min)':>16}")

    mma_summary = []  # (name, N, xi, G[0], dG[0])

    hmc = load_hmc_3d(str(hmc_path))
    k_lat, G_hmc, Ge_hmc = diagonal_propagator_3d(hmc, max_diagonality=args.max_diagonality)
    xi_hmc = fit_xi_3d(k_lat ** 2, G_hmc)
    write_dat(mma_dir / "G_k_diag_HMC.dat", k_lat, G_hmc, Ge_hmc)
    mma_summary.append(("HMC", hmc.shape[0], xi_hmc, G_hmc[0], Ge_hmc[0]))
    print(f"{'HMC':>20}  {hmc.shape[0]:>5}  {xi_hmc:7.2f}  {G_hmc[0]:8.4f}±{Ge_hmc[0]:6.4f}")
    del hmc

    dm = {}
    for ep in eps:
        path = data_dir / f"samples_{args.method}_steps{steps}_epoch={ep}.npy"
        if not path.is_file():
            print(f"  [skip] {path.name}")
            continue
        x = load_dm_3d(path)
        _, G, Ge = diagonal_propagator_3d(x, max_diagonality=args.max_diagonality)
        xi = fit_xi_3d(k_lat ** 2, G)
        dm[ep] = (G, Ge, xi)
        write_dat(mma_dir / f"G_k_diag_DM_{args.method}_epoch={ep}.dat", k_lat, G, Ge)
        mma_summary.append((f"DM_ep={ep}", x.shape[0], xi, G[0], Ge[0]))
        print(f"{'DM ep=' + ep:>20}  {x.shape[0]:>5}  {xi:7.2f}  {G[0]:8.4f}±{Ge[0]:6.4f}")
        del x

    write_xi_summary(mma_dir / "xi.dat", mma_summary)

    plot_compare(k_lat, G_hmc, Ge_hmc, xi_hmc, dm, args, out)
    print(f"\nsaved: {out}")
    print(f"saved: {mma_dir}/  ({len(mma_summary)} .dat + xi.dat)")


def write_dat(path, k, G, dG):
    np.savetxt(path, np.column_stack([k, G, dG]),
               fmt=("%.10e", "%.10e", "%.10e"),
               header="k  G(k)  dG(k)   (3D diagonal modes n_x=n_y=n_z, n=1..L/2)")


def write_xi_summary(path, rows):
    with open(path, "w") as f:
        f.write("# name           N      xi      G(k_min)     dG(k_min)\n")
        for name, N, xi, g0, dg0 in rows:
            f.write(f"{name:<14s}  {N:6d}  {xi:7.3f}  {g0:12.6e}  {dg0:12.6e}\n")


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
        ax.set_xlabel(r"$|\hat k|$  (diagonal, $n_x=n_y=n_z$)")
        ax.set_ylabel(r"$G(k)$")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    draw(axes[0])
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_title("log-log")

    draw(axes[1])
    axes[1].set_xscale("log")
    axes[1].set_title("log-x, linear-y")

    fig.suptitle(f"3D DM ({args.method}) vs HMC | κ={args.k}, L={args.L}{args.run_suffix}",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    if outpath.endswith(".pdf"):
        plt.savefig(outpath.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
