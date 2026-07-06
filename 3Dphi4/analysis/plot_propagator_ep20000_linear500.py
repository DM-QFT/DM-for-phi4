"""
Quick check: DM (ep=20000, EM, linear schedule, 500 steps, 64 samples) vs HMC L=64.

Two panels side by side (log-log + log-linear), both with log x-axis, no ratio.
"""
from __future__ import annotations
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from compare_L64_crossL_3d import (
    diagonal_propagator_3d, load_dm_3d, load_hmc_3d,
)


def main():
    k_phys = 0.1923
    lam = 0.9
    L = 64

    run_dir = Path(f"runs/phi4_3d_L{L}_k{k_phys}_l{lam}_ncsnpp_sigma2048")
    f_dm  = run_dir / "data" / "samples_em_steps500_20000_lin500.npy"
    f_hmc = Path(f"trainingdata/cfgs_wolff_fahmc_k={k_phys}_l={lam}_{L}^3.jld2")
    for f in (f_dm, f_hmc):
        if not f.exists():
            raise FileNotFoundError(f)

    hmc = load_hmc_3d(f_hmc, max_n=4096)
    dm  = load_dm_3d(f_dm)

    sources = [
        ("HMC L=64",                                "C0", "o", "-",  hmc),
        ("DM ep=20000, EM, linear, steps=500, N=64", "C3", "s", "-",  dm),
    ]

    spec = {}
    print(f"{'method':>42}  {'N':>5}  {'G(k_min)':>16}")
    print("-" * 70)
    for name, color, marker, ls, d in sources:
        k_lat, G, E = diagonal_propagator_3d(d)
        spec[name] = (color, marker, ls, k_lat, G, E)
        print(f"{name:>42}  {d.shape[0]:>5}  {G[0]:8.3f}±{E[0]:6.3f}")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"L{L}_k{k_phys}_sigma2048_ep20000_linear500_N64_propagator"
    title = (
        f"3D phi^4 L={L}, k={k_phys}, lambda={lam}, sigma=2048   "
        f"DM: EM/linear/steps=500/N=64, ckpt ep=20000"
    )

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_xlabel(r"$|\hat k|$  (lattice units)")
        ax.set_ylabel(r"$G(|\hat k|)$")
        ax.grid(alpha=0.3, which="both")
    axL.set_yscale("log")
    axL.set_title("log-log")
    axR.set_title("log-linear (y linear)")

    for name, (color, marker, ls, k_lat, G, E) in spec.items():
        lab = f"{name}  G(k_min)={G[0]:.2f}"
        for ax in (axL, axR):
            ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color,
                        lw=1.5, ms=5, capsize=2, label=lab)
    axL.legend(fontsize=9, loc="lower left")
    axR.legend(fontsize=9, loc="upper right")
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()

    out = out_dir / f"{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
