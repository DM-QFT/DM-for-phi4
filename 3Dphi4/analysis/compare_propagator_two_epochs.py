"""
Compare diagonal propagator G(|k_hat|) of two DM checkpoints
(ep=10973 and ep=20000) against HMC L=64 reference.

Run: phi4_3d_L64_k0.1923_l0.9_ncsnpp_sigma2048
Sampler: EM, steps=2000
Outputs:
  - results/L64_k0.1923_sigma2048_propagator_two_ep_loglog.pdf
  - results/L64_k0.1923_sigma2048_propagator_two_ep_loglin.pdf
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from compare_L64_crossL_3d import (
    diagonal_propagator_3d, load_dm_3d, load_hmc_3d,
)


def main():
    k_phys = 0.1923
    lam = 0.9
    L = 64

    run_dir = Path(
        f"runs/phi4_3d_L{L}_k{k_phys}_l{lam}_ncsnpp_sigma2048"
    )
    f_ep1 = run_dir / "data" / "samples_em_steps2000_epoch=10973.npy"
    f_ep2 = run_dir / "data" / "samples_em_steps2000_epoch=20000.npy"
    f_hmc = Path(
        f"trainingdata/cfgs_wolff_fahmc_k={k_phys}_l={lam}_{L}^3.jld2"
    )
    for f in (f_ep1, f_ep2, f_hmc):
        if not f.exists():
            raise FileNotFoundError(f)

    hmc = load_hmc_3d(f_hmc, max_n=4096)
    s1 = load_dm_3d(f_ep1)
    s2 = load_dm_3d(f_ep2)

    sources = [
        ("HMC L=64",   "C0", "o", "-",  hmc),
        ("DM ep=10973", "C2", "s", "-",  s1),
        ("DM ep=20000", "C3", "v", "--", s2),
    ]

    spec = {}
    print(f"{'method':>14}  {'N':>5}  {'G(k_min)':>16}")
    print("-" * 44)
    for name, color, marker, ls, d in sources:
        k_lat, G, E = diagonal_propagator_3d(d)
        spec[name] = (color, marker, ls, k_lat, G, E)
        print(f"{name:>14}  {d.shape[0]:>5}  {G[0]:8.3f}±{E[0]:6.3f}")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"L{L}_k{k_phys}_sigma2048_propagator_two_ep"

    title = (
        f"3D phi^4 L={L}, k={k_phys}, lambda={lam}, sigma=2048, "
        f"EM/steps=2000  —  diagonal G(|k_hat|)"
    )

    # log-log
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.5, 7.0), sharex=True,
        gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.08),
    )
    for name, (color, marker, ls, k_lat, G, E) in spec.items():
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color,
                    lw=1.5, ms=5, capsize=2, label=f"{name}  G(k_min)={G[0]:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$G(|\hat k|)$")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")

    _, _, _, k_h, G_h, E_h = spec["HMC L=64"]
    axr.fill_between(k_h, 1 - E_h / G_h, 1 + E_h / G_h, color="C0", alpha=0.2)
    axr.axhline(1.0, color="C0", lw=1.0)
    for name, (color, marker, ls, k_lat, G, E) in spec.items():
        if name == "HMC L=64":
            continue
        ratio = G / G_h
        rerr = ratio * np.sqrt((E / G) ** 2 + (E_h / G_h) ** 2)
        axr.errorbar(k_lat, ratio, yerr=rerr, fmt=marker + ls, color=color,
                     lw=1.4, ms=4, capsize=2, label=name)
    axr.set_xlabel(r"$|\hat k|$  (lattice units)")
    axr.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
    axr.set_ylim(0.5, 1.5)
    axr.grid(alpha=0.3, which="both")
    axr.legend(fontsize=8, loc="upper right")

    f1 = out_dir / f"{tag}_loglog.pdf"
    fig.savefig(f1, bbox_inches="tight")
    fig.savefig(str(f1).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {f1}")

    # log-linear: linear x, log y
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.5, 7.0), sharex=True,
        gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.08),
    )
    for name, (color, marker, ls, k_lat, G, E) in spec.items():
        ax.errorbar(k_lat, G, yerr=E, fmt=marker + ls, color=color,
                    lw=1.5, ms=5, capsize=2, label=f"{name}  G(k_min)={G[0]:.2f}")
    ax.set_yscale("log")
    ax.set_ylabel(r"$G(|\hat k|)$")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper right")

    axr.fill_between(k_h, 1 - E_h / G_h, 1 + E_h / G_h, color="C0", alpha=0.2)
    axr.axhline(1.0, color="C0", lw=1.0)
    for name, (color, marker, ls, k_lat, G, E) in spec.items():
        if name == "HMC L=64":
            continue
        ratio = G / G_h
        rerr = ratio * np.sqrt((E / G) ** 2 + (E_h / G_h) ** 2)
        axr.errorbar(k_lat, ratio, yerr=rerr, fmt=marker + ls, color=color,
                     lw=1.4, ms=4, capsize=2, label=name)
    axr.set_xlabel(r"$|\hat k|$  (lattice units)")
    axr.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
    axr.set_ylim(0.5, 1.5)
    axr.grid(alpha=0.3, which="both")
    axr.legend(fontsize=8, loc="upper right")

    f2 = out_dir / f"{tag}_loglin.pdf"
    fig.savefig(f2, bbox_inches="tight")
    fig.savefig(str(f2).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {f2}")


if __name__ == "__main__":
    main()
