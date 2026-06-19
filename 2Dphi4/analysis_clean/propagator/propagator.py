"""
Diagonal momentum-space propagator G(|k̂|) for 2D phi^4 configurations.

A propagator only needs (N, L, L) field configurations; whether the DM samples
are in-distribution or cross-L (a model trained on small lattices evaluated at a
larger L) is irrelevant to the computation. This single script takes one HMC
reference plus any number of DM ensembles, computes the diagonal-mode propagator
of each (with an Ornstein-Zernike ξ fit), and emits a comparison figure plus
per-source G_k_<label>.dat files (|k| G G_err) for Mathematica re-plotting.

Diagonal modes (n, n), n = 1..L/2 (match CorrelationUtils.jl):
    k̂² = 8 sin²(πn/L),  |k̂| = √8·sin(πn/L),  G(k) = ⟨|φ̃(k)|²⟩/V,  φ̃ = FFT2(φ-⟨φ⟩)
Bootstrap errors over configurations (n_boot=200).

Run from 2Dphi4/. Example (cross-L L=128: multi-L vs L=64-only vs HMC):
    python analysis/propagator/propagator.py --out crossL_k0.2705 \
        --hmc trainingdata/cfgs_wolff_fahmc_k=0.2705_l=0.022_128^2.jld2 \
        --dm "multi-L {8,16,32,64}=runs/phi4_Lmulti8-16-32-64_k0.2705_l0.022_ncsnpp/data_crossL/samples_....npy" \
        --dm "L=64-only=runs/phi4_L64_k0.2705_l0.022_ncsnpp/data_crossL/samples_....npy"
Outputs results/<out>/<out>.pdf and results/<out>/G_k_*.dat
"""

from __future__ import annotations
from pathlib import Path
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt
import h5py


def load_dm(path):
    """DM samples (canonical save (L, L, N)) -> (N, L, L)."""
    return np.load(path).transpose(2, 0, 1).astype(np.float64)


def load_hmc(path, max_n=8192):
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"]).astype(np.float64)
    sa = int(np.argmax(cfgs.shape))
    if sa != 0:
        cfgs = np.moveaxis(cfgs, sa, 0)
    return cfgs[:max_n]


def diagonal_propagator(cfgs, n_boot=200, seed=0, mode="diagonal"):
    """Momentum-space G(|k̂|) for cfgs (N, L, L). Returns (k_lat, G, dG).

    mode='diagonal' (default): modes (n,n),  k̂² = 8 sin²(πn/L).
    mode='axial'             : modes (n,0),  k̂² = 4 sin²(πn/L)
        (the former compare_multiL_propagator.py axial_G).
    """
    N, L, _ = cfgs.shape
    V = L * L
    phi = cfgs - cfgs.mean()
    pk = (np.abs(np.fft.fft2(phi, axes=(1, 2))) ** 2) / V
    n = np.arange(1, L // 2 + 1)                       # skip n=0
    if mode == "axial":
        g = pk[:, n, 0]                                # modes (n, 0)
        k_lat = 2.0 * np.sin(np.pi * n / L)            # k̂² = 4 sin²(πn/L)
    else:
        g = pk[:, n, n]                                # modes (n, n)
        k_lat = np.sqrt(8.0) * np.sin(np.pi * n / L)   # k̂² = 8 sin²(πn/L)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(n)))
    for b in range(n_boot):
        boots[b] = g[rng.integers(0, N, size=N)].mean(axis=0)
    return k_lat, boots.mean(axis=0), boots.std(axis=0)


def fit_xi(k_lat, G, n_low=5):
    """OZ fit 1/G = a k² + b on the n_low lowest modes -> ξ = sqrt(a/b)."""
    if len(k_lat) < n_low:
        return float("nan")
    k2 = k_lat[:n_low] ** 2
    a, b = np.polyfit(k2, 1.0 / G[:n_low], 1)
    return 1.0 / np.sqrt(b / a) if (a > 0 and b > 0) else float("nan")


def parse_dm(spec):
    """'label=path' -> (label, path). A bare path uses its filename as label."""
    label, sep, path = spec.partition("=")
    return (label, path) if sep else (Path(spec).stem, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hmc", required=True, help="HMC reference .jld2")
    ap.add_argument("--dm", action="append", default=[], metavar="LABEL=PATH",
                    help="a DM ensemble .npy (repeatable)")
    ap.add_argument("--out", required=True, help="output stem (dir under results/)")
    ap.add_argument("--hmc_label", default="HMC")
    ap.add_argument("--max_n_hmc", type=int, default=8192)
    ap.add_argument("--mode", choices=["diagonal", "axial"], default="diagonal",
                    help="momentum modes: diagonal (n,n) or axial (n,0)")
    ap.add_argument("--z2", action="store_true",
                    help="Z_2 symmetrize every ensemble: concat([s, -s])")
    ap.add_argument("--no_dat", action="store_true")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    sources = [(args.hmc_label, load_hmc(args.hmc, args.max_n_hmc))]
    for spec in args.dm:
        label, path = parse_dm(spec)
        sources.append((label, load_dm(path)))
    if args.z2:
        sources = [(name, np.concatenate([d, -d], axis=0)) for name, d in sources]

    results = []   # (label, N, k_lat, G, dG, xi)
    print(f"{'ensemble':>42}  {'N':>6}  {'xi':>6}  {'G(k_min)':>16}")
    for name, d in sources:
        k_lat, G, dG = diagonal_propagator(d, mode=args.mode)
        xi = fit_xi(k_lat, G)
        results.append((name, d.shape[0], k_lat, G, dG, xi))
        print(f"{name:>42}  {d.shape[0]:>6}  {xi:6.2f}  {G[0]:8.4f} ± {dG[0]:6.4f}")

    out_dir = Path("results") / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_dat:
        for name, N, k_lat, G, dG, xi in results:
            safe = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
            np.savetxt(out_dir / f"G_k_{safe}.dat",
                       np.column_stack([k_lat, G, dG]), fmt="%.8e", delimiter="\t")
        print(f"\nSaved .dat -> {out_dir}/G_k_*.dat")

    if not args.no_plot:
        ref_name, _, k_ref, G_ref, E_ref, _ = results[0]   # first source = HMC
        fig, (ax, ax_r) = plt.subplots(
            2, 1, figsize=(7, 7), sharex=True,
            gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.08))
        for i, (name, N, k_lat, G, dG, xi) in enumerate(results):
            lbl = f"{name}  (ξ={xi:.1f})" if not np.isnan(xi) else name
            ax.errorbar(k_lat, G, yerr=dG, fmt=("o-" if i == 0 else "s--"),
                        color=f"C{i}", lw=1.5, ms=5, capsize=2, label=lbl)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(r"$G(|\hat k|)$  (diagonal modes, $n_x=n_y$)")
        ax.set_title(args.out)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="lower left")

        ax_r.fill_between(k_ref, 1 - E_ref / G_ref, 1 + E_ref / G_ref,
                          color="C0", alpha=0.2)
        ax_r.axhline(1.0, color="C0", lw=1.0)
        for i, (name, N, k_lat, G, dG, xi) in enumerate(results[1:], start=1):
            ratio = G / G_ref
            rerr = ratio * np.sqrt((dG / G) ** 2 + (E_ref / G_ref) ** 2)
            ax_r.errorbar(k_lat, ratio, yerr=rerr, fmt="s--", color=f"C{i}",
                          lw=1.4, ms=4, capsize=2)
        ax_r.set_ylim(0.4, 1.6)
        ax_r.set_xlabel(r"$|\hat k|$  (lattice units)")
        ax_r.set_ylabel(r"$G_{\rm DM}/G_{\rm HMC}$")
        ax_r.grid(alpha=0.3, which="both")

        pdf = out_dir / f"{args.out}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(str(pdf).replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
        print(f"Saved figure: {pdf}")


if __name__ == "__main__":
    main()
