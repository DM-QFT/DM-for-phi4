"""
Diagonal momentum-space propagator G(|k̂|) for 3D phi^4 configurations.

A propagator only needs a set of field configurations of shape (N, L, L, L).
Whether the DM samples are in-distribution or cross-L (a model trained on small
lattices evaluated at a larger L) is irrelevant to the computation. So this
single script takes one HMC reference plus any number of DM ensembles, computes
the diagonal-mode propagator of each, and emits both a comparison figure and a
CSV (the latter for Mathematica re-plotting).

Diagonal modes (n, n, n), n = 1..L/2:
    |k̂| = sqrt(3)·2·sin(πn/L),   G(k) = ⟨|φ̃(k)|²⟩ / V,   φ̃ = FFT3(φ - ⟨φ⟩)
Bootstrap errors over configurations (n_boot=200).

Run from 3Dphi4/. Examples reproducing the cross-L comparisons in DM.tex
(loop over κ / variants externally by passing different --dm paths):

  # κ=0.1923 multi-L {4,8,16,32} vs L=32-only vs HMC
  python analysis/propagator/propagator.py --out L64_crossL_3d_k0.1923 \
      --hmc trainingdata/cfgs_wolff_fahmc_k=0.1923_l=0.9_64^3.jld2 \
      --dm "multi-L {4,8,16,32}=runs/phi4_3d_Lmulti4-8-16-32_k0.1923_l0.9_ncsnpp/data_crossL/samples_crossL64_multiL_ep10000_em_log_steps2000_512.npy" \
      --dm "L=32-only=runs/phi4_3d_L32_k0.1923_l0.9_ncsnpp/data_crossL/samples_crossL64_L32only_ep10000_em_log_steps2000_512.npy"

  # κ=0.2 σ-variant sweep: pass each variant's sample file as one --dm
  python analysis/propagator/propagator.py --out L64_crossL_3d_k0.2_variants --z2 \
      --hmc trainingdata/cfgs_wolff_fahmc_k=0.2_l=0.9_64^3.jld2 \
      --dm "σ=400=runs/..._sigma400/data_crossL/...ep10000.npy" \
      --dm "σ=800=runs/...   /data_crossL/...ep10000.npy"
"""

from __future__ import annotations
from pathlib import Path
import argparse
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.append(str(Path(__file__).resolve().parents[2]))   # 3Dphi4/
from _layout import to_NLLL


def load_dm(path):
    """DM samples (canonical save (L,L,L,N)) -> (N, L, L, L)."""
    return to_NLLL(np.load(path)).astype(np.float64)


def load_hmc(path, max_n=4096):
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    return to_NLLL(cfgs)[:max_n]


def diagonal_propagator(cfgs, n_boot=200, seed=0):
    """Diagonal G(|k̂|) for cfgs (N, L, L, L). Returns (k_lat, G, dG)."""
    L = cfgs.shape[1]
    phi = cfgs - cfgs.mean()
    g = (np.abs(np.fft.fftn(phi, axes=(1, 2, 3))) ** 2) / (L ** 3)
    n = np.arange(1, L // 2 + 1)
    g_diag = g[:, n, n, n]                                  # (N, L/2)
    k_lat = np.sqrt(3.0) * 2.0 * np.sin(np.pi * n / L)
    rng = np.random.default_rng(seed)
    N = g_diag.shape[0]
    boots = np.empty((n_boot, len(n)))
    for b in range(n_boot):
        boots[b] = g_diag[rng.integers(0, N, size=N)].mean(axis=0)
    return k_lat, boots.mean(axis=0), boots.std(axis=0)


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
    ap.add_argument("--out", required=True, help="output stem under results/")
    ap.add_argument("--hmc_label", default="HMC L=64")
    ap.add_argument("--epoch", default="", help="value for the legacy CSV 'epoch' column")
    ap.add_argument("--max_n_hmc", type=int, default=4096)
    ap.add_argument("--z2", action="store_true",
                    help="Z_2 symmetrize every ensemble: concat([s, -s])")
    ap.add_argument("--no_csv", action="store_true")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    sources = [(args.hmc_label, load_hmc(args.hmc, args.max_n_hmc))]
    for spec in args.dm:
        label, path = parse_dm(spec)
        sources.append((label, load_dm(path)))
    if args.z2:
        sources = [(name, np.concatenate([d, -d], axis=0)) for name, d in sources]

    results = []   # (label, N, k_lat, G, dG)
    print(f"{'ensemble':>42}  {'N':>6}  {'G(k_min)':>16}")
    for name, d in sources:
        k_lat, G, dG = diagonal_propagator(d)
        results.append((name, d.shape[0], k_lat, G, dG))
        print(f"{name:>42}  {d.shape[0]:>6}  {G[0]:8.3f} ± {dG[0]:6.3f}")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_csv:
        csv_path = out_dir / f"{args.out}_propagator.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "epoch", "N_samples", "k_lat", "G", "dG"])
            for name, N, k_lat, G, dG in results:
                for i in range(len(k_lat)):
                    w.writerow([name, args.epoch, N, f"{k_lat[i]:.6f}",
                                f"{G[i]:.6e}", f"{dG[i]:.6e}"])
        print(f"\nSaved CSV: {csv_path}")

    if not args.no_plot:
        ref_name, _, k_ref, G_ref, E_ref = results[0]   # first source = HMC
        fig, (ax, ax_r) = plt.subplots(
            2, 1, figsize=(7, 7), sharex=True,
            gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.08))
        for i, (name, N, k_lat, G, dG) in enumerate(results):
            fmt = "o-" if i == 0 else "s--"
            ax.errorbar(k_lat, G, yerr=dG, fmt=fmt, lw=1.5, ms=5, capsize=2,
                        color=f"C{i}", label=name)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(r"$G(|\hat k|)$  (diagonal modes)")
        ax.set_title(args.out)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="lower left")

        ax_r.fill_between(k_ref, 1 - E_ref / G_ref, 1 + E_ref / G_ref,
                          color="C0", alpha=0.2)
        ax_r.axhline(1.0, color="C0", lw=1.0)
        for i, (name, N, k_lat, G, dG) in enumerate(results[1:], start=1):
            ratio = G / G_ref
            rerr = ratio * np.sqrt((dG / G) ** 2 + (E_ref / G_ref) ** 2)
            ax_r.errorbar(k_lat, ratio, yerr=rerr, fmt="s--", lw=1.4, ms=4,
                          capsize=2, color=f"C{i}")
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
