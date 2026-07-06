"""
Export the diagonal propagator G(|k_lat|) for the 3D L=64 cross-L comparison
for k ∈ {0.18, 0.2} into long-format CSVs.

Mirrors export_L64_crossL_3d_propagator.py (which targets k=0.1923) but only
emits the multi-L vs HMC subset, since L4/L8/L16-only crossL64 samples do not
exist for these kappas.

Curves: HMC L=64, multi-L {4,8}, multi-L {4,8,16}, multi-L {4,8,16,32}.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from compare_L64_crossL_3d import (
    diagonal_propagator_3d, load_dm_3d, load_hmc_3d,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, required=True, choices=[0.18, 0.2])
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--ep", type=int, default=10000)
    ap.add_argument("--num_steps", type=int, default=2000)
    ap.add_argument("--schedule", type=str, default="linear",
                    choices=["log", "linear"])
    args = ap.parse_args()

    L = 64
    suffix = f"em_{args.schedule}_steps{args.num_steps}_ep{args.ep}"

    # κ=0.2 has a sigma=400 variant for multi-L 4-8-16-32; default ("ncsnpp")
    # was trained at sigma=800. Use sigma=400 for κ=0.2 only — κ=0.18 has no
    # such variant. See training_config.yaml in each run dir.
    multi32_suffix = "_sigma400" if abs(args.k - 0.2) < 1e-9 else ""

    cases = [
        ("HMC_L64",          None,
         f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.lam}_{L}^3.jld2"),
        ("multiL_4-8",       args.ep,
         f"runs/phi4_3d_Lmulti4-8_k{args.k}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL_train8_sample{L}_{suffix}.npy"),
        ("multiL_4-8-16",    args.ep,
         f"runs/phi4_3d_Lmulti4-8-16_k{args.k}_l{args.lam}_ncsnpp/data_crossL/"
         f"samples_crossL_train16_sample{L}_{suffix}.npy"),
        ("multiL_4-8-16-32", args.ep,
         f"runs/phi4_3d_Lmulti4-8-16-32_k{args.k}_l{args.lam}_ncsnpp{multi32_suffix}/data_crossL/"
         f"samples_crossL_train32_sample{L}_{suffix}.npy"),
    ]

    rows = []
    print(f"\nκ={args.k}, λ={args.lam}, L={L}, ep={args.ep}")
    print(f"{'method':>22}  {'ep':>5}  {'N':>5}  {'G(k_min)':>16}")
    print("-" * 60)
    for name, ep, path in cases:
        p = Path(path)
        if not p.exists():
            print(f"[skip] {name}: {p}")
            continue
        cfgs = load_hmc_3d(p) if name.startswith("HMC") else load_dm_3d(p)
        k_lat, G, dG = diagonal_propagator_3d(cfgs)
        N = cfgs.shape[0]
        ep_str = "" if ep is None else str(ep)
        for kk, gg, ee in zip(k_lat, G, dG):
            rows.append((name, ep_str, N, kk, gg, ee))
        print(f"{name:>22}  {ep_str:>5}  {N:>5}  {G[0]:8.3f}±{dG[0]:6.3f}")

    out = Path(f"results/L{L}_crossL_3d_k{args.k}_propagator.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "epoch", "N_samples", "k_lat", "G", "dG"])
        for row in rows:
            w.writerow([row[0], row[1], row[2],
                        f"{row[3]:.10g}", f"{row[4]:.10g}", f"{row[5]:.10g}"])
    print(f"\nSaved: {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
