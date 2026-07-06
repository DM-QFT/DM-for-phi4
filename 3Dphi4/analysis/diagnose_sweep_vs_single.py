"""
Side-by-side diagonal propagator for the SAME ep=NNNN ckpt, comparing samples
produced by sample_phi4_3d_sweep.py vs sample_phi4.py (single-ckpt version).

Use to localise whether the n=3 propagator anomaly comes from the sweep script
itself (state residue between ckpts, manual_seed timing, etc.) or from the
Miyabi numerical environment (cuDNN/bf16/GPU arch).

Run from 3Dphi4/:
    python analysis/diagnose_sweep_vs_single.py \\
        --k 0.1923 --L 64 --run_suffix _sigma2048 --ep 20000 --sde_steps 1000
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))
from compare_L64_crossL_3d import (  # noqa: E402
    diagonal_propagator_3d, load_dm_3d, load_hmc_3d,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=float, default=0.1923)
    p.add_argument("--l", type=float, default=0.9)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--run_suffix", default="_sigma2048")
    p.add_argument("--ep", type=int, default=20000,
                   help="epoch tag (integer; 4-digit padded for sweep path)")
    p.add_argument("--method", default="em")
    p.add_argument("--sde_steps", type=int, default=1000)
    args = p.parse_args()

    run_dir = PROJ / f"runs/phi4_3d_L{args.L}_k{args.k}_l{args.l}_ncsnpp{args.run_suffix}"
    data_dir = run_dir / "data"

    ep_str = f"{args.ep:04d}"
    sweep_path = data_dir / f"samples_{args.method}_steps{args.sde_steps}_epoch={ep_str}.npy"
    # sample_phi4.py uses --ep as a string verbatim; try both 4-digit and bare int
    single_paths = [
        data_dir / f"samples_{args.method}_steps{args.sde_steps}_{args.ep}.npy",
        data_dir / f"samples_{args.method}_steps{args.sde_steps}_{ep_str}.npy",
    ]
    hmc_path = PROJ / f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^3.jld2"

    print(f"HMC    : {hmc_path}")
    print(f"sweep  : {sweep_path}    {'OK' if sweep_path.exists() else 'MISSING'}")
    single_path = next((p_ for p_ in single_paths if p_.exists()), None)
    print(f"single : {single_path or single_paths[0]}    {'OK' if single_path else 'MISSING'}")
    print()

    hmc = load_hmc_3d(str(hmc_path))
    k_lat, G_hmc, dG_hmc = diagonal_propagator_3d(hmc)
    N_hmc = hmc.shape[0]

    rows = []
    G_sweep = G_single = dG_sweep = dG_single = None
    if sweep_path.exists():
        x = load_dm_3d(sweep_path)
        _, G_sweep, dG_sweep = diagonal_propagator_3d(x)
        N_sw = x.shape[0]
        del x
    else:
        N_sw = 0
    if single_path is not None:
        x = load_dm_3d(single_path)
        _, G_single, dG_single = diagonal_propagator_3d(x)
        N_si = x.shape[0]
        del x
    else:
        N_si = 0

    print(f"N_hmc={N_hmc}  N_sweep={N_sw}  N_single={N_si}")
    print()
    hdr = (f"{'n':>3}  {'|k|':>7}  "
           f"{'G_HMC':>12}  {'G_sweep':>14}  {'sweep/HMC':>10}  "
           f"{'G_single':>14}  {'single/HMC':>11}  {'sweep/single':>12}")
    print(hdr)
    print("-" * len(hdr))
    for i, n in enumerate(range(1, args.L // 2 + 1)):
        line = f"{n:>3}  {k_lat[i]:>7.3f}  {G_hmc[i]:>8.4f}±{dG_hmc[i]:5.4f}"
        if G_sweep is not None:
            line += (f"  {G_sweep[i]:>8.4f}±{dG_sweep[i]:5.4f}"
                     f"  {G_sweep[i] / G_hmc[i]:>10.3f}")
        else:
            line += "  " + " " * 14 + "  " + " " * 10
        if G_single is not None:
            line += (f"  {G_single[i]:>8.4f}±{dG_single[i]:5.4f}"
                     f"  {G_single[i] / G_hmc[i]:>11.3f}")
        else:
            line += "  " + " " * 14 + "  " + " " * 11
        if G_sweep is not None and G_single is not None:
            line += f"  {G_sweep[i] / G_single[i]:>12.3f}"
        print(line)


if __name__ == "__main__":
    main()
