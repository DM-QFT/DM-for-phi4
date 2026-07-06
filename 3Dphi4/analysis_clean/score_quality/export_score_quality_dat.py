"""
Export score-quality-vs-epoch .npz files to whitespace-delimited .dat files
for Mathematica / gnuplot.  CPU-only; mirror of the ESS export utility.

Per-kappa output (7 columns):
    # k=<kappa>  L=<L>  sigma_ve=<sigma_ve>  t_eval=<t_eval>
    # epoch  cos_sim  cos_sim_sem  rel_mse  rel_mse_sem  mag_ratio  mag_ratio_sem
    1   ...

Usage:
    python3 analysis_clean/score_quality/export_score_quality_dat.py \
        --inputs analysis/results/score_quality_vs_epoch_3d_L64_k0.1923_sigma2048_ema.npz \
        --out_dir analysis/results/dat
"""

import argparse
from pathlib import Path

import numpy as np


def export_one(npz_path, out_dir):
    d = dict(np.load(npz_path, allow_pickle=True))
    epochs = np.asarray(d["epochs"]).astype(int)
    cos = np.asarray(d["cos_sim"]).astype(float)
    cos_sem = np.asarray(d["cos_sim_sem"]).astype(float)
    mse = np.asarray(d["rel_mse"]).astype(float)
    mse_sem = np.asarray(d["rel_mse_sem"]).astype(float)
    mag = np.asarray(d["mag_ratio"]).astype(float)
    mag_sem = np.asarray(d["mag_ratio_sem"]).astype(float)
    k = float(d["k"])
    L = int(d["L"])
    sigma_ve = float(d["sigma_ve"])
    t_eval = float(d["t_eval"])

    out_path = Path(out_dir) / (Path(npz_path).stem + ".dat")
    header = (
        f"k={k}  L={L}  sigma_ve={sigma_ve}  t_eval={t_eval}\n"
        f"epoch\tcos_sim\tcos_sim_sem\trel_mse\trel_mse_sem\tmag_ratio\tmag_ratio_sem"
    )
    arr = np.column_stack([epochs, cos, cos_sem, mse, mse_sem, mag, mag_sem])
    np.savetxt(out_path, arr,
               fmt=("%d", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e"),
               delimiter="\t", header=header)
    print(f"  wrote {out_path}  ({len(epochs)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for npz in args.inputs:
        export_one(npz, out_dir)


if __name__ == "__main__":
    main()
