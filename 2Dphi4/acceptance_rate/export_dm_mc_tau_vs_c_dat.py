"""
Export dm_mc_tau_vs_c .npz files to whitespace-delimited .dat files for
Mathematica / gnuplot.

Per-kappa output (11 columns):
  # k=...  L=...  epoch=...  sigma_ve=...  t_mh=...  M=...  N=...
  # c  tau_M_fit  tau_S_fit  tau_M_sokal  tau_M_sokal_se  tau_S_sokal  tau_S_sokal_se  ess_M  ess_S  ess_eff  accept
  0.01  ...

The "fit" columns (cols 2 and 3) are the recommended ones for plotting tau vs c.
The Sokal columns (4-7) are kept for cross-check but unreliable when N << tau_true.

Usage (CPU only):
  python3 export_dm_mc_tau_vs_c_dat.py \
      --inputs path/to/dm_mc_tau_vs_c_*.npz \
      --out_dir path/to/dat_out
"""

import argparse
from pathlib import Path

import numpy as np


def export_one(npz_path, out_dir):
    d = dict(np.load(npz_path, allow_pickle=True))
    c = np.asarray(d["c_list"]).astype(float)
    tau_M_fit = np.asarray(d["tau_M_fit"]).astype(float)
    tau_S_fit = np.asarray(d["tau_S_fit"]).astype(float)
    tau_M  = np.asarray(d["tau_sokal"]).astype(float)
    tau_Ms = np.asarray(d["tau_se"]).astype(float)
    tau_S  = np.asarray(d["tau_S"]).astype(float)
    tau_Ss = np.asarray(d["tau_S_se"]).astype(float)
    ess_M  = np.asarray(d["ess_M"]).astype(float)
    ess_S  = np.asarray(d["ess_S"]).astype(float)
    ess_e  = np.asarray(d["ess_eff"]).astype(float)
    acc    = np.asarray(d["accept"]).astype(float)

    k = float(d["k"]); L = int(d["L"]); epoch = int(d["epoch"])
    sigma_ve = float(d["sigma_ve"]); t_mh = float(d["t_mh"])
    M = int(d["num_chains"]); N = int(d["mh_steps"])

    out_path = Path(out_dir) / (Path(npz_path).stem + ".dat")
    header = (
        f"k={k}  L={L}  epoch={epoch}  sigma_ve={sigma_ve}  "
        f"t_mh={t_mh}  M={M}  N={N}\n"
        f"c\ttau_M_fit\ttau_S_fit\ttau_M_sokal\ttau_M_sokal_se"
        f"\ttau_S_sokal\ttau_S_sokal_se\tess_M\tess_S\tess_eff\taccept"
    )
    arr = np.column_stack([
        c,
        tau_M_fit, tau_S_fit,
        tau_M, tau_Ms,
        tau_S, tau_Ss,
        ess_M, ess_S, ess_e, acc,
    ])
    np.savetxt(out_path, arr, fmt="%.8e", delimiter="\t", header=header)
    print(f"  wrote {out_path}  ({len(c)} rows)")


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
