"""
Export dm_mc_tau_vs_epoch .npz files to whitespace-delimited .dat files
for Mathematica / gnuplot.

Per-kappa output (9 columns), one row per epoch (L is single-valued in our
in-distribution sweeps so we collapse the L axis):

    # k=<kappa>  L=<L>  sigma_ve=<sigma_ve>  c=<c>  t_mh=<t_mh>  M=<num_chains>  N=<mh_steps>
    # epoch  tau_M  tau_M_se  tau_S  tau_S_se  ess_M  ess_S  ess_eff  accept
    1   ...

Usage (CPU only):
    python3 export_dm_mc_tau_dat.py \
        --inputs path/to/dm_mc_tau_vs_epoch_*.npz \
        --out_dir path/to/dat_out
"""

import argparse
from pathlib import Path

import numpy as np


def export_one(npz_path, out_dir):
    d = dict(np.load(npz_path, allow_pickle=True))
    epochs = np.asarray(d["epochs"]).astype(int)
    # All arrays are (n_epochs, n_L); collapse the single-L axis.
    def col(name):
        a = np.asarray(d[name])
        return a[:, 0] if a.ndim == 2 else a
    tau_M  = col("tau_sokal")
    tau_Ms = col("tau_se")
    tau_S  = col("tau_S")
    tau_Ss = col("tau_S_se")
    ess_M  = col("ess_M")
    ess_S  = col("ess_S")
    ess_e  = col("ess_eff")
    acc    = col("accept")

    k = float(d["k"]); L = int(np.asarray(d["L_list"])[0])
    sigma_ve = float(d["sigma_ve"])
    c = float(d["c"]); t_mh = float(d["t_mh"])
    M = int(d["num_chains"]); N = int(d["mh_steps"])

    out_path = Path(out_dir) / (Path(npz_path).stem + ".dat")
    header = (
        f"k={k}  L={L}  sigma_ve={sigma_ve}  c={c}  t_mh={t_mh}  M={M}  N={N}\n"
        f"epoch\ttau_M\ttau_M_se\ttau_S\ttau_S_se\tess_M\tess_S\tess_eff\taccept"
    )
    arr = np.column_stack([epochs, tau_M, tau_Ms, tau_S, tau_Ss, ess_M, ess_S, ess_e, acc])
    fmt = ("%d",) + ("%.8e",) * 8
    np.savetxt(out_path, arr, fmt=fmt, delimiter="\t", header=header)
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
