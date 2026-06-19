"""
Export acceptance-vs-epoch .npz files to whitespace-delimited .dat files
for use in Mathematica / gnuplot.

Per-kappa output (3 columns):
    # k=<kappa>  L=<L>  sigma_ve=<sigma_ve>  c=<c>  t_mh=<t_mh>
    # epoch    acc    sem
    1          0.013  0.002
    2          0.015  0.003
    ...

Also writes a combined file with all kappas stacked (4 columns: kappa, epoch,
acc, sem) for convenience.

Usage (CPU only, run from any dir):
    python3 export_acceptance_vs_epoch_dat.py \
        --inputs path/to/acceptance_vs_epoch_*.npz \
        --out_dir path/to/dat_out

Mathematica:
    data = Import["acceptance_vs_epoch_L128_k0.26_sigma300_ema.dat", "Table"];
    ListLogLinearPlot[data[[All, {1, 2}]]]
"""

import argparse
from pathlib import Path

import numpy as np


def export_one(npz_path, out_dir):
    d = dict(np.load(npz_path, allow_pickle=True))
    epochs = np.asarray(d["epochs"]).astype(int)
    acc = np.asarray(d["acc"]).astype(float)
    sem = np.asarray(d["sem"]).astype(float)
    k = float(d["k"])
    L = int(d["L"])
    sigma_ve = float(d["sigma_ve"])
    c = float(d["c"])
    t_mh = float(d["t_mh"])

    out_path = Path(out_dir) / (Path(npz_path).stem + ".dat")
    header = (
        f"k={k}  L={L}  sigma_ve={sigma_ve}  c={c}  t_mh={t_mh}\n"
        f"epoch\tacc\tsem"
    )
    arr = np.column_stack([epochs, acc, sem])
    np.savetxt(out_path, arr, fmt=("%d", "%.8e", "%.8e"),
               delimiter="\t", header=header)
    print(f"  wrote {out_path}  ({len(epochs)} rows)")
    return out_path, (k, L, sigma_ve, c, t_mh, epochs, acc, sem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Paths to acceptance_vs_epoch_*.npz files.")
    p.add_argument("--out_dir", required=True,
                   help="Directory to write .dat files into.")
    p.add_argument("--combined_name", type=str, default=None,
                   help="Optional: basename for the stacked-kappa .dat "
                        "(4 columns: kappa, epoch, acc, sem). Skip to omit.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for npz in args.inputs:
        _, info = export_one(npz, out_dir)
        k, L, sigma_ve, c, t_mh, epochs, acc, sem = info
        for ep, a, s in zip(epochs, acc, sem):
            rows.append((k, int(ep), float(a), float(s)))

    if args.combined_name:
        rows.sort()
        rows_arr = np.array(rows)
        combined_path = out_dir / args.combined_name
        header = (
            "Stacked acceptance-vs-epoch table\n"
            "kappa\tepoch\tacc\tsem"
        )
        np.savetxt(combined_path, rows_arr,
                   fmt=("%g", "%d", "%.8e", "%.8e"),
                   delimiter="\t", header=header)
        print(f"  wrote {combined_path}  ({len(rows_arr)} rows)")


if __name__ == "__main__":
    main()
