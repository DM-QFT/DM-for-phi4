"""
Export ess_mse_vs_epoch_*.npz to whitespace-delimited .dat files for
Mathematica / gnuplot.  Only the conservative N_eff,ref efficiency ratio
is exported (the paper reports only that curve and notes in prose that
the pure-shrinkage N_eff differs from it by at most the HMC reference
variance).

Per-input, per-observable output (4 columns), one row per epoch:

    # k=<k>  L=<L>  N_DM=<K>  N_HMC=<N_HMC>  obs=<O>  tau_hmc=<tau>
    # epoch  eff_ref  eff_ref_p16  eff_ref_p84
    <values>

When more than one input npz is given, also write per-observable
stacked-kappa files (5 columns: kappa, epoch, eff_ref, p16, p84) under
the basename "combined__<obs>.dat".

Usage (CPU only, run from any dir):
    python3 analysis/export_ess_mse_dat.py \
        --inputs analysis/results/ess_mse_vs_epoch_L128_k*.npz \
        --out_dir analysis/dat

Mathematica:
    data = Import["ess_mse_vs_epoch_L128_k0.2705_sigma450_ema__absMbar.dat",
                  "Table"];
    eps    = data[[All, 1]];
    eff    = data[[All, 2]];
    errLo  = data[[All, 2]] - data[[All, 3]];
    errHi  = data[[All, 4]] - data[[All, 2]];
    ListLogLogPlot[
        {Transpose@{eps, eff}},
        Joined -> True, PlotMarkers -> Automatic,
        IntervalMarkers -> "Bars",
        AxesLabel -> {"epoch", "N_eff,ref / N_DM"}]
"""

import argparse
from pathlib import Path

import numpy as np


OBS_NAMES = ("absMbar", "Mbar2", "Mbar4", "SoverV",
             "Gk_min", "Gk_mid", "Gk_large")


def export_one(npz_path, out_dir):
    """Write per-observable .dat files for a single npz.

    Returns dict {obs_name: (k, rows)} for the combined-file writer.
    """
    d = dict(np.load(npz_path, allow_pickle=True))
    epochs = np.asarray(d["epochs"]).astype(int)
    k = float(d["k"]);   L = int(d["L"]);   n_hmc = int(d["n_hmc"])

    base = Path(npz_path).stem
    per_obs_rows = {}
    for obs in OBS_NAMES:
        try:
            eff      = np.asarray(d[f"{obs}__eff_ref"], dtype=float)
            eff_p16  = np.asarray(d[f"{obs}__eff_ref_p16"], dtype=float)
            eff_p84  = np.asarray(d[f"{obs}__eff_ref_p84"], dtype=float)
            tau      = float(np.asarray(d[f"{obs}__tau_hmc"])[0])
            K_dm     = int(np.asarray(d[f"{obs}__N_DM"])[0])
        except KeyError as e:
            print(f"  [skip] {base}: missing {e!s}")
            continue

        out_path = Path(out_dir) / f"{base}__{obs}.dat"
        header = (
            f"k={k}  L={L}  N_DM={K_dm}  N_HMC={n_hmc}  "
            f"obs={obs}  tau_hmc={tau:.4g}\n"
            "epoch\teff_ref\teff_ref_p16\teff_ref_p84"
        )
        arr = np.column_stack([epochs, eff, eff_p16, eff_p84])
        np.savetxt(out_path, arr,
                   fmt=("%d", "%.8e", "%.8e", "%.8e"),
                   delimiter="\t", header=header)
        print(f"  wrote {out_path}  ({len(epochs)} rows)")

        per_obs_rows[obs] = (k, list(zip(epochs.tolist(),
                                         eff.tolist(),
                                         eff_p16.tolist(),
                                         eff_p84.tolist())))
    return per_obs_rows


def write_combined(per_obs_collected, out_dir):
    """One stacked-kappa file per observable: kappa epoch eff_ref p16 p84."""
    out_dir = Path(out_dir)
    for obs in OBS_NAMES:
        rows = []
        for k, pairs in per_obs_collected.get(obs, []):
            for ep, eff, p16, p84 in pairs:
                rows.append((k, int(ep), float(eff), float(p16), float(p84)))
        if not rows:
            continue
        rows.sort()
        arr = np.array(rows)
        out_path = out_dir / f"combined__{obs}.dat"
        header = (
            f"Stacked-kappa N_eff,ref / N_DM table  (obs={obs})\n"
            "kappa\tepoch\teff_ref\teff_ref_p16\teff_ref_p84"
        )
        np.savetxt(out_path, arr,
                   fmt=("%g", "%d", "%.8e", "%.8e", "%.8e"),
                   delimiter="\t", header=header)
        print(f"  wrote {out_path}  ({len(arr)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Paths to ess_mse_vs_epoch_*.npz files.")
    p.add_argument("--out_dir", required=True,
                   help="Directory to write .dat files into.")
    p.add_argument("--no_combined", action="store_true",
                   help="Skip writing per-observable stacked-kappa files "
                        "even when more than one input is given.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_obs_collected = {obs: [] for obs in OBS_NAMES}
    for npz in args.inputs:
        rows_by_obs = export_one(npz, out_dir)
        for obs, (k, pairs) in rows_by_obs.items():
            per_obs_collected[obs].append((k, pairs))

    if len(args.inputs) > 1 and not args.no_combined:
        write_combined(per_obs_collected, out_dir)


if __name__ == "__main__":
    main()
