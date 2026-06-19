"""
Build a combined multi-kappa LaTeX table for the 2D multi-L cross-L
snapshot diagnostic at L=128. Reads each per-kappa run's
``diagnostic_observables_multiL_crossL_L128_n1024.json`` and the HMC
reference at L=128, emits a multi-kappa block table analogous to the
single-L summary already in the paper.
"""

import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, "/home/tyywork/DM/2Dphi4")
from analyze_diagnostic_observables import (
    block_jackknife, z2_augment, fmt_err, TEX_NAMES,
)

BASE = "/home/tyywork/DM/2Dphi4"
RUNS = [
    ("0.26",   "phi4_Lmulti8-16-32-64_k0.26_l0.022_ncsnpp",
     "trainingdata/cfgs_wolff_fahmc_k=0.26_l=0.022_128^2.jld2"),
    ("0.2705", "phi4_Lmulti8-16-32-64_k0.2705_l0.022_ncsnpp",
     "trainingdata/cfgs_wolff_fahmc_k=0.2705_l=0.022_128^2.jld2"),
    ("0.28",   "phi4_Lmulti8-16-32-64_k0.28_l0.022_ncsnpp",
     "trainingdata/cfgs_wolff_fahmc_k=0.28_l=0.022_128^2.jld2"),
]
ORDER = ["abs_phi_bar","chi","U4","abs_phi","kappa2_s","kappa4_s","kappa6_s","kappa8_s","U4_s"]


def fmt_sci(x, prec=1):
    if x == 0 or not np.isfinite(x):
        return f"{x:g}"
    if abs(x) >= 1e-2 and abs(x) < 1e3:
        return f"{x:.{prec+1}g}"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10**exp
    return f"{mant:.{prec}f}\\!\\times\\!10^{{{exp}}}"


def main():
    rows = []
    for k_str, rdir, hmc_path in RUNS:
        # DM 9-obs summary
        json_path = os.path.join(BASE, "runs", rdir,
                                 "diagnostic_observables_multiL_crossL_L128_n1024.json")
        with open(json_path) as f:
            dm = json.load(f)
        dm_summary = {s["obs"]: s for s in dm["summary"]}

        # HMC reference at L=128
        with h5py.File(os.path.join(BASE, hmc_path), "r") as fh:
            cfgs = np.array(fh["cfgs"], dtype=np.float64)
        L = cfgs.shape[1]
        x_z2 = z2_augment(cfgs.reshape(cfgs.shape[0], -1))
        hmc_vals, hmc_errs = block_jackknife(x_z2, L, n_blocks=32)

        rows.append({"k": k_str, "dm": dm_summary, "hmc": hmc_vals, "hmc_err": hmc_errs})

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-$L$ generalisation of the 2D $\phi^4$ multi-$L$ diffusion models. Training set $L\in\{8,16,32,64\}$, $\lambda=0.022$; samples generated at the held-out volume $L=128$ for three $\kappa$ values spanning the symmetric ($\kappa=0.26$), near-critical ($\kappa\approx\kappa_c=0.2705$) and broken ($\kappa=0.28$) phases. Each DM block averages over six fine-tune snapshots spanning $100$ additional epochs (every $20$ epochs), with $N=1024$ outlier-filtered, $Z_2$-symmetrised samples per snapshot. HMC values are computed on the $L=128$ reference ensemble ($N=10240$). $\sigma_{\rm snap}$ is the standard deviation across DM snapshots and $\delta_{\rm stat}$ is the mean within-snapshot block-jackknife error; $r=\sigma_{\rm snap}/\delta_{\rm stat}$. Values quoted as $value(err)$ in the last displayed digits.}")
    lines.append(r"\label{tab:multiL_crossL_summary}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"$\kappa$ & observable & DM mean & HMC & $\sigma_{\rm snap}$ & $\delta_{\rm stat}$ & $r$ \\")
    for r in rows:
        lines.append(r"\midrule")
        for j, obs in enumerate(ORDER):
            dm = r["dm"][obs]
            hv = r["hmc"][obs]; he = r["hmc_err"][obs]
            dm_str = fmt_err(dm["mean"], dm["delta_within"], sig_digits=1)
            hmc_str = fmt_err(hv, he, sig_digits=1)
            sig_str = fmt_sci(dm["sigma_between"])
            del_str = fmt_sci(dm["delta_within"])
            first = f"${r['k']}$" if j == 0 else ""
            lines.append(f"{first} & {TEX_NAMES[obs]} & ${dm_str}$ & ${hmc_str}$ & "
                         f"${sig_str}$ & ${del_str}$ & ${dm['ratio_r']:.2f}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out = "\n".join(lines).replace("^{(s)}", "^{(\\text{site})}")
    out_path = os.path.join(BASE, "runs", "_multiL_crossL_logs",
                            "multiL_crossL_summary.tex")
    with open(out_path, "w") as f:
        f.write(out + "\n")
    print(out)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
