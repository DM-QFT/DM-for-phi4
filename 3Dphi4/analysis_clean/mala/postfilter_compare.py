"""
Post-filter comparison for MALA-refined ensembles.

Loads final.npy of one or more MALA tags, applies the paper's two support
filters (site range + max|M| from the HMC reference) to the OUTPUT, and
reports zero-mode + site observables before and after the post-filter,
against HMC.

Run from 3Dphi4/:
    python analysis_clean/mala/postfilter_compare.py --k 0.2 \
        --tags f64_shift f64_noshift
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CMP = HERE.parent / "physical observable" / "compare_L64_multiL_HMC_cumulants.py"
spec = importlib.util.spec_from_file_location("cmp_pipeline", CMP)
cmp_mod = importlib.util.module_from_spec(spec)
sys.modules["cmp_pipeline"] = cmp_mod
spec.loader.exec_module(cmp_mod)


def report(name, cfgs, k, lam, V, n_boot, seed):
    z2 = np.concatenate([cfgs, -cfgs], axis=0)
    M = np.concatenate([cfgs.mean(axis=(1, 2, 3)), -cfgs.mean(axis=(1, 2, 3))])
    mean, err = cmp_mod.bootstrap_stats(M, V, n_boot=n_boot, seed=seed, n_raw=None)
    S = cmp_mod.action_per_cfg(z2, k=k, l=lam)
    sv_m, sv_e = cmp_mod.bootstrap_S_per_V(S, V, n_boot=n_boot, seed=seed, n_raw=None)
    ms, _ = cmp_mod.bootstrap_stats_per_site(z2, n_boot=50, seed=seed,
                                             n_raw=None, method="persite_avg")
    print(f"  {name:>26}  N={cfgs.shape[0]:>3}  "
          f"<|M|>={mean['abs_M']:.5f}({err['abs_M']:.5f})  "
          f"chi={mean['chi']:.3f}({err['chi']:.3f})  "
          f"<S>/V={sv_m:+.6f}({sv_e:.6f})  U4={mean['U4']:.5f}  "
          f"k2s={ms['k2']:+.5f}  k8s={ms['k8']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--tags", type=str, nargs="+", required=True)
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    L, V, k = 64, 64 ** 3, args.k
    sigma_tag = ""
    if abs(k - 0.2) < 1e-9:  sigma_tag = "_sigma400"
    if abs(k - 0.18) < 1e-9: sigma_tag = "_sigma200"
    run_dir = Path(f"runs/phi4_3d_Lmulti4-8-16-32_k{k:g}_l{args.lam}_ncsnpp{sigma_tag}")

    hmc = cmp_mod.load_hmc_3d(cmp_mod.hmc_path(k, args.lam, L))
    lo, hi = float(hmc.min()), float(hmc.max())
    max_abs_M = float(np.abs(hmc.mean(axis=(1, 2, 3))).max())
    print(f"kappa={k}: filter thresholds range=[{lo:.4f},{hi:.4f}] "
          f"max|M|={max_abs_M:.5f}")
    report("HMC (full)", hmc, k, args.lam, V, args.n_boot, args.seed)

    for tag in args.tags:
        f = run_dir / "data_crossL_mala" / tag / "final.npy"
        cfgs = cmp_mod.to_NLLL(np.load(f)).astype(np.float64)
        print(f"\n--- {tag} ---")
        report("raw output", cfgs, k, args.lam, V, args.n_boot, args.seed)
        filt, n_drop, n_amp = cmp_mod.filter_oor(cfgs, lo, hi, max_abs_M=max_abs_M)
        print(f"  post-filter: dropped {n_drop} "
              f"({n_amp} site-OOR + {n_drop - n_amp} |M|)")
        if filt.shape[0] > 2:
            report("post-filtered", filt, k, args.lam, V, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
