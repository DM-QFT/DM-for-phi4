"""
Apply the paper's two support filters to a DM cross-L ensemble and save the
survivors, so a MALA chain can be initialized from the filtered configs.

Filters (identical to compare_L64_multiL_HMC_cumulants.py, i.e. the paper
DM rows):
    (a) site support:   drop any cfg with a site outside the HMC field
                        range [lo, hi].
    (b) magnetisation:  drop any cfg with |M|=|<phi>_V| beyond the HMC max.

Also reports zero-mode and single-site observables for the UNFILTERED and
FILTERED ensembles vs HMC, so the effect of filtering alone (before any
MALA) is visible.

Run from 3Dphi4/:
    python analysis_clean/mala/filter_init.py --k 0.2 --n_keep 128
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


def report(name, cfgs_NLLL, k, lam, V, n_boot, seed):
    """cfgs_NLLL: (N,L,L,L) raw physical. Z2-double, print zero-mode + site."""
    z2 = np.concatenate([cfgs_NLLL, -cfgs_NLLL], axis=0)
    M = np.concatenate([cfgs_NLLL.mean(axis=(1, 2, 3)),
                        -cfgs_NLLL.mean(axis=(1, 2, 3))])
    mean, err = cmp_mod.bootstrap_stats(M, V, n_boot=n_boot, seed=seed, n_raw=None)
    S = cmp_mod.action_per_cfg(z2, k=k, l=lam)
    sv_m, sv_e = cmp_mod.bootstrap_S_per_V(S, V, n_boot=n_boot, seed=seed, n_raw=None)
    ms, es = cmp_mod.bootstrap_stats_per_site(z2, n_boot=n_boot, seed=seed,
                                              n_raw=None, method="persite_avg")
    print(f"  {name:>22}  N={cfgs_NLLL.shape[0]:>4}  "
          f"<|M|>={mean['abs_M']:.5f}({err['abs_M']:.5f})  "
          f"chi={mean['chi']:.3f}({err['chi']:.3f})  "
          f"<S>/V={sv_m:+.6f}({sv_e:.6f})  "
          f"U4={mean['U4']:.5f}  "
          f"k2s={ms['k2']:+.5f}  k8s={ms['k8']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--n_keep", type=int, default=128,
                    help="Number of filtered survivors to save (first n).")
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    L, V, k = 64, 64 ** 3, args.k
    sigma_tag = ""
    if abs(k - 0.2) < 1e-9:  sigma_tag = "_sigma400"
    if abs(k - 0.18) < 1e-9: sigma_tag = "_sigma200"
    run_dir = Path(f"runs/phi4_3d_Lmulti4-8-16-32_k{k:g}_l{args.lam}_ncsnpp{sigma_tag}")
    init_file = (run_dir / "data_crossL" /
                 "samples_crossL_train32_sample64_em_linear_steps2000_ep10000.npy")

    # HMC support thresholds (paper convention).
    hmc = cmp_mod.load_hmc_3d(cmp_mod.hmc_path(k, args.lam, L))
    try:
        lo, hi = cmp_mod.hmc_norm_range(k, args.lam, L)
    except FileNotFoundError:
        lo, hi = float(hmc.min()), float(hmc.max())  # == cfgs_min/max
    max_abs_M = float(np.abs(hmc.mean(axis=(1, 2, 3))).max())
    print(f"kappa={k}: HMC field range [{lo:.4f},{hi:.4f}]  "
          f"max|M|_HMC={max_abs_M:.5f}  (N_HMC={hmc.shape[0]})")

    dm = cmp_mod.load_dm_3d(init_file)           # (N,L,L,L)
    filt, n_drop, n_drop_amp = cmp_mod.filter_oor(dm, lo, hi, max_abs_M=max_abs_M)
    n_drop_M = n_drop - n_drop_amp
    print(f"DM init: {dm.shape[0]} cfgs -> {filt.shape[0]} kept "
          f"(dropped {n_drop}: {n_drop_amp} site-OOR + {n_drop_M} by |M|)")

    keep = filt[:args.n_keep]
    print(f"\nObservables (Z2-doubled, n_boot={args.n_boot}):")
    report("HMC (full)", hmc, k, args.lam, V, args.n_boot, args.seed)
    report("DM unfiltered (512)", dm, k, args.lam, V, args.n_boot, args.seed)
    report("DM filtered (all)", filt, k, args.lam, V, args.n_boot, args.seed)
    report(f"DM filtered (first{args.n_keep})", keep, k, args.lam, V,
           args.n_boot, args.seed)

    out = Path(args.out) if args.out else (
        run_dir / "data_crossL" /
        f"samples_filtered{args.n_keep}_k{k:g}_ep10000.npy")
    np.save(out, keep.transpose(1, 2, 3, 0).astype(np.float32))  # (L,L,L,n)
    print(f"\nsaved filtered init: {out}  shape={(L,L,L,keep.shape[0])}")


if __name__ == "__main__":
    main()
