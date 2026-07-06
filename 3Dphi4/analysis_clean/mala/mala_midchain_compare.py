"""
Mid-chain observable comparison for a running/finished MALA refinement.

For a given kappa, compares at a requested MALA step (default 500):
    HMC (full)  |  HMC (N=512 subsample)  |  DM init (step 0)  |  DM+MALA

Zero-mode block (<|M|>, chi, <S>/V, U4, kappa_2..kappa_8) is evaluated at
EXACTLY the requested step from the per-step traces. Single-site cumulants
need full configurations, so they are evaluated on the nearest saved
snapshot <= the requested step.

All statistics reuse the functions of the trusted table pipeline
(analysis_clean/physical observable/compare_L64_multiL_HMC_cumulants.py)
via importlib, so conventions (Z2 doubling, bootstrap, cumulant recursion)
are identical to the paper tables.

Run from 3Dphi4/:
    python analysis_clean/mala/mala_midchain_compare.py --k 0.2 --step 500
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


def z2(a):
    return np.concatenate([a, -a], axis=0)


def report_M_block(name, M, S, V, n_boot, seed):
    """M: raw per-cfg magnetisation (N,); S: per-cfg action (N,) or None."""
    Mz = np.concatenate([M, -M])
    mean, err = cmp_mod.bootstrap_stats(Mz, V, n_boot=n_boot, seed=seed,
                                        n_raw=None)
    if S is not None:
        sv_m, sv_e = cmp_mod.bootstrap_S_per_V(np.concatenate([S, S]), V,
                                               n_boot=n_boot, seed=seed,
                                               n_raw=None)
        mean["S_V"], err["S_V"] = sv_m, sv_e
    return name, mean, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--step", type=int, default=500)
    ap.add_argument("--tag", type=str, default="prod_c3")
    ap.add_argument("--init_npy", type=str, default=None,
                    help="Override the DM_init file (e.g. a filtered ensemble). "
                         "Default: the standard 512-cfg cross-L init.")
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per_site", action="store_true", default=True)
    args = ap.parse_args()

    L, V = 64, 64 ** 3
    k = args.k
    sigma_tag = ""
    if abs(k - 0.2) < 1e-9:  sigma_tag = "_sigma400"
    if abs(k - 0.18) < 1e-9: sigma_tag = "_sigma200"
    run_dir = Path(f"runs/phi4_3d_Lmulti4-8-16-32_k{k:g}_l{args.lam}_ncsnpp{sigma_tag}")
    mala_dir = run_dir / "data_crossL_mala" / args.tag

    # ---- traces at exactly the requested step ----
    tr = np.load(mala_dir / "traces.npz")
    S_tr, M_tr = tr["S_trace"], tr["M_trace"]
    n_done = S_tr.shape[0]
    step = min(args.step, n_done)
    print(f"traces: {n_done} steps done; zero-mode block at step {step}")

    # ---- nearest snapshot <= step for site cumulants ----
    snaps = sorted(mala_dir.glob("snap_step*.npy"),
                   key=lambda p: int(p.stem.split("step")[1]))
    snap = None
    for p in snaps:
        s = int(p.stem.split("step")[1])
        if s <= args.step:
            snap = (s, p)
    if snap:
        print(f"site-cumulant block from snapshot at step {snap[0]}")

    # ---- sources ----
    hmc_file = f"trainingdata/cfgs_wolff_fahmc_k={k}_l={args.lam}_{L}^3.jld2"
    hmc = cmp_mod.load_hmc_3d(Path(hmc_file))
    rng_sub = np.random.default_rng(args.seed + 1000)
    idx = rng_sub.choice(hmc.shape[0], 512, replace=False)
    hmc_sub = hmc[idx]

    init_file = Path(args.init_npy) if args.init_npy else (
        run_dir / "data_crossL" /
        "samples_crossL_train32_sample64_em_linear_steps2000_ep10000.npy")
    dm0 = cmp_mod.load_dm_3d(init_file)

    M_rows = [
        report_M_block("HMC", hmc.mean(axis=(1, 2, 3)),
                       cmp_mod.action_per_cfg(hmc, k=k, l=args.lam),
                       V, args.n_boot, args.seed),
        report_M_block("HMC_N512", hmc_sub.mean(axis=(1, 2, 3)),
                       cmp_mod.action_per_cfg(hmc_sub, k=k, l=args.lam),
                       V, args.n_boot, args.seed),
        report_M_block("DM_init", dm0.mean(axis=(1, 2, 3)),
                       cmp_mod.action_per_cfg(dm0, k=k, l=args.lam),
                       V, args.n_boot, args.seed),
        report_M_block(f"MALA@{step}", M_tr[step - 1].astype(np.float64),
                       S_tr[step - 1], V, args.n_boot, args.seed),
    ]

    print(f"\n--- kappa={k}  zero-mode observables (Z2-doubled, n_boot={args.n_boot}) ---")
    hdr = (f"{'source':>12}  {'<|M|>':>16}  {'chi':>16}  {'<S>/V':>18}  "
           f"{'U4':>16}  {'k2':>14}  {'k4':>14}  {'k6':>14}  {'k8':>14}")
    print(hdr)
    for name, m, e in M_rows:
        print(f"{name:>12}  "
              f"{m['abs_M']:.5f}({e['abs_M']:.5f})  "
              f"{m['chi']:.4g}({e['chi']:.2g})  "
              f"{m['S_V']:+.6f}({e['S_V']:.6f})  "
              f"{m['U4']:.5f}({e['U4']:.5f})  "
              f"{m['k2']:.4g}({e['k2']:.2g})  "
              f"{m['k4']:.4g}({e['k4']:.2g})  "
              f"{m['k6']:.4g}({e['k6']:.2g})  "
              f"{m['k8']:.4g}({e['k8']:.2g})")

    if args.per_site and snap:
        cfgs_mala = cmp_mod.to_NLLL(np.load(snap[1])).astype(np.float64)
        site_rows = [("HMC", z2(hmc)), ("HMC_N512", z2(hmc_sub)),
                     ("DM_init", z2(dm0)),
                     (f"MALA@{snap[0]}", z2(cfgs_mala))]
        print(f"\n--- single-site cumulants (persite_avg, paper convention) ---")
        print(f"{'source':>12}  {'<|phi|>':>14}  {'k2s':>12}  {'k4s':>12}  "
              f"{'k6s':>12}  {'k8s':>13}  {'U4s':>14}")
        for name, cfgs in site_rows:
            m, e = cmp_mod.bootstrap_stats_per_site(
                cfgs, n_boot=args.n_boot, seed=args.seed, n_raw=None,
                method="persite_avg")
            print(f"{name:>12}  "
                  f"{m['abs_phi']:.5f}({e['abs_phi']:.5f})  "
                  f"{m['k2']:+.5f}({e['k2']:.5f})  "
                  f"{m['k4']:+.5f}({e['k4']:.5f})  "
                  f"{m['k6']:+.4f}({e['k6']:.4f})  "
                  f"{m['k8']:+.3f}({e['k8']:.3f})  "
                  f"{m['U4_site']:+.5f}({e['U4_site']:.5f})")


if __name__ == "__main__":
    main()
