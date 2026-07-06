"""
Plateau / convergence analysis for MALA-refined DM ensembles (3D phi^4).

Reads traces.npz written by 3Dphi4/sample_phi4_crossL_mala.py, which holds
per-step, per-chain traces of the exact action S and magnetization M for
N independent MALA chains initialized from DM configurations.

At each chain step the ensemble over the N chains gives
    <S>/V,  <|M|>,  chi = V (<|M|^2> - <|M|>^2),  U_4 = 1 - <M^4>/(3 <M^2>^2)
(the same Z_2-even definitions as the paper tables; Z_2 doubling leaves
|M|, M^2, M^4 invariant, so raw chain M's give identical values).

The HMC reference band is computed from the raw reference ensemble
(trainingdata/cfgs_wolff_fahmc_k=..._64^3.jld2) with bootstrap errors.

Convergence criterion (self-consistent, no circularity with HMC): the last
two log-spaced windows must agree within 1 sigma for all four observables.

Outputs (saved next to this script by default):
    plateau_{tag}.png          4-panel observable-vs-step figure
    plateau_{tag}.csv          observable values at log-spaced steps
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def action_per_cfg_np(cfgs, k, lam):
    """Exact phi^4 action per config; cfgs (N, L, L, L) physical."""
    nb = (np.roll(cfgs, 1, axis=1) + np.roll(cfgs, 1, axis=2)
          + np.roll(cfgs, 1, axis=3))
    return (-2.0 * k * cfgs * nb + (1 - 2 * lam) * cfgs ** 2
            + lam * cfgs ** 4).sum(axis=(1, 2, 3))


def obs_from_M(M, V):
    absM = np.abs(M)
    m2, m4 = (M ** 2).mean(), (M ** 4).mean()
    chi = V * (np.mean(absM ** 2) - np.mean(absM) ** 2)
    U4 = 1.0 - m4 / (3.0 * m2 ** 2)
    return absM.mean(), chi, U4


def boot_obs(M, S, V, n_boot=400, seed=0):
    """Bootstrap over chains/configs: means and errors of the 4 observables."""
    rng = np.random.default_rng(seed)
    N = M.shape[0]
    vals = np.empty((n_boot, 4))
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        am, chi, u4 = obs_from_M(M[idx], V)
        vals[b] = (am, chi, u4, S[idx].mean() / V)
    am, chi, u4 = obs_from_M(M, V)
    cen = np.array([am, chi, u4, S.mean() / V])
    return cen, vals.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, required=True,
                    help="Path to traces.npz from sample_phi4_crossL_mala.py")
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--hmc_jld2", type=str, default=None,
                    help="HMC reference (default: trainingdata path from --k)")
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--n_points", type=int, default=30,
                    help="log-spaced evaluation steps")
    ap.add_argument("--out_dir", type=str,
                    default=str(Path(__file__).resolve().parent))
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    V = args.L ** 3
    tr = np.load(args.traces)
    S_tr = tr["S_trace"]           # (n_steps, N)
    M_tr = tr["M_trace"]           # (n_steps, N)
    acc = tr["acc_trace"]          # (n_steps,)
    n_steps, N = S_tr.shape
    tag = args.tag or Path(args.traces).parent.name
    print(f"traces: {args.traces}  n_steps={n_steps}  N={N}  "
          f"overall acc={acc.mean():.3f}")

    # ---- HMC reference ----
    hmc_path = args.hmc_jld2 or (
        f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.lam}_{args.L}^3.jld2")
    hmc_cen = hmc_err = None
    if Path(hmc_path).exists():
        with h5py.File(hmc_path, "r") as f:
            hmc = np.array(f["cfgs"]).astype(np.float64)
        if hmc.shape[0] == args.L:      # stored (L,L,L,N) -> (N,L,L,L)
            hmc = hmc.transpose(3, 0, 1, 2)
        M_h = hmc.mean(axis=(1, 2, 3))
        S_h = action_per_cfg_np(hmc, args.k, args.lam)
        hmc_cen, hmc_err = boot_obs(M_h, S_h, V, args.n_boot, seed=1)
        print(f"HMC ref ({hmc.shape[0]} cfgs): <|M|>={hmc_cen[0]:.5f}({hmc_err[0]:.5f})  "
              f"chi={hmc_cen[1]:.4g}({hmc_err[1]:.2g})  U4={hmc_cen[2]:.4f}  "
              f"<S>/V={hmc_cen[3]:+.6f}")
    else:
        print(f"[warn] no HMC reference at {hmc_path}")

    # ---- observables along the chain ----
    steps = np.unique(np.geomspace(1, n_steps, args.n_points).astype(int))
    cen = np.empty((len(steps), 4))
    err = np.empty((len(steps), 4))
    for i, s in enumerate(steps):
        cen[i], err[i] = boot_obs(M_tr[s - 1], S_tr[s - 1], V,
                                  args.n_boot, seed=2 + i)

    # step-0 (pure DM) reference from the first row
    cen0, err0 = boot_obs(M_tr[0], S_tr[0], V, args.n_boot, seed=99)

    # ---- convergence check: last two points agree within 1 sigma ----
    ok = np.abs(cen[-1] - cen[-2]) <= np.sqrt(err[-1] ** 2 + err[-2] ** 2)
    names = ["<|M|>", "chi", "U4", "<S>/V"]
    print("\nconvergence (last two log-spaced points, 1-sigma):")
    for j, nm in enumerate(names):
        print(f"  {nm:>6}: {cen[-2][j]:.6g} -> {cen[-1][j]:.6g}   "
              f"{'OK' if ok[j] else 'NOT CONVERGED'}")

    # ---- csv ----
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"plateau_{tag}.csv"
    with open(csv, "w") as f:
        f.write("step,absM,absM_err,chi,chi_err,U4,U4_err,SV,SV_err,acc_recent\n")
        for i, s in enumerate(steps):
            a = acc[max(0, s - 50):s].mean()
            f.write(f"{s},{cen[i,0]:.8g},{err[i,0]:.3g},{cen[i,1]:.8g},"
                    f"{err[i,1]:.3g},{cen[i,2]:.8g},{err[i,2]:.3g},"
                    f"{cen[i,3]:.8g},{err[i,3]:.3g},{a:.4f}\n")
    print(f"saved {csv}")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, (ax, nm) in enumerate(zip(axes.flat, names)):
        ax.errorbar(steps, cen[:, j], yerr=err[:, j], fmt="o-", ms=3,
                    lw=1, label="DM + MALA")
        ax.axhline(cen0[j], color="C2", ls=":", lw=1, label="DM (step 0)")
        if hmc_cen is not None:
            ax.axhline(hmc_cen[j], color="C1", lw=1, label="HMC")
            ax.axhspan(hmc_cen[j] - hmc_err[j], hmc_cen[j] + hmc_err[j],
                       color="C1", alpha=0.2)
        ax.set_xscale("log")
        ax.set_ylabel(nm)
        if j == 0:
            ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("MALA step")
    fig.suptitle(f"MALA plateau  k={args.k} l={args.lam} L={args.L}  "
                 f"({tag})  acc={acc.mean():.3f}")
    fig.tight_layout()
    png = out_dir / f"plateau_{tag}.png"
    fig.savefig(png, dpi=140)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
