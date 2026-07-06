"""
Merge worker NPZ files from calibrate_tau_vs_c.py and plot tau_int(|M|)
and acceptance rate as a function of c.

Usage:
  python analysis/plot_tau_vs_c.py \
      --inputs results/calib_tau_c_part_a.npz results/calib_tau_c_part_b.npz \
      --out results/tau_vs_c_L32_k0.28_e7564.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, nargs="+", required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cs, taus, taus_se, accepts, taus_cut, Ws, rhos = [], [], [], [], [], [], []
    meta = None
    for path in args.inputs:
        d = np.load(path)
        cs.append(d["c_values"])
        taus.append(d["tau_sokal"])
        taus_se.append(d["tau_se"])
        accepts.append(d["accept"])
        taus_cut.append(d["tau_cutoff"])
        Ws.append(d["sokal_W"])
        rhos.append(d["rho"])
        if meta is None:
            meta = dict(L=int(d["L"]), k=float(d["k"]), l=float(d["l"]),
                        t_mh=float(d["t_mh"]), epoch=int(d["epoch"]),
                        num_chains=int(d["num_chains"]),
                        mh_steps=int(d["mh_steps"]))

    c_all = np.concatenate(cs)
    order = np.argsort(c_all)
    c_all = c_all[order]
    tau_all = np.concatenate(taus)[order]
    tau_se_all = np.concatenate(taus_se)[order]
    accept_all = np.concatenate(accepts)[order]
    tau_cut_all = np.concatenate(taus_cut)[order]
    W_all = np.concatenate(Ws)[order]
    # rho needs special concat: align lengths if any worker used a different N.
    N_max = max(r.shape[1] for r in rhos)
    rho_padded = []
    for r in rhos:
        if r.shape[1] < N_max:
            pad = np.full((r.shape[0], N_max - r.shape[1]), np.nan)
            rho_padded.append(np.concatenate([r, pad], axis=1))
        else:
            rho_padded.append(r)
    rho_all = np.concatenate(rho_padded, axis=0)[order]

    print(f"{'c':>8s}  {'h=c/L^2':>10s}  {'accept':>7s}  {'tau_sokal':>10s} "
          f"{'+/-se':>6s}  {'W':>5s}  {'tau_cut':>8s}")
    print("-" * 70)
    for i in range(len(c_all)):
        print(f"{c_all[i]:8.3f}  {c_all[i]/meta['L']**2:10.3e}  "
              f"{accept_all[i]:7.3f}  {tau_all[i]:10.3f}  "
              f"{tau_se_all[i]:6.2f}  {int(W_all[i]):>5d}  "
              f"{tau_cut_all[i]:8.3f}")

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.6, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_tau, ax_acc, ax_ess = axes

    # tau_int vs c
    ax_tau.errorbar(c_all, tau_all, yerr=tau_se_all,
                    marker="o", ms=6, capsize=3, color="C0",
                    label="Sokal (auto window)")
    ax_tau.plot(c_all, tau_cut_all, "s--", ms=4, color="C3", alpha=0.8,
                label=r"cutoff $\rho<0.05$")
    ax_tau.set_xscale("log")
    ax_tau.set_yscale("log")
    ax_tau.set_xlabel(r"$c$  ($h = c/L^2$)")
    ax_tau.set_ylabel(r"$\tau_{\mathrm{int}}(|M|)$  [MH steps]")
    ax_tau.grid(alpha=0.3, which="both")
    ax_tau.legend()

    # acceptance rate vs c
    ax_acc.plot(c_all, accept_all, "o-", color="C2", ms=6)
    ax_acc.axhline(0.574, ls=":", color="k", lw=1,
                   label="Roberts–Rosenthal 0.574")
    ax_acc.axhspan(0.3, 0.7, color="C2", alpha=0.08, label="useful zone")
    ax_acc.set_xscale("log")
    ax_acc.set_xlabel(r"$c$")
    ax_acc.set_ylabel("acceptance rate")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.grid(alpha=0.3, which="both")
    ax_acc.legend(loc="lower left")

    # ESS / step = 1/(2 tau_int)  -> we want max
    ess_per_step = 1.0 / (2.0 * tau_all)
    ax_ess.plot(c_all, ess_per_step, "o-", color="C4", ms=6)
    best = int(np.nanargmax(ess_per_step))
    ax_ess.axvline(c_all[best], ls="--", color="gray", alpha=0.7,
                   label=rf"best $c={c_all[best]:g}$")
    ax_ess.set_xscale("log")
    ax_ess.set_yscale("log")
    ax_ess.set_xlabel(r"$c$")
    ax_ess.set_ylabel(r"ESS / step  $= 1/(2\,\tau_{\mathrm{int}})$")
    ax_ess.grid(alpha=0.3, which="both")
    ax_ess.legend()

    fig.suptitle(
        rf"MALA step-size sweep: $L={meta['L']},\;\kappa={meta['k']},\;"
        rf"\lambda={meta['l']:g}$,  ckpt ep {meta['epoch']},  "
        rf"$M={meta['num_chains']}$ chains $\times\;N={meta['mh_steps']}$ MH steps,  "
        rf"$t_{{\mathrm{{mh}}}}={meta['t_mh']:g}$,  observable $|M|$",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved figure to {out}")

    np.savez(out.with_suffix(".npz"),
             c=c_all, tau_sokal=tau_all, tau_se=tau_se_all,
             tau_cutoff=tau_cut_all, accept=accept_all, sokal_W=W_all,
             ess_per_step=ess_per_step, best_c=c_all[best],
             rho=rho_all, **meta)
    print(f"Saved merged data to {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
