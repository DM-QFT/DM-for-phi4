"""Combined forward/reverse β_conf(t) figure from all per-L npz files."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
out_dir = HERE.parent / "results" / "beta_conf_traj_multiL"
tag = "k0.28_l0.022_ep10000_steps2000_log"

data = {}
for f in sorted(out_dir.glob(f"beta_conf_fwd_rev_L*_{tag}.npz")):
    d = np.load(f)
    data[int(d["L"])] = d
Ls = sorted(data)
print("found L =", Ls)

colors = {L: c for L, c in zip(Ls, ["C0", "C1", "C2", "C3", "C4", "C5"])}
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

ax = axes[0]
ax.set_xscale("log")
for L in Ls:
    d = data[L]
    ax.plot(d["t"], d["dm_mean"], "-", color=colors[L], lw=1.5,
            label=f"L={L} DM reverse")
    ax.plot(d["t"], d["hmc_mean"], "--", color=colors[L], lw=1.2,
            label=f"L={L} HMC+noise")
ax.axhline(1.0, ls=":", color="k", lw=0.8)
ax.set_xlabel("t")
ax.set_ylabel(r"$\langle\hat\beta(x_t)\rangle$")
ax.set_title("full trajectory")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
ax.set_xscale("log")
for L in Ls:
    d = data[L]
    m = d["t"] < 1e-2
    ax.errorbar(d["t"][m], d["dm_mean"][m], yerr=d["dm_sem"][m], fmt="o-",
                ms=2, lw=1.0, color=colors[L], label=f"L={L} DM", alpha=0.9)
    ax.errorbar(d["t"][m], d["hmc_mean"][m], yerr=d["hmc_sem"][m],
                fmt="s--", ms=2, lw=0.8, color=colors[L], alpha=0.5)
ax.axhline(1.0, ls=":", color="k", lw=0.8)
ax.set_ylim(0.85, 1.1)
ax.set_xlabel("t")
ax.set_title("late stage (t < 1e-2); dashed = HMC+noise")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

ax = axes[2]
ax.set_xscale("log")
for L in Ls:
    d = data[L]
    diff = d["dm_mean"] - d["hmc_mean"]
    err = np.hypot(d["dm_sem"], d["hmc_sem"])
    ax.errorbar(d["sigma_t"], diff, yerr=err, fmt="o-", ms=2, lw=1.0,
                color=colors[L], label=f"L={L}", alpha=0.85)
ax.axhline(0.0, ls=":", color="k", lw=0.8)
ax.set_xlabel(r"$\sigma_t$")
ax.set_ylabel(r"$\langle\hat\beta\rangle_{\rm DM} - \langle\hat\beta\rangle_{\rm HMC+noise}$")
ax.set_title("reverse − forward (model error)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

fig.suptitle(f"multi-L run, L = {Ls}   κ=0.28  λ=0.022  ep=10000  σ=480")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(out_dir / f"beta_conf_fwd_rev_multiL_all_{tag}.{ext}", dpi=160)
print(f"saved → {out_dir}/beta_conf_fwd_rev_multiL_all_{tag}.pdf")
