"""C2: beta_conf sensitivity to reverse-sampler quality (steps, schedule).

Multi-L kappa=0.28 model at ep=10000, L=64; compare against HMC reference.
"""
import functools

import numpy as np
import torch
import matplotlib.pyplot as plt

from common import (FIGS, OUT, ROOT, load_hmc, beta_conf_phys, blocked_mean_sem,
                    em_reverse, norm_to_phys, phys_obs, jackknife_obs)
from diffusion_lightning import DiffusionModel, marginal_prob_std
from networks import NCSNpp2D

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--hmc_n", type=int, default=10240)
args = ap.parse_args()

K, L2, L = 0.28, 0.022, args.L
N = max(1, round(512 * (64 / L) ** 2))     # equal total dof across volumes
SUF = "" if L == 64 else f"_L{L}"
CKPT = ROOT / "2Dphi4/runs/phi4_Lmulti8-16-32-64_k0.28_l0.022_ncsnpp/models/epoch=10000.ckpt"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
hp = ck["hyper_parameters"]
score_model = NCSNpp2D(functools.partial(marginal_prob_std, sigma=hp["sigma"]))
model = DiffusionModel.load_from_checkpoint(CKPT, score_model=score_model,
                                            map_location="cuda", weights_only=False)
model = model.to("cuda").eval()
model.L = L

hmc = torch.from_numpy(load_hmc("0.28", L, n=args.hmc_n)).float().cuda()
ref = jackknife_obs(phys_obs(hmc), nb=64)
b_h, e_h = blocked_mean_sem(beta_conf_phys(hmc.double(), K, L2).cpu().numpy())
print(f"HMC ref: beta_conf={b_h:.5f}({e_h:.5f})  phi2={ref['phi2'][0]:.4f}  "
      f"chi={ref['chi'][0]:.1f}  xi={ref['xi'][0]:.1f}")

STEPS = [50, 100, 200, 350, 500, 700, 1000, 1400, 2000, 3000, 5000]
RUNS = [(sched, s) for sched in ("log", "linear") for s in STEPS]
rows = []
for sched, steps in RUNS:
    x = em_reverse(model, N, steps, sched, 1e-5, seed=steps + (0 if sched == "log" else 7))
    p = norm_to_phys(x[:, 0].double(), hp["norm_min"], hp["norm_max"])
    b = beta_conf_phys(p, K, L2).cpu().numpy()
    o = jackknife_obs(phys_obs(p.float()), nb=min(32, N))
    rows.append(dict(sched=sched, steps=steps, b=b.mean(),
                     be=b.std(ddof=1) / np.sqrt(N), o=o))
    print(f"{sched:6s} {steps:5d}: beta_conf={b.mean():.4f}({b.std(ddof=1)/np.sqrt(N):.4f})"
          f"  phi2={o['phi2'][0]:.4f}  chi={o['chi'][0]:.1f}  xi={o['xi'][0]:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = axes[0]
for sched, mk, c in [("log", "o", "C0"), ("linear", "s", "C3")]:
    r = [q for q in rows if q["sched"] == sched]
    ax.errorbar([q["steps"] for q in r], [q["b"] for q in r],
                yerr=[q["be"] for q in r], fmt=mk + "-", color=c, label=f"{sched} schedule")
ax.axhspan(b_h - e_h, b_h + e_h, color="k", alpha=0.15, label="HMC")
ax.axhline(1, ls=":", color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("EM steps"); ax.set_ylabel(r"$\beta_{\rm conf}$ (generated)")
ax.set_title("thermometer vs sampler quality")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
r = [q for q in rows if q["sched"] == "log"]
for key, mk in [("phi2", "o"), ("chi", "s")]:
    err = [abs(q["o"][key][0] / ref[key][0] - 1) for q in r]
    yerr = [q["o"][key][1] / ref[key][0] for q in r]
    ax.errorbar([q["steps"] for q in r], err, yerr=yerr, fmt=mk + "-", label=key)
ax.errorbar([q["steps"] for q in r], [abs(q["b"] - 1) for q in r],
            yerr=[q["be"] for q in r], fmt="d--", color="k",
            label=r"$|\beta_{\rm conf}-1|$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("EM steps (log schedule)"); ax.set_ylabel("relative error vs HMC")
ax.set_title("observable errors track the thermometer")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

fig.suptitle(r"C2: $\beta_{\rm conf}$ as sampler diagnostic, $\kappa=0.28$, $L=64$, ep=10000")
fig.tight_layout()
fig.savefig(FIGS / f"c2_sensitivity{SUF}.pdf")
fig.savefig(FIGS / f"c2_sensitivity{SUF}.png", dpi=160)

import json
json.dump([{k: (v if not isinstance(v, dict) else {a: list(map(float, b)) for a, b in v.items()})
            for k, v in q.items()} for q in rows],
          open(OUT / f"c2_results{SUF}.json", "w"), indent=1, default=float)
print(f"saved -> {FIGS}/c2_sensitivity.pdf, c2_results.json")
