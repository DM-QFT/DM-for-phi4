"""C1: locality conjecture — does beta_conf converge before long-range observables?

Near-critical kappa=0.2705 (xi ~ 40 at L=64), multi-L run with log-spaced ckpts.
"""
import functools

import numpy as np
import torch
import matplotlib.pyplot as plt

from common import (FIGS, HERE, OUT, ROOT, load_hmc, beta_conf_phys, blocked_mean_sem,
                    em_reverse, norm_to_phys, phys_obs, jackknife_obs)
from diffusion_lightning import DiffusionModel, marginal_prob_std
from networks import NCSNpp2D

K, L2, L, N = 0.2705, 0.022, 64, 512
RUN = ROOT / "2Dphi4/runs/phi4_Lmulti8-16-32-64_k0.2705_l0.022_ncsnpp"
EPOCHS = [1, 2, 4, 8, 16, 31, 57, 108, 201, 376, 703, 1314, 2453, 4581, 8554, 10000]

hmc = torch.from_numpy(load_hmc("0.2705", L)).cuda()
ref = jackknife_obs(phys_obs(hmc.float()), nb=64)
b_h, e_h = blocked_mean_sem(beta_conf_phys(hmc, K, L2).cpu().numpy())
print(f"HMC ref: beta_conf={b_h:.5f}({e_h:.5f})  phi2={ref['phi2'][0]:.4f}  "
      f"chi={ref['chi'][0]:.1f}({ref['chi'][1]:.1f})  xi={ref['xi'][0]:.1f}({ref['xi'][1]:.1f})")

rows = []
for ep in EPOCHS:
    ckpt = RUN / "models" / f"epoch={ep:04d}.ckpt" if ep < 10000 else RUN / "models" / "epoch=10000.ckpt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    score_model = NCSNpp2D(functools.partial(marginal_prob_std, sigma=hp["sigma"]))
    model = DiffusionModel.load_from_checkpoint(ckpt, score_model=score_model,
                                                map_location="cuda", weights_only=False)
    model = model.to("cuda").eval()
    model.L = L
    x = em_reverse(model, N, 2000, "log", 1e-5, seed=ep)
    p = norm_to_phys(x[:, 0].double(), hp["norm_min"], hp["norm_max"])
    b = beta_conf_phys(p, K, L2).cpu().numpy()
    o = jackknife_obs(phys_obs(p.float()), nb=32)
    rows.append(dict(ep=ep, b=b.mean(), be=b.std(ddof=1) / np.sqrt(N), o=o))
    print(f"ep={ep:5d}: beta_conf={b.mean():.4f}({b.std(ddof=1)/np.sqrt(N):.4f})  "
          f"phi2={o['phi2'][0]:.4f}  chi={o['chi'][0]:6.1f}  xi={o['xi'][0]:5.1f}")
    del model, ck
    torch.cuda.empty_cache()

eps_arr = np.array([r["ep"] for r in rows])
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax = axes[0]
ax.errorbar(eps_arr, [r["b"] for r in rows], yerr=[r["be"] for r in rows],
            fmt="o-", color="C3", label=r"$\beta_{\rm conf}$ (DM)")
ax.axhspan(b_h - e_h, b_h + e_h, color="k", alpha=0.15, label="HMC")
# reverse stops at t=eps: residual smearing biases beta_hat by ~ -6e-4
ax.axhspan(1 - 6e-4, 1, color="C2", alpha=0.25, label=r"$p_\epsilon$ smearing")
ax.axhline(1, ls=":", color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("training epoch"); ax.set_ylabel(r"$\beta_{\rm conf}$")
ax.set_title("thermometer vs training time")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
for key, mk, c in [("phi2", "o", "C0"), ("chi", "s", "C1"), ("xi", "^", "C2"),
                   ("Fk1", "d", "C4")]:
    rel = [abs(r["o"][key][0] / ref[key][0] - 1) for r in rows]
    err = [np.hypot(r["o"][key][1] / ref[key][0],
                    r["o"][key][0] * ref[key][1] / ref[key][0] ** 2) for r in rows]
    ax.errorbar(eps_arr, rel, yerr=err, fmt=mk + "-", color=c, ms=3, lw=0.9, label=key)
    # statistical floor: |x_hat| ~ 0.8 SEM once converged; deviations below it are noise
    floor = 0.8 * np.median([r["o"][key][1] / ref[key][0] for r in rows[-4:]])
    ax.axhline(floor, ls=":", color=c, lw=0.8, alpha=0.7)
ax.errorbar(eps_arr, [abs(r["b"] - 1) for r in rows], yerr=[r["be"] for r in rows],
            fmt="d--", color="k", lw=1.2, label=r"$|\beta_{\rm conf}-1|$")
ax.axhline(0.8 * np.median([r["be"] for r in rows[-4:]]), ls=":", color="k",
           lw=0.8, alpha=0.7)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("training epoch"); ax.set_ylabel("relative error vs HMC")
ax.set_title("local vs long-range convergence")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

fig.suptitle(rf"C1: locality of $\beta_{{\rm conf}}$, near-critical $\kappa={K}$, $L={L}$")
fig.tight_layout()
fig.savefig(FIGS / "c1_epoch_locality.pdf")
fig.savefig(FIGS / "c1_epoch_locality.png", dpi=160)

import json
json.dump([{**{k: float(r[k]) for k in ("ep", "b", "be")},
            "o": {a: list(map(float, v)) for a, v in r["o"].items()}} for r in rows],
          open(OUT / "c1_results.json", "w"), indent=1)
print(f"saved -> {FIGS}/c1_epoch_locality.pdf, c1_results.json")
