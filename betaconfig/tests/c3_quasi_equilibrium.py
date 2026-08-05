"""C3: is p_t ~ exp(-beta_conf(t) S)?  Two tests, kappa=0.28, L=64.

A) at fixed sigma_t compare forward-noised HMC vs exact MALA samples of
   exp(-beta_fwd(sigma_t) S): beta_conf matches by construction, other
   observables measure the failure of the quasi-equilibrium family.
B) score-free annealed reverse SDE with drift -beta_fwd(sigma_t) c g(phi),
   the scalar analogue of the U(2) 'action as drift' MAALA run.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common import (FIGS, HERE, OUT, load_hmc, beta_conf_phys, blocked_mean_sem,
                    action_phys, grad_phys, phys_obs, jackknife_obs, norm_to_phys)

K, L2, L = 0.28, 0.022, 64
SIGMA = 480.0
DEV = "cuda"
torch.manual_seed(3)

d = np.load(HERE.parent / "results" / "beta_conf_traj_multiL" /
            "beta_conf_fwd_rev_L64_k0.28_l0.022_ep10000_steps2000_log.npz")
NORM_MIN, NORM_MAX = float(d["norm_min"]), float(d["norm_max"])
C = (NORM_MAX - NORM_MIN) / 2
# forward curve as interpolator in log sigma_t (npz stores decreasing t)
lg_s, b_fwd = np.log(d["sigma_t"][::-1]), d["hmc_mean"][::-1]


def beta_of_sigma(s):
    return float(np.interp(np.log(s), lg_s, b_fwd))


hmc = torch.from_numpy(load_hmc("0.28", L)).to(DEV)
hmc_norm = 2 * (hmc - NORM_MIN) / (NORM_MAX - NORM_MIN) - 1

# self-check: grad_phys is exactly the gradient of action_phys
_pt = torch.randn(2, 8, 8, dtype=torch.float64, device=DEV, requires_grad=True)
action_phys(_pt, K, L2).sum().backward()
assert torch.allclose(_pt.grad, grad_phys(_pt.detach(), K, L2), atol=1e-12)


@torch.no_grad()
def mala(beta, x0, nsteps, h, adapt_until):
    """Exact MALA for exp(-beta S_phys); h is the proposal variance."""
    x = x0.clone()
    u = beta * action_phys(x, K, L2)
    g = beta * grad_phys(x, K, L2)
    for i in range(nsteps):
        xi = torch.randn_like(x)
        xp = x - 0.5 * h * g + np.sqrt(h) * xi
        up = beta * action_phys(xp, K, L2)
        gp = beta * grad_phys(xp, K, L2)
        lq_fwd = -(((xp - x + 0.5 * h * g) ** 2).sum(dim=(1, 2))) / (2 * h)
        lq_rev = -(((x - xp + 0.5 * h * gp) ** 2).sum(dim=(1, 2))) / (2 * h)
        la = -up + u + lq_rev - lq_fwd
        acc = torch.log(torch.rand_like(la)) < la
        x = torch.where(acc[:, None, None], xp, x)
        u = torch.where(acc, up, u)
        g = torch.where(acc[:, None, None], gp, g)
        rate = acc.double().mean().item()
        if i < adapt_until:
            h *= 1.05 if rate > 0.574 else 0.95
    return x, h, rate


# ---------- A: quasi-equilibrium family vs true diffusion marginals ----------
SIGTS = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.5, 0.7, 1.0]
NCH, NSTEPS = 1024, 6000
rowsA = []
gen = torch.Generator(DEV).manual_seed(11)
for st in SIGTS:
    beta = beta_of_sigma(st)
    noise = torch.randn(hmc_norm.shape, generator=gen, device=DEV, dtype=hmc.dtype)
    p_fwd = norm_to_phys(hmc_norm + st * noise, NORM_MIN, NORM_MAX)

    # init configs disjoint from the measurement set; second chain set starts
    # overdispersed (x1.5) so init bias would show up as chain-set disagreement
    perm = torch.randperm(p_fwd.shape[0], generator=gen, device=DEV)
    p_meas = p_fwd[perm[NCH:]]
    x0 = p_fwd[perm[:NCH]]
    xw, h, rate = mala(beta, x0, nsteps=NSTEPS, h=1e-3, adapt_until=1000)
    xo, _, _ = mala(beta, 1.5 * x0, nsteps=NSTEPS, h=h, adapt_until=0)

    b_eq = np.concatenate([beta_conf_phys(y, K, L2).cpu().numpy() for y in (xw, xo)])
    assert abs(b_eq.mean() / beta - 1) < 0.01, (st, beta, b_eq.mean())

    o_w = jackknife_obs(phys_obs(xw.float()), nb=32)
    o_o = jackknife_obs(phys_obs(xo.float()), nb=32)
    # IR equilibration gate: warm and overdispersed chain sets must agree
    for key in ("chi_c", "Fk1"):
        zir = (o_w[key][0] - o_o[key][0]) / np.hypot(o_w[key][1], o_o[key][1])
        assert abs(zir) < 4, (st, key, o_w[key], o_o[key])

    o_fwd = jackknife_obs(phys_obs(p_meas.float()), nb=64)
    o_eq = jackknife_obs(phys_obs(torch.cat([xw, xo]).float()), nb=32)
    b_fwd_m, b_fwd_e = blocked_mean_sem(beta_conf_phys(p_meas, K, L2).cpu().numpy())
    rowsA.append(dict(st=st, beta=beta, o_fwd=o_fwd, o_eq=o_eq))
    print(f"sigma_t={st}: beta_fwd={beta:.4f}  MALA acc={rate:.2f} h={h:.2e}  "
          f"beta_conf: fwd={b_fwd_m:.4f}({b_fwd_e:.4f}) eq={b_eq.mean():.4f} "
          f"| chi_c fwd={o_fwd['chi_c'][0]:.1f} eq={o_eq['chi_c'][0]:.1f} "
          f"(warm/over z={zir:+.1f})")

# ---------- B: score-free annealed reverse using beta_fwd(sigma_t) ----------
NB, STEPS, EPS = 512, 2000, 1e-5
ts = torch.exp(torch.linspace(0.0, np.log(EPS), STEPS + 1, device=DEV)).double()
dts = ts[:-1] - ts[1:]
std1 = np.sqrt((SIGMA ** 2 - 1) / (2 * np.log(SIGMA)))
x = torch.randn(NB, L, L, device=DEV, dtype=torch.float64) * std1
rec_t, rec_b = [], []
for i in range(STEPS):
    t = ts[i].item()
    st = np.sqrt((SIGMA ** (2 * t) - 1) / (2 * np.log(SIGMA)))
    beta = beta_of_sigma(max(st, 1e-8))
    gc = SIGMA ** t
    score = -beta * C * grad_phys(norm_to_phys(x, NORM_MIN, NORM_MAX), K, L2)
    x = x + gc ** 2 * score * dts[i] + gc * torch.sqrt(dts[i]) * torch.randn_like(x)
    if (i + 1) % 40 == 0:
        b = beta_conf_phys(norm_to_phys(x, NORM_MIN, NORM_MAX), K, L2)
        rec_t.append(ts[i + 1].item()); rec_b.append(b.mean().item())

p_ann = norm_to_phys(x, NORM_MIN, NORM_MAX)
b_ann = beta_conf_phys(p_ann, K, L2).cpu().numpy()
o_ann = jackknife_obs(phys_obs(p_ann.float()), nb=32)
o_hmc = jackknife_obs(phys_obs(hmc.float()), nb=64)
print(f"\nannealed (score-free) final: beta_conf={b_ann.mean():.4f}"
      f"({b_ann.std(ddof=1) / np.sqrt(NB):.4f})  phi2={o_ann['phi2'][0]:.4f} "
      f"chi={o_ann['chi'][0]:.1f} xi={o_ann['xi'][0]:.1f}")
print(f"HMC reference:              beta_conf=1        phi2={o_hmc['phi2'][0]:.4f} "
      f"chi={o_hmc['chi'][0]:.1f} xi={o_hmc['xi'][0]:.1f}")

# ---------- figures ----------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
ax = axes[0]
for key, mk in [("phi2", "o"), ("binder", "s"), ("chi_c", "^"), ("Fk1", "d")]:
    rel, err = [], []
    for r in rowsA:
        a, ea = r["o_eq"][key]; b_, eb = r["o_fwd"][key]
        rel.append(a / b_ - 1); err.append(np.hypot(ea / b_, a * eb / b_ ** 2))
    ax.errorbar(SIGTS, rel, yerr=err, fmt=mk + "-", label=key)
ax.axhline(0, ls=":", color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"$\sigma_t$"); ax.set_ylabel(r"$O_{e^{-\beta(t)S}}/O_{p_t}-1$")
ax.set_title("A: quasi-equilibrium vs true marginal")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
ax.plot(d["sigma_t"], d["hmc_mean"], "-", color="C0", label="forward (exact)")
ax.plot(d["sigma_t"], d["dm_mean"], "--", color="C3", label="DM reverse")
st_rec = np.sqrt((SIGMA ** (2 * np.array(rec_t)) - 1) / (2 * np.log(SIGMA)))
ax.plot(st_rec, rec_b, ":", color="C2", lw=2, label=r"score-free $\beta(t)\,g$ drift")
ax.axhline(1, ls=":", color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"$\sigma_t$"); ax.set_ylabel(r"$\beta_{\rm conf}$")
ax.set_title("B: annealed trajectory")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

ax = axes[2]
keys = ["phi2", "binder", "chi_c", "Fk1"]
relA = [o_ann[k][0] / o_hmc[k][0] - 1 for k in keys]
ax.bar(range(len(keys)), relA, color="C2")
ax.set_xticks(range(len(keys)), keys)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("relative error vs HMC")
ax.set_title("B: score-free sampler, final ensemble")
ax.grid(alpha=0.3, axis="y")

fig.suptitle(r"C3: quasi-equilibrium hypothesis $p_t \sim e^{-\beta_{\rm conf}(t) S}$, "
             rf"$\kappa=0.28$, $L={L}$")
fig.tight_layout()
fig.savefig(FIGS / "c3_quasi_equilibrium.pdf")
fig.savefig(FIGS / "c3_quasi_equilibrium.png", dpi=160)

import json
json.dump(dict(
    A=[{**{k: r[k] for k in ("st", "beta")},
        "o_fwd": {a: list(map(float, v)) for a, v in r["o_fwd"].items()},
        "o_eq": {a: list(map(float, v)) for a, v in r["o_eq"].items()}} for r in rowsA],
    B=dict(beta_conf=float(b_ann.mean()),
           obs={a: list(map(float, v)) for a, v in o_ann.items()},
           hmc={a: list(map(float, v)) for a, v in o_hmc.items()})),
    open(OUT / "c3_results.json", "w"), indent=1)
print(f"saved -> {FIGS}/c3_quasi_equilibrium.pdf, c3_results.json")
