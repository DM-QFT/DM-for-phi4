"""A2: term2 = -2 g^T H g/(g^2)^2 is O(1/V); term1-only bias = -<term2> exactly.

Raw + forward-noised HMC only, no diffusion model involved.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common import FIGS, OUT, load_hmc, beta_terms_phys, blocked_mean_sem

KAPPAS = ["0.26", "0.2705", "0.27088", "0.28"]
LS = [8, 16, 32, 64, 128]
# physical-space noise std; 0.48/4.8 correspond to sigma_t ~ 0.1/1 in [-1,1] space
NOISE = [0.0, 0.48, 4.8]
DEV = "cuda"
L2 = 0.022

torch.manual_seed(0)
rows = []
for k in KAPPAS:
    for L in LS:
        p0 = torch.from_numpy(load_hmc(k, L)).to(DEV)
        for s in NOISE:
            p = p0 if s == 0 else p0 + s * torch.randn(
                p0.shape, generator=torch.Generator(DEV).manual_seed(L), device=DEV,
                dtype=p0.dtype)
            t1, t2 = beta_terms_phys(p, float(k), L2)
            m1, e1 = blocked_mean_sem(t1.cpu().numpy())
            m2, e2 = blocked_mean_sem(t2.cpu().numpy())
            ms, es = blocked_mean_sem((t1 + t2).cpu().numpy())
            rows.append((float(k), L, s, m1, e1, m2, e2, ms, es))
            if s == 0.0:
                z = (ms - 1.0) / es
                print(f"k={k} L={L:3d} raw:  <b>={ms:.5f}({es:.5f}) z={z:+.1f}  "
                      f"<t1>-1={m1 - 1:+.2e}  <t2>={m2:+.2e}")
        del p0
        torch.cuda.empty_cache()

rows = np.array(rows)
np.savetxt(OUT / "a2_terms.csv", rows, delimiter=",", header=(
    "k,L,noise,term1_mean,term1_sem,term2_mean,term2_sem,beta_mean,beta_sem"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# (1) -<term2> vs L at t=0, all kappas, with 1/L^2 guide
ax = axes[0]
for k, c in zip(KAPPAS, "C0 C1 C2 C3".split()):
    r = rows[(rows[:, 0] == float(k)) & (rows[:, 2] == 0.0)]
    ax.errorbar(r[:, 1], -r[:, 5], yerr=r[:, 6], fmt="o-", color=c, label=f"κ={k}")
ax.plot(LS, -rows[(rows[:, 0] == 0.28) & (rows[:, 2] == 0.0)][0, 5]
        * (LS[0] / np.array(LS, float)) ** 2, "k:", label=r"$\propto 1/L^2$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("L"); ax.set_ylabel(r"$-\langle t_2\rangle$")
ax.set_title("directional term, raw HMC (t=0)")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

# (2) same for noised ensembles, kappa=0.28
ax = axes[1]
def fit_slope(r):
    # only points where -<t2> is negative beyond noise; log(<=0) would poison the fit
    m = -r[:, 5] > 2 * r[:, 6]
    return np.polyfit(np.log(r[m, 1]), np.log(-r[m, 5]), 1)[0], m


for s, c in zip(NOISE, "C0 C1 C2".split()):
    r = rows[(rows[:, 0] == 0.28) & (rows[:, 2] == s)]
    lg, m = fit_slope(r)
    ax.errorbar(r[m, 1], -r[m, 5], yerr=r[m, 6], fmt="o-", color=c,
                label=f"noise s={s}  (slope {lg:.2f})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("L"); ax.set_ylabel(r"$-\langle t_2\rangle$")
ax.set_title(r"κ=0.28, HMC+noise; expect slope $-2$")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

# (3) identity test with correct statistics: z = (<t1+t2>-1)/SEM, expect |z|<~2
ax = axes[2]
for k, c in zip(KAPPAS, "C0 C1 C2 C3".split()):
    r = rows[(rows[:, 0] == float(k)) & (rows[:, 2] == 0.0)]
    ax.plot(r[:, 1], (r[:, 7] - 1) / r[:, 8], "o-", color=c, label=f"κ={k}")
ax.axhspan(-2, 2, color="k", alpha=0.1)
ax.axhline(0, ls=":", color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("L"); ax.set_ylabel(r"$(\langle\hat\beta\rangle-1)/{\rm SEM}$")
ax.set_title(r"exact identity $\langle\hat\beta\rangle=1$, raw HMC")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

fig.suptitle("A2: volume scaling of the directional (Hessian) term, 2D $\\phi^4$")
fig.tight_layout()
fig.savefig(FIGS / "a2_term_scaling.pdf")
fig.savefig(FIGS / "a2_term_scaling.png", dpi=160)

# slope fits at t=0 per kappa
for k in KAPPAS:
    r = rows[(rows[:, 0] == float(k)) & (rows[:, 2] == 0.0)]
    sl, m = fit_slope(r)
    print(f"k={k}: log-log slope of -<t2> vs L (raw, {m.sum()} pts) = {sl:.3f}  (expect -2)")
print(f"saved -> {FIGS}/a2_term_scaling.pdf, a2_terms.csv")
