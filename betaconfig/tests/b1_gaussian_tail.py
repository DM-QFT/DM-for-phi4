"""B1: forward beta_conf(sigma_t) vs Gaussian (Wick) prediction.

Prediction: field = HMC + white noise treated as Gaussian with the measured
2-pt function; exact at large sigma_t. Quartic term gives a sigma_t^-4 tail,
not the sigma_t^-2 of the draft's quadratic-action example.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common import FIGS, HERE, OUT, ROOT, load_hmc, beta_conf_phys

K, L2 = 0.28, 0.022
A = 1 - 2 * L2


def beta_wick(G0, G1, G2, G2d, k, l):
    """<TrH>/<g^2> per site for a Gaussian field with these 2-pt moments (Wick)."""
    a = 1 - 2 * l
    Eg2 = (4 * k ** 2 * (4 * G0 + 4 * G2 + 8 * G2d) + 4 * a ** 2 * G0
           + 240 * l ** 2 * G0 ** 3 - 32 * k * a * G1 - 192 * k * l * G0 * G1
           + 48 * a * l * G0 ** 2)
    return (2 * a + 12 * l * G0) / Eg2


# --- internal check of the Wick algebra: iid Gaussian field ---
# the formula predicts E[TrH]/E[g^2]; <TrH/g^2> differs from it at O(1/V)
# because Var(g^2)/E[g^2]^2 ~ 45/V at large s (phi^6 fluctuations)
from common import grad_phys  # noqa: E402

torch.manual_seed(1)
for s in [0.5, 2.0, 20.0]:
    p = s * torch.randn(4096, 32, 32, dtype=torch.float64, device="cuda")
    g = grad_phys(p, K, L2)
    trH = (2 * A + 12 * L2 * p ** 2).sum(dim=(1, 2))
    mc = (trH.mean() / (g * g).sum(dim=(1, 2)).mean()).item()
    an = beta_wick(s ** 2, 0.0, 0.0, 0.0, K, L2)
    assert abs(mc / an - 1) < 0.01, (s, mc, an)
    mor = beta_conf_phys(p, K, L2).mean().item()
    print(f"Wick self-test iid s={s}: E-ratio MC={mc:.4e} formula={an:.4e} ok  "
          f"(mean-of-ratios={mor:.4e}, 1/V shift {(mor / an - 1) * 100:+.1f}%)")

# --- measured 2-pt moments of the kappa=0.28 target (physical space) ---
curves = {}
for L in [64, 128]:
    p = torch.from_numpy(load_hmc("0.28", L)).cuda()
    G0 = (p * p).mean().item()
    G1 = (p * torch.roll(p, 1, 1)).mean().item() / 2 + \
         (p * torch.roll(p, 1, 2)).mean().item() / 2
    G2 = (p * torch.roll(p, 2, 1)).mean().item() / 2 + \
         (p * torch.roll(p, 2, 2)).mean().item() / 2
    G2d = (p * torch.roll(torch.roll(p, 1, 1), 1, 2)).mean().item() / 2 + \
          (p * torch.roll(torch.roll(p, 1, 1), -1, 2)).mean().item() / 2
    d = np.load(HERE.parent / "results" / "beta_conf_traj_multiL" /
                f"beta_conf_fwd_rev_L{L}_k0.28_l0.022_ep10000_steps2000_log.npz")
    c = (d["norm_max"] - d["norm_min"]) / 2          # normalised -> physical scale
    s_n = c * d["sigma_t"]                            # white noise only shifts G(0)
    pred = np.array([beta_wick(G0 + s ** 2, G1, G2, G2d, K, L2) for s in s_n])
    pred_iid = np.array([beta_wick(G0 + s ** 2, 0, 0, 0, K, L2) for s in s_n])
    curves[L] = (d, s_n, pred, pred_iid)
    print(f"L={L}: G0={G0:.4f} G1={G1:.4f} G2={G2:.4f} G2d={G2d:.4f}")
    del p
    torch.cuda.empty_cache()

np.savez(OUT / "b1_data.npz",
         **{f"{k}_L{L}": v for L in [64, 128]
            for k, v in zip(("s_n", "meas", "pred", "pred_iid"),
                            (curves[L][1], curves[L][0]["hmc_mean"],
                             curves[L][2], curves[L][3]))})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axes[0]
for L, c in zip([64, 128], ["C0", "C1"]):
    d, s_n, pred, _ = curves[L]
    ax.plot(s_n, d["hmc_mean"], "o", ms=2.5, color=c, label=f"measured L={L}")
    ax.plot(s_n, pred, "-", color="k" if L == 128 else "gray", lw=1.2,
            label=f"Wick-Gaussian L={L}")
s_ref = np.logspace(0.5, 2.7, 10)
ax.plot(s_ref, 1 / (20 * L2 * s_ref ** 4) * 3, ":", color="C3",
        label=r"slope $-4$ (quartic tail)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"physical noise std $s_t = c\,\sigma_t$")
ax.set_ylabel(r"$\beta_{\rm conf}$")
ax.set_title("forward curve vs Gaussian prediction")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
d, s_n, pred, pred_iid = curves[64]
ax.plot(s_n, d["hmc_mean"] / pred, "o-", ms=2.5, lw=0.8, label="measured / Wick")
ax.plot(s_n, d["hmc_mean"] / pred_iid, "s--", ms=2.5, lw=0.8,
        label="measured / iid-Wick")
ax.axhline(1, ls=":", color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"$s_t$"); ax.set_title("L=64 ratio: non-Gaussianity of the target")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

# local log-slope of the measured curve: -2 (Gaussian) vs -4 (quartic) windows
ax = axes[2]
d, s_n, pred, _ = curves[64]
m = d["hmc_mean"]
sl = np.gradient(np.log(m), np.log(s_n))
ax.plot(s_n, sl, "o-", ms=2, lw=0.8, label="measured")
ax.plot(s_n, np.gradient(np.log(pred), np.log(s_n)), "-", color="k", lw=1,
        label="Wick-Gaussian")
ax.axhline(-2, ls=":", color="C2", label="-2 (quadratic action)")
ax.axhline(-4, ls=":", color="C3", label="-4 (quartic action)")
ax.set_xscale("log")
ax.set_xlabel(r"$s_t$"); ax.set_ylabel(r"$d\log\beta_{\rm conf}/d\log s_t$")
ax.set_title("local slope")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

fig.suptitle(r"B1: large-noise asymptotics of $\beta_{\rm conf}(t)$, $\kappa=0.28$")
fig.tight_layout()
fig.savefig(FIGS / "b1_gaussian_tail.pdf")
fig.savefig(FIGS / "b1_gaussian_tail.png", dpi=160)
print(f"saved -> {FIGS}/b1_gaussian_tail.pdf")
