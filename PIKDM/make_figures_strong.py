"""Figures for the 0D PIK strong-form (reference-subtracted) benchmark.
Same layout as make_figures.py but loads strong_refsub_0d.pt and writes
figures/v_compare_strong.{pdf,png} and figures/reverse_traj_strong.{pdf,png}.
Kept separate so the weak-form figures used by archive/PIKDM_weakform.tex are
not overwritten."""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import action_matching_0d as A

phi, w, N, DEV, dphi = A.phi, A.w, A.N, A.DEV, A.dphi
phin = phi.cpu().numpy()
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)

model = A.SNet().to(DEV)
model.load_state_dict(torch.load(os.path.join(HERE, "strong_refsub_0d.pt"), map_location=DEV))
model.eval()


def cdf(t):
    p = A.density(t)
    C = torch.cat([phi.new_zeros(1), torch.cumsum(0.5 * (p[1:] + p[:-1]) * dphi, 0)])
    return (C / C[-1]).cpu().numpy()


# ---------- Figure 1: v_theta vs v_exact ----------
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ts = [0.1, 0.3, 0.5, 0.7, 0.9]
colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(ts)))
for t, c in zip(ts, colors):
    tt = phi.new_tensor(t)
    ve = A.v_exact(tt).cpu().numpy()
    vth = model.velocity(phi, tt).cpu().numpy()
    p = A.density(tt).cpu().numpy()
    m = p > 1e-4 * p.max()
    xx, vee, vtt = phin[m], ve[m], vth[m]
    ax.plot(xx, vee, color=c, lw=2.0, label=f"$t={t}$")
    k = max(1, len(xx) // 14)
    ax.plot(xx[::k], vtt[::k], color=c, lw=0, marker="o", ms=5, mfc="none", mew=1.3)
ax.set_xlim(-5, 5)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$v_t(\phi)$")
ax.set_title("lines: exact (quadrature)   circles: learned (strong, ref. subtraction)")
ax.legend(ncol=2, fontsize=9, framealpha=0.9)
ax.grid(alpha=0.25)
fig.savefig(os.path.join(HERE, "figures/v_compare_strong.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figures/v_compare_strong.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote figures/v_compare_strong.{pdf,png}")


# ---------- Figure 2: reverse trajectories ----------
t_out = np.linspace(0.0, 1.0, 101)
c_levels = np.linspace(0.025, 0.975, 31)

exact_traj = np.zeros((len(c_levels), len(t_out)))
for j, t in enumerate(t_out):
    Ct = cdf(phi.new_tensor(t))
    exact_traj[:, j] = np.interp(c_levels, Ct, phin)


def vf(t, q):
    vg = model.velocity(phi, phi.new_tensor(float(t)))
    q = torch.clamp(q, phi[0], phi[-1])
    return torch.clamp(A.interp1d(q, vg), -50.0, 50.0)


def seg(x, t_hi, t_lo, sub=20):
    h = (t_lo - t_hi) / sub; t = t_hi
    for _ in range(sub):
        k1 = vf(t, x); k2 = vf(t + 0.5 * h, x + 0.5 * h * k1)
        k3 = vf(t + 0.5 * h, x + 0.5 * h * k2); k4 = vf(t + h, x + h * k3)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4); t += h
    return x


x = torch.tensor(exact_traj[:, -1], device=DEV, dtype=phi.dtype)
learned = np.zeros_like(exact_traj)
learned[:, -1] = x.cpu().numpy()
for j in range(len(t_out) - 1, 0, -1):
    x = seg(x, t_out[j], t_out[j - 1])
    learned[:, j - 1] = x.cpu().numpy()

fig, ax = plt.subplots(figsize=(6.2, 4.2))
for i in range(len(c_levels)):
    ax.plot(t_out, exact_traj[i], color="0.62", lw=1.0,
            label="exact (CDF map)" if i == 0 else None)
    ax.plot(t_out, learned[i], color="C3", lw=0.9, ls="--",
            label="learned (reverse ODE)" if i == 0 else None)
ax.axhline(2, color="0.3", lw=0.6, ls=":"); ax.axhline(-2, color="0.3", lw=0.6, ls=":")
ax.set_xlabel(r"$t$  (target $t{=}0 \leftarrow$ base $t{=}1$)")
ax.set_ylabel(r"$\phi_t$")
ax.set_title("reverse transport (strong form): base Gaussian $\\to$ bimodal target")
ax.legend(loc="upper center", fontsize=9)
ax.grid(alpha=0.25)
fig.savefig(os.path.join(HERE, "figures/reverse_traj_strong.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figures/reverse_traj_strong.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote figures/reverse_traj_strong.{pdf,png}")
