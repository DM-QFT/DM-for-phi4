"""C4: beta_hat is Z2-even, so sector collapse is invisible to beta_conf.

Exact statement: g is odd and H even under phi -> -phi, hence
beta_hat(phi) = beta_hat(-phi) per configuration. Broken phase kappa=0.28.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common import FIGS, load_hmc, beta_conf_phys, blocked_mean_sem

K, L2, L = 0.28, 0.022, 64
p = torch.from_numpy(load_hmc("0.28", L)).cuda()
m = p.mean(dim=(1, 2))

# sector projection: flip m<0 configs -> "mode-collapsed" single-sector ensemble
p_proj = torch.where((m < 0)[:, None, None], -p, p)
m_proj = p_proj.mean(dim=(1, 2))

b = beta_conf_phys(p, K, L2)
b_proj = beta_conf_phys(p_proj, K, L2)
diff = (b - b_proj).abs().max().item()
# the symmetry is exact; the ~1e-16 residue is the affine normalisation
# round-trip inside phi4_beta_conf ((phi+1)-1 != phi in fp64)
assert diff < 1e-12, diff

mb, sb = blocked_mean_sem(b.cpu().numpy())
print(f"full ensemble:      <m>={m.mean():+.4f}  chi=V<m^2>={L * L * (m ** 2).mean():.1f}  "
      f"beta_conf={mb:.5f}({sb:.5f})")
print(f"sector-projected:   <m>={m_proj.mean():+.4f}  "
      f"chi_c=V(<m^2>-<m>^2)={L * L * (m_proj.var()):.1f}  beta_conf identical "
      f"to machine precision (max per-config diff = {diff:.1e})")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.hist(m.cpu().numpy(), bins=80, alpha=0.55, label="full (Z2-symmetric)")
ax.hist(m_proj.cpu().numpy(), bins=40, alpha=0.55, label="sector-collapsed")
ax.set_xlabel("magnetisation m"); ax.set_ylabel("count")
ax.set_title(rf"$\kappa=0.28$, $L={L}$: broken phase")
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(b.cpu().numpy(), bins=80, alpha=0.55, label="full")
ax.hist(b_proj.cpu().numpy(), bins=80, histtype="step", lw=1.5,
        label="sector-collapsed (identical)")
ax.axvline(1.0, ls=":", color="k", lw=0.8)
ax.set_xlabel(r"$\hat\beta$ per configuration")
ax.set_title(r"$\hat\beta(\phi)=\hat\beta(-\phi)$: $\beta_{\rm conf}$ blind to collapse")
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(FIGS / "c4_z2_blindness.pdf")
fig.savefig(FIGS / "c4_z2_blindness.png", dpi=160)
print(f"saved -> {FIGS}/c4_z2_blindness.pdf")
