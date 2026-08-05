"""C3 part B standalone: score-free annealed reverse with drift -beta_fwd(sigma_t) c g.

Saves the beta_conf trajectory for the draft figure.
"""
import numpy as np
import torch

from common import HERE, OUT, beta_conf_phys, grad_phys, norm_to_phys

K, L2, L = 0.28, 0.022, 64
SIGMA = 480.0
DEV = "cuda"
torch.manual_seed(7)

d = np.load(HERE.parent / "results" / "beta_conf_traj_multiL" /
            "beta_conf_fwd_rev_L64_k0.28_l0.022_ep10000_steps2000_log.npz")
NORM_MIN, NORM_MAX = float(d["norm_min"]), float(d["norm_max"])
C = (NORM_MAX - NORM_MIN) / 2
lg_s, b_fwd = np.log(d["sigma_t"][::-1]), d["hmc_mean"][::-1]

NB, STEPS, EPS = 512, 2000, 1e-5
ts = torch.exp(torch.linspace(0.0, np.log(EPS), STEPS + 1, device=DEV)).double()
dts = ts[:-1] - ts[1:]
std1 = np.sqrt((SIGMA ** 2 - 1) / (2 * np.log(SIGMA)))
x = torch.randn(NB, L, L, device=DEV, dtype=torch.float64) * std1
rec_s, rec_b = [], []
for i in range(STEPS):
    t = ts[i].item()
    st = np.sqrt((SIGMA ** (2 * t) - 1) / (2 * np.log(SIGMA)))
    beta = float(np.interp(np.log(max(st, 1e-8)), lg_s, b_fwd))
    gc = SIGMA ** t
    score = -beta * C * grad_phys(norm_to_phys(x, NORM_MIN, NORM_MAX), K, L2)
    x = x + gc ** 2 * score * dts[i] + gc * torch.sqrt(dts[i]) * torch.randn_like(x)
    if (i + 1) % 25 == 0 or i == STEPS - 1:
        t2 = ts[i + 1].item()
        s2 = np.sqrt(max((SIGMA ** (2 * t2) - 1) / (2 * np.log(SIGMA)), 1e-12))
        b = beta_conf_phys(norm_to_phys(x, NORM_MIN, NORM_MAX), K, L2)
        rec_s.append(s2)
        rec_b.append(b.mean().item())

bf = beta_conf_phys(norm_to_phys(x, NORM_MIN, NORM_MAX), K, L2).cpu().numpy()
print(f"final beta_conf = {bf.mean():.4f} ± {bf.std(ddof=1) / np.sqrt(NB):.4f}")
np.savez(OUT / "c3b_anneal.npz", sigma=np.array(rec_s), beta=np.array(rec_b),
         final=bf.mean(), final_sem=bf.std(ddof=1) / np.sqrt(NB))
print(f"saved -> {OUT}/c3b_anneal.npz")
