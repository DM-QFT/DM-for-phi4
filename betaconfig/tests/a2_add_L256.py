"""Append L=256 rows (3 kappas, 5120 cfgs) to a2_terms.csv and print slopes."""
import numpy as np
import torch

from common import OUT, load_hmc, beta_terms_phys, blocked_mean_sem

KAPPAS = ["0.26", "0.2705", "0.28"]
L = 256
NOISE = [0.0, 0.48, 4.8]
L2 = 0.022

torch.manual_seed(0)
rows = []
for k in KAPPAS:
    p0 = torch.from_numpy(load_hmc(k, L, n=10240)).cuda()
    for s in NOISE:
        p = p0 if s == 0 else p0 + s * torch.randn(
            p0.shape, generator=torch.Generator("cuda").manual_seed(L),
            device="cuda", dtype=p0.dtype)
        t1, t2 = beta_terms_phys(p, float(k), L2)
        m1, e1 = blocked_mean_sem(t1.cpu().numpy())
        m2, e2 = blocked_mean_sem(t2.cpu().numpy())
        ms, es = blocked_mean_sem((t1 + t2).cpu().numpy())
        rows.append((float(k), L, s, m1, e1, m2, e2, ms, es))
        if s == 0.0:
            print(f"k={k} L={L} raw: <b>={ms:.5f}({es:.5f}) z={(ms - 1) / es:+.1f}  "
                  f"<t2>={m2:+.2e}({e2:.0e})")
    del p0
    torch.cuda.empty_cache()

csv = OUT / "a2_terms.csv"
old = np.loadtxt(csv, delimiter=",")
old = old[old[:, 1] != L]                      # idempotent on rerun
all_rows = np.vstack([old, np.array(rows)])
np.savetxt(csv, all_rows, delimiter=",", header=(
    "k,L,noise,term1_mean,term1_sem,term2_mean,term2_sem,beta_mean,beta_sem"))

for k in KAPPAS:
    r = all_rows[(all_rows[:, 0] == float(k)) & (all_rows[:, 2] == 0.0)]
    sl = np.polyfit(np.log(r[:, 1]), np.log(-r[:, 5]), 1)[0]
    print(f"k={k}: slope with L=8..256 = {sl:.3f}  (expect -2)")
print(f"updated {csv}")
