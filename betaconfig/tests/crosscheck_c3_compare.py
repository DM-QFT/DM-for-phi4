"""Compare the Wolff+FAHMC cross-check ensembles with the C3 MALA ensembles."""
import glob
import json

import h5py
import numpy as np
import torch

from common import OUT, beta_conf_phys, phys_obs, jackknife_obs

K, L2 = 0.28, 0.022
c3 = json.load(open(OUT / "c3_results.json"))

print(f"{'beta':>7} {'obs':>22}  {'MALA':>12}  {'Wolff':>12}  {'z':>5}")
worst = 0.0
for r in c3["A"]:
    beta = r["beta"]
    f = glob.glob(str(OUT / "crosscheck_wolff" / f"wolff_beta{beta:.4f}.jld2"))
    assert len(f) == 1, beta
    with h5py.File(f[0], "r") as h:
        phi = np.array(h["cfgs"], dtype=np.float64)     # (n, L, L) via h5py
        c = float(np.array(h["c"]))
    x = torch.from_numpy(c * phi).cuda()

    b = beta_conf_phys(x, K, L2).cpu().numpy()
    zb = (b.mean() - beta) / (b.std(ddof=1) / np.sqrt(len(b)))
    o_w = jackknife_obs(phys_obs(x.float()), nb=32)
    for key in ("phi2", "binder", "chi_c", "Fk1"):
        m, em = r["o_eq"][key]
        w, ew = o_w[key]
        z = (w - m) / np.hypot(ew, em)
        worst = max(worst, abs(z))
        print(f"{beta:7.4f} {key:>22}  {m:12.4f}  {w:12.4f}  {z:+5.1f}")
    print(f"{beta:7.4f} {'<beta_hat>-beta':>22}  {'':>12}  {b.mean():12.4f}  {zb:+5.1f}")

print(f"\nworst |z| over all observables: {worst:.1f}")
