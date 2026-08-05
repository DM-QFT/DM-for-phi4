"""Rebuild the C3-A comparison with Wolff+FAHMC Boltzmann ensembles.

Replaces the MALA o_eq (IR-under-relaxed near the effective critical point)
by cluster-sampled ensembles mapped back via x = c*phi. Writes c3_wolff.json.
"""
import json

import h5py
import numpy as np
import torch

from common import OUT, beta_conf_phys, blocked_mean_sem, phys_obs, jackknife_obs

K, L2 = 0.28, 0.022
c3 = json.load(open(OUT / "c3_results.json"))

rows = []
for r in c3["A"]:
    beta = r["beta"]
    with h5py.File(OUT / "crosscheck_wolff" / f"wolff_beta{beta:.4f}.jld2", "r") as h:
        phi = np.array(h["cfgs"], dtype=np.float64)
        c = float(np.array(h["c"]))
    x = torch.from_numpy(c * phi).cuda()

    b = beta_conf_phys(x, K, L2).cpu().numpy()
    mb, eb = blocked_mean_sem(b, nb=64)
    zb = (mb - beta) / eb
    assert abs(zb) < 5, (beta, mb, eb)      # exact identity on the Wolff ensemble

    o_eq = jackknife_obs(phys_obs(x.float()), nb=64)
    rows.append(dict(st=r["st"], beta=beta, o_fwd=r["o_fwd"],
                     o_eq={k: list(map(float, v)) for k, v in o_eq.items()}))
    print(f"st={r['st']}: beta={beta:.4f}  <beta_hat>={mb:.4f}({eb:.4f}) z={zb:+.1f}  "
          f"chi_c: fwd={r['o_fwd']['chi_c'][0]:.1f} eq={o_eq['chi_c'][0]:.1f}  "
          f"Fk1 eq={o_eq['Fk1'][0]:.1f}")
    del x
    torch.cuda.empty_cache()

json.dump(dict(A=rows, B=c3["B"]), open(OUT / "c3_wolff.json", "w"), indent=1)
print(f"saved -> {OUT}/c3_wolff.json")
