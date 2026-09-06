"""C2 (3D): beta_conf sensitivity to reverse-sampler steps/schedule.

multiL4-8-16-32 k=0.1923 model at ep=10000 (final ckpt), as the 2D figure uses
its multiL model. L=32 in-range, L=64 cross-L; HMC reference at L=64.
Compile(default) + EMA as in sample_phi4_crossL.py production sampling.
"""
import argparse
import functools
import json
import sys
import time

import h5py
import numpy as np
import torch

from common import OUT, ROOT, blocked_mean_sem, em_reverse, norm_to_phys

sys.path.append(str(ROOT / "3Dphi4"))
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from networks_3d import NCSNpp3D
from phi4_action import phi4_beta_conf

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--sched", default="both", choices=["both", "log", "linear"])
args = ap.parse_args()
SCHEDS = ("log", "linear") if args.sched == "both" else (args.sched,)

K, LAM, L = 0.1923, 0.9, args.L
N = 32 if L == 64 else 64
CKPT = (ROOT / "3Dphi4/runs/phi4_3d_Lmulti4-8-16-32_k0.1923_l0.9_ncsnpp"
        / "models/epoch=10000.ckpt")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
hp = ck["hyper_parameters"]
score_model = NCSNpp3D(functools.partial(marginal_prob_std, sigma=hp["sigma"]))
# compile(default) as in sample_phi4_crossL.py; reduce-overhead corrupts
# F.interpolate(nearest) upsampling. EMA weights via em_reverse.
score_model = torch.compile(score_model, mode="default", dynamic=False)
# map_location cpu: ckpts were saved on a multi-GPU box (cuda:3)
model = DiffusionModel3D.load_from_checkpoint(CKPT, score_model=score_model,
                                              map_location="cpu")
model = model.to("cuda").eval()
assert model.spatial_dims == 3
model.L = L


def beta3d(p, chunk=64):
    # phi_min=-1, phi_max=1 makes the internal normalisation the identity
    out = [phi4_beta_conf(p[i:i + chunk, None], K, LAM, -1.0, 1.0, spatial_dims=3)
           for i in range(0, p.shape[0], chunk)]
    return torch.cat(out).cpu().numpy()


if L == 64 and SCHEDS[0] == "log":
    with h5py.File(ROOT / "3Dphi4/trainingdata"
                   / "cfgs_wolff_fahmc_k=0.1923_l=0.9_64^3.jld2", "r") as f:
        hmc = torch.from_numpy(np.array(f["cfgs"], dtype=np.float64)).cuda()
    b_h, e_h = blocked_mean_sem(beta3d(hmc), nb=64)
    phi2_h, phi2_he = blocked_mean_sem((hmc ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
    print(f"HMC ref (n={hmc.shape[0]}): beta_conf={b_h:.5f}({e_h:.5f})  "
          f"phi2={phi2_h:.4f}({phi2_he:.4f})", flush=True)
    json.dump(dict(b=b_h, be=e_h, phi2=phi2_h, phi2e=phi2_he),
              open(OUT / "c2_hmc_3d_L64.json", "w"))
    del hmc
    torch.cuda.empty_cache()

STEPS = [50, 100, 200, 350, 500, 700, 1000, 1400, 2000, 3000, 4000]
for sched in SCHEDS:
    rows = []
    for steps in STEPS:
        t0 = time.time()
        x = em_reverse(model, N, steps, sched, 1e-5,
                       seed=steps + (0 if sched == "log" else 7))
        p = norm_to_phys(x[:, 0].double(), hp["norm_min"], hp["norm_max"])
        b = beta3d(p)
        phi2 = (p ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        m2 = (p.mean(dim=(1, 2, 3)) ** 2).cpu().numpy()
        rows.append(dict(sched=sched, steps=steps,
                         b=b.mean(), be=b.std(ddof=1) / np.sqrt(N),
                         phi2=phi2.mean(), phi2e=phi2.std(ddof=1) / np.sqrt(N),
                         chi=L ** 3 * m2.mean(),
                         chie=L ** 3 * m2.std(ddof=1) / np.sqrt(N)))
        print(f"{sched:6s} {steps:5d}: beta_conf={b.mean():.4f}"
              f"({b.std(ddof=1) / np.sqrt(N):.4f})  phi2={phi2.mean():.4f}  "
              f"chi={L ** 3 * m2.mean():.1f}  [{time.time() - t0:.0f}s]", flush=True)

    json.dump(rows, open(OUT / f"c2_results_3d_L{L}_{sched}.json", "w"),
              indent=1, default=float)
    print(f"saved -> {OUT}/c2_results_3d_L{L}_{sched}.json")
