"""Probe which combination of (kinematic, potential) reproduces a high cos in
the mid-epoch region. cos is sigma-independent, so this is a clean test of the
combination rule (and, indirectly, of the input normalisation)."""
import sys, functools
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[4]))
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))

import numpy as np, torch, h5py
from diffusion_lightning import marginal_prob_std
from networks import ScoreNet
import loss_eval as le

DEV = "cuda:0"; K, LAM, L = 0.26, 0.022, 16; T = 1e-4; N = 512
CK = str(HERE.parent / "ckpt_L16_k0.26_l0.022.pth")
DATA = str(HERE.parents[3] / f"trainingdata/cfgs_wolff_fahmc_k={K}_l={LAM}_{L}^2.jld2")

ck = torch.load(CK, map_location="cpu", weights_only=False)
epochs = sorted(ck.keys(), key=lambda s: int(s.split("_")[1]))
strip = lambda sd: {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
mp = functools.partial(marginal_prob_std, sigma=25.0)

kin = ScoreNet(mp).to(DEV).eval()
pot = ScoreNet(mp).to(DEV).eval()

with h5py.File(DATA, "r") as f:
    cfgs = np.asarray(f["cfgs"])
pmin, pmax = float(cfgs.min()), float(cfgs.max())
idx = np.random.default_rng(42).choice(cfgs.shape[0], N, replace=False)
xn = ((cfgs[idx] - pmin) / (pmax - pmin) - 0.5) * 2.0
x = torch.tensor(xn, dtype=torch.float32, device=DEV)[:, None]
target = le.phi4_score(x, K, LAM, phi_min=pmin, phi_max=pmax, spatial_dims=2)

def cos_of(s):
    sm = s.reshape(N, -1); st = target.reshape(N, -1)
    c = (sm * st).sum(1) / (sm.norm(dim=1) * st.norm(dim=1) + 1e-12)
    return c.mean().item()

sel = epochs[::8] + [epochs[-1]]
print(f"norm [{pmin:.3f},{pmax:.3f}]   probing {len(sel)} epochs\n")
print(f"{'epoch':>6} {'kin':>8} {'pot':>8} {'kin+pot':>8} {'kin-pot':>8} {'pot-kin':>8}")
with torch.no_grad():
    for key in sel:
        sd = strip(ck[key])
        kin.load_state_dict({k[len('kinematic.'):]: v for k, v in sd.items() if k.startswith('kinematic.')}, strict=True)
        pot.load_state_dict({k[len('potential.'):]: v for k, v in sd.items() if k.startswith('potential.')}, strict=True)
        sk = kin(x, torch.full((N,), T, device=DEV))
        sp = pot(x, torch.full((N,), T, device=DEV))
        e = int(key.split('_')[1])
        print(f"{e:>6} {cos_of(sk):8.4f} {cos_of(sp):8.4f} {cos_of(sk+sp):8.4f} "
              f"{cos_of(sk-sp):8.4f} {cos_of(sp-sk):8.4f}")
