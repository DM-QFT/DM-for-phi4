"""Assume score = kinematic + potential and reproduce cos/mag vs epoch for the
collaborator's ckpt_L16_k0.26_l0.022.pth, using his own loss_eval.py metrics.

- Each snapshot stores TWO ScoreNet-topology sub-nets (kinematic, potential),
  DataParallel-wrapped ('module.' prefix).
- We recover the exact channels/embed_dim from the weights, build a wrapper
  whose forward returns kinematic(x,t)+potential(x,t), and load each epoch.
- cos_sim is INDEPENDENT of the (unknown) VE sigma; mag_ratio/rel_mse absolute
  scale depends on sigma, so treat those as trend-only here.
- phi_min/phi_max assumed = global min/max of the k=0.26 dataset (nm1).
"""
import sys, functools
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
ROOT2D = HERE.parents[3]
sys.path.insert(0, str(REPO))          # networks, diffusion_lightning
sys.path.insert(0, str(ROOT2D))
sys.path.insert(0, str(HERE.parent))   # loss_eval

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_lightning import marginal_prob_std
from networks import ScoreNet
import loss_eval as le

DEVICE = "cuda:0"
K, LAM, L = 0.26, 0.022, 16
T_EVAL = 1e-4
N = 512
SIGMA = 25.0   # placeholder; cos is sigma-independent

CKPT = str(HERE.parent / "ckpt_L16_k0.26_l0.022.pth")
DATA = str(ROOT2D / f"trainingdata/cfgs_wolff_fahmc_k={K}_l={LAM}_{L}^2.jld2")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
epochs = sorted(ck.keys(), key=lambda s: int(s.split("_")[1]))
ep_int = [int(s.split("_")[1]) for s in epochs]
print(f"{len(epochs)} snapshots: {ep_int[0]}..{ep_int[-1]} step {ep_int[1]-ep_int[0]}")


def strip_dp(sd):
    return {(k[len('module.'):] if k.startswith('module.') else k): v
            for k, v in sd.items()}


sd0 = strip_dp(ck[epochs[0]])
channels = [sd0[f"kinematic.conv{i}.weight"].shape[0] for i in (1, 2, 3, 4)]
embed_dim = sd0["kinematic.dense1.dense.weight"].shape[1]
print(f"recovered ScoreNet: channels={channels}  embed_dim={embed_dim}")

mp = functools.partial(marginal_prob_std, sigma=SIGMA)


class KinPot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kinematic = ScoreNet(mp, channels=channels, embed_dim=embed_dim)  # default periodic=False (exact match)
        self.potential = ScoreNet(mp, channels=channels, embed_dim=embed_dim)

    def forward(self, x, t):
        return self.kinematic(x, t) + self.potential(x, t)


model = KinPot().to(DEVICE).eval()
res = model.load_state_dict(sd0, strict=False)
print(f"epoch0 strict-load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
if res.missing_keys or res.unexpected_keys:
    print("  missing[:5]   :", res.missing_keys[:5])
    print("  unexpected[:5]:", res.unexpected_keys[:5])
    sys.exit("architecture mismatch -- aborting before sweep")

# --- reference data, normalised to [-1,1] by GLOBAL min/max (nm1 assumption) ---
with h5py.File(DATA, "r") as f:
    cfgs = np.asarray(f["cfgs"])
phi_min, phi_max = float(cfgs.min()), float(cfgs.max())
idx = np.random.default_rng(42).choice(cfgs.shape[0], size=N, replace=False)
xn = ((cfgs[idx] - phi_min) / (phi_max - phi_min) - 0.5) * 2.0
x_ref = torch.tensor(xn, dtype=torch.float32, device=DEVICE)
if x_ref.ndim == 3:
    x_ref = x_ref.unsqueeze(1)
print(f"data global norm range [{phi_min:.4f}, {phi_max:.4f}]  x_ref {tuple(x_ref.shape)}")

grad_S_fn = functools.partial(le.phi4_grad_S, k=K, l=LAM,
                              phi_min=phi_min, phi_max=phi_max, spatial_dims=2)

cos = np.zeros(len(epochs)); mse = np.zeros(len(epochs)); mag = np.zeros(len(epochs))
print(f"\n{'epoch':>6} {'cos':>8} {'rel_mse':>10} {'mag':>8}")
for i, key in enumerate(epochs):
    model.load_state_dict(strip_dp(ck[key]), strict=True)
    m = le.one_score_quality(model, grad_S_fn, x_ref, T_EVAL)
    cos[i], mse[i], mag[i] = m['cos_sim'][0], m['rel_mse'][0], m['mag_ratio'][0]
    print(f"{ep_int[i]:>6} {cos[i]:8.4f} {mse[i]:10.4f} {mag[i]:8.4f}")

# --- plot + save ---
outdir = HERE.parent / "results"; outdir.mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
e = np.array(ep_int)
for a, (v, lbl) in zip(ax, [(cos, 'cosine similarity'),
                            (mse, 'relative MSE'), (mag, 'magnitude ratio')]):
    a.plot(e, v, 'o-', ms=3); a.set_xscale('log')
    a.set_xlabel('epoch'); a.set_ylabel(lbl); a.grid(alpha=0.3)
ax[2].axhline(1.0, color='gray', ls='--', lw=0.8)
fig.suptitle(f'score = kinematic + potential  (L16, k=0.26, t={T_EVAL:.0e})', y=1.02)
fig.tight_layout()
pdf = outdir / "eval_kinpot_L16_k0.26.pdf"
fig.savefig(pdf, dpi=150, bbox_inches='tight')
np.savez(str(pdf).replace('.pdf', '.npz'),
         epochs=e, cos_sim=cos, rel_mse=mse, mag_ratio=mag,
         channels=channels, embed_dim=embed_dim,
         phi_min=phi_min, phi_max=phi_max, t_eval=T_EVAL)
print(f"\nsaved {pdf}")
print(f"cos: max={cos.max():.4f} @ epoch {e[cos.argmax()]};  final={cos[-1]:.4f} @ epoch {e[-1]}")
