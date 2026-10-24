"""DSM test loss per log-sigma bin with the true X0 and with the X0 of another event. The gap is the conditional information the network uses.
Usage: python analysis/conditional/loss_bins.py <run> <step> <L> <split>"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
import modelb_palette as pal
from config import SIGMA_MIN, SIGMA_MAX, H_COARSE
from simulate import read_pairs
from model import ParticleUNet, ParticleGNN, collate

RUN, STEP, L, SPLIT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
N_BINS = 12
OUT = os.path.join(PROJECT_DIR, "results", "conditional")
plt.style.use(os.path.join(PROJECT_DIR, "cjp_prl_nature.mplstyle"))

torch.set_default_device("cuda")
ckpt = torch.load(f"{PROJECT_DIR}/runs/{RUN}/models/step={STEP}.pt")
model = {"unet": ParticleUNet, "gnn": ParticleGNN}[ckpt["arch"]](n_blocks=ckpt.get("n_blocks", 1))
model.load_state_dict(ckpt["model"])
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
ema.load_state_dict(ckpt["ema"])
ema.copy_to(model.parameters())
model.eval()
torch.manual_seed(0)

x0s, x10s, _, _, _, _ = read_pairs(f"{PROJECT_DIR}/trainingdata/{ckpt['tag']}_L{L}_{SPLIT}.h5")
B, Ls = len(x0s), [float(L)] * len(x0s)
sig = np.geomspace(SIGMA_MIN, SIGMA_MAX, N_BINS)
perm = np.random.default_rng(0).permutation(B)   # consecutive events belong to one trajectory, so a shift by one would hand over the clean x10
curves = {}
with torch.no_grad():
    for lab, cond in (("true", x0s), ("shuffled", [x0s[p] for p in perm])):
        pos, typ, batch = collate(cond, x10s)
        isz = typ == 1
        batch_z = batch[isz]
        M = torch.bincount(batch_z, minlength=B).to(pos.dtype).clamp_min(1)
        x10 = pos[isz].clone()
        losses = []
        for sigma in sig:
            torch.manual_seed(0)      # same noise for both conditions
            eps = torch.randn(x10.shape)
            pos[isz] = torch.remainder(x10 + sigma * eps, L)
            pred = model(pos, typ, batch, Ls, torch.full((B,), float(sigma)), torch.rand(B, 2) * H_COARSE)
            losses.append((torch.zeros(B).index_add_(0, batch_z, (pred - eps).pow(2).sum(-1)) / M).mean().item())
        curves[lab] = np.array(losses)
os.makedirs(OUT, exist_ok=True)
name = f"loss_bins_{ckpt['tag']}_L{L}_{SPLIT}_{RUN}_step{STEP}"
np.savetxt(f"{OUT}/{name}.tsv", np.column_stack([sig, curves["true"], curves["shuffled"]]), delimiter="\t", fmt="%.6g", header="sigma loss_true_x0 loss_shuffled_x0")
for sg, a, b in zip(sig, curves["true"], curves["shuffled"]):
    print(f"sigma {sg:8.3g}  loss true x0 {a:.3f}  shuffled x0 {b:.3f}  gap {b - a:+.3f}")
fig, ax = plt.subplots()
ax.axhline(2, color=pal.ZERO, linewidth=0.8, zorder=0)
ax.plot(sig, curves["true"], color=pal.DM, label="true $X_0$")
ax.plot(sig, curves["shuffled"], color=pal.ALT, label="shuffled $X_0$")
ax.set_xscale("log")
ax.set_xlabel(r"$\sigma / r$")
ax.set_ylabel(r"$\langle |\hat\epsilon - \epsilon|^2 \rangle$")
ax.legend()
fig.savefig(f"{OUT}/{name}.pdf")
