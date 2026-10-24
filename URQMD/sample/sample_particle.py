"""Reverse VE diffusion (stochastic Heun, shared noise) of a whole test set at once with the true future particle number M.
Usage: python sample/sample_particle.py <run> <step> <L> <split> <K>"""
import os
import sys
import time
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from config import SIGMA_MIN, SIGMA_MAX, H_COARSE
from simulate import read_pairs, write_pairs
from model import ParticleUNet, ParticleGNN, collate

RUN, STEP, L, SPLIT, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
SEED = 1
RUN_DIR = os.path.join(PROJECT_DIR, "runs", RUN)

torch.set_default_device("cuda")
ckpt = torch.load(f"{RUN_DIR}/models/step={STEP}.pt")
model = {"unet": ParticleUNet, "gnn": ParticleGNN}[ckpt["arch"]](n_blocks=ckpt.get("n_blocks", 1))
model.load_state_dict(ckpt["model"])
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
ema.load_state_dict(ckpt["ema"])
ema.copy_to(model.parameters())
model.eval()
# static shapes: the sizes are known at compile time, so inductor picks 64-bit indexing for the L=64 message tensors (> 2^31 bytes),
# which the dynamic-shape kernels got wrong. the padding buckets keep the number of distinct shapes small
model.core = torch.compile(model.core, mode="default", dynamic=False)
torch.manual_seed(SEED)

x0s, x10s, _, traj, k, init = read_pairs(f"{PROJECT_DIR}/trainingdata/{ckpt['tag']}_L{L}_{SPLIT}.h5")
B, Ls = len(x0s), [float(L)] * len(x0s)
pos, typ, batch = collate(x0s, [np.zeros((len(x), 2), np.float32) for x in x10s])
isz = typ == 1
batch_z = batch[isz]
sigmas = torch.tensor(np.geomspace(SIGMA_MAX, SIGMA_MIN, K + 1), dtype=torch.float32)
z = torch.rand(int(isz.sum()), 2) * L
t0 = time.time()
with torch.no_grad():
    for step, (hi, lo) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        # predictor at sigma_hi, corrector at sigma_lo, same Gaussian increment in both (stochastic Heun in the variance variable)
        dv = hi * hi - lo * lo
        xi = dv.sqrt() * torch.randn_like(z)
        pos[isz] = z
        e1 = model(pos, typ, batch, Ls, hi.expand(B), torch.rand(B, 2) * H_COARSE)
        pos[isz] = torch.remainder(z - dv / hi * e1 + xi, L)
        e2 = model(pos, typ, batch, Ls, lo.expand(B), torch.rand(B, 2) * H_COARSE)
        z = torch.remainder(z - dv / 2 * (e1 / hi + e2 / lo) + xi, L)
        if step % 32 == 0:
            print(f"step {step}/{K}  sigma {hi.item():.3g}  {time.time() - t0:.0f} s", flush=True)
z = z.cpu().numpy()
batch_z = batch_z.cpu().numpy()
os.makedirs(f"{RUN_DIR}/samples", exist_ok=True)
write_pairs(f"{RUN_DIR}/samples/{ckpt['tag']}_L{L}_{SPLIT}_step{STEP}_K{K}.h5", x0s, [z[batch_z == b] for b in range(B)], L, traj, k, init)
