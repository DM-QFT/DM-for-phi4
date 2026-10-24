"""Mixed-size DSM training of the particle noise predictor. Usage: python train/train_particle.py reaction|brownian|single unet|gnn <run_name>"""
import os
import sys
import json
import time
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from config import SIGMA_MIN, SIGMA_MAX, H_COARSE
from simulate import read_pairs
from model import ParticleUNet, ParticleGNN, collate

TAG, ARCH, RUN = sys.argv[1], sys.argv[2], sys.argv[3]
N_BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 1   # blocks per level, the plan's depth check
L_TRAIN = list(range(8, 33, 2))
BATCH = 26 * (12 if TAG == "single" else 1)   # events per step, equal count per size. one particle per event needs many events for a useful gradient
STEPS = 5000
LR = 1e-3
WARMUP = 100
EMA_DECAY = 0.995     # window ~200 steps, torch_ema warms the decay up from 0 by itself
SEED = 0
RUN_DIR = os.path.join(PROJECT_DIR, "runs", RUN)
CKPT_AT = set(np.unique(np.round(np.geomspace(10, STEPS, 25)).astype(int)).tolist()) | {STEPS}

torch.set_default_device("cuda")
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
data = {}
for L in L_TRAIN:
    # whole training set on the GPU as concatenated coordinates plus event offsets
    x0s, x10s = read_pairs(f"{PROJECT_DIR}/trainingdata/{TAG}_L{L}_train.h5")[:2]
    data[L] = (torch.tensor(np.concatenate(x0s)), torch.tensor(np.concatenate(x10s)),
               np.cumsum([0] + [len(a) for a in x0s]), np.cumsum([0] + [len(a) for a in x10s]))
model = {"unet": ParticleUNet, "gnn": ParticleGNN}[ARCH](n_blocks=N_BLOCKS)
# compiling the method keeps the state_dict keys unchanged. probe on this torch 2.12 build (6-event batch, 2026-10-12):
# eager 65 ms/step, default+dynamic 35 ms, reduce-overhead+dynamic segfaults in backward, reduce-overhead+static 100 ms with one padded shape
model.core = torch.compile(model.core, mode="default", dynamic=True)
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
ema = ExponentialMovingAverage(model.parameters(), decay=EMA_DECAY)
os.makedirs(f"{RUN_DIR}/models", exist_ok=True)
os.makedirs(f"{RUN_DIR}/logs", exist_ok=True)
json.dump({"tag": TAG, "arch": ARCH, "n_blocks": N_BLOCKS, "L_train": L_TRAIN, "batch": BATCH, "steps": STEPS, "lr": LR, "ema": EMA_DECAY,
           "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX, "n_params": sum(p.numel() for p in model.parameters()),
           "n_events": {L: len(d[2]) - 1 for L, d in data.items()}}, open(f"{RUN_DIR}/meta.json", "w"), indent=1)
print(f"{ARCH}: {sum(p.numel() for p in model.parameters())} parameters, events per size {[len(data[L][2]) - 1 for L in L_TRAIN]}")

log = open(f"{RUN_DIR}/logs/train.tsv", "w")
log.write("step\tloss\tsec\n")
t0 = time.time()
for step in range(1, STEPS + 1):
    # equal number of events per size, one sigma per event, grid phase random per event
    picks = [(L, k) for L in L_TRAIN for k in rng.integers(len(data[L][2]) - 1, size=BATCH // len(L_TRAIN))]
    Ls = [float(L) for L, _ in picks]
    B = len(Ls)
    pos, typ, batch = collate([data[L][0][data[L][2][k]:data[L][2][k + 1]] for L, k in picks],
                              [data[L][1][data[L][3][k]:data[L][3][k + 1]] for L, k in picks])
    isz = typ == 1
    batch_z = batch[isz]
    M = torch.bincount(batch_z, minlength=B).to(pos.dtype).clamp_min(1)   # a box emptied by voids has M = 0 and contributes nothing
    sigma = torch.exp(torch.rand(B) * (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) + np.log(SIGMA_MIN))
    eps = torch.randn(int(isz.sum()), 2)
    L_z = torch.tensor(Ls)[batch_z]
    pos[isz] = torch.remainder(pos[isz] + sigma[batch_z, None] * eps, L_z[:, None])
    pred = model(pos, typ, batch, Ls, sigma, torch.rand(B, 2) * H_COARSE)
    loss_event = torch.zeros(B).index_add_(0, batch_z, (pred - eps).pow(2).sum(-1)) / M
    loss = loss_event.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    sched.step()
    ema.update()
    log.write(f"{step}\t{loss.item():.6f}\t{time.time() - t0:.1f}\n")
    if step % 50 == 0 or step == 1:
        print(f"step {step:6d}  loss {loss.item():.4f}  {(time.time() - t0) / step * 1000:.0f} ms/step", flush=True)
        log.flush()
    if step in CKPT_AT:
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "step": step, "arch": ARCH, "tag": TAG, "n_blocks": N_BLOCKS},
                   f"{RUN_DIR}/models/step={step}.pt")
log.close()
