# Continue the residual 13-tap model from the epoch-50 checkpoint to epoch 100: the training loop of train_su2res_dm.main
# with the network and EMA restored, a fresh Adam (the checkpoint holds no optimizer state), the EMA update count continuing
# from the checkpoint, seed 50 for the data order and the noise.
import os
import sys
import time
import torch
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, "/work/gw97/w97000/project/DM/2DSUN/src/SU2res")
import train_su2res_dm as R
from train_su2_dm import loss_fn, load_cfgs, check_equivariance

CKPT = "/work/gw97/w97000/project/DM/2DSUN/results/su2res_L128_beta200.0_lcnn_hk_c32l16d3_n5120_b64_e50_betanoise/ckpt_epoch0050.pt"
OUT = "/work/gw97/w97000/project/DM/2DSUN/results/su2res_L128_beta200.0_lcnn_hk_c32l16d3_n5120_b64_e100_betanoise"
EPOCHS = 100
SAVE_EPOCHS = [60, 70, 80, 90, 100]
SEED = 50

torch.manual_seed(SEED)
os.makedirs(OUT, exist_ok=True)
ck = torch.load(CKPT)
print(f"Continue {CKPT} (epoch {ck['epoch']}) to epoch {EPOCHS}\nOutput {OUT}\nbatch {R.BATCH}, lr {R.LR}, EMA decay {R.EMA_DECAY}, "
      f"seed {SEED}, Adam reinitialised, save epochs {SAVE_EPOCHS}", flush=True)
U_data = load_cfgs()
N = U_data.shape[0]
model = R.LCNN(ck["ch"], ck["nlayers"], ck["sigma"], noise_beta=ck["noise_beta"], dilations=ck["dilations"])
model.load_state_dict(ck["model"])
params = list(model.parameters())
ema = ExponentialMovingAverage(params, decay=R.EMA_DECAY)
ema.load_state_dict(ck["ema"])
print(f"EMA updates so far {ema.num_updates}, {sum(p.numel() for p in params)} parameters", flush=True)
opt = torch.optim.Adam(params, lr=R.LR, fused=True)
train_loss = torch.compile(loss_fn, mode="reduce-overhead", fullgraph=True, dynamic=False)
check_equivariance(model, U_data[:2])
t0 = time.time()
for epoch in range(ck["epoch"] + 1, EPOCHS + 1):
    perm = torch.randperm(N)
    tot = torch.zeros(())
    nb = 0
    for i in range(0, N - R.BATCH + 1, R.BATCH):
        torch.compiler.cudagraph_mark_step_begin()
        opt.zero_grad(set_to_none=True)
        loss = train_loss(model, U_data[perm[i:i + R.BATCH]])
        loss.backward()
        opt.step()
        ema.num_updates += 1
        decay = min(ema.decay, (1 + ema.num_updates) / (10 + ema.num_updates))
        with torch.no_grad():
            torch._foreach_lerp_(ema.shadow_params, params, 1 - decay)
        tot.add_(loss.detach())
        nb += 1
    print(f"epoch {epoch:4d}  loss {(tot / nb).item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    if epoch in SAVE_EPOCHS:
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "ch": ck["ch"], "nlayers": ck["nlayers"],
                    "dilations": ck["dilations"], "sigma": ck["sigma"], "noise_beta": ck["noise_beta"], "L": ck["L"],
                    "beta": ck["beta"], "epoch": epoch, "arch": "residual", "continued_from": CKPT},
                   f"{OUT}/ckpt_epoch{epoch:04d}.pt")
