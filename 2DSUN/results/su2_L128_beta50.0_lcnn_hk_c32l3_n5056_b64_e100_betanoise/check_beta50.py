# Checks for a beta = 50, L = 128 checkpoint: Stein ratios on the 64 held-out configurations (the last 64 of the file) at
# several noise levels, then 64 configurations generated from Haar with the standard sampler, Wilson loops against p^{R T}.
import sys
import math
import h5py
import numpy as np
import torch
from scipy.special import ive
sys.path.insert(0, "/work/gw97/w97000/project/DM/2DSUN/src/SU2")
from train_su2_dm import brownian
from sample_su2_em import wilson, em_sample
from stein_su2 import time_of, tangent_grad, load, LOOPS, SIGMA2, BATCH

DATA = "/work/gw97/w97000/project/DM/2DSUN/trainningdata/cfgs_su2_beta=50.0_128^2.h5"
N = 64
with h5py.File(DATA) as f:
    U = f["cfgs"][-N:]
q = np.stack([U[..., 0, 0].real, U[..., 0, 1].imag, U[..., 0, 1].real, U[..., 0, 0].imag], -1)
U0 = torch.tensor(q.transpose(0, 3, 2, 1, 4), dtype=torch.float32).contiguous()
torch.manual_seed(82026)
noisy = {s: brownian(U0, torch.full((N,), s), torch.ones(N))[0] for s in SIGMA2}
grads = {(s, R, T): [tangent_grad(noisy[s][i:i + BATCH], R, T) for i in range(0, N, BATCH)] for s in SIGMA2 for R, T in LOOPS}
p = ive(2, 50.0) / ive(1, 50.0)
for path in sys.argv[1:]:
    model, ck = load(path)
    print(f"\n{path}\nepoch {ck['epoch']}, {ck['nlayers']} layers, beta {ck['beta']}, noise_beta {ck['noise_beta']}")
    with torch.no_grad():
        for s in SIGMA2:
            t = torch.full((BATCH,), time_of(s, model.noise_beta))
            S = [model(noisy[s][i:i + BATCH], t) for i in range(0, N, BATCH)]
            line = f"Stein Sigma2 {s:5.3f}"
            for R, T in LOOPS:
                num = torch.cat([(Sb * gb).sum((1, 2, 3, 4)) for Sb, (_, gb) in zip(S, grads[s, R, T])]).double()
                den = 0.75 * 2 * (R + T) * torch.cat([fb for fb, _ in grads[s, R, T]]).double()
                ratio = num.mean() / den.mean()
                line += f"   W{R}x{T} {ratio:.4f}({(num - ratio * den).std() / math.sqrt(N) / den.mean():.4f})"
            print(line, flush=True)
        torch.manual_seed(1)
        Ug = em_sample(model, N, ck["L"])
        Ug = (Ug / Ug.norm(dim=-1, keepdim=True)).double()
        line = "Haar start 2000 steps, exact vs generated:"
        for R, T in LOOPS:
            w = wilson(Ug, R, T).mean((1, 2))
            line += f"   W{R}x{T} {p ** (R * T):.4f} vs {w.mean():.4f}({w.std() / math.sqrt(N):.4f})"
        print(line, flush=True)
