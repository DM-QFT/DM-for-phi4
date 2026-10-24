# Stein identity on SU(2)^links for the EMA score of each checkpoint given on the command line.  For f = site average of
# the R×T Wilson loop, E[S·∇f] = −E[Δf] = (3/4)·2(R+T)·E[f]; the ratio of the two sides is 1 for the exact score at the
# tested noise and falls below 1 where the network misses the long-range part of the score (loops larger than its receptive
# field).  Held-out heat-bath configurations, one exact heat-kernel draw per link, the same noisy inputs for every checkpoint.
import os
import sys
import math
import h5py
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage
from train_su2_dm import LCNN, qmul, conj, brownian, SIGMA, BETA, DELTA_MAX
from sample_su2_em import wilson

SRC = "/work/gw97/w97000/project/DM/2DSUN/trainningdata/cfgs_su2_beta=200.0_128^2.h5"
TRAIN = "/work/gw97/w97000/project/DM/2DSUN/trainningdata/cfgs_su2_beta=200.0_128^2_n5120.h5"
N_HELD = 64
SIGMA2 = [0.005, 0.02, 0.05, 0.1, 0.2]
LOOPS = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
BATCH = 16
SEED = 82026


def held_out():
    with h5py.File(TRAIN) as f:
        pid = set(f["parent_id"][:].tolist())
    ids = [i for i in range(10 * N_HELD) if i not in pid][:N_HELD]
    with h5py.File(SRC) as f:
        U = f["cfgs"][ids]
    q = np.stack([U[..., 0, 0].real, U[..., 0, 1].imag, U[..., 0, 1].real, U[..., 0, 0].imag], -1)
    return ids, torch.tensor(q.transpose(0, 3, 2, 1, 4), dtype=torch.float32).contiguous()


def time_of(Sigma2, noise_beta):
    if noise_beta is None:
        return math.log1p(2 * Sigma2 * math.log(SIGMA)) / (2 * math.log(SIGMA))
    return math.log1p(noise_beta * Sigma2) / math.log1p(noise_beta * (SIGMA**2 - 1) / (2 * math.log(SIGMA)))


def tangent_grad(U, R, T):
    # ∂f/∂ε^a under U_μ(x) → exp(iεσ^a/2) U_μ(x) is the vector part of ½ g Ū with g = ∂f/∂U the Euclidean gradient
    U = U.detach().requires_grad_(True)
    f = wilson(U, R, T).mean((1, 2))
    g, = torch.autograd.grad(f.sum(), U)
    return f.detach(), 0.5 * qmul(g, conj(U.detach()))[..., 1:]


def load(path):
    ck = torch.load(path)
    if ck.get("arch") == "residual":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SU2res"))
        from train_su2res_dm import LCNN as Residual
        model = Residual(ck["ch"], ck["nlayers"], ck["sigma"], noise_beta=ck.get("noise_beta"), dilations=ck["dilations"],
                         skip=ck.get("skip", True))
    else:
        model = LCNN(ck["ch"], ck["nlayers"], ck["sigma"], noise_beta=ck.get("noise_beta"))
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    ema.load_state_dict(ck["ema"])
    ema.copy_to(model.parameters())
    return model.eval(), ck


def main():
    torch.manual_seed(SEED)
    ids, U0 = held_out()
    print(f"{N_HELD} held-out heat-bath configurations, source ids {ids[0]} to {ids[-1]}, none in the training subset")
    assert all(s <= DELTA_MAX for s in SIGMA2)
    noisy = {}
    for s in SIGMA2:
        Ut, _ = brownian(U0, torch.full((N_HELD,), s), torch.ones(N_HELD))
        noisy[s] = Ut
    grads = {(s, R, T): [tangent_grad(noisy[s][i:i + BATCH], R, T) for i in range(0, N_HELD, BATCH)] for s in SIGMA2 for R, T in LOOPS}
    for path in sys.argv[1:]:
        model, ck = load(path)
        print(f"\n{path}\nepoch {ck['epoch']}, {ck['nlayers']} layers, {ck['ch']} channels")
        with torch.no_grad():
            for s in SIGMA2:
                t = torch.full((BATCH,), time_of(s, model.noise_beta))
                S = [model(noisy[s][i:i + BATCH], t) for i in range(0, N_HELD, BATCH)]
                line = f"Sigma2 {s:5.3f}"
                for R, T in LOOPS:
                    num = torch.cat([(Sb * gb).sum((1, 2, 3, 4)) for Sb, (_, gb) in zip(S, grads[s, R, T])]).double()
                    den = 0.75 * 2 * (R + T) * torch.cat([fb for fb, _ in grads[s, R, T]]).double()
                    ratio = num.mean() / den.mean()
                    err = (num - ratio * den).std() / math.sqrt(N_HELD) / den.mean()
                    line += f"   W{R}x{T} {ratio:.4f}({err:.4f})"
                print(line, flush=True)


if __name__ == "__main__":
    main()
