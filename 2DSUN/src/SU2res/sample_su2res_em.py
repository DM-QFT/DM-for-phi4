# Reverse Euler–Maruyama sampling for the residual model of train_su2res_dm.py with the sampler of ../SU2/sample_su2_em.py
# (Haar start at t = 1, N_STEPS uniform steps to EPS, drift-only last step).  Saves the configurations, the average
# plaquette and the R×R Wilson loops per configuration, and prints them against the infinite-volume values p^{R²}.
import os
import sys
import h5py
import numpy as np
import torch
from scipy.special import ive
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SU2"))
import sample_su2_em as sampler
from sample_su2_em import em_sample, wilson, to_matrix
from train_su2res_dm import LCNN, OUT, EPOCHS

CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/ckpt_epoch{EPOCHS:04d}.pt"
N_CFGS = 64
LOOPS = [1, 2, 4, 8, 16]


def main():
    torch.manual_seed(sampler.SEED)
    ck = torch.load(CKPT)
    model = LCNN(ck["ch"], ck["nlayers"], ck["sigma"], noise_beta=ck.get("noise_beta"), dilations=ck["dilations"])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    ema.load_state_dict(ck["ema"])
    ema.copy_to(model.parameters())
    model.eval()

    U = em_sample(model, N_CFGS, ck["L"])
    unit = (U.norm(dim=-1) - 1).abs().max()
    U = U / U.norm(dim=-1, keepdim=True)
    print(f"epoch {ck['epoch']}, {N_CFGS} configs, {sampler.N_STEPS} EM steps, unitarity drift before projection {unit:.1e}")
    P = ive(2, ck["beta"]) / ive(1, ck["beta"])
    W = {R: wilson(U.double(), R, R).mean((1, 2)) for R in LOOPS}
    for R in LOOPS:
        print(f"W{R}x{R}   exact {P ** (R * R):.5f}   generated {W[R].mean():.5f} ± {W[R].std() / np.sqrt(N_CFGS):.5f}")

    out = f"{os.path.dirname(CKPT)}/samples_epoch{ck['epoch']:04d}_steps{sampler.N_STEPS}.h5"
    with h5py.File(out, "w") as f:
        f["cfgs"] = to_matrix(U)
        f["plaq"] = W[1].cpu().numpy()
        f["wilson"] = torch.stack([W[R] for R in LOOPS], 1).cpu().numpy()
        f["wilson"].attrs["R"] = LOOPS
        for k, v in (("beta", ck["beta"]), ("L", ck["L"]), ("sigma", ck["sigma"]), ("noise_beta", ck["noise_beta"]),
                     ("epoch", ck["epoch"]), ("nlayers", ck["nlayers"]), ("dilations", ck["dilations"]),
                     ("n_steps", sampler.N_STEPS), ("eps", sampler.EPS), ("seed", sampler.SEED), ("plaq_exact", P)):
            f.attrs[k] = v
    print(f"saved {out}")


if __name__ == "__main__":
    main()
