"""Single free particle: network eps_hat versus the exact wrapped-Gaussian value as a function of |z - x0| at several sigma.
Usage: python analysis/brownian/single_particle_score.py <run> <step>"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
import modelb_palette as pal
from config import D, DT_PAIR
from model import ParticleUNet, ParticleGNN, collate

RUN, STEP = sys.argv[1], int(sys.argv[2])
L = 16.0
SIGMAS = [0.3, 1.0, 3.0, 10.0]
N_EV = 256                       # z positions per sigma, each its own event with x0 at the box centre
OUT = os.path.join(PROJECT_DIR, "results", "brownian")
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


def exact(dz, sigma):
    # score of the wrapped Gaussian with variance sigma^2 + 2 D DT_PAIR per component, eps_hat = -sigma * score
    var = sigma ** 2 + 2 * D * DT_PAIR
    w = np.stack([np.exp(-(dz + n * L) ** 2 / (2 * var)) for n in range(-3, 4)])
    shifted = np.stack([dz + n * L for n in range(-3, 4)])
    return sigma * (w * shifted).sum(0) / (var * w.sum(0))


os.makedirs(OUT, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.06))
rows = []
for ax, sigma, lab in zip(axes.ravel(), SIGMAS, "abcd"):
    x0 = np.full((N_EV, 1, 2), L / 2, np.float32)
    dz = np.linspace(-L / 2, L / 2, N_EV, endpoint=False).astype(np.float32)
    z = np.stack([L / 2 + dz, np.full(N_EV, L / 2, np.float32)], 1)[:, None, :]    # along x through x0
    pos, typ, batch = collate(list(x0), list(z))
    with torch.no_grad():
        pred = model(pos, typ, batch, [L] * N_EV, torch.full((N_EV,), sigma), torch.zeros(N_EV, 2))
    ex = exact(dz.astype(np.float64), sigma)
    ax.plot(dz, ex, color=pal.REFERENCE, marker="", label="exact")
    ax.plot(dz, pred[:, 0].cpu().numpy(), color=pal.DM, linestyle="none", marker="o", markersize=2.5, label="network")
    ax.plot(dz, pred[:, 1].cpu().numpy(), color=pal.ALT, linestyle="none", marker="s", markersize=2.5, label="network, $y$")
    ax.set_xlabel(r"$z_x - x_{0,x}$")
    ax.set_ylabel(r"$\hat\epsilon_x$")
    ax.text(0.04, 0.9, f"({lab}) $\\sigma={sigma}$", transform=ax.transAxes)
    err = np.sqrt(np.mean((pred[:, 0].cpu().numpy() - ex) ** 2))
    rows.append((sigma, err, np.sqrt(np.mean(ex ** 2))))
    print(f"sigma {sigma}: rms error {err:.3f}  rms exact {np.sqrt(np.mean(ex ** 2)):.3f}")
axes[0, 0].legend(loc="lower right")
fig.savefig(f"{OUT}/single_particle_score_{RUN}_step{STEP}.pdf")
np.savetxt(f"{OUT}/single_particle_score_{RUN}_step{STEP}.tsv", np.array(rows), delimiter="\t", fmt="%.6g", header="sigma rms_err rms_exact")
