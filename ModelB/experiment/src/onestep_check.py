"""One-step kernel of saved checkpoints vs the training-window pairs at the same lag.
Usage: python src/onestep_check.py runs/traj_eq_dt10s [epoch ...]"""

import functools
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from diffusion_lightning import marginal_prob_std
from traj_lightning import ScoreMLP, TrajKernel, cond_features

torch.set_float32_matmul_precision("highest")
EXP_DIR = Path(__file__).resolve().parents[1]
RUN = EXP_DIR / sys.argv[1]
EPOCHS = [int(e) for e in sys.argv[2:]] or [max(int(p.stem.split("=")[1]) for p in (RUN / "models").glob("epoch=*.ckpt"))]
N_ORIGINS, NUM_STEPS, SEED = 200000, 2000, 1
COLS = ("std_x", "std_y", "mean_dx", "P_hop-1", "P_hop+1", "kurt_x", "bottom_std", "bottom_kurt", "barrier_mean", "barrier_std")


def kurt(v):
    v = v - v.mean()
    return (v ** 4).mean() / (v ** 2).mean() ** 2 - 3


def stats(d, u0, u1):
    f = (u0 + 0.5) % 1 - 0.5
    bottom, barrier = np.abs(f) < 0.05, np.abs(f) > 0.45
    dn = np.floor(u1 + 0.5) - np.floor(u0 + 0.5)
    return (d[:, 0].std(), d[:, 1].std(), d[:, 0].mean(), (dn == -1).mean(), (dn == 1).mean(), kurt(d[:, 0]),
            d[bottom, 0].std(), kurt(d[bottom, 0]), d[barrier, 0].mean(), d[barrier, 0].std())


def main():
    dev = torch.device("cuda:0")
    hp = torch.load(RUN / "models" / f"epoch={EPOCHS[0]:04d}.ckpt", map_location="cpu", weights_only=False)["hyper_parameters"]
    W, dt = hp["window_frames"], hp["dt_frames"]
    with h5py.File(hp["data_path"], "r") as h:
        g = h[hp["group"]]
        pos = g["pos"][:, :W]
        fi = g["file_index"][:]
        x0, tilt, lam = (g[k][:][fi].astype(np.float32) for k in ("x0", "tilt", "lam"))
    o = np.arange(0, W - dt, 25)
    r0, r1 = pos[:, o], pos[:, o + dt]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, r0.shape[0] * r0.shape[1], N_ORIGINS)
    par = [np.repeat(v[:, None], len(o), 1).reshape(-1)[idx] for v in (x0, tilt, lam)]
    r0, r1 = r0.reshape(-1, 2)[idx], r1.reshape(-1, 2)[idx]
    u = lambda r: (r[:, 0] - par[0]) / par[2] - par[1] * r[:, 1]
    rows = [("data", *stats(r1 - r0, u(r0), u(r1)))]
    fov = torch.tensor(hp["fov"], device=dev) if hp.get("cond_xy", False) else None
    cond = cond_features(torch.from_numpy(r0).to(dev), *(torch.from_numpy(p).to(dev) for p in par), fov, hp.get("grid", 4))
    for epoch in EPOCHS:
        ckpt = RUN / "models" / f"epoch={epoch:04d}.ckpt"
        hp = torch.load(ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
        score_model = ScoreMLP(functools.partial(marginal_prob_std, sigma=hp["sigma"]), n_cond=cond.shape[1],
                               embed_logstd=hp.get("embed_logstd", False))
        model = TrajKernel.load_from_checkpoint(ckpt, score_model=score_model, map_location=dev, weights_only=False).to(dev)
        torch.manual_seed(SEED)
        with torch.inference_mode():
            d = model.sample(cond, NUM_STEPS, schedule="log", use_ema=True).cpu().numpy()
        rows.append((f"ep{epoch}", *stats(d, u(r0), u(r0 + d))))
    print(f"{RUN.name}: {N_ORIGINS} window origins, lag {dt} frames, {NUM_STEPS} sampler steps")
    print("%-8s" % "" + "".join("%13s" % c for c in COLS))
    for name, *vals in rows:
        print("%-8s" % name + "".join("%13.4f" % v for v in vals))


if __name__ == "__main__":
    main()
