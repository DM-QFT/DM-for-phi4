"""Iterate the learned kernel from the true positions at t = 10 min to the end of the record.
Usage: python src/rollout_traj.py runs/traj_eq_dt10s"""

import functools
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from diffusion_lightning import marginal_prob_std
from traj_lightning import ScoreMLP, TrajKernel, cond_features

torch.set_float32_matmul_precision("highest")

EXP_DIR = Path(__file__).resolve().parents[1]
RUN = EXP_DIR / sys.argv[1]
EPOCH = max(int(p.stem.split("=")[1]) for p in (RUN / "models").glob("epoch=*.ckpt"))   # numeric, not lexicographic
R, NUM_STEPS, SCHEDULE, SEED0 = 32, 2000, "log", 20260914


def main():
    ckpt = RUN / "models" / f"epoch={EPOCH:04d}.ckpt"
    out = RUN / "samples" / f"rollout_ep{EPOCH}_R{R}.h5"
    assert not out.exists(), out
    hp = torch.load(ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
    device = torch.device("cuda:0")
    cond_xy = hp.get("cond_xy", False)
    score_model = ScoreMLP(functools.partial(marginal_prob_std, sigma=hp["sigma"]), n_cond=4 if cond_xy else 2,
                           embed_logstd=hp.get("embed_logstd", False))
    model = TrajKernel.load_from_checkpoint(
        ckpt, score_model=score_model, map_location=device, weights_only=False
    ).to(device).eval()

    dt, origin = hp["dt_frames"], hp["window_frames"] - 1
    with h5py.File(hp["data_path"], "r") as h:
        g = h[hp["group"]]
        T = g["pos"].shape[1]
        fov, dt_s = h.attrs["fov_um"], float(h.attrs["dt_s"])
        r0 = torch.from_numpy(g["pos"][:, origin]).to(device)
        fi = g["file_index"][:].astype(np.int64)
        par = [torch.from_numpy(g[k][:].astype(np.float32))[fi].to(device).repeat(R)
               for k in ("x0", "tilt", "lam")]
    fov_t = torch.tensor(fov, dtype=torch.float32, device=device) if cond_xy else None
    N = r0.shape[0]
    K = (T - 1 - origin) // dt
    r = r0.repeat(R, 1)                     # row = replica * N + track
    print(f"{ckpt}\nN={N} R={R} K={K} steps of {dt} frames, A_res={hp['norm_max']:.4f}", flush=True)

    out.parent.mkdir(exist_ok=True)
    h = h5py.File(out, "w")
    h.create_dataset("r_gen", (K, R, N, 2), dtype="f4", chunks=(1, R, N, 2))
    h["r0"] = r0.cpu().numpy()
    for name, dtype in (("times", "f8"), ("seeds", "i8"), ("sampling_seconds", "f8")):
        h.create_dataset(name, (K,), dtype=dtype)
    h.attrs.update(checkpoint=str(ckpt), epoch=EPOCH, group=hp["group"], sigma=hp["sigma"],
                   A_res=hp["norm_max"], num_steps=NUM_STEPS, schedule=SCHEDULE, R=R,
                   origin_index=origin, dt_frames=dt, layout="r_gen[step, replica, track, xy]")
    with torch.inference_mode(), model.ema.average_parameters():
        for k in range(K):
            torch.manual_seed(SEED0 + k)
            torch.cuda.manual_seed_all(SEED0 + k)
            start = time.perf_counter()
            r = r + model.sample(cond_features(r, *par, fov_t, hp.get("grid", 4)), NUM_STEPS, schedule=SCHEDULE, use_ema=False)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            h["r_gen"][k] = r.view(R, N, 2).cpu().numpy()
            h["times"][k] = dt_s * (origin + dt * (k + 1))
            h["seeds"][k] = SEED0 + k
            h["sampling_seconds"][k] = seconds
            h.flush()
            outside = ((r[:, 0] < 0) | (r[:, 0] > fov[0]) | (r[:, 1] < 0) | (r[:, 1] > fov[1])).float().mean()
            print(f"step {k + 1}/{K}: {seconds:.1f} s, <dx>={float((r[:, 0] - r0.repeat(R, 1)[:, 0]).mean()):+.3f} um, "
                  f"outside FOV {float(outside):.3f}", flush=True)
    h.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
