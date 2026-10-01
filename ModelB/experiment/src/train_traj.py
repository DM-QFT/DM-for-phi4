"""Train the displacement kernel at lag DT frames on the first 10 min of every track.
The whole training step (batch draw, noise, loss, backward, Adam, EMA) replays as one CUDA graph.
Usage: python src/train_traj.py eq|driven"""

import functools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from diffusion_lightning import marginal_prob_std
from traj_lightning import ScoreMLP, TrajKernel, TrajPairData, cond_features

# bf16/TF32 quantise a normalised input of 0.1 at 2e-4 to 5e-4, comparable to std(eps) = 3e-3
torch.set_float32_matmul_precision("highest")

EXP_DIR = Path(__file__).resolve().parents[1]
GROUP = sys.argv[1]
DT, WINDOW = 250, 15000
SIGMA, LR, LR_MIN, EMA_DECAY = 12.0, 1e-3, 1e-5, 0.9999
BATCH, STEPS_PER_EPOCH, EPOCHS, NUM_CKPTS = 8192, 100, 10000, 50
T_LOG, EMBED_LOGSTD = False, False       # ablation in README 3.6: log-uniform t sharpens the tails but the sampler diverged once
COND_XY, GRID = False, 4                 # condition also on the 4x4 field-of-view cell of r: the beam envelope makes the barrier depend on the position
DATA = EXP_DIR / "data/trajectories.h5"
RUN_DIR = EXP_DIR / "runs" / (f"traj_{GROUP}_dt{DT // 25}s" + "_tlog" * T_LOG + "_logstd" * EMBED_LOGSTD
                              + f"_g{GRID}" * COND_XY + f"_ep{EPOCHS}" * (EPOCHS != 1000))


def save_checkpoint(model, epoch, step, path):
    """Same keys Lightning writes, so TrajKernel.load_from_checkpoint and on_load_checkpoint work unchanged."""
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "ema_state_dict": model.ema.state_dict(), "epoch": epoch - 1, "global_step": step,
                "pytorch-lightning_version": pl.__version__}, path)


def main():
    data = TrajPairData(DATA, GROUP, DT, WINDOW, BATCH, STEPS_PER_EPOCH)
    data.setup()
    (RUN_DIR / "models").mkdir(parents=True, exist_ok=False)
    meta = dict(group=GROUP, dt_frames=DT, window_frames=WINDOW, sigma=SIGMA, lr=LR, lr_min=LR_MIN,
                ema_decay=EMA_DECAY, batch=BATCH, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                t_log=T_LOG, embed_logstd=EMBED_LOGSTD, cond_xy=COND_XY, grid=GRID,
                A_res=data.A_res, D_max_norm=data.D_max_norm, N=int(data.pos.shape[0]), data=str(DATA))
    (RUN_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    dev = data.device
    fov = data.fov if COND_XY else None
    score_model = ScoreMLP(functools.partial(marginal_prob_std, sigma=SIGMA), n_cond=4 if COND_XY else 2,
                           embed_logstd=EMBED_LOGSTD)
    model = TrajKernel(score_model, sigma=SIGMA, lr=LR, lr_min=LR_MIN, ema_decay=EMA_DECAY,
                       norm_min=-data.A_res, norm_max=data.A_res, group=GROUP, dt_frames=DT,
                       window_frames=WINDOW, data_path=str(DATA), run_dir=str(RUN_DIR),
                       t_log=T_LOG, embed_logstd=EMBED_LOGSTD, cond_xy=COND_XY,
                       fov=data.fov.tolist(), grid=GRID).to(dev)
    model.ema.to(dev)
    model.train()
    params = list(model.score_model.parameters())            # same order as model.ema.shadow_params
    shadow = model.ema.shadow_params
    lr_t = torch.tensor(LR, device=dev)                       # read inside the graph, updated once per epoch
    decay_t, one_minus_t = torch.tensor(0.0, device=dev), torch.tensor(1.0, device=dev)   # EMA weights, same
    opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=lr_t, fused=True, capturable=True)
    pos, par, A_res = data.pos, data.par, data.A_res
    N, W = pos.shape[:2]

    def step():
        i = torch.randint(N, (BATCH,), device=dev)
        t0 = torch.randint(W - DT, (BATCH,), device=dev)
        r0 = pos[i, t0]
        loss = model.loss_fn((pos[i, t0 + DT] - r0) / A_res, cond_features(r0, *(p[i] for p in par), fov, GRID))
        loss.backward()
        opt.step()
        torch._foreach_mul_(shadow, decay_t)
        torch._foreach_add_(shadow, torch._foreach_mul(params, one_minus_t))
        return loss.detach()

    # whole-network capture (PyTorch CUDA-graph notes): warm up on a side stream, grads set to None before capture
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        static_loss = step()

    std1 = float(marginal_prob_std(torch.tensor(1.0), SIGMA))
    print(f"run: {RUN_DIR}\nparams: {sum(p.numel() for p in params)}\n"
          f"sigma={SIGMA}: std(1)={std1:.2f} = {std1 / data.D_max_norm:.2f} D_max_norm", flush=True)
    save_epochs = set(np.unique(np.geomspace(1, EPOCHS, num=NUM_CKPTS).astype(int)).tolist())
    loss_sum = torch.zeros((), device=dev)
    start = time.perf_counter()
    with open(RUN_DIR / "loss.tsv", "w") as f:
        f.write("epoch\tlr\tloss\n")
        for epoch in range(1, EPOCHS + 1):
            lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * (epoch - 1) / EPOCHS))
            lr_t.fill_(lr)
            n = (epoch - 1) * STEPS_PER_EPOCH                  # torch_ema warm-up, so early checkpoints track the weights
            decay_t.fill_(min(EMA_DECAY, (1 + n) / (10 + n)))
            one_minus_t.fill_(1 - min(EMA_DECAY, (1 + n) / (10 + n)))
            for _ in range(STEPS_PER_EPOCH):
                graph.replay()
                loss_sum += static_loss
            f.write(f"{epoch}\t{lr:.4e}\t{loss_sum.item() / STEPS_PER_EPOCH:.6f}\n")
            loss_sum.zero_()
            if epoch in save_epochs:
                save_checkpoint(model, epoch, epoch * STEPS_PER_EPOCH, RUN_DIR / "models" / f"epoch={epoch:04d}.ckpt")
            if epoch % max(EPOCHS // 10, 1) == 0:
                print(f"epoch {epoch}/{EPOCHS}: {time.perf_counter() - start:.0f} s", flush=True)
    print("Training complete!")


if __name__ == "__main__":
    main()
