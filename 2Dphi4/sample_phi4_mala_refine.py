"""
MALA + global-shift + embedded-cluster refinement of 2D phi^4 DM ensembles.

2D port of 3Dphi4/sample_phi4_crossL_mala.py (same kernels, same exactness
arguments):
  * MALA:    y = x + h s_theta(x,t_mh) + sqrt(2h) xi,  h = c/L^2,
             exact-action MH ratio in float64 outside autocast.
  * shift:   phi -> phi + delta*1, delta ~ N(0, shift_std^2) (physical),
             alpha = min(1, e^{-dS}).
  * cluster: embedded Swendsen-Wang (freeze aligned bonds with
             p = 1-exp(-4k phi_x phi_y), flip cluster signs with prob 1/2).

Run from 2Dphi4/:
    python sample_phi4_mala_refine.py --k 0.2705 \
        --run_dir runs/phi4_L128_k0.2705_l0.022_ncsnpp_sigma450 \
        --init_npy <...>.npy --c 3 --n_steps 2000 \
        --shift_every 2 --shift_std 0.513 --cluster_every 10
"""

import sys
sys.path.append("..")

import os
import argparse
import functools
from pathlib import Path

import numpy as np
import torch

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel
from diffusion_lightning import marginal_prob_std
from phi4_action import phi4_action, phi4_grad_S


def parse_args():
    p = argparse.ArgumentParser(description="2D MALA+shift+cluster refinement")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--ep", type=int, default=10000)
    p.add_argument("--init_npy", type=str, required=True,
                   help="Initial ensemble .npy, (L,L,N) or (N,L,L), physical")
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--drift", type=str, default="score", choices=["score", "exact"])
    p.add_argument("--t_mh", type=float, default=1e-4)
    p.add_argument("--c", type=float, default=3.0, help="Step size h = c / L^2")
    p.add_argument("--shift_every", type=int, default=0)
    p.add_argument("--shift_std", type=float, default=None)
    p.add_argument("--cluster_every", type=int, default=0)
    p.add_argument("--n_steps", type=int, required=True)
    p.add_argument("--max_cfgs", type=int, default=None)
    p.add_argument("--microbatch", type=int, default=1024)
    p.add_argument("--n_snapshots", type=int, default=12)
    p.add_argument("--state_every", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    if args.shift_every > 0 and args.shift_std is None:
        raise SystemExit("--shift_std required when --shift_every > 0")
    device = torch.device(args.device)
    L, V = args.L, args.L ** 2
    h = args.c / V

    ckpt_path = Path(args.run_dir) / "models" / f"epoch={args.ep:04d}.ckpt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    sigma, norm_min, norm_max = hp["sigma"], hp["norm_min"], hp["norm_max"]
    print(f"Checkpoint: {ckpt_path}")
    print(f"hparams: sigma={sigma}  norm=[{norm_min:.4f},{norm_max:.4f}]")
    print(f"MALA: drift={args.drift}  c={args.c}  h={h:.3e}  t_mh={args.t_mh}  "
          f"n_steps={args.n_steps}  seed={args.seed}")
    if args.shift_every > 0:
        print(f"shift: every {args.shift_every}, std={args.shift_std} (phys)")
    if args.cluster_every > 0:
        print(f"cluster: every {args.cluster_every} (embedded SW)")

    # ---- init ensemble ----
    phys = np.load(args.init_npy).astype(np.float64)
    if phys.shape[0] == L and phys.ndim == 3 and phys.shape[1] == L:
        phys = phys.transpose(2, 0, 1)          # (L,L,N) -> (N,L,L)
    assert phys.shape[1:] == (L, L), f"bad init shape {phys.shape}"
    if args.max_cfgs is not None:
        phys = phys[:args.max_cfgs]
    N = phys.shape[0]
    x = torch.from_numpy(
        (2.0 * (phys - norm_min) / (norm_max - norm_min) - 1.0)[:, None].copy()
    ).float().to(device)                        # (N,1,L,L)
    print(f"Init: {args.init_npy}  N={N}")

    tag = args.tag or (f"mala2d_{args.drift}_c{args.c:g}_se{args.shift_every}"
                       f"_ce{args.cluster_every}_ep{args.ep}_N{N}_seed{args.seed}")
    out_dir = Path(args.run_dir) / "data_mala_refine" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.pt"

    snap_steps = set()
    if args.n_snapshots > 0:
        snap_steps = set(np.unique(np.geomspace(
            1, args.n_steps, args.n_snapshots).astype(int)).tolist())

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp2D(marginal_prob_std_fn)
    score_model = torch.compile(score_model, mode="default", dynamic=False)
    model = DiffusionModel.load_from_checkpoint(str(ckpt_path),
                                                score_model=score_model,
                                                map_location="cpu",
                                                weights_only=False)
    model.L = L
    model = model.to(device).eval()

    def drift_fn(z):
        if args.drift == "exact":
            return -phi4_grad_S(z, args.k, args.l, norm_min, norm_max,
                                spatial_dims=2)
        out = torch.empty_like(z)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            for i in range(0, z.shape[0], args.microbatch):
                zc = z[i:i + args.microbatch]
                tc = torch.full((zc.shape[0],), args.t_mh, device=device)
                out[i:i + args.microbatch] = model(zc, tc).float()
        return out

    def action_fn(z):
        return phi4_action(z.double(), args.k, args.l, norm_min, norm_max,
                           spatial_dims=2)

    def to_phys_np(z):
        p = (z[:, 0].float().cpu().numpy() + 1.0) / 2.0 * (norm_max - norm_min) + norm_min
        return p.transpose(1, 2, 0).astype(np.float32)   # (L,L,N)

    def magnetization(z):
        scale, off = (norm_max - norm_min) / 2.0, (norm_max + norm_min) / 2.0
        return z[:, 0].mean(dim=(1, 2)) * scale + off

    def cluster_update(z):
        """Embedded SW update, 2D version (see 3D script for derivation)."""
        pf = (z[:, 0].double() + 1) / 2 * (norm_max - norm_min) + norm_min
        Nc = pf.shape[0]
        bonds = []
        for d in (1, 2):
            prod = pf * torch.roll(pf, -1, dims=d)
            pr = 1.0 - torch.exp(-4.0 * args.k * prod.clamp(min=0.0))
            bonds.append(torch.rand_like(pr) < pr)
        lbl = (torch.arange(V, device=z.device, dtype=torch.int64)
               .reshape(1, L, L).expand(Nc, -1, -1).contiguous())
        while True:
            old = lbl
            for d, b in zip((1, 2), bonds):
                nb = torch.roll(lbl, -1, dims=d)
                lbl = torch.where(b, torch.minimum(lbl, nb), lbl)
                nb = torch.roll(lbl, +1, dims=d)
                bb = torch.roll(b, +1, dims=d)
                lbl = torch.where(bb, torch.minimum(lbl, nb), lbl)
            if torch.equal(lbl, old):
                break
        coin = torch.rand(Nc, V, device=z.device) < 0.5
        flip = torch.gather(coin, 1, lbl.reshape(Nc, V)).reshape(Nc, L, L)
        pf = torch.where(flip, -pf, pf)
        zn = (2.0 * (pf - norm_min) / (norm_max - norm_min) - 1.0).float()
        return zn.unsqueeze(1)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    start_step = 0
    accept_total = torch.zeros(N, device=device, dtype=torch.float64)
    shift_accept_total = torch.zeros(N, device=device, dtype=torch.float64)
    n_shift_trials = 0
    S_trace, M_trace, acc_trace = [], [], []
    shift_scale = (norm_max - norm_min) / 2.0

    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        x = st["x"].to(device)
        accept_total = st["accept_total"].to(device)
        shift_accept_total = st["shift_accept_total"].to(device)
        n_shift_trials = st["n_shift_trials"]
        start_step = st["step"]
        S_trace, M_trace = list(st["S_trace"]), list(st["M_trace"])
        acc_trace = list(st["acc_trace"])
        torch.random.set_rng_state(st["cpu_rng"].cpu().to(torch.uint8))
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8)
                                      for s in st["cuda_rng"]])
        print(f"Resumed at step {start_step}")

    sqrt2h = float(np.sqrt(2.0 * h))

    def save_state(step):
        tmp = state_path.with_suffix(".pt.tmp")
        torch.save({"step": step, "x": x.detach().cpu(),
                    "accept_total": accept_total.cpu(),
                    "shift_accept_total": shift_accept_total.cpu(),
                    "n_shift_trials": n_shift_trials,
                    "S_trace": S_trace, "M_trace": M_trace,
                    "acc_trace": acc_trace,
                    "cpu_rng": torch.random.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "args": vars(args)}, tmp)
        os.replace(tmp, state_path)

    def save_traces():
        np.savez(out_dir / "traces.npz",
                 S_trace=np.asarray(S_trace, dtype=np.float64),
                 M_trace=np.asarray(M_trace, dtype=np.float32),
                 acc_trace=np.asarray(acc_trace, dtype=np.float64),
                 accept_total=accept_total.cpu().numpy(),
                 shift_accept_total=shift_accept_total.cpu().numpy(),
                 n_shift_trials=n_shift_trials, n_done=len(acc_trace))

    with torch.no_grad(), model.ema.average_parameters():
        s_x = drift_fn(x)
        S_x = action_fn(x)
        print(f"start: <S>/V = {S_x.mean().item() / V:+.6f}   "
              f"<|M|> = {magnetization(x).abs().mean().item():.5f}")

        for step in range(start_step, args.n_steps):
            xi = torch.randn_like(x)
            y = x + h * s_x + sqrt2h * xi
            s_y = drift_fn(y)
            S_y = action_fn(y)
            r = (sqrt2h * xi + h * (s_x + s_y)).double()
            log_q = (-(0.25 / h) * r.pow(2).sum(dim=(1, 2, 3))
                     + 0.5 * xi.double().pow(2).sum(dim=(1, 2, 3)))
            log_alpha = (S_x - S_y) + log_q
            u = torch.rand(N, device=device, dtype=torch.float64)
            accept = torch.log(u) < log_alpha
            m = accept.view(N, 1, 1, 1)
            x = torch.where(m, y, x)
            s_x = torch.where(m, s_y, s_x)
            S_x = torch.where(accept, S_y, S_x)
            accept_total += accept.double()

            if args.shift_every > 0 and (step + 1) % args.shift_every == 0:
                delta = (torch.randn(N, 1, 1, 1, device=device)
                         * (args.shift_std / shift_scale))
                y_s = x + delta
                S_ys = action_fn(y_s)
                acc_s = (torch.log(torch.rand(N, device=device,
                                              dtype=torch.float64))
                         < (S_x - S_ys))
                m_s = acc_s.view(N, 1, 1, 1)
                x = torch.where(m_s, y_s, x)
                S_x = torch.where(acc_s, S_ys, S_x)
                shift_accept_total += acc_s.double()
                n_shift_trials += 1
                if bool(acc_s.any()):
                    s_x = drift_fn(x)

            if (args.cluster_every > 0
                    and (step + 1) % args.cluster_every == 0):
                x = cluster_update(x)
                S_x = action_fn(x)
                s_x = drift_fn(x)

            S_trace.append(S_x.cpu().numpy())
            M_trace.append(magnetization(x).cpu().numpy())
            acc_trace.append(accept.double().mean().item())

            n_done = step + 1
            if n_done in snap_steps:
                np.save(out_dir / f"snap_step{n_done:06d}.npy", to_phys_np(x))
            if n_done % args.state_every == 0 or n_done == args.n_steps:
                save_state(n_done)
                save_traces()
            if n_done % 50 == 0 or n_done == args.n_steps:
                acc_s_str = ""
                if n_shift_trials > 0:
                    acc_s_str = (f"  acc_shift="
                                 f"{(shift_accept_total / n_shift_trials).mean().item():.3f}")
                print(f"step {n_done:6d}/{args.n_steps}  "
                      f"acc(last50)={np.mean(acc_trace[-50:]):.3f}{acc_s_str}  "
                      f"<S>/V={S_x.mean().item() / V:+.6f}  "
                      f"<|M|>={magnetization(x).abs().mean().item():.5f}",
                      flush=True)

    np.save(out_dir / "final.npy", to_phys_np(x))
    save_traces()
    print(f"done: overall acceptance = "
          f"{(accept_total / max(len(acc_trace), 1)).mean().item():.4f}")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
