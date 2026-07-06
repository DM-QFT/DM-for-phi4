"""
2D phi^4 test: reverse SDE with a Metropolis-corrected tail.

Phase 1 (t: 1 -> t_switch): plain EM reverse SDE on the production log grid.
Phase 2 (t: t_switch -> eps): the SAME EM update is kept as the proposal, but
every step is accepted/rejected against the exact Boltzmann target
pi(phi) ~ exp(-S[phi]).

Per step at time t (dt > 0, g = g(t)):
    proposal:  y = x + g^2 dt s_theta(x,t) + g sqrt(dt) xi
    q(y|x) = N(y; x + g^2 dt s(x,t), g^2 dt I)   -> log q(y|x) = -||xi||^2/2
    q(x|y) = N(x; y + g^2 dt s(y,t), g^2 dt I)
    log alpha = S(x) - S(y) + log q(x|y) - log q(y|x)

Each step is a pi-invariant MH kernel (time-inhomogeneity is irrelevant for
invariance: every kernel individually preserves exp(-S)). The action and all
MH reductions run in float64 outside autocast. S(x) is cached across steps;
the score cannot be cached because its time argument changes every step.

Run from 2Dphi4/:
    python sample_phi4_mh_tail.py --k 0.28 --num_samples 1024 --t_switch 1e-2
"""

import sys
sys.path.append("..")

import argparse
import functools
from pathlib import Path

import numpy as np
import torch

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from phi4_action import phi4_action


def parse_args():
    p = argparse.ArgumentParser(description="2D reverse SDE + MH-corrected tail")
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--k", type=float, default=0.28)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--ep", type=int, default=10000)
    p.add_argument("--run_dir", type=str,
                   default="runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640")
    p.add_argument("--num_samples", type=int, default=1024)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--schedule", type=str, default="log")
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--t_switch", type=float, default=1e-2,
                   help="Below this diffusion time every EM step gets an MH test")
    p.add_argument("--n_head", type=int, default=1600,
                   help="(two-segment mode) log-grid steps from t=1 to t_switch")
    p.add_argument("--n_tail", type=int, default=0,
                   help="If >0: use a two-segment grid, with n_tail log-grid "
                        "steps from t_switch to eps (all MH-corrected)")
    p.add_argument("--microbatch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    L, V = args.L, args.L ** 2
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ckpt_path = Path(args.run_dir) / "models" / f"epoch={args.ep:04d}.ckpt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    h = ckpt["hyper_parameters"]
    sigma, norm_min, norm_max = h["sigma"], h["norm_min"], h["norm_max"]
    print(f"Checkpoint: {ckpt_path}")
    print(f"hparams: sigma={sigma}  norm=[{norm_min:.4f},{norm_max:.4f}]")

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp2D(marginal_prob_std_fn)
    # compile(default): reduce-overhead corrupts F.interpolate(nearest).
    score_model = torch.compile(score_model, mode="default", dynamic=False)
    model = DiffusionModel.load_from_checkpoint(str(ckpt_path),
                                                score_model=score_model,
                                                map_location="cpu",
                                                weights_only=False)
    model.L = L
    model = model.to(device).eval()

    N = args.num_samples

    def drift(z, t_val):
        out = torch.empty_like(z)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            for i in range(0, N, args.microbatch):
                zc = z[i:i + args.microbatch]
                tc = torch.full((zc.shape[0],), t_val, device=device)
                out[i:i + args.microbatch] = model(zc, tc).float()
        return out

    def action_fn(z):
        return phi4_action(z.double(), args.k, args.l, norm_min, norm_max,
                           spatial_dims=2)

    if args.n_tail > 0:
        # two-segment log grid: 1 -> t_switch (n_head), t_switch -> eps (n_tail)
        ts1 = torch.exp(torch.linspace(0.0, float(np.log(args.t_switch)),
                                       args.n_head + 1, device=device))
        ts2 = torch.exp(torch.linspace(float(np.log(args.t_switch)),
                                       float(np.log(args.eps)),
                                       args.n_tail + 1, device=device))[1:]
        time_steps = torch.cat([ts1, ts2])
        args.num_steps = args.n_head + args.n_tail
    else:
        time_steps = model._build_time_steps(args.num_steps, args.eps,
                                             args.schedule, device)
    dt_all = time_steps[:-1] - time_steps[1:]
    n_mh = int((time_steps[:-1] < args.t_switch).sum().item())
    print(f"grid: {args.num_steps} steps ({args.schedule}, eps={args.eps}); "
          f"MH-corrected tail: {n_mh} steps below t_switch={args.t_switch}")

    tag = args.tag or (f"mhtail_k{args.k:g}_N{N}_steps{args.num_steps}"
                       f"_tsw{args.t_switch:g}_ep{args.ep}_seed{args.seed}")
    out_dir = Path(args.run_dir) / "data_mh_tail" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    acc_t, acc_rate, sv_trace, t_trace = [], [], [], []
    S_x = None
    accept_total = torch.zeros(N, device=device, dtype=torch.float64)
    n_mh_done = 0

    with torch.no_grad(), model.ema.average_parameters():
        init_std = model.marginal_prob_std_fn(torch.tensor(1.0, device=device))
        x = torch.randn(N, 1, L, L, device=device) * init_std

        for i in range(args.num_steps):
            t_val = time_steps[i].item()
            dt = dt_all[i]
            g = model.diffusion_coeff_fn(time_steps[i])
            s_x = drift(x, t_val)
            mean_x = x + (g ** 2) * s_x * dt

            if t_val >= args.t_switch:
                # plain EM
                x = mean_x + g * torch.sqrt(dt) * torch.randn_like(x)
                continue

            # ---- MH-corrected EM step (target: exp(-S)) ----
            if S_x is None:
                S_x = action_fn(x)
            xi = torch.randn_like(x)
            gsdt = g * torch.sqrt(dt)
            y = mean_x + gsdt * xi
            S_y = action_fn(y)
            s_y = drift(y, t_val)
            mean_y = y + (g ** 2) * s_y * dt

            var = (g ** 2 * dt).double()
            log_q_fwd = -0.5 * xi.double().pow(2).sum(dim=(1, 2, 3))
            log_q_rev = (-(x - mean_y).double().pow(2).sum(dim=(1, 2, 3))
                         / (2.0 * var))
            log_alpha = (S_x - S_y) + log_q_rev - log_q_fwd

            u = torch.rand(N, device=device, dtype=torch.float64)
            accept = torch.log(u) < log_alpha
            m = accept.view(N, 1, 1, 1)
            x = torch.where(m, y, x)
            S_x = torch.where(accept, S_y, S_x)
            accept_total += accept.double()
            n_mh_done += 1

            acc_t.append(t_val)
            acc_rate.append(accept.double().mean().item())
            sv_trace.append(S_x.mean().item() / V)
            t_trace.append(t_val)
            if n_mh_done % 50 == 0 or i == args.num_steps - 1:
                print(f"step {i+1:5d}/{args.num_steps}  t={t_val:.3e}  "
                      f"acc(this)={acc_rate[-1]:.3f}  "
                      f"acc(last50)={np.mean(acc_rate[-50:]):.3f}  "
                      f"<S>/V={sv_trace[-1]:+.6f}", flush=True)

    # final state x is the sample (no mean trick inside the MH phase)
    phys = ((x[:, 0].float().cpu().numpy() + 1.0) / 2.0
            * (norm_max - norm_min) + norm_min)          # (N, L, L)
    np.save(out_dir / "samples.npy", phys.astype(np.float32))
    np.savez(out_dir / "mh_trace.npz",
             t=np.asarray(acc_t), acc=np.asarray(acc_rate),
             SV=np.asarray(sv_trace),
             accept_total=accept_total.cpu().numpy(), n_mh=n_mh_done)
    print(f"\nMH steps: {n_mh_done}   overall acceptance = "
          f"{(accept_total / max(n_mh_done, 1)).mean().item():.4f}")
    print(f"saved: {out_dir}/samples.npy  (N,L,L)={phys.shape}")
    print(f"saved: {out_dir}/mh_trace.npz")


if __name__ == "__main__":
    main()
