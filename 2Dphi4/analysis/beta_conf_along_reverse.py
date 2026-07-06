"""β_conf(t) along the reverse-SDE trajectory for 2D phi^4 DM.

For each EM step of the reverse SDE we record
    β̂(x_t) = Tr(H)/g² − 2·g^T H g/(g²)²
computed with the *physical* phi^4 action (κ, λ). The expected behaviour:

    t = 1     β̂ ≈ 0   (pure noise, |g|² huge)
    t → eps   β̂ → 1   (DM has reproduced the Boltzmann distribution)

A "forward-noised HMC" curve gives the ground-truth marginal P_t. Any gap
between the DM curve and the HMC-noised curve is a direct measure of where
the model misrepresents the diffusion marginals.

Run from `2Dphi4/`:
    python analysis/beta_conf_along_reverse.py \
        --L 64 --k 0.2705 --l 0.022 --output_suffix _sigma250 \
        --ep 8554 --num_steps 2000 --schedule log \
        --num_samples 256 --hmc_n 1024 --device cuda:2 --seed 0
"""
from __future__ import annotations

import argparse
import functools
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

# repo layout: this file is at 2Dphi4/analysis/, repo root is parents[2]
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.append(str(ROOT))                  # repo root: phi4_action, diffusion_lightning, networks, ...
sys.path.append(str(HERE.parent))           # 2Dphi4/: sample_phi4, ...

from diffusion_lightning import DiffusionModel, marginal_prob_std, diffusion_coeff
from networks import NCSNpp2D
from phi4_action import phi4_beta_conf, phi4_beta_conf_term1


# ---------- HMC loading + forward-noising ----------

def load_hmc_2d(path: Path, max_n: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        cfgs = np.array(f["cfgs"], dtype=np.float64)
    sample_axis = int(np.argmax(cfgs.shape))
    if sample_axis != 0:
        cfgs = np.moveaxis(cfgs, sample_axis, 0)
    return cfgs[:max_n]


# ---------- main reverse-SDE + diagnostic ----------

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


@torch.no_grad()
def reverse_sde_with_beta(
    model: DiffusionModel, num_samples: int, num_steps: int, schedule: str,
    eps: float, k: float, l: float, norm_min: float, norm_max: float,
    record_every: int = 1, use_autocast: bool = True,
    first_term_only: bool = False,
):
    """Run EM reverse SDE; at every `record_every`-th step record β̂ stats."""
    device = model.device
    sigma_model = model.sigma
    beta_fn = phi4_beta_conf_term1 if first_term_only else phi4_beta_conf

    with model.ema.average_parameters():
        time_steps = model._build_time_steps(num_steps, eps, schedule, device)
        dt_all = time_steps[:-1] - time_steps[1:]

        init_std = model.marginal_prob_std_fn(torch.tensor(1.0, device=device))
        x = torch.randn(*model._sample_shape(num_samples), device=device) * init_std
        batch_t = torch.empty(num_samples, device=device)

        rec_t, rec_sigma_t, rec_mean, rec_sem, rec_p05, rec_p50, rec_p95 = ([] for _ in range(7))

        # record initial pure-noise state at t=1
        b0 = beta_fn(x.float(), k, l, norm_min, norm_max,
                     spatial_dims=model.spatial_dims)
        b0_np = b0.detach().cpu().numpy()
        rec_t.append(1.0)
        rec_sigma_t.append(float(init_std.item()))
        rec_mean.append(float(b0_np.mean()))
        rec_sem.append(float(b0_np.std(ddof=1) / np.sqrt(b0_np.size)))
        rec_p05.append(float(np.percentile(b0_np, 5)))
        rec_p50.append(float(np.percentile(b0_np, 50)))
        rec_p95.append(float(np.percentile(b0_np, 95)))

        # EM loop. autocast(bf16) is the production default; pass
        # use_autocast=False to force fp32 for precision-isolation tests.
        autocast_ctx = (
            torch.autocast(device.type, dtype=torch.bfloat16) if use_autocast
            else _NullCtx()
        )
        with autocast_ctx:
            for i in range(num_steps):
                t_val = time_steps[i].item()
                batch_t.fill_(t_val)
                g_coef = model.diffusion_coeff_fn(time_steps[i])
                dt = dt_all[i]
                mean_x = x + g_coef ** 2 * model(x, batch_t) * dt
                x = mean_x + g_coef * torch.sqrt(dt) * torch.randn_like(x)

                if (i + 1) % record_every == 0 or i == num_steps - 1:
                    t_next = time_steps[i + 1].item()
                    sigma_next = model.marginal_prob_std_fn(time_steps[i + 1]).item()
                    b = beta_fn(
                        x.float(), k, l, norm_min, norm_max,
                        spatial_dims=model.spatial_dims,
                    )
                    b_np = b.detach().cpu().numpy()
                    rec_t.append(t_next)
                    rec_sigma_t.append(sigma_next)
                    rec_mean.append(float(b_np.mean()))
                    rec_sem.append(float(b_np.std(ddof=1) / np.sqrt(b_np.size)))
                    rec_p05.append(float(np.percentile(b_np, 5)))
                    rec_p50.append(float(np.percentile(b_np, 50)))
                    rec_p95.append(float(np.percentile(b_np, 95)))

    return dict(
        t=np.array(rec_t),
        sigma_t=np.array(rec_sigma_t),
        mean=np.array(rec_mean),
        sem=np.array(rec_sem),
        p05=np.array(rec_p05),
        p50=np.array(rec_p50),
        p95=np.array(rec_p95),
    )


def forward_noised_hmc_curve(
    hmc_norm: torch.Tensor, t_grid: np.ndarray, sigma_grid: np.ndarray,
    k: float, l: float, norm_min: float, norm_max: float, seed: int,
    spatial_dims: int = 2, first_term_only: bool = False,
):
    """Compute β̂ on x_t = x_0 + σ_t · ε for each t, using all HMC cfgs."""
    g = torch.Generator(device=hmc_norm.device).manual_seed(seed)
    beta_fn = phi4_beta_conf_term1 if first_term_only else phi4_beta_conf

    mean, sem, p05, p50, p95 = [], [], [], [], []
    for sigma_t in sigma_grid:
        noise = torch.randn(hmc_norm.shape, generator=g, device=hmc_norm.device,
                            dtype=hmc_norm.dtype)
        x_t = hmc_norm + sigma_t * noise
        b = beta_fn(x_t, k, l, norm_min, norm_max,
                    spatial_dims=spatial_dims)
        b_np = b.detach().cpu().numpy()
        mean.append(float(b_np.mean()))
        sem.append(float(b_np.std(ddof=1) / np.sqrt(b_np.size)))
        p05.append(float(np.percentile(b_np, 5)))
        p50.append(float(np.percentile(b_np, 50)))
        p95.append(float(np.percentile(b_np, 95)))

    return dict(
        t=np.asarray(t_grid),
        sigma_t=np.asarray(sigma_grid),
        mean=np.array(mean),
        sem=np.array(sem),
        p05=np.array(p05),
        p50=np.array(p50),
        p95=np.array(p95),
    )


# ---------- plotting ----------

def plot_curves(dm, hmc_fwd, out_pdf: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # (1) full t range, log-x
    ax = axes[0]
    ax.set_xscale("log")
    ax.plot(dm["t"], dm["mean"], "-", color="C3", lw=1.6, label="DM reverse SDE")
    ax.fill_between(dm["t"], dm["mean"] - dm["sem"], dm["mean"] + dm["sem"],
                    color="C3", alpha=0.25, lw=0)
    ax.plot(hmc_fwd["t"], hmc_fwd["mean"], "--", color="C0", lw=1.6,
            label="HMC + forward noise")
    ax.fill_between(hmc_fwd["t"], hmc_fwd["mean"] - hmc_fwd["sem"],
                    hmc_fwd["mean"] + hmc_fwd["sem"], color="C0", alpha=0.2, lw=0)
    ax.axhline(1.0, ls=":", color="k", lw=0.8, label=r"$\beta_{\rm conf}=1$ target")
    ax.set_xlabel("t  (reverse time)")
    ax.set_ylabel(r"$\langle\hat\beta(x_t)\rangle$")
    ax.set_title(f"{title} — full")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    # (2) late-stage zoom: small t, y in [0.9, 1.05]
    ax = axes[1]
    ax.set_xscale("log")
    m_late = dm["t"] < 1e-2
    ax.errorbar(dm["t"][m_late], dm["mean"][m_late], yerr=dm["sem"][m_late],
                fmt="o-", ms=2.5, lw=1.0, capsize=0, color="C3",
                label="DM reverse SDE", alpha=0.9)
    m_late_h = hmc_fwd["t"] < 1e-2
    ax.errorbar(hmc_fwd["t"][m_late_h], hmc_fwd["mean"][m_late_h],
                yerr=hmc_fwd["sem"][m_late_h], fmt="s--", ms=2.5, lw=1.0,
                capsize=0, color="C0", label="HMC + fwd noise", alpha=0.9)
    ax.axhline(1.0, ls=":", color="k", lw=0.8)
    ax.set_ylim(0.85, 1.05)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle\hat\beta(x_t)\rangle$")
    ax.set_title("late stage  (t < 1e-2)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    # (3) DM − HMC difference vs σ_t (log-x); flat at 0 means perfect match
    ax = axes[2]
    ax.set_xscale("log")
    diff = dm["mean"] - hmc_fwd["mean"]
    err  = np.hypot(dm["sem"], hmc_fwd["sem"])
    ax.errorbar(dm["sigma_t"], diff, yerr=err, fmt="o-",
                ms=2.5, lw=1.0, capsize=0, color="C2", alpha=0.85)
    ax.axhline(0.0, ls=":", color="k", lw=0.8)
    ax.set_xlabel(r"$\sigma_t$")
    ax.set_ylabel(r"$\langle\hat\beta\rangle_{\rm DM} - \langle\hat\beta\rangle_{\rm HMC+noise}$")
    ax.set_title("model − marginal target")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_pdf)
    print(f"saved figure → {out_pdf}")


# ---------- driver ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--k", type=float, default=0.2705)
    p.add_argument("--l", type=float, default=0.022)
    p.add_argument("--output_suffix", type=str, default="_sigma250")
    p.add_argument("--ep", type=int, default=8554)
    p.add_argument("--num_steps", type=int, default=2000)
    p.add_argument("--schedule", type=str, default="log")
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--num_samples", type=int, default=256,
                   help="DM reverse-SDE batch size")
    p.add_argument("--record_every", type=int, default=4)
    p.add_argument("--hmc_path", type=str, default=None)
    p.add_argument("--hmc_n", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--first_term_only", action="store_true",
                   help="Use Tr(H)/g² only (drop −2 g^T H g/(g²)² term).")
    args = p.parse_args()

    run_dir = Path(f"runs/phi4_L{args.L}_k{args.k}_l{args.l}_ncsnpp{args.output_suffix}")
    ckpts = sorted(run_dir.glob(f"models/*{args.ep}*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt for ep={args.ep} in {run_dir}/models")
    ckpt_path = ckpts[-1]
    print(f"checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma_model = hparams.get("sigma", 250.0)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    print(f"  sigma_model={sigma_model}  norm=[{norm_min:.4f}, {norm_max:.4f}]")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma_model)
    score_model = NCSNpp2D(marginal_prob_std_fn)
    model = DiffusionModel.load_from_checkpoint(ckpt_path, score_model=score_model, map_location=args.device, weights_only=False)
    model = model.to(args.device).eval()
    # the loaded model's `spatial_dims` should be 2; just assert
    assert model.spatial_dims == 2

    # 1) DM reverse-SDE β̂(t)
    print(f"reverse SDE: num_steps={args.num_steps}  schedule={args.schedule}  "
          f"batch={args.num_samples}")
    dm = reverse_sde_with_beta(
        model, args.num_samples, args.num_steps, args.schedule, args.eps,
        args.k, args.l, norm_min, norm_max, record_every=args.record_every,
        first_term_only=args.first_term_only,
    )
    print(f"  recorded {len(dm['t'])} points; "
          f"final β̂={dm['mean'][-1]:.4f} ± {dm['sem'][-1]:.4f} "
          f"at t={dm['t'][-1]:.2e}")

    # 2) Forward-noised HMC curve on the same (t, σ_t) grid
    if args.hmc_path is None:
        args.hmc_path = (f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}"
                         f"_{args.L}^2.jld2")
    hmc_path = Path(args.hmc_path)
    if not hmc_path.exists():
        raise FileNotFoundError(hmc_path)
    print(f"HMC: {hmc_path.name}  using {args.hmc_n} cfgs")

    hmc_cfgs = load_hmc_2d(hmc_path, args.hmc_n)
    pmin_hmc, pmax_hmc = float(hmc_cfgs.min()), float(hmc_cfgs.max())
    print(f"  HMC range [{pmin_hmc:.3f}, {pmax_hmc:.3f}]")
    # Use the SAME normalization as training (norm_min, norm_max from ckpt)
    hmc_norm = 2.0 * (hmc_cfgs - norm_min) / (norm_max - norm_min) - 1.0
    hmc_norm_t = torch.from_numpy(hmc_norm).unsqueeze(1).float().to(args.device)

    hmc_fwd = forward_noised_hmc_curve(
        hmc_norm_t, dm["t"], dm["sigma_t"],
        args.k, args.l, norm_min, norm_max, seed=args.seed + 1,
        spatial_dims=2, first_term_only=args.first_term_only,
    )
    print(f"  HMC@t=0 (no noise) β̂ would be ≈ 1.0; here at smallest grid t "
          f"({dm['t'][-1]:.2e}): β̂={hmc_fwd['mean'][-1]:.4f} "
          f"± {hmc_fwd['sem'][-1]:.4f}")

    # 3) Save
    out_dir = Path("results/beta_conf_traj")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"L{args.L}_k{args.k}_l{args.l}_ep{args.ep}_steps{args.num_steps}"
           f"_{args.schedule}_N{args.num_samples}"
           + ("_term1" if args.first_term_only else ""))
    npz_path = out_dir / f"beta_conf_t_{tag}.npz"
    np.savez(npz_path,
             t=dm["t"], sigma_t=dm["sigma_t"],
             dm_mean=dm["mean"], dm_sem=dm["sem"],
             dm_p05=dm["p05"], dm_p50=dm["p50"], dm_p95=dm["p95"],
             hmc_mean=hmc_fwd["mean"], hmc_sem=hmc_fwd["sem"],
             hmc_p05=hmc_fwd["p05"], hmc_p50=hmc_fwd["p50"], hmc_p95=hmc_fwd["p95"],
             k=args.k, l=args.l, L=args.L, sigma_model=sigma_model,
             norm_min=norm_min, norm_max=norm_max, ep=args.ep)
    print(f"saved npz → {npz_path}")

    pdf_path = out_dir / f"beta_conf_t_{tag}.pdf"
    plot_curves(dm, hmc_fwd, pdf_path,
                title=f"L={args.L}  κ={args.k}  λ={args.l}  ep={args.ep}")


if __name__ == "__main__":
    main()
