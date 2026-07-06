"""
ESS via integrated autocorrelation time of a MALA chain driven by the
diffusion model's score, for the 2D phi^4 model.

Pipeline (per checkpoint):
  1. Draw M reference cfgs from HMC training data (chain start, ~equilibrium).
  2. Run N MH steps of MALA with proposal
        y = x + h * s_theta(x, t_mh) + sqrt(2 h) * eta,
        accept = min(1, exp(action(x) - action(y) + log_q_correction)),
     mirroring acceptance_vs_epoch.py / calibrate_step_size.py.
  3. Record per-step magnetization phi-bar(t, m) per chain.
  4. Estimate the chain-averaged ACF rho_bar(tau) via FFT.
  5. tau_int via Sokal automatic windowing
        tau_hat(W) = 1/2 + sum_{tau=1..W} rho_bar(tau);  pick smallest W with W >= c * tau_hat(W).
  6. ESS = M * N / (2 tau_int).

Parameters c=0.2, t_mh=1e-4 are the calibrated 2D values from DM.tex
(Sec.~Acceptance), with h = c / L^2.

Usage (from 2Dphi4/):
  python analysis/dm_mc_ess_2d.py \
      --device cuda:2 \
      --run_dir runs/phi4_L32_k0.28_l0.022_ncsnpp_sigma180 \
      --early_epoch 49 --late_epoch 7564 \
      --num_chains 256 --mh_steps 1000
"""

import argparse
import functools
import io
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))      # 2Dphi4/
sys.path.append(str(HERE.parents[2]))      # repo root

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std
from sample_phi4 import phi4_action


# -------------------- ckpt + model loading --------------------

def find_epoch_checkpoints(model_dir):
    """Match either 'epoch=NNNN.ckpt' or 'epoch=epoch=NNNN.ckpt'."""
    ckpts = {}
    for f in sorted(model_dir.glob("epoch=*.ckpt")):
        if "-v" in f.stem:
            continue
        m = re.search(r"epoch=(\d+)$", f.stem)
        if m:
            ckpts[int(m.group(1))] = f
    return dict(sorted(ckpts.items()))


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 150.0)
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19

    path_str = str(ckpt_path)
    k = float(re.search(r"_k([\d.]+)", path_str).group(1))
    l = float(re.search(r"_l([\d.]+)", path_str).group(1))

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp2D(marginal_prob_std_fn)

    sd = ckpt.get("state_dict", {})
    if any("_orig_mod" in key for key in sd):
        ckpt["state_dict"] = {key.replace("._orig_mod.", "."): v for key, v in sd.items()}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    del ckpt
    buf.seek(0)
    model = DiffusionModel.load_from_checkpoint(buf, score_model=score_model, map_location=device, weights_only=False)
    model = model.to(device).eval()
    action_fn = functools.partial(
        phi4_action, k=k, l=l, phi_min=norm_min, phi_max=norm_max
    )
    meta = dict(k=k, l=l, sigma_ve=sigma, norm_min=norm_min, norm_max=norm_max)
    return model, action_fn, meta


def load_reference_data(data_path, num_samples, norm_min, norm_max, device, seed=42):
    """Load training cfgs, normalise to [-1,1], return (M, 1, L, L) torch tensor."""
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(cfgs.shape[0], size=num_samples, replace=False)
    cfgs = cfgs[idx]
    cfgs_norm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 3:
        x = x.unsqueeze(1)
    return x


# -------------------- MALA chain + observables --------------------

@torch.no_grad()
def run_mala_chain(model, action_fn, x0, h, t_mh, n_steps, batch_size=None, log_every=50):
    """
    Drive a batch of M parallel MALA chains for n_steps and record:
      - phi_bar[t, m] = mean over space of x_m at step t (incl. t=0)
      - accept[t, m] = 0/1 for step t (t=1..n_steps)

    h is the MALA step size; t_mh is the diffusion time at which the score is queried.
    """
    device = x0.device
    M = x0.shape[0]
    h_t = torch.tensor(h, device=device, dtype=x0.dtype)
    sqrt_2h = torch.sqrt(2 * h_t)

    if batch_size is None or batch_size >= M:
        def score(z):
            return model(z, torch.full((z.shape[0],), t_mh, device=device, dtype=z.dtype))
    else:
        def score(z):
            outs = []
            for i in range(0, z.shape[0], batch_size):
                zi = z[i:i + batch_size]
                ti = torch.full((zi.shape[0],), t_mh, device=device, dtype=zi.dtype)
                outs.append(model(zi, ti))
            return torch.cat(outs, dim=0)

    spatial_dims = tuple(range(1, x0.ndim))  # collapse over (C, L, L)

    phi_bar = torch.empty((n_steps + 1, M), device=device, dtype=x0.dtype)
    accept_log = torch.empty((n_steps, M), device=device, dtype=x0.dtype)
    x = x0.clone()
    phi_bar[0] = x.mean(dim=spatial_dims)

    for step in range(n_steps):
        score_x = score(x)
        noise = torch.randn_like(x)
        y = x + h_t * score_x + sqrt_2h * noise
        score_y = score(y)
        drift = h_t * (score_x + score_y)
        # log q(x|y)/q(y|x), see calibrate_step_size.py / acceptance_vs_epoch.py
        log_q = -(0.25 / h_t) * torch.sum((sqrt_2h * noise + drift) ** 2,
                                          dim=spatial_dims) \
                + 0.5 * torch.sum(noise ** 2, dim=spatial_dims)
        log_pi = action_fn(x) - action_fn(y)
        accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
        accept = torch.rand(M, device=device) < accept_prob
        x = torch.where(accept[(slice(None),) + (None,) * (x.ndim - 1)], y, x)
        accept_log[step] = accept.to(x.dtype)
        phi_bar[step + 1] = x.mean(dim=spatial_dims)

        if log_every and ((step + 1) % log_every == 0 or step == 0):
            ar = accept_log[: step + 1].mean().item()
            print(f"    step {step + 1:>5d}/{n_steps}  cumulative accept = {ar:.4f}")

    return phi_bar.cpu().numpy(), accept_log.cpu().numpy()


# -------------------- autocorrelation + tau_int --------------------

def acf_per_chain_fft(series):
    """
    Normalised ACF for each chain, computed via FFT.

    series: (T, M) array of an observable for M chains of length T.
    returns: rho of shape (T, M) with rho[0, :] = 1.
    """
    T, M = series.shape
    x = series - series.mean(axis=0, keepdims=True)
    n2 = 1 << int(np.ceil(np.log2(2 * T)))  # zero-pad to avoid wrap-around
    f = np.fft.rfft(x, n=n2, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n2, axis=0)[:T]
    norm = (T - np.arange(T))[:, None]      # unbiased divisor
    acf = acf / norm
    rho = acf / acf[0:1]
    return rho


def sokal_window(rho_mean, c=5.0):
    """
    Sokal automatic windowing. rho_mean is the chain-averaged ACF (length T).

    tau_hat(W) = 1/2 + sum_{t=1..W} rho_mean(t)
    Pick the smallest W >= 1 with W >= c * tau_hat(W); fall back to last W if none.
    Return (tau_int, W_chosen, tau_hat_curve).
    """
    T = len(rho_mean)
    cum = np.cumsum(rho_mean[1:])           # cum[k-1] = sum_{t=1..k} rho(t)
    tau_hat = 0.5 + cum                     # tau_hat[W-1] for W = 1..T-1
    Ws = np.arange(1, T)
    ok = Ws >= c * tau_hat
    if ok.any():
        W = int(Ws[np.argmax(ok)])
    else:
        W = int(Ws[-1])
    tau = float(tau_hat[W - 1])
    return tau, W, tau_hat


def cutoff_window(rho_mean, cutoff=0.05):
    """Simple cutoff: sum rho(t) for t >= 1 until rho(t) < cutoff. Returns tau_int."""
    tau = 0.5
    for t in range(1, len(rho_mean)):
        if rho_mean[t] < cutoff:
            break
        tau += rho_mean[t]
    return tau


# -------------------- driver --------------------

def analyse_one_ckpt(label, ckpt_path, args, x_ref):
    print(f"\n{'='*60}\n{label}: epoch ckpt = {ckpt_path}\n{'='*60}")
    model, action_fn, meta = load_model(ckpt_path, args.device)

    L = args.L
    h = args.c / (L ** 2)
    sigma_ve = meta["sigma_ve"]
    sigma_at_tmh = float(np.sqrt((sigma_ve ** (2 * args.t_mh) - 1.0)
                                 / (2.0 * np.log(sigma_ve))))
    print(f"  k={meta['k']}, sigma_VE={sigma_ve}, sigma(t_mh={args.t_mh:.0e}) ~ {sigma_at_tmh:.3e}")
    print(f"  c={args.c}, h={h:.3e}, M={x_ref.shape[0]} chains, N={args.mh_steps} steps")

    phi_bar, accept_log = run_mala_chain(
        model, action_fn, x_ref, h, args.t_mh, args.mh_steps,
        batch_size=args.batch_size, log_every=args.log_every,
    )
    accept_rate = float(accept_log.mean())
    print(f"  -> overall acceptance rate = {accept_rate:.4f}")

    # ACF on the post-step series (length n_steps), per chain, then average.
    series = phi_bar[1:]                          # (N, M)
    rho_per_chain = acf_per_chain_fft(series)     # (N, M)
    rho_mean = rho_per_chain.mean(axis=1)         # average across chains

    tau_sokal, W, tau_hat_curve = sokal_window(rho_mean, c=args.sokal_c)
    tau_cut = cutoff_window(rho_mean, cutoff=args.acf_cutoff)

    N, M = series.shape
    # Madras-Sokal s.e. on tau_int: var(tau_hat(W)) ~ 2(2W+1)/N_eff_total * tau^2
    n_eff_total = M * N
    tau_se = float(np.sqrt(2.0 * (2 * W + 1) / n_eff_total) * tau_sokal)

    ess_per_chain = N / (2.0 * tau_sokal)
    ess_total = M * N / (2.0 * tau_sokal)
    print(f"  Sokal window: W={W}, tau_int={tau_sokal:.3f} +/- {tau_se:.3f}")
    print(f"  cutoff (rho<{args.acf_cutoff}): tau_int={tau_cut:.3f}")
    print(f"  ESS_per_chain = {ess_per_chain:.2f},  ESS_total = {ess_total:.1f}")

    out = dict(
        label=label, epoch=int(re.search(r"epoch=(\d+)$", ckpt_path.stem).group(1)),
        c=args.c, t_mh=args.t_mh, h=h,
        sigma_ve=sigma_ve, sigma_at_tmh=sigma_at_tmh,
        meta_k=meta["k"], meta_l=meta["l"],
        n_chains=M, n_steps=N, accept_rate=accept_rate,
        phi_bar=phi_bar, accept_log=accept_log,
        rho_mean=rho_mean, tau_hat_curve=tau_hat_curve,
        tau_int_sokal=tau_sokal, sokal_W=W, tau_int_se=tau_se,
        tau_int_cutoff=tau_cut, ess_per_chain=ess_per_chain, ess_total=ess_total,
    )
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def make_plot(results, out_path, args):
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "lines.linewidth": 1.4, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_trace, ax_rho, ax_w, ax_summary = axes.flatten()
    colors = {"early": "C3", "late": "C0"}

    for r in results:
        c = colors.get(r["label"].lower(), "C2")
        lbl = f"{r['label']} (ep {r['epoch']})"

        # (a) trace of one chain
        ax_trace.plot(r["phi_bar"][:, 0], color=c, lw=0.8, alpha=0.9, label=lbl)

        # (b) rho_bar(tau)
        rho = r["rho_mean"]
        tau_axis = np.arange(len(rho))
        keep = min(len(rho), max(50, 5 * r["sokal_W"]))
        ax_rho.plot(tau_axis[:keep], rho[:keep], color=c, label=lbl)

        # (c) tau_hat(W) curves with Sokal window
        Ws = np.arange(1, len(r["tau_hat_curve"]) + 1)
        keep_w = min(len(Ws), max(50, 8 * r["sokal_W"]))
        ax_w.plot(Ws[:keep_w], r["tau_hat_curve"][:keep_w], color=c, label=lbl)
        ax_w.axvline(r["sokal_W"], ls=":", color=c, alpha=0.7)

    ax_trace.set_xlabel("MH step")
    ax_trace.set_ylabel(r"$\bar\phi$ (chain 0, normalised space)")
    ax_trace.set_title(rf"Single-chain trace ($L={args.L},\;\kappa={args.k}$)")
    ax_trace.legend(loc="best")
    ax_trace.grid(alpha=0.3)

    ax_rho.axhline(0, color="k", lw=0.5)
    ax_rho.set_xlabel(r"lag $\tau$ (MH steps)")
    ax_rho.set_ylabel(r"$\bar\rho(\tau)$")
    ax_rho.set_title(r"Chain-averaged ACF of $\bar\phi$")
    ax_rho.legend()
    ax_rho.grid(alpha=0.3)

    ax_w.set_xlabel(r"window $W$")
    ax_w.set_ylabel(r"$\hat\tau_{\mathrm{int}}(W)$")
    ax_w.set_title(rf"Sokal windowing (vertical: chosen $W^*$, $c={args.sokal_c}$)")
    ax_w.legend()
    ax_w.grid(alpha=0.3)

    ax_summary.axis("off")
    rows = [
        ["Quantity"] + [f"{r['label']} (ep {r['epoch']})" for r in results],
        ["accept rate"] + [f"{r['accept_rate']:.3f}" for r in results],
        ["tau_int (Sokal)"] + [f"{r['tau_int_sokal']:.2f} ± {r['tau_int_se']:.2f}"
                               for r in results],
        ["Sokal W*"] + [f"{r['sokal_W']}" for r in results],
        ["tau_int (cutoff)"] + [f"{r['tau_int_cutoff']:.2f}" for r in results],
        ["ESS / chain"] + [f"{r['ess_per_chain']:.1f}" for r in results],
        ["ESS total"] + [f"{r['ess_total']:.0f}" for r in results],
    ]
    ax_summary.table(cellText=rows, loc="center", cellLoc="center")
    ax_summary.set_title("Summary")

    fig.suptitle(
        rf"DM-MALA ESS, 2D $\phi^4$  $L={args.L},\;\kappa={args.k}$,  "
        rf"$c={args.c},\;t_{{\mathrm{{mh}}}}={args.t_mh:g}$,  "
        rf"$M={args.num_chains},\;N={args.mh_steps}$",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved figure to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--run_dir", type=str,
                        default="runs/phi4_L32_k0.28_l0.022_ncsnpp_sigma180")
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--k", type=float, default=0.28)
    parser.add_argument("--l", type=float, default=0.022)
    parser.add_argument("--data_path", type=str, default=None,
                        help="HMC training data .jld2 (default: trainingdata/cfgs_wolff_fahmc_*).")
    parser.add_argument("--early_epoch", type=int, default=49)
    parser.add_argument("--late_epoch", type=int, default=7564)
    parser.add_argument("--c", type=float, default=0.2,
                        help="MALA step coefficient, h = c/L^2 (DM.tex 2D: 0.2).")
    parser.add_argument("--t_mh", type=float, default=1e-4,
                        help="Diffusion time at which score is queried.")
    parser.add_argument("--num_chains", type=int, default=256)
    parser.add_argument("--mh_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Sub-batch over chains for the score forward pass; default = num_chains.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sokal_c", type=float, default=5.0)
    parser.add_argument("--acf_cutoff", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--out_tag", type=str, default=None,
                        help="Tag appended to output files; default uses late epoch.")
    args = parser.parse_args()

    # All paths in run_dir / trainingdata / results are written relative to 2Dphi4/.
    os.chdir(str(HERE.parents[1]))

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = HERE.parents[1] / args.run_dir
    ckpts = find_epoch_checkpoints(run_dir / "models")
    if not ckpts:
        raise RuntimeError(f"No checkpoints in {run_dir/'models'}")
    available = sorted(ckpts.keys())

    def _nearest(target):
        return min(available, key=lambda e: abs(e - target))
    early_ep = _nearest(args.early_epoch)
    late_ep = _nearest(args.late_epoch)
    if early_ep == late_ep:
        raise RuntimeError(f"Early and late epochs collapsed to {early_ep}.")
    print(f"Available {len(available)} ckpts, [{available[0]}, {available[-1]}]")
    print(f"Early -> epoch {early_ep}    Late  -> epoch {late_ep}")

    if args.data_path is None:
        args.data_path = f"trainingdata/cfgs_wolff_fahmc_k={args.k}_l={args.l}_{args.L}^2.jld2"
    if not Path(args.data_path).exists():
        raise RuntimeError(f"Training data not found: {args.data_path}")

    # Read norm parameters once from late ckpt (assumed consistent across run).
    ref_ckpt = torch.load(ckpts[late_ep], map_location="cpu", weights_only=False)
    hparams = ref_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min") or -6.22
    norm_max = hparams.get("norm_max") or 6.19
    del ref_ckpt
    print(f"norm range: [{norm_min}, {norm_max}]")

    x_ref = load_reference_data(args.data_path, args.num_chains,
                                norm_min, norm_max, args.device, seed=args.seed)
    print(f"Loaded {x_ref.shape[0]} cfgs from {args.data_path}, shape {tuple(x_ref.shape)}")

    torch.manual_seed(args.seed)

    results = []
    for label, ep in [("early", early_ep), ("late", late_ep)]:
        results.append(analyse_one_ckpt(label, ckpts[ep], args, x_ref))

    tag = args.out_tag or f"e{late_ep:04d}"
    out_dir = HERE.parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    out_npz = out_dir / f"dm_mc_ess_2D_L{args.L}_k{args.k}_{tag}.npz"
    out_pdf = out_dir / f"dm_mc_ess_2D_L{args.L}_k{args.k}_{tag}.pdf"

    save_dict = {
        "args_c": args.c, "args_t_mh": args.t_mh,
        "L": args.L, "k": args.k, "l": args.l,
        "num_chains": args.num_chains, "mh_steps": args.mh_steps,
        "seed": args.seed, "sokal_c": args.sokal_c,
        "available_epochs": np.array(available),
    }
    for r in results:
        prefix = r["label"]
        for key, val in r.items():
            if key == "label":
                continue
            save_dict[f"{prefix}_{key}"] = val
    np.savez(out_npz, **save_dict)
    print(f"Saved data to {out_npz}")

    make_plot(results, out_pdf, args)


if __name__ == "__main__":
    main()
