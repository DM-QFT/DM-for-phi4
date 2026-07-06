"""
3D phi^4 DM-MALA tau_int / ESS sweep over the step-size coefficient c at a
fixed (converged) training epoch.  Multi-L checkpoint evaluated on a single
fixed lattice size L.  Mirror of 2Dphi4/acceptance_rate/dm_mc_tau_vs_c.py
with h = c / L^3.

Usage:
  python analysis/dm_mc_tau_vs_c_3d.py \
      --device cuda:0 --k 0.2 \
      --run_dir runs/phi4_3d_Lmulti4-8-16-32_k0.2_l0.9_ncsnpp \
      --epoch 20000 \
      --L 64 \
      --c_list 0.01,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0 \
      --t_mh 1e-4 --num_chains 64 --mh_steps 200 --ema \
      --out_npz analysis/results/dm_mc_tau_vs_c_3d_L64_multiL_k0.2_ep20000_ema.npz
"""

import argparse
import functools
import io
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.append(str(HERE.parents[1]))      # 3Dphi4/
sys.path.append(str(HERE.parents[2]))      # repo root

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from phi4_action import phi4_action as _phi4_action_nd
phi4_action_3d = functools.partial(_phi4_action_nd, spatial_dims=3)


def find_epoch_checkpoints(model_dir):
    ckpts = {}
    for f in sorted(model_dir.glob("epoch=*.ckpt")):
        if "-v" in f.stem:
            continue
        m = re.search(r"epoch=(\d+)$", f.stem)
        if m:
            ckpts[int(m.group(1))] = f
    return dict(sorted(ckpts.items()))


def load_model(ckpt_path, device, k, l):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    sigma = hparams.get("sigma", 800.0)
    norm_min = hparams.get("norm_min")
    norm_max = hparams.get("norm_max")

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    score_model = NCSNpp3D(marginal_prob_std_fn)

    sd = ckpt.get("state_dict", {})
    if any("_orig_mod" in key for key in sd):
        ckpt["state_dict"] = {key.replace("._orig_mod.", "."): v for key, v in sd.items()}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    del ckpt
    buf.seek(0)
    model = DiffusionModel3D.load_from_checkpoint(
        buf, score_model=score_model, map_location=device, weights_only=False
    )
    model = model.to(device).eval()
    if hasattr(model, "ema"):
        model.ema.to(device)
    action_fn = functools.partial(
        phi4_action_3d, k=k, l=l, phi_min=norm_min, phi_max=norm_max
    )
    return model, action_fn, dict(sigma_ve=sigma, norm_min=norm_min, norm_max=norm_max)


def load_reference_data(data_path, num_samples, norm_min, norm_max, device, seed=42):
    with h5py.File(data_path, "r") as f:
        cfgs = np.array(f["cfgs"])
    rng = np.random.default_rng(seed)
    n_avail = cfgs.shape[0]
    if num_samples > n_avail:
        idx = rng.choice(n_avail, size=num_samples, replace=True)
    else:
        idx = rng.choice(n_avail, size=num_samples, replace=False)
    cfgs = cfgs[idx]
    cfgs_norm = ((cfgs - norm_min) / (norm_max - norm_min) - 0.5) * 2.0
    x = torch.tensor(cfgs_norm, dtype=torch.float32, device=device)
    if x.ndim == 4:                    # (M, L, L, L) -> (M, 1, L, L, L)
        x = x.unsqueeze(1)
    return x


# -------------------- ACF + Sokal (identical to vs_epoch) --------------------

def acf_per_chain_fft(series):
    T, M = series.shape
    x = series - series.mean(axis=0, keepdims=True)
    n2 = 1 << int(np.ceil(np.log2(2 * T)))
    f = np.fft.rfft(x, n=n2, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n2, axis=0)[:T]
    norm = (T - np.arange(T))[:, None]
    acf = acf / norm
    rho = acf / acf[0:1]
    return rho


def sokal_window(rho_mean, c=5.0):
    T = len(rho_mean)
    cum = np.cumsum(rho_mean[1:])
    tau_hat = 0.5 + cum
    Ws = np.arange(1, T)
    ok = Ws >= c * tau_hat
    if ok.any():
        W = int(Ws[np.argmax(ok)])
    else:
        W = int(Ws[-1])
    return float(tau_hat[W - 1]), W


def cutoff_window(rho_mean, cutoff=0.05):
    tau = 0.5
    for t in range(1, len(rho_mean)):
        if rho_mean[t] < cutoff:
            break
        tau += rho_mean[t]
    return tau


def estimate_tau(series, sokal_c=5.0, acf_cutoff=0.05):
    T, M = series.shape
    var = series.var(axis=0)
    if (var <= 0).all():
        return T / 2.0, T - 1, 0.0, T / 2.0
    rho_pc = acf_per_chain_fft(series)
    bad = ~np.isfinite(rho_pc).all(axis=0)
    if bad.any():
        rho_pc[:, bad] = 1.0
    rho_mean = rho_pc.mean(axis=1)
    tau_s, W = sokal_window(rho_mean, c=sokal_c)
    tau_cut = cutoff_window(rho_mean, cutoff=acf_cutoff)
    tau_se = float(np.sqrt(2.0 * (2 * W + 1) / (M * T)) * tau_s)
    return tau_s, W, tau_se, tau_cut


def acf_grand_mean(series):
    """ACF with grand-mean (over chains and time) subtraction, robust for
    short chains starting from i.i.d. equilibrium samples."""
    T, M = series.shape
    grand_mean = series.mean()
    x = series - grand_mean
    n2 = 1 << int(np.ceil(np.log2(2 * T)))
    f = np.fft.rfft(x, n=n2, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n2, axis=0)[:T]
    norm = (T - np.arange(T))[:, None]
    acf = acf / norm
    acf_mean = acf.mean(axis=1)
    if acf_mean[0] <= 0 or not np.isfinite(acf_mean[0]):
        return np.full(T, np.nan)
    return acf_mean / acf_mean[0]


def estimate_tau_fit(series, threshold=0.1, min_points=5):
    """Log-linear single-exp fit of the grand-mean ACF over the leading
    segment where rho(t) > threshold.  More robust than Sokal for N << tau_true."""
    T, M = series.shape
    rho = acf_grand_mean(series)
    if not np.isfinite(rho).all():
        return np.nan, 0, rho
    valid = rho > threshold
    if not valid[0]:
        return np.nan, 0, rho
    first_invalid = np.where(~valid)[0]
    last_t = first_invalid[0] if len(first_invalid) > 0 else T
    if last_t < min_points:
        return np.nan, last_t, rho
    t_fit = np.arange(last_t, dtype=np.float64)
    y_fit = np.log(rho[:last_t])
    slope, _ = np.polyfit(t_fit, y_fit, 1)
    if slope >= 0:
        return np.nan, last_t, rho
    return -1.0 / slope, last_t, rho


def ess_from_tau(M, N, tau):
    if not np.isfinite(tau) or tau <= 0:
        return 0.0
    return float(M * N / (2.0 * tau))


# -------------------- MALA chain (3D) --------------------

@torch.no_grad()
def run_mala_chain_3d(model, action_fn, x0, h, t_mh, n_steps, batch_size=None):
    device = x0.device
    M = x0.shape[0]
    h_t = torch.tensor(h, device=device, dtype=x0.dtype)
    sqrt_2h = torch.sqrt(2 * h_t)

    if batch_size is None or batch_size >= M:
        def score(z):
            return model(z, torch.full((z.shape[0],), t_mh,
                                       device=device, dtype=z.dtype))
    else:
        def score(z):
            outs = []
            for i in range(0, z.shape[0], batch_size):
                zi = z[i:i + batch_size]
                ti = torch.full((zi.shape[0],), t_mh, device=device, dtype=zi.dtype)
                outs.append(model(zi, ti))
            return torch.cat(outs, dim=0)

    spatial_dims = tuple(range(1, x0.ndim))     # (1,2,3,4)

    phi_bar = np.empty((n_steps + 1, M), dtype=np.float64)
    S_chain = np.empty((n_steps + 1, M), dtype=np.float64)
    accept_log = np.empty((n_steps, M), dtype=np.float32)

    x = x0.clone()
    S_x = action_fn(x)
    phi_bar[0] = x.mean(dim=spatial_dims).cpu().numpy()
    S_chain[0] = S_x.cpu().numpy()

    for step in range(n_steps):
        score_x = score(x)
        noise = torch.randn_like(x)
        y = x + h_t * score_x + sqrt_2h * noise
        score_y = score(y)
        drift = h_t * (score_x + score_y)
        log_q = -(0.25 / h_t) * torch.sum((sqrt_2h * noise + drift) ** 2,
                                          dim=spatial_dims) \
                + 0.5 * torch.sum(noise ** 2, dim=spatial_dims)
        S_y = action_fn(y)
        log_pi = S_x - S_y
        accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
        accept = torch.rand(M, device=device) < accept_prob
        x = torch.where(accept[(slice(None),) + (None,) * (x.ndim - 1)], y, x)
        S_x = torch.where(accept, S_y, S_x)
        accept_log[step] = accept.cpu().numpy().astype(np.float32)
        phi_bar[step + 1] = x.mean(dim=spatial_dims).cpu().numpy()
        S_chain[step + 1] = S_x.cpu().numpy()

    return phi_bar, S_chain, accept_log


def nearest_epoch(available, target):
    return min(available, key=lambda e: abs(e - target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Multi-L checkpoint dir, e.g. "
                             "runs/phi4_3d_Lmulti4-8-16-32_k0.2_l0.9_ncsnpp")
    parser.add_argument("--epoch", type=int, default=20000)
    parser.add_argument("--L", type=int, required=True,
                        help="Single lattice size at which to sample.")
    parser.add_argument("--c_list", type=str,
                        default="0.01,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0",
                        help="Comma-separated c values; h = c / L^3.")
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--data_root", type=str, default="trainingdata")
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--num_chains", type=int, default=64)
    parser.add_argument("--mh_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--out_npz", required=True)
    args = parser.parse_args()

    os.chdir(str(HERE.parents[1]))

    L = args.L
    c_list = [float(x) for x in args.c_list.split(",") if x.strip()]
    print(f"[{args.device}] L = {L}")
    print(f"[{args.device}] c_list = {c_list}")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = HERE.parents[1] / args.run_dir
    ckpts = find_epoch_checkpoints(run_dir / "models")
    available = sorted(ckpts.keys())
    if not available:
        raise RuntimeError(f"No ckpts in {run_dir / 'models'}")
    ep = nearest_epoch(available, args.epoch)
    print(f"[{args.device}] target epoch = {args.epoch}, using nearest = {ep}")
    print(f"[{args.device}] ckpt: {ckpts[ep]}")

    ref_ckpt = torch.load(ckpts[ep], map_location="cpu", weights_only=False)
    hparams = ref_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min")
    norm_max = hparams.get("norm_max")
    sigma_ve = hparams.get("sigma", None)
    del ref_ckpt
    print(f"[{args.device}] pooled norm = [{norm_min:.4f}, {norm_max:.4f}], "
          f"sigma_VE = {sigma_ve}")

    data_path = (Path(args.data_root) /
                 f"cfgs_wolff_fahmc_k={args.k}_l={args.l}_{L}^3.jld2")
    if not data_path.exists():
        raise RuntimeError(f"missing training data: {data_path}")
    x_ref = load_reference_data(
        str(data_path), args.num_chains,
        norm_min, norm_max, args.device, seed=args.seed,
    )
    print(f"[{args.device}] L={L}: x_ref {tuple(x_ref.shape)}")

    print(f"[{args.device}] loading model ...")
    model, _, _ = load_model(ckpts[ep], args.device, args.k, args.l)
    if args.ema and hasattr(model, "ema"):
        model.ema.copy_to()
    action_fn = functools.partial(
        phi4_action_3d, k=args.k, l=args.l,
        phi_min=norm_min, phi_max=norm_max,
    )

    n_c = len(c_list)
    def _empty():
        return np.full(n_c, np.nan)
    res_tau_M, res_tau_M_se, res_W_M, res_cut_M = _empty(), _empty(), _empty(), _empty()
    res_tau_S, res_tau_S_se, res_W_S, res_cut_S = _empty(), _empty(), _empty(), _empty()
    res_tau_M_fit, res_n_M_fit = _empty(), _empty()
    res_tau_S_fit, res_n_S_fit = _empty(), _empty()
    res_ess_M, res_ess_S, res_ess_eff = _empty(), _empty(), _empty()
    res_acc = _empty()

    torch.manual_seed(args.seed)
    for ic, c in enumerate(c_list):
        t1 = time.time()
        h = c / L ** 3
        phi_bar, S_chain, acc_log = run_mala_chain_3d(
            model, action_fn, x_ref,
            h, args.t_mh, args.mh_steps,
            batch_size=args.batch_size,
        )
        M, N = phi_bar.shape[1], phi_bar.shape[0] - 1
        series_M = np.abs(phi_bar[1:])
        series_S = S_chain[1:]
        tau_M, W_M, se_M, cut_M = estimate_tau(series_M)
        tau_S, W_S, se_S, cut_S = estimate_tau(series_S)
        tau_M_fit, n_M_fit, _ = estimate_tau_fit(series_M)
        tau_S_fit, n_S_fit, _ = estimate_tau_fit(series_S)
        tau_M_use = tau_M_fit if np.isfinite(tau_M_fit) else tau_M
        tau_S_use = tau_S_fit if np.isfinite(tau_S_fit) else tau_S
        ess_M  = ess_from_tau(M, N, tau_M_use)
        ess_S  = ess_from_tau(M, N, tau_S_use)
        ess_eff = ess_from_tau(M, N, max(tau_M_use, tau_S_use))
        acc_rate = float(acc_log.mean())
        res_tau_M[ic], res_tau_M_se[ic] = tau_M, se_M
        res_W_M[ic], res_cut_M[ic] = W_M, cut_M
        res_tau_S[ic], res_tau_S_se[ic] = tau_S, se_S
        res_W_S[ic], res_cut_S[ic] = W_S, cut_S
        res_tau_M_fit[ic], res_n_M_fit[ic] = tau_M_fit, n_M_fit
        res_tau_S_fit[ic], res_n_S_fit[ic] = tau_S_fit, n_S_fit
        res_ess_M[ic], res_ess_S[ic] = ess_M, ess_S
        res_ess_eff[ic] = ess_eff
        res_acc[ic] = acc_rate
        print(f"[{args.device}] c={c:7.3f}  h={h:.2e}  acc={acc_rate:.3f}  "
              f"tau_fit|M|={tau_M_fit:6.2f}(n={int(n_M_fit) if np.isfinite(n_M_fit) else 0})  "
              f"tau_fit_S={tau_S_fit:6.2f}(n={int(n_S_fit) if np.isfinite(n_S_fit) else 0})  "
              f"[Sokal: {tau_M:5.2f}/{tau_S:5.2f}]  "
              f"ESS_eff={ess_eff:7.1f}  ({time.time()-t1:.1f}s)")

    out = Path(args.out_npz)
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        out,
        c_list=np.array(c_list), L=L,
        epoch=ep,
        # Sokal-window (back-compat)
        tau_sokal=res_tau_M, tau_se=res_tau_M_se,
        sokal_W=res_W_M, tau_cutoff=res_cut_M,
        tau_S=res_tau_S, tau_S_se=res_tau_S_se,
        sokal_W_S=res_W_S, tau_cutoff_S=res_cut_S,
        # Exponential-fit estimates (recommended primary)
        tau_M_fit=res_tau_M_fit, n_M_fit=res_n_M_fit,
        tau_S_fit=res_tau_S_fit, n_S_fit=res_n_S_fit,
        ess_M=res_ess_M, ess_S=res_ess_S, ess_eff=res_ess_eff,
        accept=res_acc,
        t_mh=args.t_mh, k=args.k, l=args.l,
        num_chains=args.num_chains, mh_steps=args.mh_steps,
        norm_min=norm_min, norm_max=norm_max, sigma_ve=sigma_ve,
        run_dir=str(run_dir),
    )
    print(f"[{args.device}] Saved npz -> {out}")

    # --- quick-look 2-panel plot (tau and ESS vs c) ---
    plt.rcParams.update({
        'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'lines.linewidth': 1.5, 'lines.markersize': 5,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })
    c_arr = np.array(c_list)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(c_arr, res_tau_M_fit, marker='o', color='C0',
             label=r'$\tau_{\mathrm{fit}}(|\bar\phi|)$')
    ax1.plot(c_arr, res_tau_S_fit, marker='s', color='C1',
             label=r'$\tau_{\mathrm{fit}}(S)$')
    ax1.plot(c_arr, res_tau_M, marker='.', ls=':', color='C0', alpha=0.5,
             label=r'$\tau_{\mathrm{Sokal}}(|\bar\phi|)$')
    ax1.plot(c_arr, res_tau_S, marker='.', ls=':', color='C1', alpha=0.5,
             label=r'$\tau_{\mathrm{Sokal}}(S)$')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel(r'$c\;(h = c / L^3)$'); ax1.set_ylabel(r'$\tau_{\mathrm{int}}$')
    ax1.grid(alpha=0.3); ax1.legend(loc='best', fontsize=8)

    ax2.plot(c_arr, res_ess_M, marker='o', color='C0', label=r'$\mathrm{ESS}_{|\bar\phi|}$')
    ax2.plot(c_arr, res_ess_S, marker='s', color='C1', label=r'$\mathrm{ESS}_S$')
    ax2.plot(c_arr, res_ess_eff, marker='x', ls='--', color='gray',
             label=r'$\mathrm{ESS}_{\mathrm{eff}}$')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel(r'$c\;(h = c / L^3)$'); ax2.set_ylabel('ESS')
    ax2.grid(alpha=0.3); ax2.legend(loc='best')

    fig.suptitle(rf'3D: $L={L},\;\kappa={args.k},\;\text{{epoch}}={ep},\;t_{{\mathrm{{mh}}}}={args.t_mh:.0e}$',
                 y=1.03)
    fig.tight_layout()
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out_pdf)
    fig.savefig(out.with_suffix(".png"))
    print(f"[{args.device}] Saved pdf -> {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
