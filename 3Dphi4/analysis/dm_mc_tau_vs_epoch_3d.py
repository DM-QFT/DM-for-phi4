"""
For a multi-L 3D phi^4 score model, sweep training epoch and measure
tau_int(|M|) under DM-driven MALA at several volumes L. Each (L, epoch)
runs M parallel chains starting from HMC training cfgs.

Designed to be split across GPUs by giving disjoint L_list to each worker.

Convention: use the model's stored (pooled) norm range for both data
normalisation and the action_fn, matching the existing 3D acceptance
scripts (acceptance_vs_epoch.py, calibrate_step_size.py).

Usage (one worker per device):
  python analysis/dm_mc_tau_vs_epoch_3d.py \
      --device cuda:2 --L_list 8,16,32 \
      --num_chains 64 --mh_steps 200 \
      --out_npz results/dm_mc_tau_vs_epoch_3d_part_a.npz
"""

import argparse
import functools
import io
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

from networks_3d import NCSNpp3D
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
# Import phi4_action from the root module directly: 3Dphi4/sample_phi4.py
# pulls in scipy at module level (cumulants_from_moments), which isn't
# installed in nenv2 on Miyabi.
from phi4_action import phi4_action as _phi4_action_nd
phi4_action_3d = functools.partial(_phi4_action_nd, spatial_dims=3)


# -------------------- ckpt + model loading --------------------

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
    """Returns (M, 1, L, L, L) torch tensor in normalised space."""
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
    if x.ndim == 4:                   # (M, L, L, L) -> (M, 1, L, L, L)
        x = x.unsqueeze(1)
    return x


# -------------------- ACF + Sokal --------------------

def acf_per_chain_fft(series):
    """Normalised ACF per chain. series: (T, M)."""
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


# -------------------- MALA chain (3D) --------------------

@torch.no_grad()
def run_mala_chain_3d(model, action_fn, x0, h, t_mh, n_steps, batch_size=None):
    """Run M parallel MALA chains, tracking both |phi_bar| and S along the
    chain. action_fn(x) is computed only once per step (the current x's
    action is cached and rolled forward through accept/reject)."""
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

    spatial_dims = tuple(range(1, x0.ndim))    # collapse C, L, L, L

    phi_bar = np.empty((n_steps + 1, M), dtype=np.float64)
    S_chain = np.empty((n_steps + 1, M), dtype=np.float64)
    accept_log = np.empty((n_steps, M), dtype=np.float32)

    x = x0.clone()
    S_x = action_fn(x)                                  # initial action
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
        log_pi = S_x - S_y                              # cached S_x, fresh S_y
        accept_prob = torch.exp(log_pi + log_q).clamp(max=1.0)
        accept = torch.rand(M, device=device) < accept_prob
        x = torch.where(accept[(slice(None),) + (None,) * (x.ndim - 1)], y, x)
        S_x = torch.where(accept, S_y, S_x)             # roll cached S forward
        accept_log[step] = accept.cpu().numpy().astype(np.float32)
        phi_bar[step + 1] = x.mean(dim=spatial_dims).cpu().numpy()
        S_chain[step + 1] = S_x.cpu().numpy()

    return phi_bar, S_chain, accept_log


def estimate_tau(series, sokal_c=5.0, acf_cutoff=0.05):
    """Generic Sokal-window tau_int for an arbitrary (T, M) series.

    Returns (tau_sokal, W, tau_se, tau_cutoff).  When all chains have zero
    variance (e.g. the order parameter never moved), we fall back to T/2 to
    flag the "fully autocorrelated" regime."""
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


def ess_from_tau(M, N, tau):
    """ESS = M*N / (2 tau).  Falls back to 0 if tau is non-finite."""
    if not np.isfinite(tau) or tau <= 0:
        return 0.0
    return float(M * N / (2.0 * tau))


# -------------------- main --------------------

def select_log_epochs(available, n_target):
    if len(available) <= n_target:
        return list(available)
    log_idx = np.unique(np.round(
        np.geomspace(1, len(available), n_target)
    ).astype(int) - 1)
    return [available[i] for i in log_idx[:n_target]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--run_dir", type=str,
                        default="runs/phi4_3d_Lmulti4-8-16-32_k0.1923_l0.9_ncsnpp")
    parser.add_argument("--L_list", required=True,
                        help="comma-separated, e.g. 8,16,32 or 64")
    parser.add_argument("--epochs", type=str, default=None,
                        help="explicit epoch list; default = log-spaced from available.")
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--k", type=float, default=0.1923)
    parser.add_argument("--l", type=float, default=0.9)
    parser.add_argument("--data_root", type=str, default="trainingdata")
    parser.add_argument("--c", type=float, default=0.5,
                        help="3D MALA step coefficient, h = c/L^3 (DM.tex 3D: 0.5).")
    parser.add_argument("--t_mh", type=float, default=1e-4)
    parser.add_argument("--num_chains", type=int, default=64)
    parser.add_argument("--mh_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Sub-batch over chains for the score forward; default = M.")
    parser.add_argument("--ema", action="store_true",
                        help="Apply EMA-averaged weights before sampling.")
    parser.add_argument("--out_npz", required=True)
    args = parser.parse_args()

    import os
    os.chdir(str(HERE.parents[1]))      # 3Dphi4/

    L_list = [int(x) for x in args.L_list.split(",") if x.strip()]
    print(f"[{args.device}] L_list = {L_list}")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = HERE.parents[1] / args.run_dir
    ckpts = find_epoch_checkpoints(run_dir / "models")
    available = sorted(ckpts.keys())

    if args.epochs:
        epochs = [int(e) for e in args.epochs.split(",") if e.strip()]
        epochs = [min(available, key=lambda e: abs(e - tgt)) for tgt in epochs]
    else:
        epochs = select_log_epochs(available, args.n_epochs)
    epochs = sorted(set(epochs))
    print(f"[{args.device}] {len(epochs)} epochs: {epochs}")

    # Read pooled norm from any ckpt; assumed identical across ckpts of same run.
    ref_ckpt = torch.load(ckpts[epochs[0]], map_location="cpu", weights_only=False)
    hparams = ref_ckpt.get("hyper_parameters", {})
    norm_min = hparams.get("norm_min")
    norm_max = hparams.get("norm_max")
    sigma_ve = hparams.get("sigma", None)
    del ref_ckpt
    print(f"[{args.device}] pooled norm = [{norm_min:.4f}, {norm_max:.4f}], "
          f"sigma_VE = {sigma_ve}")

    # Pre-sample HMC starting cfgs for each L (same set across all epochs).
    x_ref_per_L = {}
    for L in L_list:
        data_path = (Path(args.data_root) /
                     f"cfgs_wolff_fahmc_k={args.k}_l={args.l}_{L}^3.jld2")
        if not data_path.exists():
            raise RuntimeError(f"missing training data: {data_path}")
        x_ref_per_L[L] = load_reference_data(
            str(data_path), args.num_chains,
            norm_min, norm_max, args.device, seed=args.seed,
        )
        print(f"[{args.device}] L={L}: x_ref {tuple(x_ref_per_L[L].shape)}")

    # Result arrays: (n_epochs, n_L), separate columns for |M| and S.
    n_e, n_L = len(epochs), len(L_list)
    def _empty():
        return np.full((n_e, n_L), np.nan)
    res_tau_M, res_tau_M_se, res_W_M, res_cut_M = _empty(), _empty(), _empty(), _empty()
    res_tau_S, res_tau_S_se, res_W_S, res_cut_S = _empty(), _empty(), _empty(), _empty()
    res_ess_M, res_ess_S, res_ess_eff = _empty(), _empty(), _empty()
    res_acc = _empty()

    torch.manual_seed(args.seed)
    for ie, ep in enumerate(epochs):
        t0 = time.time()
        model, _, _ = load_model(ckpts[ep], args.device, args.k, args.l)
        if args.ema and hasattr(model, "ema"):
            model.ema.copy_to()
        for iL, L in enumerate(L_list):
            t1 = time.time()
            h = args.c / L ** 3
            action_fn = functools.partial(
                phi4_action_3d, k=args.k, l=args.l,
                phi_min=norm_min, phi_max=norm_max,
            )
            phi_bar, S_chain, acc_log = run_mala_chain_3d(
                model, action_fn, x_ref_per_L[L],
                h, args.t_mh, args.mh_steps,
                batch_size=args.batch_size,
            )
            M, N = phi_bar.shape[1], phi_bar.shape[0] - 1
            tau_M, W_M, se_M, cut_M = estimate_tau(np.abs(phi_bar[1:]))
            tau_S, W_S, se_S, cut_S = estimate_tau(S_chain[1:])
            ess_M  = ess_from_tau(M, N, tau_M)
            ess_S  = ess_from_tau(M, N, tau_S)
            ess_eff = ess_from_tau(M, N, max(tau_M, tau_S))
            acc_rate = float(acc_log.mean())
            res_tau_M[ie, iL], res_tau_M_se[ie, iL] = tau_M, se_M
            res_W_M[ie, iL], res_cut_M[ie, iL] = W_M, cut_M
            res_tau_S[ie, iL], res_tau_S_se[ie, iL] = tau_S, se_S
            res_W_S[ie, iL], res_cut_S[ie, iL] = W_S, cut_S
            res_ess_M[ie, iL], res_ess_S[ie, iL] = ess_M, ess_S
            res_ess_eff[ie, iL] = ess_eff
            res_acc[ie, iL] = acc_rate
            print(f"[{args.device}] ep={ep:>5d}  L={L:>3d}  acc={acc_rate:.3f}  "
                  f"tau|M|={tau_M:6.2f}+/-{se_M:4.2f}  tau_S={tau_S:6.2f}+/-{se_S:4.2f}  "
                  f"ESS|M|={ess_M:7.1f}  ESS_S={ess_S:7.1f}  ESS_eff={ess_eff:7.1f}  "
                  f"({time.time()-t1:.1f}s)")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[{args.device}] epoch {ep} total {time.time()-t0:.1f}s")

    out = Path(args.out_npz)
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        out,
        epochs=np.array(epochs), L_list=np.array(L_list),
        # |M| diagnostics (back-compat: same field names as before)
        tau_sokal=res_tau_M, tau_se=res_tau_M_se,
        sokal_W=res_W_M, tau_cutoff=res_cut_M,
        # S diagnostics
        tau_S=res_tau_S, tau_S_se=res_tau_S_se,
        sokal_W_S=res_W_S, tau_cutoff_S=res_cut_S,
        # ESS
        ess_M=res_ess_M, ess_S=res_ess_S, ess_eff=res_ess_eff,
        accept=res_acc,
        c=args.c, t_mh=args.t_mh, k=args.k, l=args.l,
        num_chains=args.num_chains, mh_steps=args.mh_steps,
        norm_min=norm_min, norm_max=norm_max, sigma_ve=sigma_ve,
    )
    print(f"[{args.device}] Saved -> {out}")


if __name__ == "__main__":
    main()
