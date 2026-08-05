"""Effective inverse propagator R(k,t) = -u_k^T J u_k from the trained score network.

For s = grad log p_t and near-Gaussian p_t: R(k,t) ~ 1/(G(k) + sigma_t^2), so
G_net(k,t) = 1/R - sigma_t^2 exposes the critical propagator k^-(2-eta).
"""

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from networks import NCSNpp2D
from diffusion_lightning import DiffusionModel, marginal_prob_std


def plane_wave_probes(L, device):
    # cos/sin along x and y for n=0..L/2; downsampling breaks exact translation
    # invariance, so projecting the response back onto the same wave extracts the
    # translation-averaged diagonal of J.
    x = torch.arange(L, device=device, dtype=torch.float32)
    probes, labels = [], []
    for n in range(L // 2 + 1):
        k = 2 * np.pi * n / L
        row_c = torch.cos(k * x).expand(L, L)
        waves = [row_c, row_c.T]
        if 0 < n < L // 2:
            row_s = torch.sin(k * x).expand(L, L)
            waves += [row_s, row_s.T]
        for w in waves:
            probes.append(w)
            labels.append(n)
    U = torch.stack(probes)[:, None]
    return U / U.flatten(1).norm(dim=1).view(-1, 1, 1, 1), np.array(labels)


def response(score_model, phi0, t, U, chunk=256):
    # J is block-diagonal over the batch, so one jvp call gives J u per sample
    B, P = phi0.shape[0], U.shape[0]
    x = phi0.repeat_interleave(P, 0)
    v = U.repeat(B, 1, 1, 1)
    R = torch.empty(B * P, device=x.device)
    for i in range(0, B * P, chunk):
        xc, vc = x[i:i + chunk], v[i:i + chunk]
        tt = torch.full((xc.shape[0],), t, device=x.device)
        _, Jv = torch.func.jvp(lambda y: score_model(y, tt), (xc,), (vc,))
        R[i:i + chunk] = -(Jv * vc).flatten(1).sum(1)  # probes are unit-norm
    return R.view(B, P).mean(0)


def score_covariance(score_model, phi_t, t, L, chunk=256):
    # axis modes of <|s_k|^2>/V; for a Gaussian theory this equals 1/(G+sigma_t^2)
    S = torch.zeros(L, L // 2 + 1, device=phi_t.device)
    for i in range(0, phi_t.shape[0], chunk):
        xc = phi_t[i:i + chunk]
        tt = torch.full((xc.shape[0],), t, device=xc.device)
        s = score_model(xc, tt)[:, 0]
        S += (torch.fft.rfft2(s).abs() ** 2).sum(0)
    S /= phi_t.shape[0] * L * L
    Sx = S[0, : L // 2 + 1]                       # k along x
    Sy = 0.5 * (S[1: L // 2, 0] + S[L // 2 + 1:, 0].flip(0))  # k along y, +-k averaged
    R = Sx.clone()
    R[1: L // 2] = 0.5 * (Sx[1: L // 2] + Sy)
    R[L // 2] = 0.5 * (Sx[L // 2] + S[L // 2, 0])
    return R.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--probe", choices=["zero", "ensemble", "cov"], default="zero")
    p.add_argument("--cfgs", default=None, help="jld2 file for ensemble/cov probes")
    p.add_argument("--n_cfg", type=int, default=16)
    p.add_argument("--t_min", type=float, default=1e-4)
    p.add_argument("--t_max", type=float, default=1.0)
    p.add_argument("--n_t", type=int, default=25)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    sigma, nmin, nmax = hp["sigma"], hp["norm_min"], hp["norm_max"]
    score_model = NCSNpp2D(functools.partial(marginal_prob_std, sigma=sigma))
    model = DiffusionModel.load_from_checkpoint(
        args.ckpt, score_model=score_model, map_location=args.device,
        weights_only=False)
    model.eval()

    L = args.L
    U, ns = plane_wave_probes(L, args.device)
    if args.probe == "zero":
        phi0_base = torch.zeros(1, 1, L, L, device=args.device)
    else:
        import h5py
        with h5py.File(args.cfgs, "r") as f:
            cfgs = f["cfgs"][: args.n_cfg].astype(np.float32)
        phi0_base = torch.from_numpy(
            2 * (cfgs - nmin) / (nmax - nmin) - 1).to(args.device)[:, None]

    ts = np.geomspace(args.t_min, args.t_max, args.n_t)
    ema_ctx = (torch.no_grad() if args.no_ema else model.ema.average_parameters())
    rows = []
    with ema_ctx, torch.no_grad():
        for t in ts:
            std2 = marginal_prob_std(torch.tensor(t), sigma).item() ** 2
            phi0 = phi0_base
            if args.probe != "zero":
                phi0 = phi0_base + np.sqrt(std2) * torch.randn_like(phi0_base)
            if args.probe == "cov":
                # exact-score identity <s s^T> = -<grad s> makes this a second
                # estimator of R; its gap to the jvp probe maps score error in k
                R = score_covariance(model.score_model, phi0, float(t), L)
            else:
                R = response(model.score_model, phi0, float(t), U).cpu().numpy()
            for n in range(L // 2 + 1):
                k = 2 * np.pi * n / L
                khat = 2 * np.sin(k / 2)  # lattice momentum reduces UV artifacts
                Rn = R[n] if args.probe == "cov" else R[ns == n].mean()
                rows.append((t, std2, n, k, khat, Rn, 1.0 / Rn - std2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hdr = "t\tsigma_t2\tn\tk\tkhat\tR\tG_net"
    np.savetxt(out, np.array(rows), fmt="%.8g", delimiter="\t", header=hdr)
    print(f"wrote {out}  ({len(rows)} rows, L={L}, probe={args.probe}, "
          f"ema={not args.no_ema})")


if __name__ == "__main__":
    main()
