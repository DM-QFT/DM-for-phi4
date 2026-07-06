"""
MALA refinement of cross-L DM ensembles for 3D phi^4.

Phase-2-only sampler. Initial configurations come from a saved reverse-SDE
ensemble (.npy, physical field, shape (L, L, L, N)). This script runs one
independent Metropolis-adjusted Langevin chain per configuration, targeting
pi(phi) ~ exp(-S[phi]) with the exact lattice action. The stationary
distribution is exact for ANY drift; the drift only affects acceptance and
mixing speed.

Proposal in normalized field space x in [-1, 1], step h = c / L^3:

    y = x + h * s(x) + sqrt(2h) * xi,      xi ~ N(0, 1)

with s = s_theta(x, t_mh) (learned score, --drift score) or
s = -dS/dx (exact gradient, --drift exact).

MH ratio, derived from q(y|x) = N(y; x + h s(x), 2h I):

    log q(y|x) = -||y - x - h s(x)||^2 / (4h)          = -||xi||^2 / 2
    log q(x|y) = -||x - y - h s(y)||^2 / (4h)
               = -||sqrt(2h) xi + h (s(x) + s(y))||^2 / (4h)
    log alpha  = S(x) - S(y) + log q(x|y) - log q(y|x)

Implementation notes:
  * The action and all MH reductions are evaluated in float64 outside any
    autocast region (S ~ O(1e5) at L=64; low precision would corrupt dS).
  * Score/action of the current state are cached: an accepted proposal
    reuses s(y) -> s(x) and S(y) -> S(x), so each step costs one drift
    evaluation and one action evaluation.
  * The network is evaluated in microbatches to bound memory at 64^3.
  * State (fields, RNG, traces) is checkpointed every --state_every steps
    and the chain can be resumed with --resume (walltime safety).
  * Per-step traces of per-chain S and magnetization M are recorded, so
    plateau analysis has full resolution without relying on snapshots.

Outputs under {run_dir}/data_crossL_mala/{tag}/:
    snap_step{n:06d}.npy   physical field (L, L, L, N) at log-spaced steps
    final.npy              physical field (L, L, L, N) after n_steps
    traces.npz             S_trace, M_trace (n_steps, N), acc_trace (n_steps,)
    state.pt               resumable chain state
"""

import os
import sys
import argparse
import functools
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from networks_3d import NCSNpp3D, ScoreNet3D, ScoreNet3DUNetPeriodic
from diffusion_lightning_3d import DiffusionModel3D, marginal_prob_std
from phi4_action import phi4_action, phi4_grad_S


def parse_args():
    p = argparse.ArgumentParser(description="MALA refinement of DM ensembles (3D phi^4)")
    p.add_argument("--run_dir", type=str, required=True,
                   help="Training run dir (checkpoint + output location)")
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--l", type=float, default=0.9)
    p.add_argument("--ep", type=int, default=10000)
    p.add_argument("--network", type=str, default="ncsnpp",
                   choices=["scorenet", "unet", "ncsnpp"])
    p.add_argument("--init_npy", type=str, default=None,
                   help="Initial ensemble (.npy, physical, (L,L,L,N)). Default: "
                        "{run_dir}/data_crossL/samples_crossL_train32_sample64_"
                        "em_linear_steps2000_ep{ep}.npy")
    p.add_argument("--L_sample", type=int, default=64)
    p.add_argument("--drift", type=str, default="score", choices=["score", "exact"])
    p.add_argument("--t_mh", type=float, default=1e-4)
    p.add_argument("--c", type=float, default=0.2, help="Step size h = c / L^3")
    p.add_argument("--shift_every", type=int, default=0,
                   help="Every N MALA steps, insert one exact global-shift MH "
                        "update phi -> phi + delta*1 (symmetric proposal, "
                        "alpha = min(1, e^{-dS})). Targets the slow zero mode. "
                        "0 = off.")
    p.add_argument("--shift_std", type=float, default=None,
                   help="Std of the global shift delta in PHYSICAL field "
                        "units (e.g. 2.4*sqrt(chi_HMC/V)). Required if "
                        "--shift_every > 0.")
    p.add_argument("--cluster_every", type=int, default=0,
                   help="Every N MALA steps, apply one embedded Swendsen-Wang "
                        "cluster update (Brower-Tamayo embedding: freeze "
                        "aligned bonds with p=1-exp(-4k phi_x phi_y), flip "
                        "each cluster sign with prob 1/2). Heals minority-"
                        "sign domains / Z2 sector. 0 = off.")
    p.add_argument("--fourier_every", type=int, default=0,
                   help="Every N MALA steps, do one band-limited Fourier "
                        "random-walk: perturb modes with |k|<=fourier_kmax by "
                        "eps*sqrt(G(k))*noise (propagator-preconditioned, "
                        "G measured from the ensemble), symmetric proposal, "
                        "MH-accept against exact S. Drift-free, exact; the "
                        "FA-preconditioned generalization of the global shift "
                        "to a low-k band. 0 = off.")
    p.add_argument("--fourier_kmax", type=float, default=3.0,
                   help="Include Fourier modes with integer radius |k|<=this")
    p.add_argument("--fourier_eps", type=float, default=0.3,
                   help="Random-walk amplitude in units of sqrt(G(k))")
    p.add_argument("--hmc_leap", type=int, default=0,
                   help="If >0, replace the single MALA step by an HMC "
                        "trajectory of this many leapfrog steps. The leapfrog "
                        "force is the LEARNED score (drift), the mass matrix "
                        "is M(k)=1/G_DM(k) (Fourier-accelerated), and the MH "
                        "accepts on the EXACT Hamiltonian H=T+S. Exact for "
                        "e^{-S} for any force (leapfrog is reversible + "
                        "volume-preserving); force quality only sets the "
                        "acceptance. 0 = plain MALA.")
    p.add_argument("--hmc_eps", type=float, default=0.1,
                   help="Leapfrog step size for --hmc_leap")
    p.add_argument("--n_steps", type=int, required=True)
    p.add_argument("--max_cfgs", type=int, default=None,
                   help="Use only the first N configs (short tuning runs)")
    p.add_argument("--cfg_offset", type=int, default=0,
                   help="Skip this many configs before taking max_cfgs "
                        "(multi-node sharding: node i uses offset i*shard)")
    p.add_argument("--microbatch", type=int, default=64)
    p.add_argument("--n_snapshots", type=int, default=24,
                   help="Number of log-spaced ensemble snapshots (0 = only final)")
    p.add_argument("--state_every", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def load_hparams(run_dir, ep):
    ckpt_path = Path(run_dir) / "models" / f"epoch={ep:04d}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    h = ckpt.get("hyper_parameters", {})
    return str(ckpt_path), h["sigma"], h["norm_min"], h["norm_max"]


def build_model(ckpt_path, sigma, network, L_sample, device):
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    if network == "ncsnpp":
        score_model = NCSNpp3D(marginal_prob_std_fn)
    elif network == "scorenet":
        score_model = ScoreNet3D(marginal_prob_std_fn, periodic=True)
    else:
        score_model = ScoreNet3DUNetPeriodic(marginal_prob_std_fn)
    # compile(default): reduce-overhead corrupts F.interpolate(nearest) in
    # NCSNpp; default is the verified-safe mode.
    score_model = torch.compile(score_model, mode="default", dynamic=False)
    # map_location: ckpts were saved on a multi-GPU box (e.g. cuda:3); this
    # node exposes a single device.
    model = DiffusionModel3D.load_from_checkpoint(ckpt_path, score_model=score_model,
                                                  map_location="cpu")
    model.L = L_sample
    return model.to(device).eval()


def main():
    args = parse_args()
    device = torch.device(args.device)
    L, V = args.L_sample, args.L_sample ** 3
    h = args.c / V

    if args.shift_every > 0 and args.shift_std is None:
        raise SystemExit("--shift_std is required when --shift_every > 0")

    ckpt_path, sigma, norm_min, norm_max = load_hparams(args.run_dir, args.ep)
    print(f"Checkpoint: {ckpt_path}")
    print(f"hparams: sigma={sigma}  norm=[{norm_min:.4f},{norm_max:.4f}]")
    print(f"MALA: drift={args.drift}  c={args.c}  h={h:.3e}  t_mh={args.t_mh}  "
          f"n_steps={args.n_steps}  seed={args.seed}")
    if args.shift_every > 0:
        print(f"global-shift MH: every {args.shift_every} steps, "
              f"std={args.shift_std} (physical units)")

    # ---- initial ensemble (physical -> normalized, (N,1,L,L,L)) ----
    init_npy = args.init_npy or str(
        Path(args.run_dir) / "data_crossL" /
        f"samples_crossL_train32_sample{L}_em_linear_steps2000_ep{args.ep}.npy")
    phys = np.load(init_npy).astype(np.float64)          # (L,L,L,N)
    assert phys.shape[:3] == (L, L, L), f"bad init shape {phys.shape}"
    if args.cfg_offset or args.max_cfgs is not None:
        end = None if args.max_cfgs is None else args.cfg_offset + args.max_cfgs
        phys = phys[..., args.cfg_offset:end]
        assert phys.shape[-1] > 0, "empty shard (cfg_offset beyond ensemble)"
    N = phys.shape[-1]
    x = torch.from_numpy(
        (2.0 * (phys - norm_min) / (norm_max - norm_min) - 1.0)
        .transpose(3, 0, 1, 2)[:, None].copy()
    ).float().to(device)                                  # (N,1,L,L,L)
    print(f"Init ensemble: {init_npy}  N={N}")

    # ---- output dir ----
    tag = args.tag or (f"mala_{args.drift}_c{args.c:g}_tmh{args.t_mh:g}"
                       f"_ep{args.ep}_N{N}_seed{args.seed}")
    out_dir = Path(args.run_dir) / "data_crossL_mala" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.pt"

    snap_steps = set()
    if args.n_snapshots > 0:
        snap_steps = set(np.unique(np.geomspace(
            1, args.n_steps, args.n_snapshots).astype(int)).tolist())

    # ---- model / drift ----
    model = None
    if args.drift == "score":
        model = build_model(ckpt_path, sigma, args.network, L, device)

    def drift_fn(z):
        """Drift s(z) in normalized space, fp32, chunked over configs."""
        if args.drift == "exact":
            return -phi4_grad_S(z, args.k, args.l, norm_min, norm_max,
                                spatial_dims=3)
        out = torch.empty_like(z)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            for i in range(0, z.shape[0], args.microbatch):
                zc = z[i:i + args.microbatch]
                tc = torch.full((zc.shape[0],), args.t_mh, device=device)
                out[i:i + args.microbatch] = model(zc, tc).float()
        return out

    def action_fn(z):
        """Exact S per config, float64."""
        return phi4_action(z.double(), args.k, args.l, norm_min, norm_max,
                           spatial_dims=3)

    def to_phys_np(z):
        p = (z[:, 0].float().cpu().numpy() + 1.0) / 2.0 * (norm_max - norm_min) + norm_min
        return p.transpose(1, 2, 3, 0).astype(np.float32)  # (L,L,L,N)

    def magnetization(z):
        scale, off = (norm_max - norm_min) / 2.0, (norm_max + norm_min) / 2.0
        return (z[:, 0].mean(dim=(1, 2, 3)) * scale + off)  # (N,) physical M

    def cluster_update(z):
        """Embedded Swendsen-Wang cluster update (exact for exp(-S)).

        |phi| is held fixed; the sign field is an embedded Ising model with
        couplings J_xy = 2k|phi_x phi_y| on aligned nearest-neighbour pairs.
        Freeze aligned bonds with p = 1-exp(-4k phi_x phi_y), decompose into
        clusters (batched min-label propagation), flip each cluster's sign
        with probability 1/2. Standard SW validity: no accept/reject needed.
        Sign flip in physical space is an affine map in normalized coords.
        """
        pf = (z[:, 0].double() + 1) / 2 * (norm_max - norm_min) + norm_min
        Nc = pf.shape[0]
        bonds = []
        for d in range(1, 4):
            prod = pf * torch.roll(pf, -1, dims=d)      # phi_x * phi_{x+e_d}
            pr = 1.0 - torch.exp(-4.0 * args.k * prod.clamp(min=0.0))
            bonds.append(torch.rand_like(pr) < pr)      # anti-aligned: never
        lbl = (torch.arange(V, device=z.device, dtype=torch.int64)
               .reshape(1, L, L, L).expand(Nc, -1, -1, -1).contiguous())
        while True:
            old = lbl
            for d, b in zip((1, 2, 3), bonds):
                nb = torch.roll(lbl, -1, dims=d)        # label at x+e_d
                lbl = torch.where(b, torch.minimum(lbl, nb), lbl)
                nb = torch.roll(lbl, +1, dims=d)        # label at x-e_d
                bb = torch.roll(b, +1, dims=d)          # bond (x-e_d, x)
                lbl = torch.where(bb, torch.minimum(lbl, nb), lbl)
            if torch.equal(lbl, old):
                break
        coin = torch.rand(Nc, V, device=z.device) < 0.5  # per (cfg, root id)
        flip = torch.gather(coin, 1, lbl.reshape(Nc, V)).reshape(Nc, L, L, L)
        pf = torch.where(flip, -pf, pf)
        zn = (2.0 * (pf - norm_min) / (norm_max - norm_min) - 1.0).float()
        return zn.unsqueeze(1)

    # ---- low-k Fourier heat-bath (drift-free, exact) setup ----
    lowk_mask = Gk = None
    if args.fourier_every > 0:
        fr = torch.fft.fftfreq(L, d=1.0, device=device) * L      # integer index
        kx, ky, kz = torch.meshgrid(fr, fr, fr, indexing="ij")
        kr = torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
        # exclude k=0 (handled by the dedicated global shift kernel)
        lowk_mask = (kr <= args.fourier_kmax) & (kr > 0)        # (L,L,L) bool
        xh0 = torch.fft.fftn(x[:, 0].double(), dim=(1, 2, 3), norm="ortho")
        Gk = (xh0.abs() ** 2).mean(0).clamp_min(1e-12)          # (L,L,L) <|phi_k|^2>
        print(f"Fourier RW: {int(lowk_mask.sum())} modes (k!=0) with "
              f"|k|<={args.fourier_kmax}, eps={args.fourier_eps}, "
              f"every {args.fourier_every} steps")

    fourier_step = None
    if args.fourier_every > 0:
        # per-mode real-space perturbation std, band-limited and preconditioned
        fourier_step = args.fourier_eps * Gk.sqrt() * lowk_mask   # (L,L,L)

    def fourier_rw(z):
        """Band-limited, propagator-preconditioned random-walk proposal.

        Perturb the |k|<=kmax modes by eps*sqrt(G(k))*noise (per mode),
        symmetric so no proposal ratio; the k=0 case reduces to a global
        shift. Returns the proposed normalized field.
        """
        wh = torch.fft.fftn(torch.randn_like(z[:, 0].double()),
                            dim=(1, 2, 3), norm="ortho")
        delta = torch.fft.ifftn(fourier_step * wh, dim=(1, 2, 3),
                                norm="ortho").real
        return (z[:, 0].double() + delta).unsqueeze(1).float()

    # ---- Fourier-accelerated HMC with the learned score as force ----
    # Mass matrix M(k) = 1/G_DM(k) (measured from the ensemble) equalizes the
    # mode frequencies. The leapfrog uses the learned score as force; the MH
    # uses the exact Hamiltonian H = T + S, so the chain is exact for e^{-S}.
    Gmass = None
    if args.hmc_leap > 0:
        xh0 = torch.fft.fftn(x[:, 0].double(), dim=(1, 2, 3), norm="ortho")
        Gmass = (xh0.abs() ** 2).mean(0).clamp_min(1e-8)       # (L,L,L) G(k)
        print(f"FA-HMC: {args.hmc_leap} leapfrog steps, eps={args.hmc_eps}, "
              f"M(k)=1/G_DM(k) cond={float(1/Gmass.min()*Gmass.max()):.1f}")

    def hmc_sample_momentum(n):
        # pi_k ~ CN(0, M_k=1/G(k));  real field via Hermitian white noise
        wh = torch.fft.fftn(torch.randn(n, L, L, L, device=device,
                                        dtype=torch.float64),
                            dim=(1, 2, 3), norm="ortho")
        return torch.fft.ifftn(wh / Gmass.sqrt(), dim=(1, 2, 3),
                               norm="ortho").real                # (n,L,L,L)

    def hmc_kinetic(pi):
        # T = 0.5 * sum_k |pi_k|^2 / M_k = 0.5 * sum_k |pi_k|^2 * G(k)
        pih = torch.fft.fftn(pi, dim=(1, 2, 3), norm="ortho")
        return 0.5 * ((pih.abs() ** 2) * Gmass).sum(dim=(1, 2, 3))  # (n,)

    def hmc_apply_Minv(pi):
        # velocity = M^{-1} pi -> in Fourier multiply by G(k)
        pih = torch.fft.fftn(pi, dim=(1, 2, 3), norm="ortho")
        return torch.fft.ifftn(pih * Gmass, dim=(1, 2, 3), norm="ortho").real

    # ---- state init / resume ----
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    start_step = 0
    accept_total = torch.zeros(N, device=device, dtype=torch.float64)
    shift_accept_total = torch.zeros(N, device=device, dtype=torch.float64)
    n_shift_trials = 0
    fourier_accept_total = torch.zeros(N, device=device, dtype=torch.float64)
    n_fourier_trials = 0
    S_trace, M_trace, acc_trace = [], [], []
    # global shift is proposed in normalized units: delta_x = delta_phys/scale
    shift_scale = (norm_max - norm_min) / 2.0

    if args.resume and state_path.exists():
        # load to CPU: RNG states must stay CPU ByteTensors for set_rng_state
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        x = st["x"].to(device)
        accept_total = st["accept_total"].to(device)
        shift_accept_total = st.get("shift_accept_total",
                                    torch.zeros(N, dtype=torch.float64)).to(device)
        n_shift_trials = st.get("n_shift_trials", 0)
        fourier_accept_total = st.get("fourier_accept_total",
                                      torch.zeros(N, dtype=torch.float64)).to(device)
        n_fourier_trials = st.get("n_fourier_trials", 0)
        start_step = st["step"]
        S_trace = list(st["S_trace"])
        M_trace = list(st["M_trace"])
        acc_trace = list(st["acc_trace"])
        torch.random.set_rng_state(st["cpu_rng"].cpu().to(torch.uint8))
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in st["cuda_rng"]])
        print(f"Resumed from {state_path} at step {start_step}")

    ema_ctx = (model.ema.average_parameters() if model is not None
               else torch.no_grad())

    sqrt2h = float(np.sqrt(2.0 * h))

    def save_state(step):
        tmp = state_path.with_suffix(".pt.tmp")
        torch.save({
            "step": step,
            "x": x.detach().cpu(),
            "accept_total": accept_total.cpu(),
            "shift_accept_total": shift_accept_total.cpu(),
            "n_shift_trials": n_shift_trials,
            "fourier_accept_total": fourier_accept_total.cpu(),
            "n_fourier_trials": n_fourier_trials,
            "S_trace": S_trace, "M_trace": M_trace, "acc_trace": acc_trace,
            "cpu_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "args": vars(args),
        }, tmp)
        os.replace(tmp, state_path)

    def save_traces():
        np.savez(out_dir / "traces.npz",
                 S_trace=np.asarray(S_trace, dtype=np.float64),
                 M_trace=np.asarray(M_trace, dtype=np.float32),
                 acc_trace=np.asarray(acc_trace, dtype=np.float64),
                 accept_total=accept_total.cpu().numpy(),
                 shift_accept_total=shift_accept_total.cpu().numpy(),
                 n_shift_trials=n_shift_trials,
                 n_done=len(acc_trace))

    with torch.no_grad(), ema_ctx:
        s_x = drift_fn(x)
        S_x = action_fn(x)
        print(f"start: <S>/V = {S_x.mean().item() / V:+.6f}   "
              f"<|M|> = {magnetization(x).abs().mean().item():.5f}")

        for step in range(start_step, args.n_steps):
            if args.hmc_leap > 0:
                # ---- FA-HMC trajectory: learned-score force, exact-H MH ----
                eps = args.hmc_eps
                pi = hmc_sample_momentum(N)                     # (N,L,L,L)
                H0 = hmc_kinetic(pi) + S_x
                xh = x.clone()
                # force = grad S ~ -score, so pi update += (eps/2)*score
                s = s_x[:, 0].double()
                pi = pi + 0.5 * eps * s
                for i in range(args.hmc_leap):
                    xh = xh + eps * hmc_apply_Minv(pi).unsqueeze(1).to(xh.dtype)
                    s = drift_fn(xh)[:, 0].double()
                    pi = pi + (eps if i < args.hmc_leap - 1 else 0.5 * eps) * s
                S_new = action_fn(xh)
                dH = (hmc_kinetic(pi) + S_new) - H0
                u = torch.rand(N, device=device, dtype=torch.float64)
                accept = torch.log(u) < -dH
                m = accept.view(N, 1, 1, 1, 1)
                x = torch.where(m, xh, x)
                S_x = torch.where(accept, S_new, S_x)
                s_x = drift_fn(x)
                accept_total += accept.double()
            else:
                # ---- single MALA step ----
                xi = torch.randn_like(x)
                y = x + h * s_x + sqrt2h * xi
                s_y = drift_fn(y)
                S_y = action_fn(y)

                # MH ratio in float64
                r = (sqrt2h * xi + h * (s_x + s_y)).double()
                log_q = (-(0.25 / h) * r.pow(2).sum(dim=(1, 2, 3, 4))
                         + 0.5 * xi.double().pow(2).sum(dim=(1, 2, 3, 4)))
                log_alpha = (S_x - S_y) + log_q

                u = torch.rand(N, device=device, dtype=torch.float64)
                accept = torch.log(u) < log_alpha
                m = accept.view(N, 1, 1, 1, 1)
                x = torch.where(m, y, x)
                s_x = torch.where(m, s_y, s_x)
                S_x = torch.where(accept, S_y, S_x)
                accept_total += accept.double()

            # exact global-shift MH update (targets the slow zero mode):
            # phi -> phi + delta*1, symmetric proposal, alpha = min(1, e^{-dS})
            if args.shift_every > 0 and (step + 1) % args.shift_every == 0:
                delta = (torch.randn(N, 1, 1, 1, 1, device=device)
                         * (args.shift_std / shift_scale))
                y_s = x + delta
                S_ys = action_fn(y_s)
                acc_s = (torch.log(
                    torch.rand(N, device=device, dtype=torch.float64))
                    < (S_x - S_ys))
                m_s = acc_s.view(N, 1, 1, 1, 1)
                x = torch.where(m_s, y_s, x)
                S_x = torch.where(acc_s, S_ys, S_x)
                shift_accept_total += acc_s.double()
                n_shift_trials += 1
                if bool(acc_s.any()):
                    # cached drift is invalid for shifted chains
                    s_x = drift_fn(x)

            # embedded SW cluster update (heals minority-sign domains)
            if (args.cluster_every > 0
                    and (step + 1) % args.cluster_every == 0):
                x = cluster_update(x)
                S_x = action_fn(x)
                s_x = drift_fn(x)

            # low-k Fourier random-walk (drift-free exact IR update)
            if (args.fourier_every > 0
                    and (step + 1) % args.fourier_every == 0):
                y_f = fourier_rw(x)
                S_yf = action_fn(y_f)
                log_a = S_x - S_yf                     # symmetric proposal
                acc_f = (torch.log(torch.rand(N, device=device,
                                              dtype=torch.float64)) < log_a)
                m_f = acc_f.view(N, 1, 1, 1, 1)
                x = torch.where(m_f, y_f, x)
                S_x = torch.where(acc_f, S_yf, S_x)
                fourier_accept_total += acc_f.double()
                n_fourier_trials += 1
                if bool(acc_f.any()):
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
                acc_recent = float(np.mean(acc_trace[-50:]))
                acc_s_str = ""
                if n_shift_trials > 0:
                    acc_s_str = (f"  acc_shift="
                                 f"{(shift_accept_total / n_shift_trials).mean().item():.3f}")
                if n_fourier_trials > 0:
                    acc_s_str += (f"  acc_fourier="
                                  f"{(fourier_accept_total / n_fourier_trials).mean().item():.3f}")
                print(f"step {n_done:6d}/{args.n_steps}  "
                      f"acc(last50)={acc_recent:.3f}{acc_s_str}  "
                      f"<S>/V={S_x.mean().item() / V:+.6f}  "
                      f"<|M|>={magnetization(x).abs().mean().item():.5f}",
                      flush=True)

    np.save(out_dir / "final.npy", to_phys_np(x))
    save_traces()
    acc_overall = (accept_total / max(len(acc_trace), 1)).mean().item()
    print(f"done: n_steps={len(acc_trace)}  overall acceptance = {acc_overall:.4f}")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
