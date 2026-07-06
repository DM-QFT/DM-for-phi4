"""
2D lattice phi^4 PIK Action Matching (weak form) -- draft, mirrors action_matching_0d.py.

Prescribed action path (unnormalized, project convention of phi4_action.py):
    S_tar(phi) = sum_x [ -2 kappa phi_x sum_mu phi_{x+mu} + (1-2 lambda) phi_x^2
                         + lambda phi_x^4 ]                      (kappa, lambda physical)
    S_B(phi)   = 1/2 sum_x phi_x^2                               (unit Gaussian base)
    S_t        = (1-alpha(t)) S_tar + alpha(t) S_B,  alpha = 3t^2-2t^3 (smoothstep)

Learn a Z2-even, translation-equivariant scalar potential s_theta(phi, t) (CNN,
circular padding, extensive sum pooling); the transport velocity is v = grad_phi s.
Weak/AM objective per time slice (configs phi ~ p_t from batched HMC):

    L = E_t E_{p_t}[ 1/2 |grad_phi s|^2 + s * (dS_dt - <dS_dt>_{p_t}) ] / V .

Only ONE network gradient per point -- no divergence, scales with volume.
Validation: reverse probability-flow ODE from the t=1 Gaussian, observables
(<phi^2>_site, <|m|>, Binder U4) against the t=0 HMC reference ensemble.
"""

import argparse
import math
import time
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


# ---------------- physics: action path (unnormalized, batched) ----------------
def alpha(t):
    return 3.0 * t**2 - 2.0 * t**3


def adot(t):
    return 6.0 * t * (1.0 - t)


class Phi4Path:
    """S_t, dS_t/dt and force dS_t/dphi for configs of shape (B, L, L)."""

    def __init__(self, kappa, lam):
        self.k, self.l = kappa, lam

    def S_tar(self, p):
        nb = torch.roll(p, 1, dims=1) + torch.roll(p, 1, dims=2)
        return (-2 * self.k * p * nb + (1 - 2 * self.l) * p**2 + self.l * p**4).sum((1, 2))

    def F_tar(self, p):  # dS_tar/dphi
        nbf = (torch.roll(p, 1, 1) + torch.roll(p, -1, 1)
               + torch.roll(p, 1, 2) + torch.roll(p, -1, 2))
        return -2 * self.k * nbf + 2 * (1 - 2 * self.l) * p + 4 * self.l * p**3

    @staticmethod
    def S_B(p):
        return 0.5 * (p**2).sum((1, 2))

    def S_t(self, p, t):
        a = alpha(t)
        return (1 - a) * self.S_tar(p) + a * self.S_B(p)

    def F_t(self, p, t):  # dS_t/dphi
        a = alpha(t)
        return (1 - a) * self.F_tar(p) + a * p

    def dS_dt(self, p, t):
        return adot(t) * (self.S_B(p) - self.S_tar(p))


# ---------------- batched HMC at fixed t (B independent chains) ----------------
def hmc_slice(path, phi, t, n_traj, n_leap=10, dt=0.1, adapt=True, gen=None):
    """Run n_traj HMC trajectories on chains phi (B,L,L) at fixed slice time t."""
    B = phi.shape[0]
    acc_hist = []
    for traj in range(n_traj):
        pi = torch.randn(phi.shape, generator=gen, device=phi.device, dtype=phi.dtype)
        H0 = path.S_t(phi, t) + 0.5 * (pi**2).sum((1, 2))
        x = phi.clone()
        pi = pi - 0.5 * dt * path.F_t(x, t)
        for _ in range(n_leap - 1):
            x = x + dt * pi
            pi = pi - dt * path.F_t(x, t)
        x = x + dt * pi
        pi = pi - 0.5 * dt * path.F_t(x, t)
        H1 = path.S_t(x, t) + 0.5 * (pi**2).sum((1, 2))
        u = torch.rand(B, generator=gen, device=phi.device, dtype=phi.dtype)
        accept = (u < torch.exp(H0 - H1))
        phi = torch.where(accept[:, None, None], x, phi)
        acc_hist.append(accept.double().mean().item())
        if adapt and (traj + 1) % 10 == 0:
            a = sum(acc_hist[-10:]) / 10
            if a > 0.85:
                dt *= 1.05
            elif a < 0.70:
                dt *= 0.90
    return phi, dt, sum(acc_hist) / len(acc_hist)


def make_dataset(path, L, t_slices, n_cfg, n_therm, device, seed):
    """Annealed HMC sweep t=1 -> 0: exact Gaussian start, thermalize each slice.

    Returns configs (Nt, n_cfg, L, L), per-slice <dS_dt>, plus the t=0 reference
    ensemble for validation (extra slice, not used in training).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    phi = torch.randn((n_cfg, L, L), generator=gen, device=device)  # p_1 exact
    cfgs, means, dt = [], [], 0.1
    for t in reversed(t_slices.tolist()):                            # anneal downwards
        tt = torch.tensor(t, device=device)
        phi, dt, acc = hmc_slice(path, phi, tt, n_therm, dt=dt, gen=gen)
        cfgs.append(phi.clone())
        means.append(path.dS_dt(phi, tt).mean().item())
        print(f"  slice t={t:.4f}  acc={acc:.2f}  dt={dt:.3f}  "
              f"<phi^2>={phi.pow(2).mean().item():.4f}", flush=True)
    cfgs.reverse(); means.reverse()
    tt0 = torch.tensor(0.0, device=device)
    phi0, _, acc0 = hmc_slice(path, phi, tt0, n_therm, dt=dt, gen=gen)
    print(f"  reference t=0  acc={acc0:.2f}", flush=True)
    return torch.stack(cfgs), torch.tensor(means, device=device), phi0


# ---------------- network: Z2-even, translation-equivariant potential ----------------
class SNet2D(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Conv2d(2, width, 3, padding=1, padding_mode="circular"), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
                       nn.Tanh()]
        layers += [nn.Conv2d(width, 1, 3, padding=1, padding_mode="circular")]
        self.net = nn.Sequential(*layers)

    def raw(self, x, t):
        # x (B,L,L), t (B,) -> extensive scalar (B,)
        tt = t[:, None, None, None].expand(-1, 1, x.shape[1], x.shape[2])
        return self.net(torch.cat([x[:, None], tt], dim=1)).sum((1, 2, 3))

    def forward(self, x, t):
        # Z2-even by symmetrization; gauge s(0,t)=0 removes the c(t) direction.
        s = 0.5 * (self.raw(x, t) + self.raw(-x, t))
        return s - self.raw(torch.zeros_like(x), t)

    def velocity(self, x, t):
        xr = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            s = self.forward(xr, t)
            v = torch.autograd.grad(s.sum(), xr)[0]
        return v.detach()


# ---------------- weak/AM loss ----------------
def am_loss(model, X, T, SIG, V):
    Xr = X.detach().clone().requires_grad_(True)
    s = model(Xr, T)
    v = torch.autograd.grad(s.sum(), Xr, create_graph=True)[0]
    return ((0.5 * (v**2).sum((1, 2)) + s * SIG).mean()) / V


# ---------------- validation: reverse probability-flow ODE ----------------
def reverse_ode(model, path, L, n, M, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((n, L, L), generator=gen, device=device)        # p_1 = N(0,1)
    h, t = -1.0 / M, 1.0
    vf = lambda tv, q: model.velocity(q, torch.full((q.shape[0],), tv, device=device))
    for _ in range(M):
        k1 = vf(t, x); k2 = vf(t + 0.5 * h, x + 0.5 * h * k1)
        k3 = vf(t + 0.5 * h, x + 0.5 * h * k2); k4 = vf(t + h, x + h * k3)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4); t += h
    return x


def observables(phi):
    m = phi.mean((1, 2))
    m2, m4 = (m**2).mean(), (m**4).mean()
    return {"phi2_site": phi.pow(2).mean().item(),
            "abs_m": m.abs().mean().item(),
            "U4": (1 - m4 / (3 * m2**2)).item()}


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--kappa", type=float, default=0.28)   # broken phase (bimodal, like 0D)
    ap.add_argument("--lam", type=float, default=0.022)
    ap.add_argument("--T", type=int, default=32, help="interior time slices")
    ap.add_argument("--n_cfg", type=int, default=512, help="HMC configs per slice")
    ap.add_argument("--n_therm", type=int, default=100, help="HMC trajectories per slice")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128, help="configs per slice per step")
    ap.add_argument("--slices_per_step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n_val", type=int, default=1024)
    ap.add_argument("--M_val", type=int, default=200, help="reverse-ODE RK4 steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="action_matching_2d.pt")
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(args.seed)
    V = args.L**2
    path = Phi4Path(args.kappa, args.lam)
    t_slices = torch.linspace(0.0, 1.0, args.T + 2)[1:-1]

    print(f"device={dev}  L={args.L}  kappa={args.kappa}  lambda={args.lam}")
    print(f"generating data: {args.T} slices x {args.n_cfg} cfgs (annealed HMC)", flush=True)
    t0 = time.time()
    CFG, MEAN, REF0 = make_dataset(path, args.L, t_slices, args.n_cfg,
                                   args.n_therm, dev, args.seed)
    print(f"data done in {time.time()-t0:.0f}s")

    # precompute per-config source sigma = dS_dt - <dS_dt>_slice
    tdev = t_slices.to(dev)
    SIGMA = torch.stack([path.dS_dt(CFG[a], tdev[a]) - MEAN[a]
                         for a in range(args.T)])                    # (Nt, n_cfg)

    model = SNet2D(width=args.width).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    for step in range(args.steps):
        sl = torch.randperm(args.T)[:args.slices_per_step]
        idx = torch.randint(0, args.n_cfg, (args.slices_per_step, args.bs))
        X = torch.cat([CFG[a, idx[j]] for j, a in enumerate(sl)])
        Tv = torch.cat([tdev[a].expand(args.bs) for a in sl])
        SG = torch.cat([SIGMA[a, idx[j]] for j, a in enumerate(sl)])
        loss = am_loss(model, X, Tv, SG, V)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:6d}  loss {loss.item():+.6e}", flush=True)

    print("\nreverse-ODE validation vs t=0 HMC reference:")
    gen_cfgs = reverse_ode(model, path, args.L, args.n_val, args.M_val, dev, args.seed + 1)
    ob_g, ob_r = observables(gen_cfgs), observables(REF0)
    for key in ob_g:
        print(f"  {key:10s}  model={ob_g[key]:+.4f}   hmc={ob_r[key]:+.4f}")

    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
