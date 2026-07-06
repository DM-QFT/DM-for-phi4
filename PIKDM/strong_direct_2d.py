"""2D lattice phi^4 STRONG-form PIK training: direct velocity + reference subtraction.

Self-contained. Prescribed action path (unnormalized, project convention):
    S_tar(phi) = sum_x [ -2 kappa phi_x sum_mu phi_{x+mu} + (1-2 lambda) phi_x^2
                         + lambda phi_x^4 ]                     (kappa, lambda physical)
    S_B(phi)   = 1/2 sum_x phi_x^2                              (unit Gaussian base)
    S_t        = (1-alpha(t)) S_tar + alpha(t) S_B,  alpha = 3t^2-2t^3 (smoothstep)

The network outputs the velocity field v_theta(phi,t) DIRECTLY, Z2-odd by
antisymmetrization so v(0,t) = 0 exactly. The strong residual is enforced pointwise:
    g_theta = div v - <v, F_t> - dS_dt,      r = g_theta(phi,t) - g_theta(0,t).
At the exact PIK velocity g = -<dS_dt>_{p_t}, a phi-independent constant, so the
reference subtraction at chi = 0 cancels the only p_t expectation analytically and
the minimizer has r = 0 for every phi, t. Direct v keeps the divergence a JACOBIAN
trace -- one autograd order lower than the tr Hess(s) of a potential
parameterization v = grad s:
  --probes 0    exact divergence: V one-hot vjp rows in ONE batched backward
                (is_grads_batched). First-order, affordable at small L.
  --probes m>=2 Hutchinson trace with U-statistic debiasing (unbiased r^2),
                each probe one first-order vjp. Use for larger L.

Collocation configs come from annealed HMC on the ANALYTIC S_t. The refsub identity
holds for any collocation measure -- unequilibrated chains introduce no bias, they
only reallocate accuracy -- HMC just places the points on the typical sets that the
reverse trajectories actually visit (iid noise would miss them in V = L^2 dims).

What direct v gives up: the gradient form v = grad s pinned the unique curl-free
(Action-Matching) solution; the strong PDE alone determines v only up to
incompressible modes w with div(p_t w) = 0. Those modes do not change the
transported marginals -- any residual-zero v solves the continuity equation -- and
are suppressed by the smooth CNN prior plus the exact Z2-odd structure.

Validation: reverse probability-flow ODE from the t=1 Gaussian, observables
(<phi^2>_site, <|m|>, Binder U4) against the t=0 HMC reference ensemble.
"""

import argparse
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


# ---------------- network: Z2-odd direct velocity field ----------------
class VNet2D(nn.Module):
    """Z2-odd, translation-equivariant direct velocity field v_theta(phi, t).

    Circular-padded conv stack whose output IS the field; no sum pooling, no
    potential.

    Field normalization: the conv stack sees psi = phi/scale (~[-1,1] like the
    image-DM convention) and its output is multiplied back by scale, so
    v_theta(phi,t) = scale * u_net(phi/scale, t). This is a pure linear change
    of variables: the strong residual g is form-invariant under it (div_phi v =
    div_psi u, v.grad_phi S = u.grad_psi S), so the objective and its minimizer
    are untouched -- only the network conditioning improves. Loss code,
    physics, and validation need no change; autograd sees the scaling inside
    forward().
    """

    def __init__(self, width=64, depth=4, scale=1.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale)))
        layers = [nn.Conv2d(2, width, 3, padding=1, padding_mode="circular"), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1, padding_mode="circular"),
                       nn.Tanh()]
        layers += [nn.Conv2d(width, 1, 3, padding=1, padding_mode="circular")]
        self.net = nn.Sequential(*layers)

    def raw(self, x, t):
        # x (B,L,L), t (B,) -> v (B,L,L); net works in normalized units
        x = x / self.scale
        tt = t[:, None, None, None].expand(-1, 1, x.shape[1], x.shape[2])
        return self.net(torch.cat([x[:, None], tt], dim=1))[:, 0] * self.scale

    def forward(self, x, t):
        # exact oddness v(-phi,t) = -v(phi,t); in particular v(0,t) = 0.
        return 0.5 * (self.raw(x, t) - self.raw(-x, t))

    def velocity(self, x, t):
        with torch.no_grad():
            return self.forward(x, t)


# ---------------- strong refsub loss ----------------
def _div_exact(v, Xr):
    """Exact divergence tr(dv/dphi), differentiable in theta.

    One batched backward: V one-hot vjps via is_grads_batched, then the
    diagonal. v:(B,L,L) network output, Xr the leaf it was computed from.
    """
    B, L, _ = v.shape
    V = L * L
    I = torch.eye(V, device=v.device, dtype=v.dtype).view(V, 1, L, L).expand(V, B, L, L)
    (rows,) = torch.autograd.grad(v, Xr, grad_outputs=I, is_grads_batched=True,
                                  create_graph=True, retain_graph=True)
    return rows.reshape(V, B, V).diagonal(dim1=0, dim2=2).sum(1)


def _rademacher(shape, device, dtype):
    return torch.randint(0, 2, shape, device=device, dtype=dtype).mul_(2).sub_(1)


def strong_slice_loss(model, path, X, t, V, probes=4):
    """Reference-subtracted strong residual on one time slice, direct v.

    X:(bs,L,L) collocation configs at slice time t (scalar tensor). Residual is
    made intensive (/V). probes=0 -> exact divergence; probes=m>=2 -> unbiased
    Hutchinson (U-statistic debiases r^2).
    """
    bs, L, _ = X.shape
    Xr = X.detach().clone().requires_grad_(True)
    v = model(Xr, t.expand(bs))                                 # direct output
    c = -(v * path.F_t(X, t)).sum((1, 2)) - path.dS_dt(X, t)    # deterministic part (bs,)
    Z = torch.zeros(1, L, L, device=X.device, dtype=X.dtype).requires_grad_(True)
    v0 = model(Z, t.expand(1))                                  # = 0 exactly, dv/dphi != 0

    if probes == 0:                                             # exact divergence
        r = (_div_exact(v, Xr) - _div_exact(v0, Z)) + c
        return ((r / V) ** 2).mean()

    # Hutchinson: delta_i = eps_i^T (J - J_0) eps_i, iid unbiased for D = div - div0
    D = []
    for _ in range(probes):
        e = _rademacher((bs, L, L), X.device, X.dtype)
        (Je,) = torch.autograd.grad((v * e).sum(), Xr, create_graph=True, retain_graph=True)
        e0 = _rademacher((1, L, L), X.device, X.dtype)
        (Je0,) = torch.autograd.grad((v0 * e0).sum(), Z, create_graph=True, retain_graph=True)
        D.append((e * Je).sum((1, 2)) - (e0 * Je0).sum((1, 2)))
    D = torch.stack(D)                                          # (m, bs)
    S1, S2 = D.sum(0), (D * D).sum(0)
    D2 = (S1 * S1 - S2) / (probes * (probes - 1))              # unbiased D^2 (U-statistic)
    dbar = S1 / probes                                         # unbiased D
    return ((D2 + 2.0 * dbar * c + c * c) / V ** 2).mean()     # unbiased r^2 / V^2


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
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=0.28)        # broken phase (bimodal)
    ap.add_argument("--lam", type=float, default=0.022)
    ap.add_argument("--probes", type=int, default=0, help="0 = exact divergence; m>=2 Hutchinson")
    ap.add_argument("--T", type=int, default=32, help="interior time slices")
    ap.add_argument("--n_cfg", type=int, default=512, help="HMC configs per slice")
    ap.add_argument("--n_therm", type=int, default=100, help="HMC trajectories per slice")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=64, help="collocation configs per slice per step")
    ap.add_argument("--slices_per_step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--scale", type=float, default=0.0,
                    help="field normalization c (net sees phi/c); 0 = auto max|phi| of dataset, 1 = off")
    ap.add_argument("--n_val", type=int, default=1024)
    ap.add_argument("--M_val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="strong_direct_2d.pt")
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    V = args.L ** 2
    path = Phi4Path(args.kappa, args.lam)
    t_slices = torch.linspace(0.0, 1.0, args.T + 2)[1:-1]

    tag = "EXACT divergence" if args.probes == 0 else f"Hutchinson ({args.probes} probes, unbiased)"
    print(f"device={dev}  L={args.L} (V={V})  kappa={args.kappa}  lambda={args.lam}  "
          f"direct-v  {tag}")
    print(f"generating data: {args.T} slices x {args.n_cfg} cfgs (annealed HMC)", flush=True)
    t0 = time.time()
    CFG, MEAN, REF0 = make_dataset(path, args.L, t_slices, args.n_cfg,
                                   args.n_therm, dev, args.seed)
    print(f"data done in {time.time()-t0:.0f}s  (<dS_dt> unused: reference subtraction "
          f"replaces it)", flush=True)
    tdev = t_slices.to(dev)

    c = args.scale if args.scale > 0 else CFG.abs().max().item()
    print(f"field scale c = {c:.4f}  (network input phi/c, output v/c)")
    model = VNet2D(width=args.width, scale=c).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    t0 = time.time()
    for step in range(args.steps):
        sl = torch.randperm(args.T)[:args.slices_per_step].tolist()
        loss = 0.0
        for a in sl:
            idx = torch.randint(0, args.n_cfg, (args.bs,))
            loss = loss + strong_slice_loss(model, path, CFG[a, idx], tdev[a], V, args.probes)
        loss = loss / args.slices_per_step
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"step {step:6d}  loss {loss.item():.6e}  ({1000*el/(step+1):.0f} ms/step)", flush=True)

    print("\nreverse-ODE validation vs t=0 HMC reference:")
    gen_cfgs = reverse_ode(model, path, args.L, args.n_val, args.M_val, dev, args.seed + 1)
    ob_g, ob_r = observables(gen_cfgs), observables(REF0)
    for key in ob_g:
        print(f"  {key:10s}  model={ob_g[key]:+.4f}   hmc={ob_r[key]:+.4f}")

    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
