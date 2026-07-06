"""2D lattice phi^4 STRONG-form PIK training with reference subtraction.

Control experiment for action_matching_2d.py (weak/AM form). Extends
strong_refsub_0d.py to a 2D field. Reuses that module's physics (Phi4Path),
annealed-HMC dataset, Z2-even CNN potential (SNet2D), reverse-ODE validation and
observables -- ONLY the loss differs, so weak vs strong is a like-for-like test.

Strong residual with v = grad_phi s_theta (Z2-odd, v(0)=0 exact):
    g_theta(phi,t) = div_phi v  -  <grad s, F_t>  -  d_t Shat_t,   F_t = grad_phi Shat_t
    r_theta(phi,t) = g_theta(phi,t) - g_theta(0,t),   g_theta(0,t) = div v(0,t)
The exact PIK velocity gives r_theta = 0 for every phi, t. Reference subtraction
at the zero field cancels the only p_t-dependent term <d_t Shat>_{p_t} exactly.

WHAT EXTENDS FROM 0D, WHAT DOES NOT:
  * reference subtraction (chi=0): dimension-free, kept verbatim.
  * v = grad s: in d>1 this is REQUIRED for uniqueness (the PIK velocity is the
    unique curl-free / gradient solution; the strong PDE alone has a huge
    incompressible null space).
  * the divergence div v = tr Hess(s) = sum_x d^2 s/d phi_x^2 is the scaling wall.
    0D put a grid on the single field axis and used a cheap FD stencil; R^{L^2}
    has no such grid. Two estimators here:
      --probes 0  : EXACT trace, O(V) backward passes. Correct but ~2-3 s/step
                    even at L=4; use only to VERIFY on tiny lattices.
      --probes m  : Hutchinson trace with m Rademacher probes, cost O(m) and
                    INDEPENDENT of V. A single probe biases this SQUARED residual
                    (E[r_hat^2] = r^2 + Var); we remove the bias with a U-statistic
                    over the m probes, so the loss is unbiased for E_p[r^2]. This
                    is the honest way to run strong form on a real lattice.
  * collocation: 0D used a p_t-free uniform grid. R^{L^2} has none, so we reuse
    the annealed-HMC p_t configs as collocation points (same data as the weak
    form). Reference subtraction still removes <d_t Shat>_{p_t} for free.
"""

import argparse
import time
import torch

from action_matching_2d import Phi4Path, SNet2D, make_dataset, reverse_ode, observables

torch.set_default_dtype(torch.float64)


def _tr_hess(v, Xr):
    """Exact field-space divergence sum_x d v_x/d phi_x for v = grad_phi s.
    O(V) backward passes; differentiable in theta. v:(B,L,L), Xr the leaf."""
    B, L, _ = v.shape
    div = v.new_zeros(B)
    for i in range(L):
        for j in range(L):
            (g,) = torch.autograd.grad(v[:, i, j].sum(), Xr,
                                       create_graph=True, retain_graph=True)
            div = div + g[:, i, j]
    return div


def _rademacher(shape, device, dtype):
    return torch.randint(0, 2, shape, device=device, dtype=dtype).mul_(2).sub_(1)


def strong_slice_loss(model, path, X, t, V, probes=4):
    """Reference-subtracted strong residual on one time slice.

    X:(bs,L,L) collocation configs at slice time t (scalar tensor). Residual is
    made intensive (/V). probes=0 -> exact divergence; probes=m>=2 -> unbiased
    Hutchinson (U-statistic debiases r^2).
    """
    bs, L, _ = X.shape
    Xr = X.detach().clone().requires_grad_(True)
    (v,) = torch.autograd.grad(model(Xr, t.expand(bs)).sum(), Xr, create_graph=True)
    c = -(v * path.F_t(X, t)).sum((1, 2)) - path.dS_dt(X, t)     # deterministic part (bs,)
    Z = torch.zeros(1, L, L, device=X.device, dtype=X.dtype).requires_grad_(True)
    (v0,) = torch.autograd.grad(model(Z, t.expand(1)).sum(), Z, create_graph=True)

    if probes == 0:                                             # exact (small L only)
        r = (_tr_hess(v, Xr) - _tr_hess(v0, Z)) + c
        return ((r / V) ** 2).mean()

    # Hutchinson: delta_i = eps_i^T (Hess_phi - Hess_0) eps_i, iid unbiased for D=div-div0
    D = []
    for _ in range(probes):
        e = _rademacher((bs, L, L), X.device, X.dtype)
        (He,) = torch.autograd.grad((v * e).sum(), Xr, create_graph=True, retain_graph=True)
        e0 = _rademacher((1, L, L), X.device, X.dtype)
        (He0,) = torch.autograd.grad((v0 * e0).sum(), Z, create_graph=True, retain_graph=True)
        D.append((e * He).sum((1, 2)) - (e0 * He0).sum((1, 2)))
    D = torch.stack(D)                                          # (m, bs)
    S1, S2 = D.sum(0), (D * D).sum(0)
    D2 = (S1 * S1 - S2) / (probes * (probes - 1))              # unbiased D^2 (U-statistic)
    dbar = S1 / probes                                         # unbiased D
    return ((D2 + 2.0 * dbar * c + c * c) / V ** 2).mean()     # unbiased r^2 / V^2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=0.28)        # broken phase (bimodal)
    ap.add_argument("--lam", type=float, default=0.022)
    ap.add_argument("--probes", type=int, default=4, help="Hutchinson probes; 0 = exact (small L)")
    ap.add_argument("--T", type=int, default=32, help="interior time slices")
    ap.add_argument("--n_cfg", type=int, default=512, help="HMC configs per slice")
    ap.add_argument("--n_therm", type=int, default=100, help="HMC trajectories per slice")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=64, help="collocation configs per slice per step")
    ap.add_argument("--slices_per_step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n_val", type=int, default=1024)
    ap.add_argument("--M_val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="strong_refsub_2d.pt")
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    V = args.L ** 2
    path = Phi4Path(args.kappa, args.lam)
    t_slices = torch.linspace(0.0, 1.0, args.T + 2)[1:-1]

    tag = "EXACT divergence" if args.probes == 0 else f"Hutchinson ({args.probes} probes, unbiased)"
    print(f"device={dev}  L={args.L} (V={V})  kappa={args.kappa}  lambda={args.lam}  {tag}")
    print(f"generating data: {args.T} slices x {args.n_cfg} cfgs (annealed HMC)", flush=True)
    t0 = time.time()
    CFG, MEAN, REF0 = make_dataset(path, args.L, t_slices, args.n_cfg,
                                   args.n_therm, dev, args.seed)
    print(f"data done in {time.time()-t0:.0f}s  (<dS_dt> unused: reference subtraction "
          f"replaces it)", flush=True)
    tdev = t_slices.to(dev)

    model = SNet2D(width=args.width).to(dev)
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
