"""
Strong-form PIK training with REFERENCE SUBTRACTION -- zero sampling from p_t.
Space derivatives via FINITE-DIFFERENCE stencils on the field grid; automatic
differentiation is used ONLY for d(loss)/d(theta) when optimizing the network.

Control experiment for action_matching_0d.py (weak form). Same SNet, same physics,
same validation. Differences vs weak:
  (1) loss = pointwise PIK residual (strong), not the p_t-weighted variational one;
  (2) <d_t Shat>_{p_t} is CANCELLED by a reference config chi=0 (PIK paper eq.20 /
      tex eq:strong_reference_residual) -> NO p_t sampling, NO p_t weighting;
  (3) collocation on a FIXED UNIFORM grid on [-Lc, Lc] -- a p_t-free measure;
  (4) d_phi s and d_phi^2 s = d_phi v are CENTRAL FINITE DIFFERENCES on the grid,
      not autograd. In d dimensions the analogue is div v = sum_mu FD-stencil, a
      nearest-neighbour combination -- no O(d) backward passes, matching how the
      lattice action itself is discretized. Autograd is kept ONLY for grad_theta.

Residual, with v = d_phi s_theta (odd, v(0,t)=0 exact via the even potential):
    g_theta(phi,t) = d_phi v - v * d_phi Shat_t - d_t Shat_t
    r_ref(phi,t)   = g_theta(phi,t) - g_theta(0,t),   g_theta(0,t) = d_phi v(0,t)
The exact PIK velocity gives r_ref=0 for every phi,t.
"""

import argparse
import torch
import action_matching_0d as A

torch.set_default_dtype(torch.float64)
DEV = A.DEV
phi, w, N = A.phi, A.w, A.N
H = (phi[1] - phi[0]).item()       # uniform grid spacing
CI = N // 2                        # index of phi == 0 (grid symmetric, N odd)


def dS_dphi(x, t):
    a = A.alpha(t)
    return (-2.0 + 3.0 * a) * x + 0.5 * (1.0 - a) * x**3


def refsub_residual(model, ts):
    """Reference-subtracted strong residual on the full grid for times ts (n_t,).

    d_phi s and d_phi^2 s use central FD stencils; the field grid is NOT an autograd
    leaf. Grad flows only into the network parameters, so loss.backward() still
    differentiates through the FD combination of s_theta values -- one backward pass,
    no create_graph, no second-order autograd in phi.
    """
    nt = ts.shape[0]
    X = phi.unsqueeze(0).expand(nt, N)                      # (nt, N)
    T = ts.unsqueeze(1).expand(nt, N)
    s = model(X.reshape(-1), T.reshape(-1)).reshape(nt, N)  # grad w.r.t. theta only
    v = torch.zeros_like(s)
    dv = torch.zeros_like(s)
    v[:, 1:-1] = (s[:, 2:] - s[:, :-2]) / (2.0 * H)                 # d_phi s
    dv[:, 1:-1] = (s[:, 2:] - 2.0 * s[:, 1:-1] + s[:, :-2]) / H**2  # d_phi^2 s = d_phi v
    tcol = ts.unsqueeze(1)                                          # (nt, 1)
    xrow = phi.unsqueeze(0)                                         # (1, N)
    g = dv - v * dS_dphi(xrow, tcol) - A.dS_dt(xrow, tcol)
    return g - g[:, CI:CI + 1]                                     # cancels <d_t Shat>_{p_t}


def train(steps, T, lr, Lc, n_t, seed):
    torch.manual_seed(seed)
    model = A.SNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t_grid = torch.linspace(0.0, 1.0, T + 2, device=DEV)[1:-1]
    u = (w * (phi.abs() <= Lc)).detach()                           # uniform, p_t-free
    u = (u / u.sum()).unsqueeze(0)                                 # (1, N)
    for step in range(steps):
        ts = t_grid[torch.randperm(T)[:n_t]]
        r = refsub_residual(model, ts)                            # (n_t, N)
        loss = (u * r**2).sum() / n_t
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:6d}  loss {loss.item():.6e}", flush=True)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--Lc", type=float, default=5.0)
    ap.add_argument("--n_t", type=int, default=4)
    args = ap.parse_args()
    print(f"device={DEV}  STRONG reference-subtracted -- ZERO p_t sampling, FD stencils")
    print(f"collocation: uniform on [-{args.Lc},{args.Lc}], N={N}, N_t={args.T}, "
          f"lr={args.lr}, h={H:.2e}", flush=True)
    model = train(args.steps, args.T, args.lr, args.Lc, args.n_t, seed=0)
    A.validate(model)
    A.transport_test(model)
    torch.save(model.state_dict(), "strong_refsub_0d.pt")
    print("\nsaved -> strong_refsub_0d.pt")
