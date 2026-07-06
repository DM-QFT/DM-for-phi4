"""
Step 2 (MAIN line): Action Matching for the 0D phi^4 PIK velocity.

Learn an even scalar potential s_theta(phi, t) = g((phi/L)^2, t); the transport
velocity is v = d_phi s (automatically odd). Objective (Neklyudov et al. 2023),
which generates the PRESCRIBED density path p_t = e^{-S_t}/Z_t and needs only
first derivatives of s (no divergence -> scales to the lattice):

    L(s) = E_{p_0}[s_0] - E_{p_1}[s_1] + E_t E_{p_t}[ 1/2 (d_phi s)^2 + d_t s ].

At the minimum d_phi s = v_exact. Validated against the quadrature ground truth
and by reverse-ODE moment reproduction (see verify_exact_0d.py).
"""

import argparse
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

L, N = 8.0, 4097
phi = torch.linspace(-L, L, N, device=DEV)
dphi = (phi[1] - phi[0]).item()
w = torch.full((N,), dphi, device=DEV); w[0] *= 0.5; w[-1] *= 0.5

# ---- physics (smoothstep schedule; identical to verify_exact_0d.py) ----
def alpha(t):     return 3.0 * t**2 - 2.0 * t**3
def adot(t):      return 6.0 * t * (1.0 - t)
def S(x, t):      a = alpha(t); return (-1.0 + 1.5 * a) * x**2 + (1.0 - a) / 8.0 * x**4
def dS_dt(x, t):  return adot(t) * (1.5 * x**2 - 0.125 * x**4)

def prob_weights(t):
    lp = -S(phi, t); lp = lp - lp.max()
    pw = torch.exp(lp) * w
    return pw / pw.sum()

def density(t):
    lp = -S(phi, t); lp = lp - lp.max()
    p = torch.exp(lp)
    return p / (p * w).sum()

def v_exact(t):
    # cum/p is the flux/density ratio; in the deep tails both underflow and the
    # ratio is pure roundoff (unobservable, zero probability), so we zero it.
    p = density(t)
    src = p * (dS_dt(phi, t) - (p * w * dS_dt(phi, t)).sum())
    cum = torch.cat([phi.new_zeros(1),
                     torch.cumsum(0.5 * (src[1:] + src[:-1]) * dphi, 0)])
    v = cum / torch.clamp(p, min=1e-300)
    return torch.where(p > 1e-10 * p.max(), v, torch.zeros_like(v))

# ---- network: even potential s -> odd velocity v = d_phi s ----
class SNet(nn.Module):
    def __init__(self, hidden=128, depth=4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # gauge-fix s(0, t) = 0 to remove the constant-in-phi (=c(t)) direction,
        # which has d_phi s = 0 and would otherwise drive the AM loss to -inf.
        if t.ndim == 0:
            t = t * torch.ones_like(x)
        feat = torch.stack([(x / L) ** 2, t], dim=-1)
        feat0 = torch.stack([torch.zeros_like(x), t], dim=-1)
        return (self.net(feat) - self.net(feat0)).squeeze(-1)

    def velocity(self, x, t):
        xr = x.detach().clone().requires_grad_(True)
        s = self.forward(xr, t)
        v = torch.autograd.grad(s.sum(), xr, create_graph=False)[0]
        return v.detach()

# ---- variational (PIK) Action Matching loss --------------------------------
# L(s) = int_0^1 E_{p_t}[ 1/2 (d_phi s)^2 + s * (dS_dt - <dS_dt>) ] dt
# No boundary terms, no d_t s; the gauge direction s=c(t) is automatically
# annihilated because <dS_dt - <dS_dt>> = 0.
def am_loss(model, PHIr, TT, PW_grid, SRC_grid):
    T, Ng = PHIr.shape
    s = model(PHIr.reshape(-1), TT.reshape(-1)).reshape(T, Ng)
    v = torch.autograd.grad(s.sum(), PHIr, create_graph=True)[0]
    return (PW_grid * (0.5 * v**2 + s * SRC_grid)).sum() / T

def train(steps=8000, T=64, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = SNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t_grid = torch.linspace(0.0, 1.0, T + 2, device=DEV)[1:-1]      # interior times
    PW_grid = torch.stack([prob_weights(t) for t in t_grid]).detach()        # (T,N)
    DSDT_grid = torch.stack([dS_dt(phi, t) for t in t_grid]).detach()        # (T,N)
    SRC_grid = (DSDT_grid - (PW_grid * DSDT_grid).sum(1, keepdim=True))      # (T,N)
    PHIr = phi.unsqueeze(0).expand(T, N).clone().requires_grad_(True)        # (T,N) leaf
    TT = t_grid.unsqueeze(1).expand(T, N).contiguous()                       # (T,N)
    for step in range(steps):
        loss = am_loss(model, PHIr, TT, PW_grid, SRC_grid)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:6d}  loss {loss.item():+.6e}", flush=True)
    return model

# ---- validation against the quadrature ground truth ----
def validate(model, times=(0.1, 0.3, 0.5, 0.7, 0.9)):
    # bulk-weighted RMSE of v_theta vs v_exact, plus the physical velocity scale.
    print("\n  t      wRMSE        |v|_rms(exact)   rel.err")
    for t in times:
        tt = phi.new_tensor(t)
        ve = v_exact(tt)
        vth = model.velocity(phi, tt)
        pw = prob_weights(tt)
        mask = (density(tt) > 1e-8 * density(tt).max()).double()
        pwm = pw * mask; pwm = pwm / pwm.sum()
        rmse = torch.sqrt((pwm * (vth - ve) ** 2).sum()).item()
        vscale = torch.sqrt((pwm * ve ** 2).sum()).item()
        print(f"  {t:.2f}   {rmse:.3e}    {vscale:.3e}       {rmse / max(vscale, 1e-30):.2%}")

# ---- acceptance test: reverse ODE with the LEARNED v reproduces p_tar -------
def interp1d(xq, yg):
    idx = torch.clamp(torch.searchsorted(phi, xq), 1, N - 1)
    x0 = phi[idx - 1]
    wgt = (xq - x0) / (phi[idx] - x0)
    return yg[idx - 1] + wgt * (yg[idx] - yg[idx - 1])

def transport_test(model, n=200_000, M=2000, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(n, generator=g, device=DEV)          # p_B = N(0,1)
    def vf(t, q):
        vg = model.velocity(phi, phi.new_tensor(t))      # detached grid velocity
        q = torch.clamp(q, phi[0], phi[-1])
        return torch.clamp(interp1d(q, vg), -50.0, 50.0)
    h, t = -1.0 / M, 1.0                                  # reverse: t 1 -> 0
    for _ in range(M):
        k1 = vf(t, x); k2 = vf(t + 0.5 * h, x + 0.5 * h * k1)
        k3 = vf(t + 0.5 * h, x + 0.5 * h * k2); k4 = vf(t + h, x + h * k3)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4); t += h
    m2, m4 = (x**2).mean().item(), (x**4).mean().item()
    p0 = density(phi.new_tensor(0.0))
    g2, g4 = (p0 * w * phi**2).sum().item(), (p0 * w * phi**4).sum().item()
    print(f"\n  reverse-ODE (learned v):  <phi^2>={m2:.4f}  <phi^4>={m4:.4f}")
    print(f"  target p_tar (quadrature):<phi^2>={g2:.4f}  <phi^4>={g4:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--T", type=int, default=64)
    args = ap.parse_args()
    print(f"device={DEV}")
    model = train(steps=args.steps, T=args.T)
    validate(model)
    transport_test(model)
    torch.save(model.state_dict(), "action_matching_0d.pt")
    print("\nsaved -> action_matching_0d.pt")
