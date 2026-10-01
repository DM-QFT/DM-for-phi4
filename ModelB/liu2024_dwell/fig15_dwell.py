"""Reproduce Fig. 15 of Liu, Chen, Xiu, Zhang, arXiv:2410.03108 (training-free conditional
diffusion model, double-well SDE dX = (X - X^3) dt + sigma dW). Follows gm_DWell.py step by step."""
import copy
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde, norm

dev = "cuda"
torch.manual_seed(12345678)
np.random.seed(12312414)
out = Path(__file__).resolve().parent
t0 = time.time()

sigma, dt, T_obs, H = 0.5, 0.01, 1.0, 100_000
L = int(T_obs / dt)

# ---- D_obs: H Euler-Maruyama trajectories from U(-2.5, 2.5), all L adjacent pairs pooled (M = 10^7) ----
x = torch.empty(H, device=dev).uniform_(-2.5, 2.5)
xs = torch.empty(L + 1, H, device=dev)
xs[0] = x
for n in range(L):
    x = x + (x - x**3) * dt + sigma * dt**0.5 * torch.randn_like(x)
    xs[n + 1] = x
# diff_scale = 10 puts the target 10*dX at O(0.5), comparable to the beta^2 = tau noise of the diffusion model
diff_scale = 10.0
x_obs = xs[:-1].reshape(-1)
z_obs = (xs[1:] - xs[:-1]).reshape(-1) * diff_scale
order = torch.argsort(x_obs)
x_obs, z_obs = x_obs[order], z_obs[order]
M = x_obs.numel()
del xs, order

# ---- J conditioning states drawn from D_obs; K exact nearest neighbours in x (faiss IndexFlatL2 in the original) ----
J, K = 60_000, 2048
x0 = x_obs[torch.randperm(M, device=dev)[:J]]
i0 = torch.searchsorted(x_obs, x0).clamp(K, M - K)
cand = i0[:, None] + torch.arange(-K, K, device=dev)[None]
nn_idx = cand.gather(1, (x_obs[cand] - x0[:, None]).abs().topk(K, dim=1, largest=False).indices)
x_nn, z_nn = x_obs[nn_idx], z_obs[nn_idx]
del cand, nn_idx
win = (x_nn - x0[:, None]).abs().max(1).values
print(f"[{time.time()-t0:6.1f}s] D_obs M={M}, kNN window half-width: median {win.median():.2e}, max {win.max():.2e}")

# ---- labels y_j = z_0: probability-flow ODE, Euler with K_ode steps, alpha = 1 - tau + dtau, beta^2 = tau + dtau ----
K_ode = 10_000
dtau = 1.0 / K_ode
z1 = torch.randn(J, device=dev)
# nu = 1 kernel in x, flat across the ~1e-3 wide kNN window, so the hard kNN cut does the conditioning
logw_x = -(x0[:, None] - x_nn) ** 2 / 2
z = z1.clone()
for k in range(K_ode):
    tau = 1.0 - (k + 1) * dtau
    alpha, beta2 = 1.0 - tau + dtau, tau + dtau
    f, g2 = -1.0 / alpha, 1.0 + 2.0 * beta2 / alpha
    r = (z[:, None] - alpha * z_nn) / beta2
    # softmax equals their normalised weights wherever those are finite (they drop the underflow NaN rows)
    w = torch.softmax(logw_x - r**2 * (beta2 / 2), dim=1)
    score = -(r * w).sum(1)
    z = z - (f * z - 0.5 * g2 * score) * dtau
y = z
del r, w, logw_x

# in 1D the probability-flow ODE is a monotone map, so y_j should be the local empirical quantile at Phi(z1_j)
u = torch.tensor(norm.cdf(z1.cpu().numpy()), device=dev, dtype=torch.float32)
zs = z_nn.sort(1).values
pos = u * (K - 1)
lo = pos.floor().long().clamp(max=K - 2)
yq = zs.gather(1, lo[:, None])[:, 0] * (1 - (pos - lo)) + zs.gather(1, (lo + 1)[:, None])[:, 0] * (pos - lo)
print(f"[{time.time()-t0:6.1f}s] labels: rms y {y.pow(2).mean().sqrt():.4f}, rms(y - local quantile) {(y-yq).pow(2).mean().sqrt():.4f}")

# ---- G_theta(x, z): 2 -> 50 -> 50 -> 1 tanh MLP, full-batch Adam, best-on-validation ----
X = torch.stack([x0, z1], 1)
Y = y[:, None]
perm = torch.randperm(J, device=dev)
X, Y = X[perm], Y[perm]
Xm, Xs, Ym, Ys = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
Xn, Yn = (X - Xm) / Xs, (Y - Ym) / Ys
ntr = int(0.8 * J)
net = torch.nn.Sequential(torch.nn.Linear(2, 50), torch.nn.Tanh(), torch.nn.Linear(50, 50), torch.nn.Tanh(), torch.nn.Linear(50, 1)).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-6)
best, best_state = float("inf"), None
for it in range(2000):
    opt.zero_grad()
    loss = (net(Xn[:ntr]) - Yn[:ntr]).pow(2).mean()
    loss.backward()
    opt.step()
    with torch.no_grad():
        vl = (net(Xn[ntr:]) - Yn[ntr:]).pow(2).mean().item()
    if vl < best:
        best, best_state = vl, copy.deepcopy(net.state_dict())
net.load_state_dict(best_state)
print(f"[{time.time()-t0:6.1f}s] trained: final train loss {loss.item():.3e}, best valid loss {best:.3e}")


@torch.no_grad()
def step_learned(xg):
    inp = (torch.stack([xg, torch.randn_like(xg)], 1) - Xm) / Xs
    return xg + (net(inp)[:, 0] * Ys + Ym) / diff_scale


# ---- Fig. 14 style sanity check: effective drift a(x) and diffusion b(x) from one learned step ----
for xc in [-1.5, -0.5, 0.0, 0.5, 1.5]:
    xg = torch.full((500_000,), xc, device=dev)
    dx = step_learned(xg) - xg
    a, b = dx.mean().item() / dt, dx.std().item() / dt**0.5
    print(f"  x={xc:5.2f}  a_learned={a:8.4f} a_true={xc-xc**3:8.4f}   b_learned={b:.4f} b_true={sigma}")

# ---- Fig. 15: N paths from X0 = 1.5, learned flow map vs Euler-Maruyama reference, snapshots at T = 0.5, 10, 30, 100 ----
N = 500_000
snaps = {50: 0.5, 1000: 10, 3000: 30, 10000: 100}
xg = torch.full((N,), 1.5, device=dev)
xr = xg.clone()
pdfs = {}
grid = np.linspace(-4, 4, 401)
for n in range(1, max(snaps) + 1):
    xg = step_learned(xg)
    xr = xr + (xr - xr**3) * dt + sigma * dt**0.5 * torch.randn_like(xr)
    if n in snaps:
        xg_np, xr_np = xg.cpu().numpy().astype(np.float64), xr.cpu().numpy().astype(np.float64)
        pdfs[snaps[n]] = (gaussian_kde(xr_np)(grid), gaussian_kde(xg_np)(grid))
        print(f"[{time.time()-t0:6.1f}s] T={snaps[n]:5.1f}  P(x<0): ref {np.mean(xr_np<0):.4f} learned {np.mean(xg_np<0):.4f}"
              f"   mean: ref {xr_np.mean():7.4f} learned {xg_np.mean():7.4f}   std: ref {xr_np.std():.4f} learned {xg_np.std():.4f}")

fig, axs = plt.subplots(1, 4, figsize=(16, 3.2))
for ax, (T, (p_ref, p_learn)) in zip(axs, pdfs.items()):
    ax.plot(grid, p_ref, label="Reference")
    ax.plot(grid, p_learn, label="Learned")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("x")
    ax.set_title(f"T = {T}")
    ax.legend()
axs[0].set_ylabel("pdf")
fig.tight_layout()
fig.savefig(out / "fig15_dwell.png", dpi=200)
fig.savefig(out / "fig15_dwell.pdf")
print(f"[{time.time()-t0:6.1f}s] saved {out/'fig15_dwell.png'}")
