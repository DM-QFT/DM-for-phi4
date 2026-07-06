"""Reproduce, on the L16 ep=9949 checkpoint (verified cos=0.9947), the wiring
mistakes that make the collaborator's cosine_similarity look 'wrong'.

The metric code is correct (verify_loss_eval_L16.py shows it matches the
baseline to ~1e-7). These are *caller-side* traps, ranked by how badly they
break cos vs only the magnitude.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

import torch
import score_quality_vs_epoch as sq
import loss_eval as le

DEVICE = "cuda:0"
L, K, LAM = 16, 0.2705, 0.022

run_dir = Path("runs/phi4_L16_k0.2705_l0.022_ncsnpp")
ckpts = sq.find_epoch_checkpoints(run_dir / "models")
last = sorted(ckpts)[-1]
model, grad_S_fn, nmin, nmax, sigma = sq.build_run_context(ckpts[last], DEVICE)
sq.swap_weights(model, ckpts[last])
model.ema.copy_to()
x = sq.load_reference_data(
    f"trainingdata/cfgs_wolff_fahmc_k={K}_l={LAM}_{L}^2.jld2", 512, nmin, nmax, DEVICE)

print(f"checkpoint epoch={last}  norm=[{nmin:.4f},{nmax:.4f}]\n")
co = lambda **kw: le.cosine_similarity(x, model, **kw)

print("scenario                                         cos_sim")
print("-" * 60)

# 0) CORRECT wiring
c = co(k=K, l=LAM, phi_min=nmin, phi_max=nmax, spatial_dims=2, eps=1e-4)
print(f"correct (phi_min/max set, t=1e-4, dims=2)        {c:8.4f}")

# 1) Forgot phi_min/phi_max  -> normalised [-1,1] field treated as physical p.
c = co(k=K, l=LAM, eps=1e-4, spatial_dims=2)
print(f"FORGOT phi_min/phi_max (None -> x used as p)     {c:8.4f}")

# 2) Evaluated at a large diffusion time (model score != -grad S there).
for t in (1e-2, 1e-1, 0.5, 1.0):
    c = co(k=K, l=LAM, phi_min=nmin, phi_max=nmax, spatial_dims=2, eps=t)
    print(f"wrong t_eval = {t:<6g}                           {c:8.4f}")

# 3) Wrong dimensionality flag (treat 2D data as 3D).
try:
    c = co(k=K, l=LAM, phi_min=nmin, phi_max=nmax, spatial_dims=3, eps=1e-4)
    print(f"wrong spatial_dims=3 on 2D data                  {c:8.4f}")
except Exception as e:
    print(f"wrong spatial_dims=3 on 2D data                  ERROR: {type(e).__name__}")

# 4) Magnitude-only trap: a network that does NOT divide by std (scale the
#    score by std(t)). cos is scale-invariant, so cos stays right while
#    mag_ratio/rel_mse blow up -> proves a wrong *cos* is never a std/mag bug.
std_t = float(model.score_model.marginal_prob_std(torch.tensor([1e-4], device=DEVICE)).item())

class NoStdDivide(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, xx, tt):
        s = self.m(xx, tt)
        return s * std_t                       # mimic raw network (un-divided)

raw = NoStdDivide(model).to(DEVICE)
cc, mm, gg = le.score_quality_each(x, raw, k=K, l=LAM, phi_min=nmin, phi_max=nmax,
                                   spatial_dims=2, eps=1e-4)
print(f"\nun-std-divided 'score' (std(t=1e-4)={std_t:.4f}):")
print(f"  cos={cc.mean():.4f}  rel_mse={mm.mean():.3e}  mag_ratio={gg.mean():.3e}")
print("  -> cos unchanged (scale-invariant); only mag/rel_mse break.")
