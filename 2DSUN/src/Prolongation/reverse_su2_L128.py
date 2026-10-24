"""Partial reverse diffusion at L=128, beta=200 from the relaxed L64 superresolution, one Sigma^2 per call.
Same parents, seeds, EMA model, time grid and loops as the Sigma^2 = 1.5, 0.8, 0.2 runs. Usage: reverse_su2_L128.py SIGMA2"""
import hashlib
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/SU2"))
sys.path.insert(0, str(Path(__file__).parent))
import train_su2_dm as train  # noqa: E402
import sample_su2_em as sampler  # noqa: E402
from refine_su2 import load_cfgs, prolong, metropolis, gauge_transform, haar, normalize, wilson, to_matrix  # noqa: E402

DIR = ROOT / "results/su2_prolongation_reverse"
# The Sigma^2 = 0.2 run fixes parents, seeds, checkpoint and the measured loops of every partial run.
REF = DIR / "reverse_L64_to_L128_beta200_Sigma2_0p2_n512_steps0834_betanoise.h5"
requested = float(sys.argv[1])
started = time.perf_counter()

with h5py.File(REF) as f:
    a = dict(f.attrs)
    rt = f["wilson_RT"][:]
    parent_ref = f["parent_id"][:]
    relaxed_ref = f["relaxed/wilson"][:]
n, L, beta, eps = int(a["n_cfgs"]), int(a["L"]), float(a["beta"]), float(a["eps"])


def loops(u):
    return np.stack([wilson(u, r, t).mean(axis=(1, 2)) for r, t in rt], axis=1)


def project(u):
    # float32 network arithmetic leaves a norm drift of order 1e-7, removed as in save_network_group.
    return u/np.linalg.norm(u, axis=-1, keepdims=True)


# 512 L64 parents drawn without replacement and sorted. Interior and full sweeps at beta_f=200, then a
# random gauge transformation. The regenerated loops must equal the stored relaxed loops.
with h5py.File(a["coarse"]) as f:
    n_coarse = len(f["cfgs"])
parent = np.sort(np.random.default_rng(int(a["selection_seed"])).choice(n_coarse, n, replace=False))
np.testing.assert_array_equal(parent, parent_ref)
coarse, _ = load_cfgs(a["coarse"], parent, int(a["L_coarse"]), a["beta_c"])
uniform, _ = prolong(coarse)
seed = int(a["preparation_seed"])
interior, acc_i, _ = metropolis(uniform, int(a["interior_sweeps"]), True, np.random.default_rng([seed, 10]), beta)
relaxed, acc_f, _ = metropolis(interior, int(a["full_sweeps"]), False, np.random.default_rng([seed, 11]), beta)
relaxed, _ = normalize(gauge_transform(relaxed, haar((n, L, L), np.random.default_rng([seed, 3, 100]))))
relaxed_loops = loops(relaxed)
np.testing.assert_allclose(relaxed_loops, relaxed_ref, rtol=0, atol=1e-12)
preparation = time.perf_counter()-started
print(f"Same {n} parents and relaxed initial states. Acceptance interior {acc_i:.6f} full {acc_f:.6f}. "
      f"Preparation {preparation:.1f} s", flush=True)

ck = torch.load(a["checkpoint"])
assert ck["L"] == L and ck["beta"] == beta and ck["epoch"] == a["epoch"]
model = train.LCNN(ck["ch"], ck["nlayers"], ck["sigma"], noise_beta=ck["noise_beta"])
ema = ExponentialMovingAverage(model.parameters(), decay=train.EMA_DECAY)
ema.load_state_dict(ck["ema"])
ema.copy_to(model.parameters())
model.eval()

# Start on the full 2000-step grid at the time whose Sigma^2 is closest to the request, the L=32 rule.
full = sampler.em_time_grid(eps=eps)
Sigma2_grid = train.marginal_prob_std(full.double(), ck["sigma"], ck["noise_beta"])**2
i = int(torch.argmin((Sigma2_grid-requested).abs()))
t_start, Sigma2, steps = float(full[i]), float(Sigma2_grid[i]), len(full)-1-i
tag = f"{requested:g}".replace(".", "p")
out = DIR / f"reverse_L64_to_L128_beta200_Sigma2_{tag}_n{n}_steps{steps:04d}_betanoise.h5"
print(f"Requested Sigma squared {requested}. Actual {Sigma2!r}. t_start {t_start!r}. Steps {steps}. Output {out}",
      flush=True)

# Forward noise from the GPU heat kernel with the recorded seed, increments of variance at most DELTA_MAX.
k = max(1, math.ceil(Sigma2/train.DELTA_MAX))
torch.manual_seed(int(a["forward_seed"]))
u = torch.tensor(relaxed, dtype=torch.float32, device="cuda")
delta = torch.full((n, 1, 1, 1), Sigma2/k, device="cuda")
for _ in range(k):
    u = train.qmul(train.qexp(train.heat_kernel_step(u.shape[:-1], delta)), u)
noisy = project(u.cpu().numpy().astype(np.float64))
noisy_loops = loops(noisy)
print(f"/noisy P {noisy_loops[:, 0].mean():.12f} +/- {noisy_loops[:, 0].std(ddof=1)/np.sqrt(n):.4g}", flush=True)

torch.manual_seed(int(a["seed"]))
torch.cuda.synchronize()
t0 = time.perf_counter()
U = sampler.em_sample(model, n, L, U_start=torch.tensor(noisy, dtype=torch.float32, device="cuda"),
                      t_start=t_start, eps=eps)
torch.cuda.synchronize()
sampling = time.perf_counter()-t0
times = sampler.em_time_grid(t_start, eps=eps)
assert len(times)-1 == steps and torch.equal(times, full[-(steps+1):])
max_variance = (train.diffusion_coeff(times[:-1], ck["sigma"], ck["noise_beta"])**2*(times[:-1]-times[1:])).max().item()
q = U.cpu().numpy().astype(np.float64)
drift = float(np.abs(np.linalg.norm(q, axis=-1)-1).max())
q = project(q)
w = loops(q)
print(f"Reverse completed in {sampling:.2f} seconds. Norm drift {drift:.3e}. Maximum increment {max_variance:.6g}", flush=True)
print(f"/ P {w[:, 0].mean():.12f} +/- {w[:, 0].std(ddof=1)/np.sqrt(n):.4g}", flush=True)

shared = ("L", "L_coarse", "beta", "beta_c", "coarse", "checkpoint", "epoch", "sigma", "noise_beta", "weights", "eps",
          "error_block", "final_noise", "full_steps", "interior_sweeps", "full_sweeps", "initialization", "n_cfgs",
          "selection_seed", "preparation_seed", "forward_seed", "seed", "proposal_width", "schedule", "plaq_exact",
          "heatbath_source", "baseline")
programs = [ROOT/"src/SU2/train_su2_dm.py", ROOT/"src/SU2/sample_su2_em.py", Path(__file__).parent/"refine_su2.py",
            Path(__file__)]
with h5py.File(out, "w-") as f:
    f.attrs.update({key: a[key] for key in shared})
    f.attrs.update(requested_Sigma2=requested, Sigma2=Sigma2, Sigma=math.sqrt(Sigma2), t_start=t_start, n_steps=steps,
                   forward_increments=k, forward_increment_variance=Sigma2/k, max_increment_variance=max_variance,
                   unitarity_drift_before_projection=drift, interior_acceptance=acc_i, full_acceptance=acc_f,
                   preparation_seconds=preparation, sampling_seconds=sampling, reference_run=str(REF),
                   torch_version=str(torch.__version__), numpy_version=np.__version__,
                   **{p.stem+"_sha256": hashlib.sha256(p.read_bytes()).hexdigest() for p in programs})
    f["cfgs"] = to_matrix(q)
    f["parent_id"] = parent
    f["times"] = times.cpu().numpy()
    f["wilson_RT"] = rt
    f["wilson"] = w
    f["plaq"] = w[:, 0]
    f["relaxed/wilson"] = relaxed_loops
    f["noisy/wilson"] = noisy_loops
    f.attrs["total_seconds"] = time.perf_counter()-started
print(f"Saved {out}. Total seconds {time.perf_counter()-started:.2f}", flush=True)
