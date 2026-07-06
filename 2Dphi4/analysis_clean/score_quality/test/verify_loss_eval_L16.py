"""Cross-check the collaborator's loss_eval.py against the verified
score_quality_vs_epoch.py on the L16 k=0.2705 checkpoints.

For every checkpoint we apply EMA weights exactly as the verified sweep does,
then compute cos_sim / rel_mse / mag_ratio three ways:

  ref      : score_quality_vs_epoch.one_score_quality (verified baseline)
  collab_A : loss_eval.one_score_quality(model, grad_S_fn, x, t)   (grad_S_fn path)
  collab_B : loss_eval.score_quality_each(x, model, k,l,phi_min,phi_max)  (internal target)

If collab_A / collab_B reproduce ref to ~1e-5, the metric module itself is
correct and any wrong numbers in the collaborator's own training come from how
it is wired there (model/score convention, k/l, phi_min/phi_max), not this file.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))   # analysis_clean/score_quality/  (score_quality_vs_epoch)
sys.path.insert(0, str(HERE.parent))       # test/                          (loss_eval)

import numpy as np
import torch

import score_quality_vs_epoch as sq        # NOTE: import does os.chdir(2Dphi4/)
import loss_eval as le

DEVICE = "cuda:0"
L, K, LAM = 16, 0.2705, 0.022
T_EVAL = 1e-4
N = 512

run_dir = Path("runs/phi4_L16_k0.2705_l0.022_ncsnpp")
ckpts = sq.find_epoch_checkpoints(run_dir / "models")
epochs = sorted(ckpts)

model, grad_S_fn, nmin, nmax, sigma = sq.build_run_context(ckpts[epochs[0]], DEVICE)
data_path = f"trainingdata/cfgs_wolff_fahmc_k={K}_l={LAM}_{L}^2.jld2"
x_ref = sq.load_reference_data(data_path, N, nmin, nmax, DEVICE)

print(f"run={run_dir}")
print(f"sigma_ve={sigma}  norm=[{nmin:.4f}, {nmax:.4f}]  N={N}  t_eval={T_EVAL}")
print(f"x_ref shape {tuple(x_ref.shape)}\n")

hdr = (f"{'epoch':>6} | {'cos_ref':>8} {'cosA':>8} {'cosB':>8} {'dcos':>9} | "
       f"{'mse_ref':>8} {'mseA':>8} | {'mag_ref':>8} {'magA':>8}")
print(hdr)
print("-" * len(hdr))

max_dcos = 0.0
for ep in epochs:
    if ckpts[ep] != ckpts[epochs[0]]:
        sq.swap_weights(model, ckpts[ep])
    model.ema.copy_to()                                  # EMA weights, like the verified sweep

    ref = sq.one_score_quality(model, grad_S_fn, x_ref, T_EVAL)
    colA = le.one_score_quality(model, grad_S_fn, x_ref, T_EVAL)
    cosB = le.cosine_similarity(x_ref, model, k=K, l=LAM,
                                phi_min=nmin, phi_max=nmax, spatial_dims=2, eps=T_EVAL)

    cr, mr, gr = ref['cos_sim'][0], ref['rel_mse'][0], ref['mag_ratio'][0]
    ca, ma, ga = colA['cos_sim'][0], colA['rel_mse'][0], colA['mag_ratio'][0]
    dcos = max(abs(cr - ca), abs(cr - cosB))
    max_dcos = max(max_dcos, dcos)
    print(f"{ep:>6} | {cr:8.4f} {ca:8.4f} {cosB:8.4f} {dcos:9.1e} | "
          f"{mr:8.4f} {ma:8.4f} | {gr:8.4f} {ga:8.4f}")

print(f"\nmax |cos_ref - cos_collab| over all epochs = {max_dcos:.2e}")
print("PASS: collaborator metric reproduces the verified baseline."
      if max_dcos < 1e-4 else
      "MISMATCH: collaborator metric diverges from baseline -- investigate.")
