#!/usr/bin/env bash
# Run from 3Dphi4/. Sequentially evaluates the multi-L acceptance on GPU 1.
set -euo pipefail
PY=/home/tyywork/miniconda3/envs/nenv2/bin/python
DEV=cuda:1

# k=0.2 trio
$PY analysis/acceptance_multiL.py --device $DEV --k 0.2 \
  --run_dir runs/phi4_3d_Lmulti4-8_k0.2_l0.9_ncsnpp \
  --L_eval 4 8 64 2>&1 | tee logs/acc_multiL_4-8_k0.2.log

$PY analysis/acceptance_multiL.py --device $DEV --k 0.2 \
  --run_dir runs/phi4_3d_Lmulti4-8-16_k0.2_l0.9_ncsnpp \
  --L_eval 4 8 16 64 2>&1 | tee logs/acc_multiL_4-8-16_k0.2.log

$PY analysis/acceptance_multiL.py --device $DEV --k 0.2 \
  --run_dir runs/phi4_3d_Lmulti4-8-16-32_k0.2_l0.9_ncsnpp_sigma400 \
  --L_eval 4 8 16 32 64 2>&1 | tee logs/acc_multiL_4-8-16-32_k0.2_sigma400.log

echo "GPU1 batch done."
