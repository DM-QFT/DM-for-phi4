#!/usr/bin/env bash
# Run from 3Dphi4/. Sequentially evaluates the multi-L acceptance on GPU 0.
set -euo pipefail
PY=/home/tyywork/miniconda3/envs/nenv2/bin/python
DEV=cuda:0

# k=0.18 trio
$PY analysis/acceptance_multiL.py --device $DEV --k 0.18 \
  --run_dir runs/phi4_3d_Lmulti4-8_k0.18_l0.9_ncsnpp \
  --L_eval 4 8 64 2>&1 | tee logs/acc_multiL_4-8_k0.18.log

$PY analysis/acceptance_multiL.py --device $DEV --k 0.18 \
  --run_dir runs/phi4_3d_Lmulti4-8-16_k0.18_l0.9_ncsnpp \
  --L_eval 4 8 16 64 2>&1 | tee logs/acc_multiL_4-8-16_k0.18.log

$PY analysis/acceptance_multiL.py --device $DEV --k 0.18 \
  --run_dir runs/phi4_3d_Lmulti4-8-16-32_k0.18_l0.9_ncsnpp_sigma200 \
  --L_eval 4 8 16 32 64 2>&1 | tee logs/acc_multiL_4-8-16-32_k0.18_sigma200.log

# k=0.1923 trio
$PY analysis/acceptance_multiL.py --device $DEV --k 0.1923 \
  --run_dir runs/phi4_3d_Lmulti4-8_k0.1923_l0.9_ncsnpp \
  --L_eval 4 8 64 2>&1 | tee logs/acc_multiL_4-8_k0.1923.log

$PY analysis/acceptance_multiL.py --device $DEV --k 0.1923 \
  --run_dir runs/phi4_3d_Lmulti4-8-16_k0.1923_l0.9_ncsnpp \
  --L_eval 4 8 16 64 2>&1 | tee logs/acc_multiL_4-8-16_k0.1923.log

$PY analysis/acceptance_multiL.py --device $DEV --k 0.1923 \
  --run_dir runs/phi4_3d_Lmulti4-8-16-32_k0.1923_l0.9_ncsnpp \
  --L_eval 4 8 16 32 64 2>&1 | tee logs/acc_multiL_4-8-16-32_k0.1923.log

echo "GPU0 batch done."
