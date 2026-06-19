#!/bin/bash
# Per-rank worker for 3D phi^4 acceptance-vs-epoch sweep.
# Dispatches by OMPI_COMM_WORLD_RANK to one of three kappa runs.
#
# Positional args:
#   $1  EMA          "1"|"0"  (default "1")
#   $2  EPOCH_WIN    epoch range, e.g. "0-1000000"

set -e

EMA="${1:-1}"
EPOCH_WIN="${2:-0-1000000}"

RANK=${OMPI_COMM_WORLD_RANK:-0}

declare -a KAP=(0.18      0.1923    0.2      )
declare -a SUF=(sigma2000 sigma2048 sigma2760)

if (( RANK >= ${#KAP[@]} )); then
    echo "[rank ${RANK} on $(hostname)] idle (no work assigned)"
    exit 0
fi

K=${KAP[$RANK]}
S=${SUF[$RANK]}
RUN_DIR=runs/phi4_3d_L64_k${K}_l0.9_ncsnpp_${S}

EMA_FLAG=""
if [[ "${EMA}" != "0" ]]; then
    EMA_FLAG="--ema"
fi

cd /work/gw97/w97000/project/DM/3Dphi4

echo "[rank ${RANK} on $(hostname)] kappa=${K} run=${RUN_DIR}  ema=${EMA}  epoch_win=${EPOCH_WIN}"
nvidia-smi -L

"$HOME/envs/nenv2/bin/python" analysis/acceptance_vs_epoch.py \
    --device cuda:0 --L 64 --k ${K} --l 0.9 \
    --run_dir ${RUN_DIR} \
    --epoch_window ${EPOCH_WIN} --max_ckpts 0 \
    --num_samples 256 --batch_size 512 --mh_steps 1 \
    --c 0.2 --t_mh 1e-4 \
    ${EMA_FLAG} \
    --out_dir analysis --out_tag ${S}

echo "[rank ${RANK} on $(hostname)] done"
