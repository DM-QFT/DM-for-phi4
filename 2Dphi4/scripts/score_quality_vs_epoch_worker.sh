#!/bin/bash
# Per-rank worker for 2D phi^4 score-quality-vs-epoch sweep.
# Dispatches by OMPI_COMM_WORLD_RANK to one of three kappa runs.
#
# Positional args:
#   $1  EMA          "1"|"0"  (default "1")
#   $2  EPOCH_WIN    epoch range (default "0-1000000" = all)

set -e

EMA="${1:-1}"
EPOCH_WIN="${2:-0-1000000}"

RANK=${OMPI_COMM_WORLD_RANK:-0}

declare -a KAP=(0.26      0.2705    0.28    )
declare -a SUF=(sigma300  sigma450  sigma640)

if (( RANK >= ${#KAP[@]} )); then
    echo "[rank ${RANK} on $(hostname)] idle (no work assigned)"
    exit 0
fi

K=${KAP[$RANK]}
S=${SUF[$RANK]}
RUN_DIR=runs/phi4_L128_k${K}_l0.022_ncsnpp_${S}

EMA_FLAG=""
if [[ "${EMA}" != "0" ]]; then
    EMA_FLAG="--ema"
fi

cd /work/gw97/w97000/project/DM/2Dphi4

echo "[rank ${RANK} on $(hostname)] kappa=${K} run=${RUN_DIR}  ema=${EMA}  epoch_win=${EPOCH_WIN}"
nvidia-smi -L

"$HOME/envs/nenv2/bin/python" score_quality_analysis/score_quality_vs_epoch.py \
    --device cuda:0 --L 128 --k ${K} --l 0.022 \
    --run_dir ${RUN_DIR} \
    --epoch_window ${EPOCH_WIN} --max_ckpts 0 \
    --num_samples 512 --t_eval 1e-4 \
    ${EMA_FLAG} \
    --out_dir score_quality_analysis --out_tag ${S}

echo "[rank ${RANK} on $(hostname)] done"
