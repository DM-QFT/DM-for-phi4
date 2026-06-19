#!/bin/bash
# 3D phi^4 DM-MALA tau_int / ESS sweep over training epoch.
# Each rank handles one kappa with the corresponding multi-L checkpoint,
# sweeping the same L_list at every saved epoch.
#
# Positional args:
#   $1  EMA           "1"|"0"  (default "1")
#   $2  NUM_CHAINS    parallel chains M (default 64)
#   $3  MH_STEPS      chain length N (default 200)
#   $4  L_LIST        comma-separated L's; default 4,8,16,32,64

set -e

EMA="${1:-1}"
NUM_CHAINS="${2:-64}"
MH_STEPS="${3:-200}"
L_LIST="${4:-4,8,16,32,64}"

RANK=${OMPI_COMM_WORLD_RANK:-0}

declare -a KAP=(0.18 0.1923 0.2)

if (( RANK >= ${#KAP[@]} )); then
    echo "[rank ${RANK} on $(hostname)] idle"; exit 0
fi

K=${KAP[$RANK]}
RUN_DIR=runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp

EMA_FLAG=""
SUFFIX=""
if [[ "${EMA}" != "0" ]]; then
    EMA_FLAG="--ema"
    SUFFIX="_ema"
fi

cd /work/gw97/w97000/project/DM/3Dphi4

OUT=analysis/results/dm_mc_tau_vs_epoch_3d_multiL_k${K}${SUFFIX}.npz

echo "[rank ${RANK} on $(hostname)] kappa=${K} run=${RUN_DIR}  ema=${EMA}  L=${L_LIST}  M=${NUM_CHAINS}  N=${MH_STEPS}"
nvidia-smi -L

"$HOME/envs/nenv2/bin/python" analysis/dm_mc_tau_vs_epoch_3d.py \
    --device cuda:0 \
    --run_dir ${RUN_DIR} \
    --L_list ${L_LIST} \
    --k ${K} --l 0.9 \
    --c 0.2 --t_mh 1e-4 \
    --num_chains ${NUM_CHAINS} --mh_steps ${MH_STEPS} \
    --n_epochs 15 \
    ${EMA_FLAG} \
    --out_npz ${OUT}

echo "[rank ${RANK} on $(hostname)] done -> ${OUT}"
