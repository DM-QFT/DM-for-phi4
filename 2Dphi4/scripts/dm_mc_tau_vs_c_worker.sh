#!/bin/bash
# 2D phi^4 DM-MALA tau_int / ESS sweep over step-size coefficient c at a
# fixed (converged) training epoch.  Uses the SINGLE-L in-distribution
# checkpoint at L=128 (one per kappa), NOT a multi-L checkpoint.  The
# multi-L checkpoint is only used by dm_mc_tau_vs_epoch (cross-L epoch
# sweep), where evaluating the same model at several L's is the point.
#
# Positional args:
#   $1  EMA           "1"|"0"  (default "1")
#   $2  NUM_CHAINS    parallel chains M (default 64)
#   $3  MH_STEPS      chain length N (default 200)
#   $4  L             single lattice size (default 128, in distribution)
#   $5  EPOCH         target epoch (default 10000)
#   $6  C_LIST        comma-separated c's (default = dense grid)

set -e

EMA="${1:-1}"
NUM_CHAINS="${2:-64}"
MH_STEPS="${3:-200}"
L="${4:-128}"
EPOCH="${5:-10000}"
C_LIST="${6:-0.01,0.02,0.03,0.05,0.07,0.1,0.13,0.17,0.2,0.25,0.3,0.4,0.5,0.7,1.0,1.5,2.0,3.0,5.0,7.0,10.0,15.0,20.0}"

RANK=${OMPI_COMM_WORLD_RANK:-0}

# (kappa, sigma_suffix) for the single-L in-distribution runs.
declare -a KAP=(0.26     0.2705   0.28    )
declare -a SUF=(sigma300 sigma450 sigma640)

if (( RANK >= ${#KAP[@]} )); then
    echo "[rank ${RANK} on $(hostname)] idle"; exit 0
fi

K=${KAP[$RANK]}
S=${SUF[$RANK]}
RUN_DIR=runs/phi4_L${L}_k${K}_l0.022_ncsnpp_${S}

EMA_FLAG=""
SUFFIX=""
if [[ "${EMA}" != "0" ]]; then
    EMA_FLAG="--ema"
    SUFFIX="_ema"
fi

cd /work/gw97/w97000/project/DM/2Dphi4

OUT=acceptance_rate/results/dm_mc_tau_vs_c_L${L}_k${K}_${S}_ep${EPOCH}${SUFFIX}.npz

echo "[rank ${RANK} on $(hostname)] kappa=${K} run=${RUN_DIR}  epoch=${EPOCH}  L=${L}  ema=${EMA}  c=${C_LIST}"
nvidia-smi -L

"$HOME/envs/nenv2/bin/python" acceptance_rate/dm_mc_tau_vs_c.py \
    --device cuda:0 \
    --run_dir ${RUN_DIR} \
    --epoch ${EPOCH} \
    --L ${L} \
    --c_list ${C_LIST} \
    --k ${K} --l 0.022 \
    --t_mh 1e-4 \
    --num_chains ${NUM_CHAINS} --mh_steps ${MH_STEPS} \
    ${EMA_FLAG} \
    --out_npz ${OUT}

echo "[rank ${RANK} on $(hostname)] done -> ${OUT}"
