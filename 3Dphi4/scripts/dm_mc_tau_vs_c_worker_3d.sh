#!/bin/bash
# 3D phi^4 DM-MALA tau_int / ESS sweep over step-size coefficient c at a
# fixed (converged) training epoch.  Uses the SINGLE-L in-distribution
# checkpoint at L=64 (one per kappa).

set -e

EMA="${1:-1}"
NUM_CHAINS="${2:-64}"
MH_STEPS="${3:-200}"
L="${4:-64}"
EPOCH="${5:-20000}"
C_LIST="${6:-0.01,0.02,0.03,0.05,0.07,0.1,0.13,0.17,0.2,0.25,0.3,0.4,0.5,0.7,1.0,1.5,2.0,3.0,5.0,7.0,10.0,15.0,20.0}"

RANK=${OMPI_COMM_WORLD_RANK:-0}

declare -a KAP=(0.18      0.1923    0.2      )
declare -a SUF=(sigma2000 sigma2048 sigma2760)

if (( RANK >= ${#KAP[@]} )); then
    echo "[rank ${RANK} on $(hostname)] idle"; exit 0
fi

K=${KAP[$RANK]}
S=${SUF[$RANK]}
RUN_DIR=runs/phi4_3d_L${L}_k${K}_l0.9_ncsnpp_${S}

EMA_FLAG=""
SUFFIX=""
if [[ "${EMA}" != "0" ]]; then
    EMA_FLAG="--ema"
    SUFFIX="_ema"
fi

cd /work/gw97/w97000/project/DM/3Dphi4

OUT=analysis/results/dm_mc_tau_vs_c_3d_L${L}_k${K}_${S}_ep${EPOCH}${SUFFIX}.npz

echo "[rank ${RANK} on $(hostname)] kappa=${K} run=${RUN_DIR}  epoch=${EPOCH}  L=${L}  ema=${EMA}  c=${C_LIST}"
nvidia-smi -L

"$HOME/envs/nenv2/bin/python" analysis/dm_mc_tau_vs_c_3d.py \
    --device cuda:0 \
    --run_dir ${RUN_DIR} \
    --epoch ${EPOCH} \
    --L ${L} \
    --c_list ${C_LIST} \
    --k ${K} --l 0.9 \
    --t_mh 1e-4 \
    --num_chains ${NUM_CHAINS} --mh_steps ${MH_STEPS} \
    ${EMA_FLAG} \
    --out_npz ${OUT}

echo "[rank ${RANK} on $(hostname)] done -> ${OUT}"
