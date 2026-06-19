#!/bin/bash
# Per-rank worker. Launched by mpirun from calibrate_and_plot.sh.
# Dispatches by OMPI_COMM_WORLD_RANK to one of six sweeps: 3 kappas
# (0.26, 0.2705, 0.28) x {h sweep, t_mh sweep}.
#
# Positional args (passed by the wrapper, identical across ranks):
#   $1  EMA          "1"|"0"  (default "1")
#   $2  EARLY        early window, e.g. "0-50"
#   $3  LATE         late  window, e.g. "6000-10000"

set -e

EMA="${1:-1}"
EARLY="${2:-0-50}"
LATE="${3:-6000-10000}"

RANK=${OMPI_COMM_WORLD_RANK:-0}

# Per-rank assignment: (kind, kappa, sigma_suffix)
declare -a KIND=(  h         h          h        t_mh      t_mh       t_mh    )
declare -a KAP=(   0.26      0.2705     0.28     0.26      0.2705     0.28    )
declare -a SUF=(   sigma300  sigma450   sigma640 sigma300  sigma450   sigma640)

if (( RANK >= ${#KIND[@]} )); then
    echo "[rank ${RANK} on $(hostname)] idle (no work assigned)"
    exit 0
fi

KIND_R=${KIND[$RANK]}
K=${KAP[$RANK]}
S=${SUF[$RANK]}
RUN_DIR=runs/phi4_L128_k${K}_l0.022_ncsnpp_${S}

EMA_FLAG=""
if [[ "${EMA}" != "0" ]]; then
    EMA_FLAG="--ema"
fi

cd /work/gw97/w97000/project/DM/2Dphi4

echo "[rank ${RANK} on $(hostname)] kind=${KIND_R} kappa=${K} run=${RUN_DIR}  ema=${EMA}  early=${EARLY}  late=${LATE}"
nvidia-smi -L

if [[ "${KIND_R}" == "h" ]]; then
    "$HOME/envs/nenv2/bin/python" acceptance_rate/calibrate_step_size.py \
        --device cuda:0 --L 128 --k ${K} --l 0.022 \
        --run_dir ${RUN_DIR} \
        --early_window ${EARLY} --late_window ${LATE} \
        --max_early_ckpts 0 --max_late_ckpts 0 \
        --num_samples 1024 --mh_steps 1 --t_mh 1e-4 \
        ${EMA_FLAG} \
        --out_dir acceptance_rate --out_tag ${S}
else
    "$HOME/envs/nenv2/bin/python" acceptance_rate/calibrate_t_mh.py \
        --device cuda:0 --L 128 --k ${K} --l 0.022 \
        --run_dir ${RUN_DIR} \
        --early_window ${EARLY} --late_window ${LATE} \
        --max_early_ckpts 0 --max_late_ckpts 0 \
        --num_samples 1024 --mh_steps 1 --c 0.2 \
        ${EMA_FLAG} \
        --out_dir acceptance_rate --out_tag ${S}
fi

echo "[rank ${RANK} on $(hostname)] done"
