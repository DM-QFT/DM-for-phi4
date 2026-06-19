#!/bin/bash
# Launched once per node by mpirun. Uses OMPI_COMM_WORLD_RANK to pick its epoch.

set -e

EPOCHS_STR="$1"
K="$2"
L="$3"
SIGMA="$4"
RUN_SUFFIX="$5"

read -ra EPOCHS <<< "$EPOCHS_STR"
EP="${EPOCHS[$OMPI_COMM_WORLD_RANK]}"

source /etc/profile.d/modules.sh
module load miniforge3
source activate nenv2
cd /work/gw97/w97000/project/DM/2Dphi4

echo "[rank ${OMPI_COMM_WORLD_RANK} on $(hostname)] epoch=${EP}"

python sample_phi4_sweep.py \
    --L "${L}" --k "${K}" --l 0.022 \
    --sigma "${SIGMA}" \
    --output_suffix "${RUN_SUFFIX}" \
    --device cuda:0 \
    --epochs "${EP}" \
    --num_samples 512 --n_repeats 1 \
    --sde_steps 2000 --schedule log \
    --compile_mode reduce-overhead \
    --seed 20260422
