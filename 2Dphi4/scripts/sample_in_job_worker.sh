#!/bin/bash
# Per-rank worker. Launched by mpirun from sample_in_job.sh.
# Args: CHUNKS_STR(';'-separated) K L RUN_SUFFIX
set -e

CHUNKS_STR="$1"
K="$2"
L="$3"
RUN_SUFFIX="$4"
EMA="${5:-true}"

IFS=";" read -ra CHUNKS <<< "$CHUNKS_STR"
MY_EP="${CHUNKS[$OMPI_COMM_WORLD_RANK]}"

if [[ -z "$MY_EP" ]]; then
    echo "[rank $OMPI_COMM_WORLD_RANK on $(hostname)] no epochs to sample"
    exit 0
fi

echo "[rank $OMPI_COMM_WORLD_RANK on $(hostname)] epochs=$MY_EP ema=$EMA"

cd /work/gw97/w97000/project/DM/2Dphi4

"$HOME/envs/nenv2/bin/python" sample_phi4_sweep.py \
    --L "$L" --k "$K" --l 0.022 \
    --output_suffix "$RUN_SUFFIX" \
    --device cuda:0 \
    --epochs "$MY_EP" \
    --num_samples 2048 --n_repeats 1 \
    --sde_steps 2000 --schedule log \
    --compile_mode default \
    --seed 44 \
    --ema "$EMA"
