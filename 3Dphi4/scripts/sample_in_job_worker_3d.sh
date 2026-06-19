#!/bin/bash
# Per-rank worker for 3D phi^4. Launched by mpirun from sample_in_job_3d.sh.
# Each rank picks its epoch chunk via OMPI_COMM_WORLD_RANK and invokes
# sample_phi4_3d_sweep.py with --method em (SDE EM) on cuda:0 of that node.
#
# Args: CHUNKS_STR(';'-separated) K L LAMBDA RUN_SUFFIX SDE_STEPS \
#       NUM_SAMPLES N_REPEATS SCHEDULE SEED COMPILE_MODE

set -e

CHUNKS_STR="$1"
K="$2"
L="$3"
LAMBDA="$4"
RUN_SUFFIX="$5"
SDE_STEPS="$6"
NUM_SAMPLES="$7"
N_REPEATS="$8"
SCHEDULE="$9"
SEED="${10}"
COMPILE_MODE="${11}"

IFS=";" read -ra CHUNKS <<< "$CHUNKS_STR"
MY_EP="${CHUNKS[$OMPI_COMM_WORLD_RANK]}"

if [[ -z "$MY_EP" ]]; then
    echo "[rank $OMPI_COMM_WORLD_RANK on $(hostname)] no epochs to sample"
    exit 0
fi

echo "[rank $OMPI_COMM_WORLD_RANK on $(hostname)] epochs=$MY_EP sde_steps=$SDE_STEPS schedule=$SCHEDULE"

cd /work/gw97/w97000/project/DM/3Dphi4

"$HOME/envs/nenv2/bin/python" sample_phi4_3d_sweep.py \
    --L "$L" --k "$K" --l "$LAMBDA" \
    --output_suffix "$RUN_SUFFIX" \
    --device cuda:0 \
    --epochs "$MY_EP" \
    --method em \
    --num_samples "$NUM_SAMPLES" --n_repeats "$N_REPEATS" \
    --sde_steps "$SDE_STEPS" \
    --schedule "$SCHEDULE" \
    --compile_mode "$COMPILE_MODE" \
    --seed "$SEED"
