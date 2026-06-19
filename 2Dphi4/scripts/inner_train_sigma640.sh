#!/bin/bash -l
# Runs inside one qsub -I session on the compute node.
# Finds the latest LogScaleCheckpoint ckpt, resumes training from it,
# and writes done.marker when training completes normally (max_epochs hit).

set -uo pipefail

WORK_DIR=/work/gw97/w97000/project/DM/2Dphi4
RUN_DIR=$WORK_DIR/runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640
CKPT_DIR=$RUN_DIR/models
DONE_MARKER=$RUN_DIR/done.marker

cd "$WORK_DIR"

source /etc/profile.d/modules.sh
module load miniforge3
source activate nenv2

echo "================================================================"
echo "[inner] Node       : $(hostname)"
echo "[inner] Start      : $(date '+%F %T')"
echo "[inner] Workdir    : $WORK_DIR"
echo "[inner] PBS_JOBID  : ${PBS_JOBID:-<not set>}"
echo "================================================================"
nvidia-smi -L

# Pick the LATEST checkpoint by epoch number (filename: epoch=NNNN.ckpt).
# `sort -V` handles the embedded integer correctly.
LATEST_CKPT=""
if [ -d "$CKPT_DIR" ]; then
    LATEST_CKPT=$(ls "$CKPT_DIR"/epoch=*.ckpt 2>/dev/null | sort -V | tail -1 || true)
fi

CKPT_ARG=()
if [ -n "$LATEST_CKPT" ]; then
    echo "[inner] Resuming from: $LATEST_CKPT"
    CKPT_ARG=(--ckpt_path "$LATEST_CKPT")
else
    echo "[inner] No checkpoint found; starting from scratch"
fi

python train_phi4.py \
    --L 128 --k 0.28 --l 0.022 \
    --sigma 640 \
    --batch_size 2560 --device cuda:0 \
    --network ncsnpp --gpu_data \
    --data_path trainingdata/cfgs_wolff_fahmc_k=0.28_l=0.022_128^2.jld2 \
    --output_suffix _sigma640 \
    --epochs 40000 --num_ckpts 200 \
    "${CKPT_ARG[@]}"
TRAIN_RC=$?

if [ $TRAIN_RC -eq 0 ]; then
    echo "[inner] Training exited cleanly (rc=0). Touching done marker."
    mkdir -p "$RUN_DIR"
    touch "$DONE_MARKER"
else
    echo "[inner] Training exited with rc=$TRAIN_RC (likely walltime kill or error)."
    echo "[inner] No done marker; outer loop will resubmit."
fi

echo "[inner] Finish     : $(date '+%F %T')"
exit $TRAIN_RC
