#!/bin/bash
# Per-rank worker, launched once per node by mpirun from
# scripts/sample_data2.pbs.
#
# Args:
#   1  CHUNKS_STR  ';'-separated, one chunk per rank.
#                  Each chunk is ','-separated epoch tags (e.g. "0001,0079").
#   2  K           kappa
#   3  L           lattice size
#   4  SIGMA       VE-SDE sigma
#   5  RUN_SUFFIX  e.g. _sigma450
#   6  OUT_SUBDIR  e.g. data2
#
# This rank picks CHUNKS[OMPI_COMM_WORLD_RANK] and runs sample_phi4_sweep.py
# for that comma-list of epochs. Saves raw samples (no Z2 augmentation):
# each .npy holds num_samples * n_repeats = 512 configs.

set -e

CHUNKS_STR="$1"
K="$2"
L="$3"
SIGMA="$4"
RUN_SUFFIX="$5"
OUT_SUBDIR="$6"

IFS=";" read -ra CHUNKS <<< "$CHUNKS_STR"
MY_EP="${CHUNKS[$OMPI_COMM_WORLD_RANK]}"

if [[ -z "$MY_EP" ]]; then
    echo "[rank $OMPI_COMM_WORLD_RANK on $(hostname)] no epochs assigned"
    exit 0
fi

cd /work/gw97/w97000/project/DM/2Dphi4

echo "[rank $OMPI_COMM_WORLD_RANK on $(hostname)] epochs=${MY_EP}"
nvidia-smi -L

# compile_mode=default — reduce-overhead corrupts F.interpolate(nearest)
# in NCSNpp's up1 stage on L=128 (see CLAUDE.md).
"$HOME/envs/nenv2/bin/python" sample_phi4_sweep.py \
    --L "$L" --k "$K" --l 0.022 \
    --sigma "$SIGMA" \
    --output_suffix "$RUN_SUFFIX" \
    --device cuda:0 \
    --epochs "$MY_EP" \
    --num_samples 512 --n_repeats 1 \
    --sde_steps 2000 --schedule log \
    --compile_mode default \
    --out_subdir "$OUT_SUBDIR" \
    --ema true \
    --seed 20260524
