#!/bin/bash
# Per-rank worker for 3D data2 sampling.
# Launched once per node by mpirun from scripts/sample_data2_3d.pbs.
#
# Args:
#   1  CHUNKS_STR   ';'-separated, one chunk per rank;
#                   chunk = ','-separated epoch tags.
#   2  K            kappa
#   3  L            lattice size
#   4  LAMBDA       coupling lambda
#   5  SIGMA        VE-SDE sigma
#   6  RUN_SUFFIX   e.g. _sigma2048
#   7  OUT_SUBDIR   e.g. data2

set -e

CHUNKS_STR="$1"
K="$2"
L="$3"
LAMBDA="$4"
SIGMA="$5"
RUN_SUFFIX="$6"
OUT_SUBDIR="$7"

IFS=";" read -ra CHUNKS <<< "$CHUNKS_STR"
MY_EP="${CHUNKS[$OMPI_COMM_WORLD_RANK]}"

if [[ -z "$MY_EP" ]]; then
    echo "[rank $OMPI_COMM_WORLD_RANK on $(hostname)] no epochs assigned"
    exit 0
fi

cd /work/gw97/w97000/project/DM/3Dphi4

echo "[rank $OMPI_COMM_WORLD_RANK on $(hostname)] epochs=${MY_EP}"
nvidia-smi -L

# compile_mode follows existing 3D production scripts. NCSNpp3D upsampling
# pattern is structurally similar to 2D, so 'default' is the safe choice
# to avoid any latent F.interpolate-style cudagraph corruption.
"$HOME/envs/nenv2/bin/python" sample_phi4_3d_sweep.py \
    --L "$L" --k "$K" --l "$LAMBDA" \
    --sigma "$SIGMA" \
    --output_suffix "$RUN_SUFFIX" \
    --device cuda:0 \
    --epochs "$MY_EP" \
    --method em \
    --num_samples 64 --n_repeats 1 \
    --sde_steps 2000 --schedule linear \
    --compile_mode default \
    --out_subdir "$OUT_SUBDIR" \
    --seed 20260524
