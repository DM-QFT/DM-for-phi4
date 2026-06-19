#!/bin/bash
# Batch sample multi-L EDM2 ckpt at L=8, 16, 32, 64, 128 in parallel.
# Usage:  bash scripts/sample_multiL_edm2.sh <CKPT_PATH>
# Example:
#   bash scripts/sample_multiL_edm2.sh \
#     runs/phi4_Lmulti8-16-32-64_k0.28_l0.022_edm2_sd0.343_lcond/models/epoch=5000.ckpt

set -e
CKPT="${1:?Usage: $0 <ckpt path>}"
N_PER_REP=5120
N_REPEATS=2          # total 10240 samples per L
NUM_STEPS=36
S_CHURN=4
SEED=0
CONDA_ENV=nenv2

source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV

mkdir -p runs/_logs/

# L=8 on cuda:0, L=16 on cuda:1, L=32 on cuda:2, L=64 on cuda:3, L=128 sequential on cuda:0 after
declare -A DEV
DEV[8]="cuda:0"
DEV[16]="cuda:1"
DEV[32]="cuda:2"
DEV[64]="cuda:3"

declare -A PIDS
for L in 8 16 32 64; do
    LOG="runs/_logs/multiL_edm2_sample_L${L}.log"
    echo "Launching L=$L on ${DEV[$L]} (log: $LOG)"
    python sample_phi4_edm2.py \
        --L $L --k 0.28 --l 0.022 \
        --checkpoint "$CKPT" \
        --num_samples $N_PER_REP --n_repeats $N_REPEATS \
        --num_steps $NUM_STEPS --S_churn $S_CHURN \
        --seed $SEED --device ${DEV[$L]} > $LOG 2>&1 &
    PIDS[$L]=$!
done

# Wait for L=8 ... 64
for L in 8 16 32 64; do
    wait ${PIDS[$L]}
    echo "L=$L done"
done

# L=128 — needs smaller batch to fit in 32GB
echo "Launching L=128 on cuda:3 (N=512 single pass to avoid OOM)"
python sample_phi4_edm2.py \
    --L 128 --k 0.28 --l 0.022 \
    --checkpoint "$CKPT" \
    --num_samples 512 --n_repeats 1 \
    --num_steps $NUM_STEPS --S_churn $S_CHURN \
    --seed $SEED --device cuda:3 > runs/_logs/multiL_edm2_sample_L128.log 2>&1
echo "L=128 done"

echo
echo "All multi-L sampling complete. Analysis:"
echo "  python analysis/compare_edm2_multiL_propagator.py \\"
echo "    --run_dir $(dirname $(dirname $CKPT)) --ep \$(echo $CKPT | grep -oE '[0-9]+' | tail -1)"
