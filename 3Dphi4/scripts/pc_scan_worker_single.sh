#!/bin/bash
# Per-rank worker for a single-setting PC scan (8 nodes = 8 shards).
# Arg 1: corrector_steps (PC). Rank r generates the r-th 64-sample shard.
# EM/linear, 2000 steps, L=64 cross-L, correctors only for t<0.1, no MALA,
# no filter.

set -e
PC="${1:?need corrector_steps arg}"
RANK=${OMPI_COMM_WORLD_RANK:?need OMPI_COMM_WORLD_RANK}
PY=/work/gw97/w97000/envs/nenv2/bin/python
export TORCHINDUCTOR_COMPILE_THREADS=8
cd /work/gw97/w97000/project/DM/3Dphi4

K=0.1923
RD="runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp"
SEED=$((1000 + RANK))
TAG="pcscan_pc${PC}_shard${RANK}"
LOG="results/mala_pipeline_logs/${TAG}.log"

echo "[rank $RANK on $(hostname)] pc=$PC shard=$RANK"
nvidia-smi -L

"$PY" sample_phi4_crossL.py \
    --L_train 32 --L_sample 64 --k "$K" --l 0.9 --ep 10000 \
    --method em --schedule linear --num_steps 2000 \
    --num_samples 64 --n_repeats 1 \
    --corrector_steps "$PC" --corrector_snr 0.16 --corrector_tmax 0.1 \
    --seed "$SEED" --device cuda:0 \
    --run_dir "$RD" --tag "$TAG" > "$LOG" 2>&1 \
    || { echo "[rank $RANK] FAIL pc=$PC"; exit 1; }
echo "[rank $RANK] done pc=$PC shard=$RANK $(date '+%T')"
