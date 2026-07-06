#!/bin/bash
# Per-rank worker: full-corrector (corrector_tmax=1.0) PC scan at kappa_c.
# 16 ranks = 2 settings {5,10} x 8 shards.
#   pc = pc_list[rank/8],  shard = rank%8
# EM/linear, 2000 steps, L=64 cross-L, correctors at ALL t, no MALA/filter.

set -e
RANK=${OMPI_COMM_WORLD_RANK:?need OMPI_COMM_WORLD_RANK}
PY=/work/gw97/w97000/envs/nenv2/bin/python
export TORCHINDUCTOR_COMPILE_THREADS=8
cd /work/gw97/w97000/project/DM/3Dphi4

PC_LIST=(5 10)
PC=${PC_LIST[$((RANK / 8))]}
SHARD=$((RANK % 8))
SEED=$((1000 + SHARD))

K=0.1923
RD="runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp"
TAG="pcfull_pc${PC}_shard${SHARD}"
LOG="results/mala_pipeline_logs/${TAG}.log"

echo "[rank $RANK on $(hostname)] pc=$PC shard=$SHARD (corrector_tmax=1.0)"
nvidia-smi -L

"$PY" sample_phi4_crossL.py \
    --L_train 32 --L_sample 64 --k "$K" --l 0.9 --ep 10000 \
    --method em --schedule linear --num_steps 2000 \
    --num_samples 64 --n_repeats 1 \
    --corrector_steps "$PC" --corrector_snr 0.16 --corrector_tmax 1.0 \
    --seed "$SEED" --device cuda:0 \
    --run_dir "$RD" --tag "$TAG" > "$LOG" 2>&1 \
    || { echo "[rank $RANK] FAIL pc=$PC shard=$SHARD"; exit 1; }
echo "[rank $RANK] done pc=$PC shard=$SHARD $(date '+%T')"
