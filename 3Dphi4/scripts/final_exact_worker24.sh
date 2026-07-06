#!/bin/bash
# Per-rank worker: final exact-drift production on 24 nodes.
# 24 ranks = 3 kappas x 8 shards, fully parallel.
#   kappa = kappa_list[rank/8],  shard = rank%8
# Recipe: exact -grad S, c=100, shift/2, cluster/10, 5000 steps.

set -e
RANK=${OMPI_COMM_WORLD_RANK:?need OMPI_COMM_WORLD_RANK}
PY=/work/gw97/w97000/envs/nenv2/bin/python
export TORCHINDUCTOR_COMPILE_THREADS=8
cd /work/gw97/w97000/project/DM/3Dphi4

KS_LIST=("0.18 _sigma200 0.0105" "0.1923 NOSUF 0.1168" "0.2 _sigma400 0.0104")
set -- ${KS_LIST[$((RANK / 8))]}
K=$1; SUF=$2; SSTD=$3
[ "$SUF" = "NOSUF" ] && SUF=""
SHARD=$((RANK % 8))
OFF=$((64 * SHARD))

RD="runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp${SUF}"
INIT="$RD/data_crossL/samples_filtered512_k${K}_ep10000.npy"
LOG="results/mala_pipeline_logs/final_exact_k${K}_shard${SHARD}.log"

echo "[rank $RANK on $(hostname)] kappa=$K shard=$SHARD offset=$OFF"
nvidia-smi -L

"$PY" sample_phi4_crossL_mala.py \
    --run_dir "$RD" --k "$K" --l 0.9 --ep 10000 \
    --drift exact --init_npy "$INIT" --cfg_offset "$OFF" --max_cfgs 64 \
    --c 100 --n_steps 5000 --t_mh 1e-4 \
    --shift_every 2 --shift_std "$SSTD" --cluster_every 10 \
    --tag "final_exact_shard${SHARD}" \
    --microbatch 64 --state_every 1000 --n_snapshots 6 \
    --seed $((300 + SHARD)) --device cuda:0 > "$LOG" 2>&1
echo "[rank $RANK] kappa=$K shard=$SHARD done $(date '+%T')  $(grep -o 'overall acceptance = [0-9.]*' "$LOG")"
