#!/bin/bash
# Per-rank worker: resume shard r of the exact-c100 kappa_c chain
# (tag kc_ex100_2000_shard{r}) to n_steps 5000. Idempotent: shards already
# at 5000 exit immediately.

set -e

RANK=${OMPI_COMM_WORLD_RANK:?need OMPI_COMM_WORLD_RANK}
OFF=$((64 * RANK))
PY=/work/gw97/w97000/envs/nenv2/bin/python
export TORCHINDUCTOR_COMPILE_THREADS=8

cd /work/gw97/w97000/project/DM/3Dphi4

K=0.1923
RD="runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp"
INIT="$RD/data_crossL/samples_filtered512_k${K}_ep10000.npy"
LOG="results/mala_pipeline_logs/kc_ex100_5000_shard${RANK}.log"

echo "[rank $RANK on $(hostname)] resume shard $RANK -> 5000 steps (offset $OFF)"
nvidia-smi -L

"$PY" sample_phi4_crossL_mala.py \
    --run_dir "$RD" --k "$K" --l 0.9 --ep 10000 \
    --drift exact --init_npy "$INIT" --cfg_offset "$OFF" --max_cfgs 64 \
    --c 100 --n_steps 5000 --t_mh 1e-4 \
    --shift_every 2 --shift_std 0.1168 --cluster_every 10 \
    --tag "kc_ex100_2000_shard${RANK}" --resume \
    --microbatch 64 --state_every 1000 --n_snapshots 6 \
    --seed $((800 + RANK)) --device cuda:0 > "$LOG" 2>&1
echo "[rank $RANK] done  $(grep -o 'overall acceptance = [0-9.]*' "$LOG")"
