#!/bin/bash
# Per-rank worker: learned-score MALA + shift + cluster starting from the
# PC5-full ensemble (kappa_c). Tests whether a better initial zero-mode
# (PC already fixed <|M|>) lets the score-drift chain reach HMC.
# 8 shards; rank r refines configs [64r, 64(r+1)) of the PC5-full init.

set -e
RANK=${OMPI_COMM_WORLD_RANK:?need OMPI_COMM_WORLD_RANK}
OFF=$((64 * RANK))
PY=/work/gw97/w97000/envs/nenv2/bin/python
export TORCHINDUCTOR_COMPILE_THREADS=8
cd /work/gw97/w97000/project/DM/3Dphi4

K=0.1923
RD="runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp"
INIT="$RD/data_crossL/samples_pcfull_pc5_filtered.npy"
LOG="results/mala_pipeline_logs/pcinit_score_shard${RANK}.log"

echo "[rank $RANK on $(hostname)] PC5-full init, score MALA, offset $OFF"
nvidia-smi -L

"$PY" sample_phi4_crossL_mala.py \
    --run_dir "$RD" --k "$K" --l 0.9 --ep 10000 \
    --drift score --init_npy "$INIT" --cfg_offset "$OFF" --max_cfgs 64 \
    --c 3 --n_steps 2000 --t_mh 1e-4 \
    --shift_every 2 --shift_std 0.1168 --cluster_every 10 \
    --tag "pcinit_score_shard${RANK}" \
    --microbatch 64 --state_every 1000 --n_snapshots 6 \
    --seed $((900 + RANK)) --device cuda:0 > "$LOG" 2>&1
echo "[rank $RANK] done  $(grep -o 'overall acceptance = [0-9.]*' "$LOG")"
