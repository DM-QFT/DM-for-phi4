#!/bin/bash
# Generic per-rank worker for kappa_c refinement jobs.
# Args: DRIFT C N_STEPS TAG_BASE SEED_BASE
# Rank r refines configs [64r, 64(r+1)) of the filtered kappa_c ensemble.

set -e

DRIFT="$1"; C="$2"; NSTEPS="$3"; TAGB="$4"; SEEDB="$5"
RANK=${OMPI_COMM_WORLD_RANK:?need OMPI_COMM_WORLD_RANK}
OFF=$((64 * RANK))
PY=/work/gw97/w97000/envs/nenv2/bin/python
export TORCHINDUCTOR_COMPILE_THREADS=8

cd /work/gw97/w97000/project/DM/3Dphi4

K=0.1923
RD="runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp"
INIT="$RD/data_crossL/samples_filtered512_k${K}_ep10000.npy"
LOG="results/mala_pipeline_logs/${TAGB}_shard${RANK}.log"

echo "[rank $RANK on $(hostname)] drift=$DRIFT c=$C n_steps=$NSTEPS offset=$OFF"
nvidia-smi -L

"$PY" sample_phi4_crossL_mala.py \
    --run_dir "$RD" --k "$K" --l 0.9 --ep 10000 \
    --drift "$DRIFT" --init_npy "$INIT" --cfg_offset "$OFF" --max_cfgs 64 \
    --c "$C" --n_steps "$NSTEPS" --t_mh 1e-4 \
    --shift_every 2 --shift_std 0.1168 \
    --cluster_every 10 \
    --tag "${TAGB}_shard${RANK}" \
    --microbatch 64 --state_every 1000 --n_snapshots 6 \
    --seed $((SEEDB + RANK)) --device cuda:0 > "$LOG" 2>&1
echo "[rank $RANK] done  $(grep -o 'overall acceptance = [0-9.]*' "$LOG")"
