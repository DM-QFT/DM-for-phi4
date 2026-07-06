#!/bin/bash
# Per-rank worker: kappa_c=0.1923 only, shift width reduced by 10x
# (shift_std 0.1168 -> 0.0117). Single-variable test against the
# prod512f_2000 baseline. Launched by scripts/mala_kc_8node.pbs.

set -e

RANK=${OMPI_COMM_WORLD_RANK:?need OMPI_COMM_WORLD_RANK}
OFF=$((64 * RANK))
PY=/work/gw97/w97000/envs/nenv2/bin/python
export TORCHINDUCTOR_COMPILE_THREADS=8

cd /work/gw97/w97000/project/DM/3Dphi4

K=0.1923
RD="runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp"
INIT="$RD/data_crossL/samples_filtered512_k${K}_ep10000.npy"
LOG="results/mala_pipeline_logs/kcS_k${K}_shard${RANK}.log"

echo "[rank $RANK on $(hostname)] cfg_offset=$OFF shift_std=0.0117"
nvidia-smi -L

"$PY" sample_phi4_crossL_mala.py \
    --run_dir "$RD" --k "$K" --l 0.9 --ep 10000 \
    --init_npy "$INIT" --cfg_offset "$OFF" --max_cfgs 64 \
    --c 3 --n_steps 2000 --t_mh 1e-4 \
    --shift_every 2 --shift_std 0.0117 \
    --cluster_every 10 \
    --tag "prod512f_2000_kcS_shard${RANK}" \
    --microbatch 64 --state_every 500 --n_snapshots 4 \
    --seed $((200 + RANK)) --device cuda:0 > "$LOG" 2>&1
echo "[rank $RANK] done  $(grep -o 'overall acceptance = [0-9.]*' "$LOG")  $(grep -o 'acc_shift=[0-9.]*' "$LOG" | tail -1)"
