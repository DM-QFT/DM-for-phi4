#!/bin/bash
# Per-rank worker for the 8-node MALA+shift+cluster production.
# Launched once per node by mpirun from scripts/mala_prod_8node.pbs.
#
# Rank r refines configs [64*r, 64*(r+1)) of each kappa's pre-filtered
# ensemble for 2000 steps with the validated recipe:
#   MALA c=3 + global shift every 2 + embedded SW cluster every 10.
# Shards write to data_crossL_mala/prod512f_2000_shard${r}/ per run dir;
# merge afterwards with scripts/merge_mala_shards.py.

set -e

RANK=${OMPI_COMM_WORLD_RANK:?need OMPI_COMM_WORLD_RANK}
OFF=$((64 * RANK))
PY=/work/gw97/w97000/envs/nenv2/bin/python
export TORCHINDUCTOR_COMPILE_THREADS=8

cd /work/gw97/w97000/project/DM/3Dphi4

echo "[rank $RANK on $(hostname)] cfg_offset=$OFF"
nvidia-smi -L

# kappa | run-dir suffix | shift_std (= 2.4*sqrt(chi_HMC/V), physical units)
for KS in "0.18 _sigma200 0.0105" "0.1923 NOSUF 0.1168" "0.2 _sigma400 0.0104"; do
    set -- $KS
    K=$1; SUF=$2; SSTD=$3
    [ "$SUF" = "NOSUF" ] && SUF=""
    RD="runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp${SUF}"
    INIT="$RD/data_crossL/samples_filtered512_k${K}_ep10000.npy"
    LOG="results/mala_pipeline_logs/prod2000_k${K}_shard${RANK}.log"
    echo "[rank $RANK] kappa=$K start $(date '+%T')"
    "$PY" sample_phi4_crossL_mala.py \
        --run_dir "$RD" --k "$K" --l 0.9 --ep 10000 \
        --init_npy "$INIT" --cfg_offset "$OFF" --max_cfgs 64 \
        --c 3 --n_steps 2000 --t_mh 1e-4 \
        --shift_every 2 --shift_std "$SSTD" \
        --cluster_every 10 \
        --tag "prod512f_2000_shard${RANK}" \
        --microbatch 64 --state_every 500 --n_snapshots 4 \
        --seed $((100 + RANK)) --device cuda:0 > "$LOG" 2>&1
    echo "[rank $RANK] kappa=$K done  $(date '+%T')  $(grep -o 'overall acceptance = [0-9.]*' "$LOG")"
done
echo "[rank $RANK] ALL DONE"
