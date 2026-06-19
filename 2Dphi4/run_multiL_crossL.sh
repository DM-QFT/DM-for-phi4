#!/usr/bin/env bash
# Full 6-snapshot diagnostic for 2D multi-L models, cross-L sampled at L=128.
#
# Mirrors run_kappa_diagnostic.sh but for multi-L training (L_train in
# {8,16,32,64}), with L_sample=128 cross-L generation.
#
# Phases per kappa:
#   1. Fine-tune 100 epochs from the baseline ckpt, save 5 new snapshots
#   2. Cross-L sample L=128 from each of the 6 snapshots (N=1024, EM 2000 steps)
#   3. Snapshot-spread analysis via analyze_diagnostic_observables.py
#
# Three kappa values run sequentially in this script (cuda assignment alternates).
#
# Launch:
#   tmux new-session -d -s phi4_multiL 'bash run_multiL_crossL.sh'

set -euo pipefail

source /home/tyywork/miniconda3/etc/profile.d/conda.sh
conda activate nenv2

cd /home/tyywork/DM/2Dphi4

ROOT_LOG_DIR="runs/_multiL_crossL_logs"
mkdir -p "$ROOT_LOG_DIR"

echo "==== multi-L cross-L diagnostic start $(date) ===="

# -------------- Per-kappa pipeline --------------
run_kappa() {
    local K="$1" BASE_EP="$2" NMIN="$3" NMAX="$4" DEVICE="$5"
    local RDIR="runs/phi4_Lmulti8-16-32-64_k${K}_l0.022_ncsnpp"
    local BASE_CKPT="${RDIR}/models/epoch=$(printf "%04d" "$BASE_EP").ckpt"
    local LOG_DIR="${RDIR}/logs_crossL"
    mkdir -p "$LOG_DIR"

    echo ""
    echo "==== k=$K  base_ep=$BASE_EP  device=$DEVICE ===="

    # ---- Phase 1: fine-tune ----
    echo "[$(date)] Phase 1 (k=$K): fine-tune 100 epochs"
    python finetune_multiL_diagnostic.py \
        --run_dir "$RDIR" \
        --base_ckpt "$BASE_CKPT" \
        --device "$DEVICE" \
        --k "$K" --l 0.022 \
        --start_epoch "$BASE_EP" \
        --extra_epochs 100 --save_every 20 \
        2>&1 | tee "$LOG_DIR/finetune.log"
    echo "[$(date)] Phase 1 (k=$K) done"

    # ---- Phase 2: cross-L sample L=128 from 6 snapshots ----
    local SNAP_EPS=()
    SNAP_EPS+=("$BASE_EP")
    for offset in 20 40 60 80 100; do
        SNAP_EPS+=("$((BASE_EP + offset))")
    done

    # Output naming alignment: sample_phi4_crossL saves to
    # $RDIR/data_crossL/samples_${tag}.npy. The analysis script
    # analyze_diagnostic_observables.py reads
    # $RDIR/data/samples_em_steps${nsteps}_${tag}.npy. Symlink each
    # cross-L output into the data/ tree so the analyser can pick it up.
    mkdir -p "${RDIR}/data"

    local TAGS=()
    for EP in "${SNAP_EPS[@]}"; do
        local TAG="ep${EP}_multiL_n1024"
        TAGS+=("$TAG")
        echo "[$(date)] sampling k=$K ep=$EP on $DEVICE"
        python sample_phi4_crossL.py \
            --L_train 64 --L_sample 128 \
            --k "$K" --l 0.022 --ep "$EP" \
            --network ncsnpp \
            --num_samples 1024 --num_steps 2000 \
            --method em --schedule log --n_repeats 1 \
            --device "$DEVICE" \
            --run_dir "$RDIR" \
            --tag "$TAG" \
            2>&1 | tee "$LOG_DIR/sample_${TAG}.log"
        # symlink for the analyser
        ln -sf "../data_crossL/samples_${TAG}.npy" \
               "${RDIR}/data/samples_em_steps2000_${TAG}.npy"
    done
    echo "[$(date)] Phase 2 (k=$K) done"

    # ---- Phase 3: snapshot-spread analysis ----
    echo "[$(date)] Phase 3 (k=$K): 9-obs analysis"
    python analyze_diagnostic_observables.py \
        --run_dir "$RDIR" \
        --tags "${TAGS[@]}" \
        --L 128 --num_steps 2000 \
        --out_tag multiL_crossL_L128_n1024 \
        --norm_min "$NMIN" --norm_max "$NMAX" \
        2>&1 | tee "$LOG_DIR/analyze.log"
    echo "[$(date)] Phase 3 (k=$K) done"
}

# Two-track parallel layout:
#   cuda:0: k=0.26 (symmetric) then k=0.28 (broken)
#   cuda:1: k=0.2705 (near-critical)
# These three full pipelines run independently.
(
    run_kappa 0.26   9111 -4.205986499786377 4.041594505310059   cuda:0
    run_kappa 0.28   9111 -4.778076171875    4.771599292755127   cuda:0
) > "$ROOT_LOG_DIR/track_cuda0.log" 2>&1 &
PID_A=$!

(
    run_kappa 0.2705 8554 -4.352510452270508 4.3837151527404785 cuda:1
) > "$ROOT_LOG_DIR/track_cuda1.log" 2>&1 &
PID_B=$!

wait "$PID_A" "$PID_B"

echo ""
echo "==== Building combined multi-kappa cross-L table $(date) ===="
python build_multiL_crossL_table.py 2>&1 | tee "$ROOT_LOG_DIR/combined_table.log"

echo "==== multi-L cross-L diagnostic done $(date) ===="
