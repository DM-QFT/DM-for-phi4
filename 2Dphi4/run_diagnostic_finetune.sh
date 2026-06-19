#!/usr/bin/env bash
# Late-training snapshot variance experiment for 2D phi^4 diffusion model.
#
# Phase 1: fine-tune from epoch 9999 for 100 more epochs on cuda:0
#          (saves finetune_ep10020 ... finetune_ep10100)
# Phase 2: sample 512 configs from each of 6 snapshots, split across cuda:0/1
# Phase 3: compute |m|, chi_conn per snapshot, report drift vs noise diagnostic
#
# Launch inside tmux to survive disconnect:
#   tmux new-session -d -s phi4_diag 'bash run_diagnostic_finetune.sh'
#   tmux attach -t phi4_diag           # to watch
#   Ctrl+B then D                       # to detach

set -euo pipefail

source /home/tyywork/miniconda3/etc/profile.d/conda.sh
conda activate nenv2

cd /home/tyywork/DM/2Dphi4

RUN_DIR="runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640"
BASE_CKPT="$RUN_DIR/lightning_logs/version_0/checkpoints/epoch=9999-step=800000.ckpt"
OUT_MODELS="$RUN_DIR/models"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

echo "==== Diagnostic experiment start $(date) ===="

# === Phase 1: fine-tune ===
echo "[$(date)] Phase 1: fine-tune 100 epochs on cuda:0"
python finetune_diagnostic.py \
    --run_dir "$RUN_DIR" \
    --base_ckpt "$BASE_CKPT" \
    --device cuda:0 \
    --L 128 --k 0.28 --l 0.022 --sigma 640 \
    2>&1 | tee "$LOG_DIR/finetune_diagnostic.log"

echo "[$(date)] Phase 1 done"

# === Phase 2: sampling, two groups in parallel ===
# Six snapshots total: the original epoch=9999 + five fine-tune snapshots
# Group A on cuda:0, group B on cuda:1.
GROUP_A=(
    "${BASE_CKPT}|ep10000"
    "${OUT_MODELS}/finetune_ep10020.ckpt|ep10020"
    "${OUT_MODELS}/finetune_ep10040.ckpt|ep10040"
)
GROUP_B=(
    "${OUT_MODELS}/finetune_ep10060.ckpt|ep10060"
    "${OUT_MODELS}/finetune_ep10080.ckpt|ep10080"
    "${OUT_MODELS}/finetune_ep10100.ckpt|ep10100"
)

sample_one() {
    local device=$1 ckpt=$2 tag=$3
    echo "[$(date)] sampling $tag on $device"
    python sample_phi4.py \
        --checkpoint "$ckpt" \
        --L 128 --k 0.28 --l 0.022 \
        --method em --num_steps 2000 --num_samples 512 --n_repeats 1 \
        --network ncsnpp --output_suffix _sigma640 \
        --device "$device" --ep "$tag" \
        2>&1 | tee -a "$LOG_DIR/sample_${tag}.log"
}

run_group() {
    local device=$1
    shift
    for entry in "$@"; do
        sample_one "$device" "${entry%|*}" "${entry#*|}"
    done
}

echo "[$(date)] Phase 2: parallel sampling on cuda:0 and cuda:1"
run_group cuda:0 "${GROUP_A[@]}" &
PID_A=$!
run_group cuda:1 "${GROUP_B[@]}" &
PID_B=$!
wait "$PID_A" "$PID_B"
echo "[$(date)] Phase 2 done"

# === Phase 3: analysis ===
echo "[$(date)] Phase 3: diagnostic analysis"
python analyze_diagnostic_spread.py \
    --run_dir "$RUN_DIR" \
    --tags ep10000 ep10020 ep10040 ep10060 ep10080 ep10100 \
    --L 128 --num_steps 2000 \
    2>&1 | tee "$LOG_DIR/analyze_diagnostic.log"

echo "==== Diagnostic experiment complete $(date) ===="
