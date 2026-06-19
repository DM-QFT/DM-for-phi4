#!/usr/bin/env bash
# Re-sample 6 diagnostic snapshots at num_samples=1024, then compute the
# Z2-symmetrized 9 observables. Reuses existing fine-tune ckpts (no retraining).
#
# Launch:
#   tmux new-session -d -s phi4_n1024 'bash /home/tyywork/DM/2Dphi4/run_diagnostic_sample_n1024.sh'

set -euo pipefail

source /home/tyywork/miniconda3/etc/profile.d/conda.sh
conda activate nenv2

cd /home/tyywork/DM/2Dphi4

RUN_DIR="runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640"
BASE_CKPT="$RUN_DIR/lightning_logs/version_0/checkpoints/epoch=9999-step=800000.ckpt"
OUT_MODELS="$RUN_DIR/models"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

echo "==== n1024+Z2 diagnostic start $(date) ===="

GROUP_A=(
    "${BASE_CKPT}|ep10000_n1024"
    "${OUT_MODELS}/finetune_ep10020.ckpt|ep10020_n1024"
    "${OUT_MODELS}/finetune_ep10040.ckpt|ep10040_n1024"
)
GROUP_B=(
    "${OUT_MODELS}/finetune_ep10060.ckpt|ep10060_n1024"
    "${OUT_MODELS}/finetune_ep10080.ckpt|ep10080_n1024"
    "${OUT_MODELS}/finetune_ep10100.ckpt|ep10100_n1024"
)

sample_one() {
    local device=$1 ckpt=$2 tag=$3
    echo "[$(date)] sampling $tag on $device"
    python sample_phi4.py \
        --checkpoint "$ckpt" \
        --L 128 --k 0.28 --l 0.022 \
        --method em --num_steps 2000 --num_samples 1024 --n_repeats 1 \
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

echo "[$(date)] Phase A (n=1024): parallel sampling on cuda:0 and cuda:1"
run_group cuda:0 "${GROUP_A[@]}" &
PID_A=$!
run_group cuda:1 "${GROUP_B[@]}" &
PID_B=$!
wait "$PID_A" "$PID_B"
echo "[$(date)] Phase A done"

echo "[$(date)] Phase B: 9-observable Z2 analysis"
python analyze_diagnostic_observables.py \
    --run_dir "$RUN_DIR" \
    --tags ep10000_n1024 ep10020_n1024 ep10040_n1024 ep10060_n1024 ep10080_n1024 ep10100_n1024 \
    --L 128 --num_steps 2000 --out_tag n1024_z2 \
    2>&1 | tee "$LOG_DIR/analyze_observables_n1024.log"

echo "==== n1024+Z2 diagnostic complete $(date) ===="
