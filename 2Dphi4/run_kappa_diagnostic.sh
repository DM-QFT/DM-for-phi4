#!/usr/bin/env bash
# Per-kappa snapshot diagnostic: fine-tune 100 epochs, sample N=1024 from 6
# snapshots, then run the 9-observable Z2-symmetrised analysis. Single GPU.
#
# Usage:
#   bash run_kappa_diagnostic.sh K SIGMA DEVICE NORM_MIN NORM_MAX
# Example:
#   bash run_kappa_diagnostic.sh 0.26   300 cuda:0 -4.217828 4.192613
#   bash run_kappa_diagnostic.sh 0.2705 450 cuda:1 -4.473008 4.743169

set -euo pipefail

if [ $# -ne 5 ]; then
    echo "Usage: $0 K SIGMA DEVICE NORM_MIN NORM_MAX" >&2
    exit 1
fi

K="$1"
SIGMA="$2"
DEVICE="$3"
NORM_MIN="$4"
NORM_MAX="$5"
SIGMA_INT="${SIGMA%.*}"

source /home/tyywork/miniconda3/etc/profile.d/conda.sh
conda activate nenv2

cd /home/tyywork/DM/2Dphi4

RUN_DIR="runs/phi4_L128_k${K}_l0.022_ncsnpp_sigma${SIGMA_INT}"
BASE_CKPT="$RUN_DIR/lightning_logs/version_0/checkpoints/epoch=9999-step=800000.ckpt"
OUT_MODELS="$RUN_DIR/models"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

echo "==== k=$K sigma=$SIGMA device=$DEVICE diagnostic start $(date) ===="

# === Phase 1: fine-tune 100 epochs ===
echo "[$(date)] Phase 1 (k=$K): fine-tune on $DEVICE"
python finetune_diagnostic.py \
    --run_dir "$RUN_DIR" \
    --base_ckpt "$BASE_CKPT" \
    --device "$DEVICE" \
    --L 128 --k "$K" --l 0.022 --sigma "$SIGMA" \
    2>&1 | tee "$LOG_DIR/finetune_diagnostic.log"

echo "[$(date)] Phase 1 (k=$K) done"

# === Phase 2: sample 6 snapshots on the same GPU (sequential) ===
SNAPSHOTS=(
    "${BASE_CKPT}|ep10000_n1024"
    "${OUT_MODELS}/finetune_ep10020.ckpt|ep10020_n1024"
    "${OUT_MODELS}/finetune_ep10040.ckpt|ep10040_n1024"
    "${OUT_MODELS}/finetune_ep10060.ckpt|ep10060_n1024"
    "${OUT_MODELS}/finetune_ep10080.ckpt|ep10080_n1024"
    "${OUT_MODELS}/finetune_ep10100.ckpt|ep10100_n1024"
)

echo "[$(date)] Phase 2 (k=$K): sequential sampling on $DEVICE"
for entry in "${SNAPSHOTS[@]}"; do
    CKPT="${entry%|*}"
    TAG="${entry#*|}"
    echo "[$(date)] sampling $TAG on $DEVICE"
    python sample_phi4.py \
        --checkpoint "$CKPT" \
        --L 128 --k "$K" --l 0.022 \
        --method em --num_steps 2000 --num_samples 1024 --n_repeats 1 \
        --network ncsnpp --output_suffix "_sigma${SIGMA_INT}" \
        --device "$DEVICE" --ep "$TAG" \
        2>&1 | tee -a "$LOG_DIR/sample_${TAG}.log"
done
echo "[$(date)] Phase 2 (k=$K) done"

# === Phase 3: analysis ===
echo "[$(date)] Phase 3 (k=$K): 9-observable Z2 analysis"
python analyze_diagnostic_observables.py \
    --run_dir "$RUN_DIR" \
    --tags ep10000_n1024 ep10020_n1024 ep10040_n1024 ep10060_n1024 ep10080_n1024 ep10100_n1024 \
    --L 128 --num_steps 2000 --out_tag n1024_z2_filt \
    --norm_min "$NORM_MIN" --norm_max "$NORM_MAX" \
    2>&1 | tee "$LOG_DIR/analyze_observables_n1024.log"

echo "==== k=$K diagnostic complete $(date) ===="
