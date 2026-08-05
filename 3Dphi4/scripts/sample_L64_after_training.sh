#!/bin/bash
# Wait for k=0.18 (cuda:2, sigma=2000) and k=0.2 (cuda:3, sigma=2760) training
# to finish, then generate 512 samples at epoch=20000 with EM/linear/2000 steps
# for both. On CUDA OOM the per-rep batch is halved while keeping the 512 total
# (1x512 -> 2x256 -> 4x128 -> 8x64 -> 16x32). Used to populate the DM L=64 rows
# of Table VI (tab:multiL_HMC_cumulants) in paper/DM/DM.tex.

set -u

source /home/tyywork/miniconda3/etc/profile.d/conda.sh
conda activate nenv2

cd /data/tyywork/DM/3Dphi4

CKPT_018=runs/phi4_3d_L64_k0.18_l0.9_ncsnpp_sigma2000/models/epoch=20000.ckpt
CKPT_02=runs/phi4_3d_L64_k0.2_l0.9_ncsnpp_sigma2760/models/epoch=20000.ckpt

PID_018=1730535
PID_02=1730661

EXPECTED_SIZE=45860901  # bytes of a healthy 20000.ckpt

echo "[wait] start: $(date)"

wait_for_training () {
    local pid="$1" ckpt="$2" tag="$3"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    echo "[wait] $tag pid $pid exited at $(date)"
    local tries=0
    while [[ ! -s "$ckpt" ]] || [[ "$(stat -c%s "$ckpt" 2>/dev/null || echo 0)" -lt "$EXPECTED_SIZE" ]]; do
        tries=$((tries+1))
        if (( tries > 30 )); then
            echo "[error] $tag: $ckpt missing or too small after 30 retries"
            return 1
        fi
        sleep 10
    done
    echo "[wait] $tag ckpt ready: $ckpt ($(stat -c%s "$ckpt") bytes)"
}

wait_for_training "$PID_018" "$CKPT_018" "k=0.18" &
W018=$!
wait_for_training "$PID_02"  "$CKPT_02"  "k=0.2"  &
W02=$!
wait "$W018" || { echo "[error] wait_for_training k=0.18 failed"; exit 1; }
wait "$W02"  || { echo "[error] wait_for_training k=0.2 failed";  exit 1; }

echo "[sample] both trainings done, launching sampling at $(date)"

mkdir -p logs

# (num_samples, n_repeats) pairs, total = num_samples * n_repeats = 512
CONFIGS=("512 1" "256 2" "128 4" "64 8" "32 16")

sample_k_with_fallback () {
    local k="$1" sigma="$2" device="$3"
    local attempt=0
    for cfg in "${CONFIGS[@]}"; do
        attempt=$((attempt+1))
        local ns nr; read ns nr <<<"$cfg"
        local log="logs/sample_L64_k${k}_ep20000_try${attempt}_ns${ns}_nr${nr}.log"
        echo "[sample] k=$k attempt $attempt: num_samples=$ns n_repeats=$nr  (device $device)"
        python -u sample_phi4.py \
            --method em --L 64 --k "$k" --l 0.9 \
            --device "$device" \
            --num_steps 2000 --schedule linear \
            --num_samples "$ns" --n_repeats "$nr" \
            --network ncsnpp --output_suffix "_sigma${sigma}" \
            --ep 20000 \
            > "$log" 2>&1
        local rc=$?
        if [[ $rc -eq 0 ]]; then
            echo "[sample] k=$k SUCCESS at attempt $attempt"
            return 0
        fi
        if grep -qiE "out of memory|outofmemory|cuda out of memory" "$log"; then
            echo "[sample] k=$k OOM at attempt $attempt (ns=$ns nr=$nr); falling back"
            continue
        else
            echo "[sample] k=$k FAILED at attempt $attempt rc=$rc (non-OOM)"
            tail -n 20 "$log"
            return $rc
        fi
    done
    echo "[sample] k=$k all attempts exhausted"
    return 1
}

sample_k_with_fallback 0.18 2000 cuda:2 &
S018=$!
sample_k_with_fallback 0.2  2760 cuda:3 &
S02=$!
wait "$S018"; r1=$?
wait "$S02";  r2=$?

echo "[sample] k=0.18 exit=$r1   k=0.2 exit=$r2   at $(date)"
exit $(( r1 + r2 ))
