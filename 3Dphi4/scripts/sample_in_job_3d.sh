#!/bin/bash
# Run INSIDE an existing multi-node interactive PBS job (3D phi^4).
# Distributes a given epoch list across all assigned nodes via mpirun
# (round-robin), each rank invokes sample_phi4_3d_sweep.py (SDE EM) for its share.
#
# Usage (from 3Dphi4/ dir):
#   bash scripts/sample_in_job_3d.sh <K> <L> <RUN_SUFFIX> "<EPOCH_LIST>" \
#        [SDE_STEPS=2000] [SCHEDULE=linear] \
#        [NUM_SAMPLES=256] [N_REPEATS=1] [LAMBDA=0.9]
#
# Example (6 epochs across 6 nodes → 1 epoch/node):
#   bash scripts/sample_in_job_3d.sh 0.2 64 _sigma2760 \
#       "0001 0009 0095 0977 9111 20000"

set -e

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <K> <L> <RUN_SUFFIX> \"<EPOCH_LIST>\" \\"
    echo "         [SDE_STEPS=2000] [SCHEDULE=linear] \\"
    echo "         [NUM_SAMPLES=256] [N_REPEATS=1] [LAMBDA=0.9]"
    exit 1
fi

if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: \$PBS_NODEFILE not set — run inside a PBS job"
    exit 1
fi

K="$1"
L="$2"
RUN_SUFFIX="$3"
EPOCH_LIST="$4"
SDE_STEPS="${5:-2000}"
SCHEDULE="${6:-linear}"
NUM_SAMPLES="${7:-256}"
N_REPEATS="${8:-1}"
LAMBDA="${9:-0.9}"
SEED="${SEED:-20260422}"
COMPILE_MODE="${COMPILE_MODE:-default}"

read -ra EPOCHS <<< "$EPOCH_LIST"
N_NODES=$(uniq "$PBS_NODEFILE" | wc -l)
N_EPOCHS=${#EPOCHS[@]}

# Pre-validate: bail before launching mpirun if any ckpt is missing.
RUN_DIR="runs/phi4_3d_L${L}_k${K}_l${LAMBDA}_ncsnpp${RUN_SUFFIX}"
MISSING=()
for ep in "${EPOCHS[@]}"; do
    [[ -f "${RUN_DIR}/models/epoch=${ep}.ckpt" ]] || MISSING+=("$ep")
done
if (( ${#MISSING[@]} > 0 )); then
    echo "ERROR: missing ckpts in ${RUN_DIR}/models/:"
    for ep in "${MISSING[@]}"; do echo "  epoch=${ep}.ckpt"; done
    echo
    echo "Available epochs nearby (head -40):"
    ls "${RUN_DIR}/models/" 2>/dev/null | sed 's/epoch=//;s/.ckpt//' | sort -n | head -40
    exit 1
fi

# Round-robin split: rank r gets epochs r, r+N_NODES, r+2*N_NODES, ...
declare -a CHUNKS
for ((i=0; i<N_NODES; i++)); do CHUNKS[$i]=""; done
for ((i=0; i<N_EPOCHS; i++)); do
    r=$((i % N_NODES))
    if [[ -z "${CHUNKS[$r]}" ]]; then
        CHUNKS[$r]="${EPOCHS[$i]}"
    else
        CHUNKS[$r]="${CHUNKS[$r]},${EPOCHS[$i]}"
    fi
done

echo "=== Distribution ($N_EPOCHS epochs → $N_NODES nodes) ==="
echo "    sde_steps=$SDE_STEPS  schedule=$SCHEDULE"
echo "    num_samples=$NUM_SAMPLES  n_repeats=$N_REPEATS  seed=$SEED"
echo "    compile_mode=$COMPILE_MODE  run_dir=$RUN_DIR"
i=0
for n in $(uniq "$PBS_NODEFILE"); do
    echo "  $n (rank $i): ${CHUNKS[$i]}"
    i=$((i+1))
done
echo

# Join CHUNKS with ';' (safer than space — no quoting through mpirun)
OLD_IFS="$IFS"; IFS=";"
CHUNKS_STR="${CHUNKS[*]}"
IFS="$OLD_IFS"

mpirun --map-by ppr:1:node -hostfile "$PBS_NODEFILE" \
    bash "$(dirname "$(realpath "$0")")/sample_in_job_worker_3d.sh" \
        "$CHUNKS_STR" "$K" "$L" "$LAMBDA" "$RUN_SUFFIX" \
        "$SDE_STEPS" "$NUM_SAMPLES" "$N_REPEATS" \
        "$SCHEDULE" "$SEED" "$COMPILE_MODE"
