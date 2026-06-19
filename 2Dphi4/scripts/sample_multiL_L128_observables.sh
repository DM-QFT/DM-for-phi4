#!/bin/bash

# Generate held-out L=128 samples from 2D phi4 multi-L checkpoints.
#
# Targets:
#   runs/phi4_Lmulti8-16-32-64_k{0.26,0.2705,0.28}_l0.022_ncsnpp/models/epoch=10000.ckpt
#
# Sampling:
#   - L_sample = 128
#   - EM reverse SDE, 2000 steps, log schedule
#   - 2048 raw samples per kappa, saved as physical fields with shape (128,128,2048)
#   - No Z2 augmentation here; apply Z2/filtering in the observable analysis.
#
# Run this script *after* you manually enter an interactive PBS allocation:
#   qsub -I -q interact-g -W group_list=gw97 -l select=8 -l walltime=00:10:00
#   cd /work/gw97/w97000/project/DM/2Dphi4
#   bash scripts/sample_multiL_L128_observables.sh

set -euo pipefail

PROJECT_ROOT="/work/gw97/w97000/project/DM"
PHI4_DIR="${PROJECT_ROOT}/2Dphi4"
PYTHON_BIN="${PYTHON_BIN:-$HOME/envs/nenv2/bin/python}"

LAMBDA="0.022"
L_TRAIN="64"
L_SAMPLE="128"
EPOCH="10000"
METHOD="em"
SCHEDULE="log"
NUM_STEPS="2000"
TOTAL_SAMPLES="2048"
COMPILE_MODE="default"

# GB200 has enough memory for the full L=128 ensemble in one reverse pass.
SAMPLES_PER_REP="2048"
N_REPEATS="1"

SEED_BASE="20260601"
OUT_TAG_PREFIX="crossL128_multiL"
KAPPAS=(0.26 0.2705 0.28)

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

worker_main() {
    local chunks_str="$1"
    local rank="${OMPI_COMM_WORLD_RANK:-0}"
    local my_chunk

    IFS=";" read -ra chunks <<< "${chunks_str}"
    my_chunk="${chunks[$rank]:-}"

    if [[ -z "${my_chunk}" ]]; then
        echo "[rank ${rank} on $(hostname)] no kappa assigned"
        exit 0
    fi

    cd "${PHI4_DIR}"

    echo "[rank ${rank} on $(hostname)] kappas=${my_chunk}"
    nvidia-smi -L || true

    IFS="," read -ra my_kappas <<< "${my_chunk}"
    for kappa in "${my_kappas[@]}"; do
        local run_dir="runs/phi4_Lmulti8-16-32-64_k${kappa}_l${LAMBDA}_ncsnpp"
        local tag="${OUT_TAG_PREFIX}_ep${EPOCH}_${METHOD}_${SCHEDULE}_steps${NUM_STEPS}_${TOTAL_SAMPLES}"
        local out_dir="${run_dir}/data_crossL"
        local out_npy="${out_dir}/samples_${tag}.npy"
        local log_dir="${run_dir}/logs_crossL"
        local ckpt="${run_dir}/models/epoch=${EPOCH}.ckpt"
        local seed_tag="${kappa//./}"
        local seed=$((SEED_BASE + 10#${seed_tag}))

        mkdir -p "${out_dir}" "${log_dir}" "${run_dir}/data"

        if [[ ! -f "${ckpt}" ]]; then
            echo "[rank ${rank}] ERROR: missing checkpoint: ${ckpt}" >&2
            exit 1
        fi

        if [[ -f "${out_npy}" && "${FORCE:-0}" != "1" ]]; then
            echo "[rank ${rank}] skip existing ${out_npy} (set FORCE=1 to overwrite)"
        else
            echo "[rank ${rank}] sampling k=${kappa} -> ${out_npy}"
            "${PYTHON_BIN}" sample_phi4_crossL.py \
                --L_train "${L_TRAIN}" --L_sample "${L_SAMPLE}" \
                --k "${kappa}" --l "${LAMBDA}" --ep "${EPOCH}" \
                --network ncsnpp \
                --num_samples "${SAMPLES_PER_REP}" \
                --n_repeats "${N_REPEATS}" \
                --num_steps "${NUM_STEPS}" \
                --method "${METHOD}" \
                --schedule "${SCHEDULE}" \
                --compile_mode "${COMPILE_MODE}" \
                --device cuda:0 \
                --run_dir "${run_dir}" \
                --tag "${tag}" \
                --seed "${seed}" \
                2>&1 | tee "${log_dir}/sample_${tag}.log"
        fi

        "${PYTHON_BIN}" - "${out_npy}" "${L_SAMPLE}" "${TOTAL_SAMPLES}" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
L = int(sys.argv[2])
N = int(sys.argv[3])
arr = np.load(path, mmap_mode="r")
expected = (L, L, N)
if arr.shape != expected:
    raise SystemExit(f"shape mismatch for {path}: got {arr.shape}, expected {expected}")
print(
    f"[verify] {path}: shape={arr.shape} dtype={arr.dtype} "
    f"min={float(arr.min()):.6g} max={float(arr.max()):.6g} mean={float(arr.mean()):.6g}"
)
PY

        # Convenience name for analysis scripts that look under run_dir/data.
        ln -sf "../data_crossL/samples_${tag}.npy" \
            "${run_dir}/data/samples_${tag}.npy"
    done
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    worker_main "$@"
    exit 0
fi

cd "${PHI4_DIR}"

cat <<EOF
========================================
Job ID       : ${PBS_JOBID:-<no-pbs>}
Start        : $(date '+%F %T')
Workdir      : $(pwd)
Python       : ${PYTHON_BIN}
L_sample     : ${L_SAMPLE}
lambda       : ${LAMBDA}
epoch        : ${EPOCH}
sampler      : ${METHOD}, ${NUM_STEPS} steps, ${SCHEDULE} schedule
compile mode : ${COMPILE_MODE}
samples      : ${SAMPLES_PER_REP} x ${N_REPEATS} = ${TOTAL_SAMPLES}
kappas       : ${KAPPAS[*]}
output tag   : ${OUT_TAG_PREFIX}_ep${EPOCH}_${METHOD}_${SCHEDULE}_steps${NUM_STEPS}_${TOTAL_SAMPLES}
========================================
EOF

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: python not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "sample_phi4_crossL.py" ]]; then
    echo "ERROR: run from a checkout containing 2Dphi4/sample_phi4_crossL.py" >&2
    exit 1
fi

missing=()
for kappa in "${KAPPAS[@]}"; do
    run_dir="runs/phi4_Lmulti8-16-32-64_k${kappa}_l${LAMBDA}_ncsnpp"
    ckpt="${run_dir}/models/epoch=${EPOCH}.ckpt"
    [[ -f "${ckpt}" ]] || missing+=("${ckpt}")
done

if (( ${#missing[@]} > 0 )); then
    echo "ERROR: missing checkpoint(s):" >&2
    for path in "${missing[@]}"; do
        echo "  ${path}" >&2
    done
    exit 1
fi

echo "Planned outputs:"
for kappa in "${KAPPAS[@]}"; do
    run_dir="runs/phi4_Lmulti8-16-32-64_k${kappa}_l${LAMBDA}_ncsnpp"
    tag="${OUT_TAG_PREFIX}_ep${EPOCH}_${METHOD}_${SCHEDULE}_steps${NUM_STEPS}_${TOTAL_SAMPLES}"
    echo "  ${run_dir}/data_crossL/samples_${tag}.npy"
done
echo

if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE is not set, so this is not running inside an interactive PBS allocation." >&2
    echo "Start the allocation manually first:" >&2
    echo "  qsub -I -q interact-g -W group_list=gw97 -l select=8 -l walltime=00:10:00" >&2
    echo "Then run:" >&2
    echo "  cd /work/gw97/w97000/project/DM/2Dphi4" >&2
    echo "  bash scripts/sample_multiL_L128_observables.sh" >&2
    exit 1
fi

mapfile -t nodes < <(uniq "${PBS_NODEFILE}")
n_nodes="${#nodes[@]}"
n_jobs="${#KAPPAS[@]}"

if (( n_nodes < n_jobs )) && [[ "${ALLOW_PACK:-0}" != "1" ]]; then
    echo "ERROR: need at least ${n_jobs} PBS nodes to run all kappas in parallel; got ${n_nodes}." >&2
    echo "       Re-submit with select>=${n_jobs}, or set ALLOW_PACK=1 to allow sequential packing." >&2
    exit 1
fi

n_workers="${n_jobs}"
if (( n_nodes < n_jobs )); then
    n_workers="${n_nodes}"
fi

declare -a chunks
for ((i=0; i<n_workers; i++)); do
    chunks[$i]=""
done

for ((i=0; i<n_jobs; i++)); do
    r=$((i % n_workers))
    if [[ -z "${chunks[$r]}" ]]; then
        chunks[$r]="${KAPPAS[$i]}"
    else
        chunks[$r]="${chunks[$r]},${KAPPAS[$i]}"
    fi
done

echo "=== Distribution (${n_jobs} kappas -> ${n_workers}/${n_nodes} allocated nodes) ==="
for ((i=0; i<n_workers; i++)); do
    echo "  ${nodes[$i]} (rank ${i}): ${chunks[$i]:-<idle>}"
done
echo

active_hostfile="$(mktemp)"
trap 'rm -f "${active_hostfile}"' EXIT
for ((i=0; i<n_workers; i++)); do
    echo "${nodes[$i]}" >> "${active_hostfile}"
done

old_ifs="${IFS}"
IFS=";"
chunks_str="${chunks[*]}"
IFS="${old_ifs}"

mpirun -np "${n_workers}" --map-by ppr:1:node -hostfile "${active_hostfile}" \
    bash "$(realpath "$0")" --worker "${chunks_str}"

echo
echo "Finished: $(date '+%F %T')"
