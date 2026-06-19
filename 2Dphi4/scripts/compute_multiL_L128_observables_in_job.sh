#!/bin/bash

# Compute 2D phi4 multi-L L=128 observables on compute nodes.
#
# Run this script only after manually entering an interactive PBS allocation:
#   qsub -I -q interact-g -W group_list=gw97 -l select=8 -l walltime=00:10:00
#   cd /work/gw97/w97000/project/DM/2Dphi4
#   bash scripts/compute_multiL_L128_observables_in_job.sh
#
# Outputs:
#   results/multiL_L128_observables/
#     multiL_L128_k*_observables.{json,txt}
#     multiL_L128_observables_summary.{csv,txt,tex}

set -euo pipefail

PROJECT_ROOT="/work/gw97/w97000/project/DM"
PHI4_DIR="${PROJECT_ROOT}/2Dphi4"
PYTHON_BIN="${PYTHON_BIN:-$HOME/envs/nenv2/bin/python}"

KAPPAS=(0.26 0.2705 0.28)
OUT_DIR="${OUT_DIR:-results/multiL_L128_observables_fp64}"
N_BOOT="${N_BOOT:-800}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
N_SITE_BLOCKS="${N_SITE_BLOCKS:-32}"
COMPUTE_PRECISION="float64"

# Optional protocol switches. Defaults follow the user's explicit request:
# Z2 symmetrization is always used; only per-site-spatial-average site
# cumulants are reported (no pooled site cumulants).
FILTER_DM_SITE_SUPPORT="${FILTER_DM_SITE_SUPPORT:-0}"
FILTER_DM_MAGNETIZATION_SUPPORT="${FILTER_DM_MAGNETIZATION_SUPPORT:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"

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

    local filter_args=()
    if [[ "${FILTER_DM_SITE_SUPPORT}" == "1" ]]; then
        filter_args+=(--filter_dm_site_support)
    fi
    if [[ "${FILTER_DM_MAGNETIZATION_SUPPORT}" == "1" ]]; then
        filter_args+=(--filter_dm_magnetization_support)
    fi

    IFS="," read -ra my_kappas <<< "${my_chunk}"
    for kappa in "${my_kappas[@]}"; do
        "${PYTHON_BIN}" analysis/compute_multiL_L128_observables.py \
            --kappas "${kappa}" \
            --out_dir "${OUT_DIR}" \
            --n_boot "${N_BOOT}" \
            --chunk_size "${CHUNK_SIZE}" \
            --n_site_blocks "${N_SITE_BLOCKS}" \
            --no_combine \
            "${filter_args[@]}"
    done
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    worker_main "$@"
    exit 0
fi

cd "${PHI4_DIR}"

if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE is not set; refusing to compute on a login node." >&2
    echo "Start the allocation manually first:" >&2
    echo "  qsub -I -q interact-g -W group_list=gw97 -l select=8 -l walltime=00:10:00" >&2
    echo "Then run:" >&2
    echo "  cd /work/gw97/w97000/project/DM/2Dphi4" >&2
    echo "  bash scripts/compute_multiL_L128_observables_in_job.sh" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: python not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

for kappa in "${KAPPAS[@]}"; do
    sample="runs/phi4_Lmulti8-16-32-64_k${kappa}_l0.022_ncsnpp/data_crossL/samples_crossL128_multiL_ep10000_em_log_steps2000_2048.npy"
    hmc="trainingdata/cfgs_wolff_fahmc_k=${kappa}_l=0.022_128^2.jld2"
    [[ -f "${sample}" ]] || { echo "ERROR: missing DM sample ${sample}" >&2; exit 1; }
    [[ -f "${hmc}" ]] || { echo "ERROR: missing HMC data ${hmc}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"

cat <<EOF
========================================
Start        : $(date '+%F %T')
Workdir      : $(pwd)
Python       : ${PYTHON_BIN}
Kappas       : ${KAPPAS[*]}
N_BOOT       : ${N_BOOT}
CHUNK_SIZE   : ${CHUNK_SIZE}
N_SITE_BLOCKS: ${N_SITE_BLOCKS}
Precision    : ${COMPUTE_PRECISION}
Out dir      : ${OUT_DIR}
Z2           : enabled for HMC and DM
Site scheme  : per-site spatial average only (ps)
DM filters   : site=${FILTER_DM_SITE_SUPPORT}, magnetization=${FILTER_DM_MAGNETIZATION_SUPPORT}
========================================
EOF

mapfile -t nodes < <(uniq "${PBS_NODEFILE}")
n_nodes="${#nodes[@]}"
n_jobs="${#KAPPAS[@]}"
if (( n_nodes < n_jobs )) && [[ "${ALLOW_PACK:-0}" != "1" ]]; then
    echo "ERROR: need at least ${n_jobs} PBS nodes for fully parallel kappa jobs; got ${n_nodes}." >&2
    echo "       Re-enter with select>=${n_jobs}, or set ALLOW_PACK=1 to pack jobs." >&2
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
echo "Combining per-kappa outputs..."
"${PYTHON_BIN}" analysis/compute_multiL_L128_observables.py \
    --combine_only \
    --out_dir "${OUT_DIR}"

echo
echo "Finished: $(date '+%F %T')"
echo "Results: ${PHI4_DIR}/${OUT_DIR}"
