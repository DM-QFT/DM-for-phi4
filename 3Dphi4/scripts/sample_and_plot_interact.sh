#!/bin/bash
# 3D phi^4: sample (mpirun fan-out, one epoch per node) + plot, all inline
# inside an interact-g job shell. Unlike sample_and_plot.sh (batch), this does
# NOT submit any qsub — it expects to already be running on PBS-allocated
# nodes and uses them directly.
#
# How to run:
#   1) From the LOGIN node, request an interactive job. Note interact-g sub-
#      queue limits (User's Guide §5.2.2, Table 5-10):
#        select=1     → interact-g_n1, walltime ≤ 02:00:00
#        select=2..8  → interact-g_n8, walltime ≤ 00:10:00 (10 minutes!)
#      So multi-node interactive is only useful for very short tests:
#        qsub -I -l select=6:mpiprocs=1 -q interact-g \
#             -W group_list=gw97 -l walltime=00:10:00
#      For single-node interactive debug (more time, one epoch at a time):
#        qsub -I -l select=1:mpiprocs=1 -q interact-g \
#             -W group_list=gw97 -l walltime=02:00:00
#   2) Once the prompt drops you on a compute node:
#        cd /work/gw97/w97000/project/DM/3Dphi4
#        bash scripts/sample_and_plot_interact.sh
#   3) When it finishes, `exit` releases the nodes.

set -e

# ============ Edit these ============
K=0.18
LAMBDA=0.9
L=64
RUN_SUFFIX=_sigma2000
EPOCH_LIST="0095 0977 2056 2477 2983 3944 4750 20000"
SDE_STEPS=1000             # SDE Euler-Maruyama integration steps
SCHEDULE=linear            # 3D default: linear
NUM_SAMPLES=64             # per repeat; mind L=64 GPU memory
N_REPEATS=1
SEED=${SEED:-20260422}
# ====================================

# Must be inside a PBS job (interact-g shell sets PBS_NODEFILE just like batch).
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    cat <<EOF
ERROR: \$PBS_NODEFILE is not set — you are not inside a PBS job.

First request an interact-g shell from the login node, e.g.:
  qsub -I -l select=1:mpiprocs=1 -q interact-g -W group_list=gw97 -l walltime=02:00:00     # interact-g_n1
  qsub -I -l select=6:mpiprocs=1 -q interact-g -W group_list=gw97 -l walltime=00:10:00     # interact-g_n8 (≤10min)

Then rerun this script.
EOF
    exit 1
fi

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
PROJ_DIR="$(dirname "$SCRIPTS_DIR")"
RUN_DIR="runs/phi4_3d_L${L}_k${K}_l${LAMBDA}_ncsnpp${RUN_SUFFIX}"

cd "$PROJ_DIR"

# ---- Pre-validate ckpts ----
MISSING=()
for ep in $EPOCH_LIST; do
    [[ -f "${RUN_DIR}/models/epoch=${ep}.ckpt" ]] || MISSING+=("$ep")
done
if (( ${#MISSING[@]} > 0 )); then
    echo "ERROR: missing ckpts in ${RUN_DIR}/models/:"
    for ep in "${MISSING[@]}"; do echo "  epoch=${ep}.ckpt"; done
    exit 1
fi

# ---- Sanity-check node count vs epoch count ----
N_NODES=$(uniq "$PBS_NODEFILE" | wc -l)
N_EPOCHS=$(echo $EPOCH_LIST | wc -w)
if (( N_NODES < N_EPOCHS )); then
    echo "WARNING: $N_NODES nodes allocated but $N_EPOCHS epochs requested —"
    echo "         epochs will be round-robin'd, some nodes get multiple."
fi

echo "============================================================"
echo "Step 1: sample via mpirun fan-out ($N_NODES nodes, $N_EPOCHS epochs)"
echo "        epochs: $EPOCH_LIST"
echo "============================================================"
bash "$SCRIPTS_DIR/sample_in_job_3d.sh" \
    "$K" "$L" "$RUN_SUFFIX" "$EPOCH_LIST" \
    "$SDE_STEPS" "$SCHEDULE" \
    "$NUM_SAMPLES" "$N_REPEATS" "$LAMBDA"

echo
echo "============================================================"
echo "Step 2: plot propagator (inline, no second job)"
echo "============================================================"
EPOCH_CSV="${EPOCH_LIST// /,}"
"$HOME/envs/nenv2/bin/python" \
    analysis/compare_dm_hmc_propagator_3d.py \
    --k "$K" --L "$L" --l "$LAMBDA" \
    --run_suffix "$RUN_SUFFIX" \
    --epochs "$EPOCH_CSV" \
    --method em \
    --sde_steps "$SDE_STEPS"

echo
echo "Done. Type 'exit' to release the interact-g nodes."
