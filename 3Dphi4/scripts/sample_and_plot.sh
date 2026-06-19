#!/bin/bash
# 3D phi^4: submit ONE PBS batch job with N nodes for SDE EM sampling
# (all epochs fan-out via mpirun, one epoch per node), then submit a dependent
# plotting job that fires after the sampling job succeeds. Run from the LOGIN
# node — the script qsub's both jobs and exits immediately; no terminal /
# tmux / expect required.
#
# Usage (from 3Dphi4/):
#   bash scripts/sample_and_plot.sh

set -e

# ============ Edit these ============
QUEUE=regular-g            # GPU queue for sampling (regular-g routes select=1..16 → small-g, 48h cap)
PLOT_QUEUE=regular-g       # plot job MUST stay on Miyabi-G (ARM). The nenv2 conda env at
                           # ~/envs/nenv2 is aarch64-compiled — submitting to regular-c
                           # (Miyabi-C = x86_64) gives "Exec format error" at python startup.
                           # See dm3d_plot.o1978073 (2026-05-22 18:55) for the failure.
K=0.1923
LAMBDA=0.9
L=64
RUN_SUFFIX=_sigma2048
EPOCH_LIST="0009 0099 0994 20000"
# EPOCH_LIST="0977 2983 3593 20000"
SDE_STEPS=2000             # SDE Euler-Maruyama integration steps
SCHEDULE=linear            # 3D default: linear
NUM_SAMPLES=512             # per repeat; mind L=64 GPU memory
N_REPEATS=1
WALLTIME=01:00:00          # small-g max 48:00:00
SEED=${SEED:-20260422}
# ====================================

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
PROJ_DIR="$(dirname "$SCRIPTS_DIR")"
RUN_DIR="runs/phi4_3d_L${L}_k${K}_l${LAMBDA}_ncsnpp${RUN_SUFFIX}"

cd "$PROJ_DIR"

# ---- Pre-validate ckpts before submitting anything ----
MISSING=()
for ep in $EPOCH_LIST; do
    [[ -f "${RUN_DIR}/models/epoch=${ep}.ckpt" ]] || MISSING+=("$ep")
done
if (( ${#MISSING[@]} > 0 )); then
    echo "ERROR: missing ckpts in ${RUN_DIR}/models/:"
    for ep in "${MISSING[@]}"; do echo "  epoch=${ep}.ckpt"; done
    exit 1
fi

N_NODES=$(echo $EPOCH_LIST | wc -w)
EPOCH_CSV="${EPOCH_LIST// /,}"

# Export so qsub -V inherits them into the batch job environment.
export K LAMBDA L RUN_SUFFIX EPOCH_LIST EPOCH_CSV \
       SDE_STEPS SCHEDULE NUM_SAMPLES N_REPEATS SEED

echo "============================================================"
echo "Step 1: qsub ONE batch job with select=$N_NODES (mpirun fan-out)"
echo "        epochs: $EPOCH_LIST"
echo "        walltime=$WALLTIME"
echo "============================================================"

sample_jid=$(qsub \
    -q "$QUEUE" \
    -l "select=${N_NODES}:mpiprocs=1" \
    -l "walltime=${WALLTIME}" \
    -V \
    "$SCRIPTS_DIR/pbs_sample_multi_epoch.sh")
echo "  sample jobid=${sample_jid}"

echo
echo "============================================================"
echo "Step 2: qsub plot job (afterok dependency on sample job)"
echo "============================================================"
plot_jid=$(qsub \
    -q "$PLOT_QUEUE" \
    -W "depend=afterok:${sample_jid}" \
    -V \
    "$SCRIPTS_DIR/pbs_plot_propagator.sh")
echo "  plot jobid=${plot_jid}  (afterok:${sample_jid}, queue=${PLOT_QUEUE})"

echo
echo "All jobs submitted. You can log out now; check status with:"
echo "  qstat -u \$USER"
