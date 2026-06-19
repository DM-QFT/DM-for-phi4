#!/bin/bash
# Sample DM checkpoints across multiple nodes, then plot diagonal propagator.
# Run from a multi-node interactive PBS job (e.g. qsub -I -q interact-g -l select=8).
#
# Edit the variables below, then:
#   bash scripts/sample_and_plot.sh

set -e

# ============ Edit these ============
K=0.2705
L=128
RUN_SUFFIX=_sigma450
# EPOCH_LIST="0010 0102 0205 1014 5013 8096 10018"
EPOCH_LIST="0001 0002 0005 0010 0104 0977 10000"
METHOD=em
EMA=true
# ====================================

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
PROJ_DIR="$(dirname "$SCRIPTS_DIR")"

echo "============================================================"
echo "Step 1: sample DM at epochs $EPOCH_LIST"
echo "============================================================"
bash "$SCRIPTS_DIR/sample_in_job.sh" "$K" "$L" "$RUN_SUFFIX" "$EPOCH_LIST" "$EMA"

echo
echo "============================================================"
echo "Step 2: compute diagonal propagator + plot vs HMC"
echo "============================================================"
"$HOME/envs/nenv2/bin/python" "$PROJ_DIR/analysis/compare_dm_hmc_propagator.py" \
    --k "$K" --L "$L" --l 0.022 \
    --run_suffix "$RUN_SUFFIX" \
    --epochs "${EPOCH_LIST// /,}" \
    --method "$METHOD"
