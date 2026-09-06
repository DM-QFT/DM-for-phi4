#!/bin/bash
# Per-rank worker: rank -> (L, schedule) for c2_sensitivity_3d.py.
set -e

PARTS=("64 log" "64 linear" "32 log" "32 linear")
read -r L SCHED <<< "${PARTS[$OMPI_COMM_WORLD_RANK]}"

echo "[rank $OMPI_COMM_WORLD_RANK on $(hostname)] L=$L sched=$SCHED"
cd /work/gw97/w97000/project/DM/betaconfig/tests

"$HOME/envs/nenv2/bin/python" c2_sensitivity_3d.py --L "$L" --sched "$SCHED" \
    > "../logs/c2_3d_L${L}_${SCHED}.log" 2>&1
