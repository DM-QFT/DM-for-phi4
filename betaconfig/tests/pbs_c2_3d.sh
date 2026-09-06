#!/bin/bash -l
# 4-node fan-out of c2_sensitivity_3d.py: one (L, schedule) combo per node.
# Submit from anywhere:  qsub betaconfig/tests/pbs_c2_3d.sh

#PBS -q regular-g
#PBS -l select=4:mpiprocs=1
#PBS -l walltime=01:00:00
#PBS -W group_list=gw97
#PBS -j oe
#PBS -N c2_3d

set -e
export PATH=/work/gw97/w97000/envs/nenv2/bin:$PATH

# nv-hpcx mpirun is on PATH at job start (UG §5.4.5); no module swap needed
command -v mpirun >/dev/null || { echo "ERROR: mpirun not on PATH"; exit 1; }
echo "[$(date '+%F %T')] nodes=$(uniq "$PBS_NODEFILE" | wc -l)"

mpirun --map-by ppr:1:node -hostfile "$PBS_NODEFILE" \
    bash /work/gw97/w97000/project/DM/betaconfig/tests/pbs_c2_3d_worker.sh

echo "[$(date '+%F %T')] all parts done"
