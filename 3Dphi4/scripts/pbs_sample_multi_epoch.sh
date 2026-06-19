#!/bin/bash -l
# `-l` (login shell) per Miyabi User's Guide §5.4.1 best practice. We don't
# strictly need the `module` function anymore (see mpirun note below), but
# `bash -l` is harmless and keeps the script aligned with UG sample scripts.
# Multi-epoch single PBS batch job for 3D phi^4 sampling. Fan out N epochs to
# N nodes within a single qsub via mpirun (one epoch per node, --map-by
# ppr:1:node), so all epochs run wall-clock parallel — bounded only by gw97's
# NODE 0/256, not the per-project RUN 0/4 job cap.
#
# Submitted by sample_and_plot.sh via:
#   qsub -l select=N:mpiprocs=1 -V pbs_sample_multi_epoch.sh
# (-V inherits K/L/LAMBDA/RUN_SUFFIX/EPOCH_LIST/... from submitter env)

#PBS -W group_list=gw97
#PBS -j oe
#PBS -N dm3d_sample
# queue (-q), select, walltime are intentionally NOT set here —
# sample_and_plot.sh passes them on the qsub command line:
#   qsub -q <queue> -l select=N:mpiprocs=1 -l walltime=...

set -e
cd "$PBS_O_WORKDIR"

: "${K:?K not set}"
: "${L:?L not set}"
: "${LAMBDA:?LAMBDA not set}"
: "${RUN_SUFFIX:?RUN_SUFFIX not set}"
: "${EPOCH_LIST:?EPOCH_LIST not set}"
: "${SDE_STEPS:?SDE_STEPS not set}"
: "${SCHEDULE:?SCHEDULE not set}"
: "${NUM_SAMPLES:?NUM_SAMPLES not set}"
: "${N_REPEATS:?N_REPEATS not set}"
: "${SEED:?SEED not set}"

# Miyabi-G compute nodes auto-load `nvidia + nv-hpcx` at job start
# (User Guide §5.4.5, p.86), which puts a working `mpirun (Open MPI 4.1.9a1)`
# on PATH. We deliberately do NOT do the documented `module purge && module
# load gcc && module load ompi` swap because on login nodes `nvidia/25.9` is
# sticky — purge cannot unload it, the subsequent `module load gcc` errors
# out with a conflict, and `set -e` would kill the job before mpirun ever
# runs. The nv-hpcx mpirun handles --map-by ppr:1:node + -hostfile correctly.
command -v mpirun >/dev/null || { echo "ERROR: mpirun not on PATH"; exit 1; }

echo "[$(date '+%F %T')] host=$(hostname) nodes=$(uniq "$PBS_NODEFILE" | wc -l)"
echo "  K=$K L=$L lambda=$LAMBDA  epochs=$EPOCH_LIST"
echo "  mpirun=$(which mpirun)"

# NB: $0 inside a PBS batch job points to /var/spool/pbs/mom_priv/jobs/<jid>.SC
# (PBS copies the script there before running). Locate sibling scripts via
# PBS_O_WORKDIR instead, which is the submitter's cwd (= 3Dphi4/).
bash "$PBS_O_WORKDIR/scripts/sample_in_job_3d.sh" \
    "$K" "$L" "$RUN_SUFFIX" "$EPOCH_LIST" \
    "$SDE_STEPS" "$SCHEDULE" \
    "$NUM_SAMPLES" "$N_REPEATS" "$LAMBDA"
