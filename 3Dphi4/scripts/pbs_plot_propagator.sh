#!/bin/bash
# Aggregate-and-plot batch job for 3D phi^4. Runs after all per-epoch sampling
# jobs succeed (PBS afterok dependency). Pure numpy/matplotlib — but MUST be
# submitted to a Miyabi-G (ARM) queue, not Miyabi-C (x86_64), because the
# nenv2 conda env at ~/envs/nenv2 is aarch64-compiled. Running it on Miyabi-C
# yields "Exec format error" at python startup. Login node is also forbidden
# (admin warning on heavy compute), so this batch job is the only path.

#PBS -l select=1:mpiprocs=1
#PBS -l walltime=00:30:00
#PBS -W group_list=gw97
#PBS -j oe
#PBS -N dm3d_plot
# queue (-q) is set on the qsub command line by sample_and_plot.sh.

set -e

cd "$PBS_O_WORKDIR"

: "${K:?K not set}"
: "${L:?L not set}"
: "${LAMBDA:?LAMBDA not set}"
: "${RUN_SUFFIX:?RUN_SUFFIX not set}"
: "${EPOCH_CSV:?EPOCH_CSV not set}"
: "${SDE_STEPS:?SDE_STEPS not set}"

echo "[$(date '+%F %T')] host=$(hostname) epochs=$EPOCH_CSV"

"$HOME/envs/nenv2/bin/python" \
    analysis/compare_dm_hmc_propagator_3d.py \
    --k "$K" --L "$L" --l "$LAMBDA" \
    --run_suffix "$RUN_SUFFIX" \
    --epochs "$EPOCH_CSV" \
    --method em \
    --sde_steps "$SDE_STEPS"
