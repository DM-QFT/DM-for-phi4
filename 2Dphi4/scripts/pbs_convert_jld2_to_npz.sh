#!/bin/bash
# Convert 2Dphi4 trainingdata/*.jld2 -> trainingdata/npz/*.npz (float64, uncompressed).
# Pure h5py + numpy (CPU only), but MUST run on Miyabi-G (ARM): the conda env at
# ~/envs/nenv2 is aarch64-compiled. Login node is forbidden for this, so batch only.
#
# Submit from 2Dphi4/:
#   qsub scripts/pbs_convert_jld2_to_npz.sh

#PBS -q regular-g
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=02:00:00
#PBS -W group_list=gw97
#PBS -j oe
#PBS -N convert_npz

set -e

echo "[$(date '+%F %T')] host=$(hostname)"

"$HOME/envs/nenv2/bin/python" \
    /work/gw97/w97000/project/DM/2Dphi4/scripts/convert_jld2_to_npz.py

echo "[$(date '+%F %T')] finished"
