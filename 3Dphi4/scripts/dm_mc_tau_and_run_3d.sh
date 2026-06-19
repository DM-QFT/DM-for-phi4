#!/bin/bash
# Fan out 3D phi^4 DM-MALA tau-int + ESS sweep over training epoch.
# 3 ranks, one per kappa in {0.18, 0.1923, 0.2}, using multi-L checkpoint
# runs/phi4_3d_Lmulti4-8-16-32_k{K}_l0.9_ncsnpp.
#
# Env overrides: EMA=0, NUM_CHAINS=128, MH_STEPS=500, L_LIST=...
#
# Usage:
#   qsub -I -q interact-g -W group_list=gw97 -l select=4 -l walltime=01:00:00
#   cd /work/gw97/w97000/project/DM/3Dphi4
#   bash scripts/dm_mc_tau_and_run_3d.sh

set -e

if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: \$PBS_NODEFILE not set -- run inside a PBS job"; exit 1
fi

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
PROJ_DIR="$(dirname "$SCRIPTS_DIR")"
cd "$PROJ_DIR"

N_NODES=$(uniq "$PBS_NODEFILE" | wc -l)
(( N_NODES >= 3 )) || { echo "ERROR: need >=3 nodes (got $N_NODES)"; exit 1; }

declare -a KAPS=(0.18 0.1923 0.2)
echo "=== Pre-flight check ==="
for K in "${KAPS[@]}"; do
    RUN_DIR="runs/phi4_3d_Lmulti4-8-16-32_k${K}_l0.9_ncsnpp"
    if [[ ! -d "$RUN_DIR/models" ]]; then echo "  MISSING: $RUN_DIR/models"; exit 1; fi
    N=$(ls "$RUN_DIR/models" | grep -cE '^epoch=[0-9]+\.ckpt$')
    echo "  $RUN_DIR  total ckpts: $N"
done

EMA=${EMA:-1}
NUM_CHAINS=${NUM_CHAINS:-64}
MH_STEPS=${MH_STEPS:-200}
L_LIST=${L_LIST:-4,8,16,32,64}

mkdir -p analysis/results

echo
echo "=== Fan out 3 kappa runs over $N_NODES nodes ==="
echo "EMA=$EMA  L_LIST=$L_LIST  NUM_CHAINS=$NUM_CHAINS  MH_STEPS=$MH_STEPS"
mpirun --map-by ppr:1:node -hostfile "$PBS_NODEFILE" \
    bash "$SCRIPTS_DIR/dm_mc_tau_worker_3d.sh" \
        "$EMA" "$NUM_CHAINS" "$MH_STEPS" "$L_LIST"

echo "Done. Outputs in analysis/results/"
