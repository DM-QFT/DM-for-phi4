#!/bin/bash
# 2D phi^4 DM-MALA tau_int / ESS sweep over step-size coefficient c at a
# fixed (converged) epoch.  Default epoch 10000, default L_list 8,16,32,64,128,
# default c grid matches calibrate_step_size.py.
#
# Env overrides: EMA, NUM_CHAINS, MH_STEPS, L_LIST, EPOCH, C_LIST
#
# Usage (inside an interactive PBS job >=3 nodes):
#   qsub -I -q interact-g -W group_list=gw97 -l select=4 -l walltime=00:30:00
#   cd /work/gw97/w97000/project/DM/2Dphi4
#   bash scripts/dm_mc_tau_vs_c_and_run.sh

set -e

if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: \$PBS_NODEFILE not set -- run inside a PBS job"; exit 1
fi

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
PROJ_DIR="$(dirname "$SCRIPTS_DIR")"
cd "$PROJ_DIR"

N_NODES=$(uniq "$PBS_NODEFILE" | wc -l)
(( N_NODES >= 3 )) || { echo "ERROR: need >=3 nodes (got $N_NODES)"; exit 1; }

EMA=${EMA:-1}
NUM_CHAINS=${NUM_CHAINS:-64}
MH_STEPS=${MH_STEPS:-200}
L=${L:-128}
EPOCH=${EPOCH:-10000}
C_LIST=${C_LIST:-0.01,0.02,0.03,0.05,0.07,0.1,0.13,0.17,0.2,0.25,0.3,0.4,0.5,0.7,1.0,1.5,2.0,3.0,5.0,7.0,10.0,15.0,20.0}

declare -a KAPS=(0.26     0.2705   0.28    )
declare -a SUFS=(sigma300 sigma450 sigma640)
echo "=== Pre-flight check (single-L ckpts at L=${L}) ==="
for i in 0 1 2; do
    K=${KAPS[$i]}; S=${SUFS[$i]}
    RUN_DIR="runs/phi4_L${L}_k${K}_l0.022_ncsnpp_${S}"
    if [[ ! -d "$RUN_DIR/models" ]]; then echo "  MISSING: $RUN_DIR/models"; exit 1; fi
    N=$(ls "$RUN_DIR/models" | grep -cE '^epoch=[0-9]+\.ckpt$')
    echo "  $RUN_DIR  total ckpts: $N"
done

mkdir -p acceptance_rate/results

echo
echo "=== Step 1: 3 kappa c-sweeps over $N_NODES nodes (one kappa per rank) ==="
echo "EMA=$EMA  EPOCH=$EPOCH  L=$L  C_LIST=$C_LIST"
echo "NUM_CHAINS=$NUM_CHAINS  MH_STEPS=$MH_STEPS"
mpirun --map-by ppr:1:node -hostfile "$PBS_NODEFILE" \
    bash "$SCRIPTS_DIR/dm_mc_tau_vs_c_worker.sh" \
        "$EMA" "$NUM_CHAINS" "$MH_STEPS" "$L" "$EPOCH" "$C_LIST"

SUFFIX=
if [[ "$EMA" != "0" ]]; then SUFFIX=_ema; fi
NPZS=()
for i in 0 1 2; do
    K=${KAPS[$i]}; S=${SUFS[$i]}
    NPZS+=("acceptance_rate/results/dm_mc_tau_vs_c_L${L}_k${K}_${S}_ep${EPOCH}${SUFFIX}.npz")
done

echo
echo "=== Step 2: export .dat for Mathematica ==="
mkdir -p acceptance_rate/dat
"$HOME/envs/nenv2/bin/python" acceptance_rate/export_dm_mc_tau_vs_c_dat.py \
    --inputs "${NPZS[@]}" \
    --out_dir acceptance_rate/dat

echo
echo "=== Step 3: multi-kappa combined pdf ==="
"$HOME/envs/nenv2/bin/python" acceptance_rate/plot_dm_mc_tau_vs_c_combined.py \
    --inputs "${NPZS[@]}" \
    --out ../draft/figures/2Dphi4/dm_mc_tau_vs_c_L${L}_multi_kappa_ep${EPOCH}${SUFFIX}.pdf

echo "Done."
echo "  per-kappa npz/pdf : acceptance_rate/results/"
echo "  per-kappa dat     : acceptance_rate/dat/"
echo "  combined pdf      : draft/figures/2Dphi4/"
