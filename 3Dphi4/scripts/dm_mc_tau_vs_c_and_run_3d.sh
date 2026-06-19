#!/bin/bash
# 3D phi^4 DM-MALA tau_int / ESS sweep over step-size coefficient c at a
# fixed (converged) epoch.  Default epoch 20000, default L_list 4,8,16,32,64.
#
# Env overrides: EMA, NUM_CHAINS, MH_STEPS, L_LIST, EPOCH, C_LIST
#
# Usage:
#   qsub -I -q interact-g -W group_list=gw97 -l select=4 -l walltime=00:30:00
#   cd /work/gw97/w97000/project/DM/3Dphi4
#   bash scripts/dm_mc_tau_vs_c_and_run_3d.sh

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
L=${L:-64}
EPOCH=${EPOCH:-20000}
C_LIST=${C_LIST:-0.01,0.02,0.03,0.05,0.07,0.1,0.13,0.17,0.2,0.25,0.3,0.4,0.5,0.7,1.0,1.5,2.0,3.0,5.0,7.0,10.0,15.0,20.0}

declare -a KAPS=(0.18      0.1923    0.2      )
declare -a SUFS=(sigma2000 sigma2048 sigma2760)
echo "=== Pre-flight check (single-L ckpts at L=${L}) ==="
for i in 0 1 2; do
    K=${KAPS[$i]}; S=${SUFS[$i]}
    RUN_DIR="runs/phi4_3d_L${L}_k${K}_l0.9_ncsnpp_${S}"
    if [[ ! -d "$RUN_DIR/models" ]]; then echo "  MISSING: $RUN_DIR/models"; exit 1; fi
    N=$(ls "$RUN_DIR/models" | grep -cE '^epoch=[0-9]+\.ckpt$')
    echo "  $RUN_DIR  total ckpts: $N"
done

mkdir -p analysis/results

echo
echo "=== Step 1: 3 kappa c-sweeps over $N_NODES nodes (one kappa per rank) ==="
echo "EMA=$EMA  EPOCH=$EPOCH  L=$L  C_LIST=$C_LIST"
echo "NUM_CHAINS=$NUM_CHAINS  MH_STEPS=$MH_STEPS"
mpirun --map-by ppr:1:node -hostfile "$PBS_NODEFILE" \
    bash "$SCRIPTS_DIR/dm_mc_tau_vs_c_worker_3d.sh" \
        "$EMA" "$NUM_CHAINS" "$MH_STEPS" "$L" "$EPOCH" "$C_LIST"

SUFFIX=
if [[ "$EMA" != "0" ]]; then SUFFIX=_ema; fi
NPZS=()
for i in 0 1 2; do
    K=${KAPS[$i]}; S=${SUFS[$i]}
    NPZS+=("analysis/results/dm_mc_tau_vs_c_3d_L${L}_k${K}_${S}_ep${EPOCH}${SUFFIX}.npz")
done

echo
echo "=== Step 2: export .dat for Mathematica ==="
mkdir -p analysis/dat
"$HOME/envs/nenv2/bin/python" analysis/export_dm_mc_tau_vs_c_dat.py \
    --inputs "${NPZS[@]}" \
    --out_dir analysis/dat

echo
echo "=== Step 3: multi-kappa combined pdf ==="
"$HOME/envs/nenv2/bin/python" analysis/plot_dm_mc_tau_vs_c_combined.py \
    --inputs "${NPZS[@]}" \
    --out ../draft/figures/3Dphi4/dm_mc_tau_vs_c_3d_L${L}_multi_kappa_ep${EPOCH}${SUFFIX}.pdf

echo "Done."
echo "  per-kappa npz/pdf : analysis/results/"
echo "  per-kappa dat     : analysis/dat/"
echo "  combined pdf      : draft/figures/3Dphi4/"
