#!/bin/bash
# Fan out the MALA calibration sweeps for 3D phi^4 at L=64,
# kappa=0.18, 0.1923, 0.2 across the nodes of an existing multi-node
# interactive PBS job (>=6 nodes), then combine the per-kappa npz into
# multi-kappa figures for the paper.
#
# Run from inside an interactive job:
#   qsub -I -q interact-g -W group_list=gw97 -l select=8 -l walltime=00:30:00
#   cd /work/gw97/w97000/project/DM/3Dphi4
#   bash scripts/calibrate_and_plot_3d.sh

set -e

if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: \$PBS_NODEFILE not set -- run inside a PBS job"
    exit 1
fi

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
PROJ_DIR="$(dirname "$SCRIPTS_DIR")"
cd "$PROJ_DIR"

N_NODES=$(uniq "$PBS_NODEFILE" | wc -l)
if (( N_NODES < 6 )); then
    echo "ERROR: need >=6 nodes (got $N_NODES) -- 6 sweeps mapped 1:1 to nodes"
    exit 1
fi

# Pre-validate: every (kappa, suffix) run must have ckpts in BOTH windows.
EARLY_LO=0;    EARLY_HI=50
LATE_LO=6000;  LATE_HI=10000
declare -a KAPS=(0.18      0.1923    0.2)
declare -a SUFS=(sigma2000 sigma2048 sigma2760)

echo "=== Pre-flight check ==="
for i in 0 1 2; do
    K=${KAPS[$i]}
    S=${SUFS[$i]}
    RUN_DIR="runs/phi4_3d_L64_k${K}_l0.9_ncsnpp_${S}"
    if [[ ! -d "$RUN_DIR/models" ]]; then
        echo "  MISSING: $RUN_DIR/models"
        exit 1
    fi
    N_EARLY=$(ls "$RUN_DIR/models" | grep -oE 'epoch=[0-9]+' | awk -F= '{print $2}' \
                | awk -v lo=$EARLY_LO -v hi=$EARLY_HI '$1>=lo && $1<=hi' | wc -l)
    N_LATE=$(ls "$RUN_DIR/models" | grep -oE 'epoch=[0-9]+' | awk -F= '{print $2}' \
                | awk -v lo=$LATE_LO -v hi=$LATE_HI '$1>=lo && $1<=hi' | wc -l)
    echo "  $RUN_DIR  early[${EARLY_LO},${EARLY_HI}]=$N_EARLY  late[${LATE_LO},${LATE_HI}]=$N_LATE"
    if (( N_EARLY == 0 || N_LATE == 0 )); then
        echo "  -> ERROR: empty window"
        exit 1
    fi
done

EMA=${EMA:-1}
EARLY=${EARLY:-0-50}
LATE=${LATE:-6000-10000}
if [[ "$EMA" != "0" ]]; then
    EMA_SUF=_ema
    FIG_SUF=_ema
else
    EMA_SUF=
    FIG_SUF=
fi

echo
echo "=== Step 1: 6 sweeps over $N_NODES nodes (ranks 0-5 active, rest idle) ==="
echo "EMA mode: $EMA  EARLY: $EARLY  LATE: $LATE"
echo "(env-overridable: EMA=0 to turn off; EARLY=... LATE=... to override windows)"
# Pass config as positional args (Miyabi Open MPI pre-sets mca_base_env_list,
# so mpirun -x conflicts and aborts).
mpirun --map-by ppr:1:node -hostfile "$PBS_NODEFILE" \
    bash "$SCRIPTS_DIR/calibrate_mh_worker_3d.sh" \
        "$EMA" "$EARLY" "$LATE"

echo
echo "=== Step 2: combine per-kappa npz into multi-kappa figures ==="
"$HOME/envs/nenv2/bin/python" analysis/plot_calibrate_combined.py --kind h \
    --inputs \
        analysis/calibrate_h_3d_L64_k0.18_sigma2000${EMA_SUF}.npz \
        analysis/calibrate_h_3d_L64_k0.1923_sigma2048${EMA_SUF}.npz \
        analysis/calibrate_h_3d_L64_k0.2_sigma2760${EMA_SUF}.npz \
    --out ../paper/DM/figures/3Dphi4/calibrate_h_3d_L64_multi_kappa${FIG_SUF}.pdf

"$HOME/envs/nenv2/bin/python" analysis/plot_calibrate_combined.py --kind t_mh \
    --inputs \
        analysis/calibrate_t_mh_3d_L64_k0.18_sigma2000${EMA_SUF}.npz \
        analysis/calibrate_t_mh_3d_L64_k0.1923_sigma2048${EMA_SUF}.npz \
        analysis/calibrate_t_mh_3d_L64_k0.2_sigma2760${EMA_SUF}.npz \
    --out ../paper/DM/figures/3Dphi4/calibrate_t_mh_3d_L64_multi_kappa${FIG_SUF}.pdf

echo "Done."
