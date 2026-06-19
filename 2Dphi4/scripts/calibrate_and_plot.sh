#!/bin/bash
# Fan out the MALA calibration sweeps for kappa=0.26, 0.2705, 0.28 across
# the nodes of an existing multi-node interactive PBS job (>=6 nodes),
# then combine the per-kappa npz into the multi-kappa figures used in
# FIG. 26 / FIG. 27 of the paper.
#
# Run from inside an interactive job:
#   qsub -I -q interact-g -W group_list=gw97 -l select=8 -l walltime=00:20:00
#   cd /work/gw97/w97000/project/DM/2Dphi4
#   bash scripts/calibrate_and_plot.sh

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
declare -a KAPS=(0.26     0.2705   0.28)
declare -a SUFS=(sigma300 sigma450 sigma640)

echo "=== Pre-flight check ==="
for i in 0 1 2; do
    K=${KAPS[$i]}
    S=${SUFS[$i]}
    RUN_DIR="runs/phi4_L128_k${K}_l0.022_ncsnpp_${S}"
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
# Pass config as positional args, NOT via -x: Miyabi's Open MPI has
# mca_base_env_list pre-set, which conflicts with -x.  Pattern matches
# sample_in_job.sh.
mpirun --map-by ppr:1:node -hostfile "$PBS_NODEFILE" \
    bash "$SCRIPTS_DIR/calibrate_mh_worker.sh" \
        "$EMA" "$EARLY" "$LATE"

echo
echo "=== Step 2: combine per-kappa npz into multi-kappa figures ==="
"$HOME/envs/nenv2/bin/python" acceptance_rate/plot_calibrate_combined.py --kind h \
    --inputs \
        acceptance_rate/calibrate_h_L128_k0.26_sigma300${EMA_SUF}.npz \
        acceptance_rate/calibrate_h_L128_k0.2705_sigma450${EMA_SUF}.npz \
        acceptance_rate/calibrate_h_L128_k0.28_sigma640${EMA_SUF}.npz \
    --out ../draft/figures/2Dphi4/calibrate_h_L128_multi_kappa${FIG_SUF}.pdf

"$HOME/envs/nenv2/bin/python" acceptance_rate/plot_calibrate_combined.py --kind t_mh \
    --inputs \
        acceptance_rate/calibrate_t_mh_L128_k0.26_sigma300${EMA_SUF}.npz \
        acceptance_rate/calibrate_t_mh_L128_k0.2705_sigma450${EMA_SUF}.npz \
        acceptance_rate/calibrate_t_mh_L128_k0.28_sigma640${EMA_SUF}.npz \
    --out ../draft/figures/2Dphi4/calibrate_t_mh_L128_multi_kappa${FIG_SUF}.pdf

echo "Done."
