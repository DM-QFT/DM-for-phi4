#!/bin/bash
# Fan out the 3D phi^4 acceptance-vs-epoch sweep for kappa = 0.18, 0.1923, 0.2
# across nodes of an interactive PBS job (>=3 nodes), then combine the three
# npz into a multi-kappa figure.
#
# Run from inside an interactive job:
#   qsub -I -q interact-g -W group_list=gw97 -l select=4 -l walltime=00:30:00
#   cd /work/gw97/w97000/project/DM/3Dphi4
#   bash scripts/acceptance_vs_epoch_and_plot_3d.sh

set -e

if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: \$PBS_NODEFILE not set -- run inside a PBS job"
    exit 1
fi

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
PROJ_DIR="$(dirname "$SCRIPTS_DIR")"
cd "$PROJ_DIR"

N_NODES=$(uniq "$PBS_NODEFILE" | wc -l)
if (( N_NODES < 3 )); then
    echo "ERROR: need >=3 nodes (got $N_NODES) -- 3 kappa runs mapped 1:1 to nodes"
    exit 1
fi

declare -a KAPS=(0.18      0.1923    0.2)
declare -a SUFS=(sigma2000 sigma2048 sigma2760)

echo "=== Pre-flight check ==="
for i in 0 1 2; do
    K=${KAPS[$i]}
    S=${SUFS[$i]}
    RUN_DIR="runs/phi4_3d_L64_k${K}_l0.9_ncsnpp_${S}"
    if [[ ! -d "$RUN_DIR/models" ]]; then
        echo "  MISSING: $RUN_DIR/models"; exit 1
    fi
    N=$(ls "$RUN_DIR/models" | grep -cE '^epoch=[0-9]+\.ckpt$')
    echo "  $RUN_DIR  total ckpts: $N"
    if (( N == 0 )); then echo "  -> ERROR: no ckpts"; exit 1; fi
done

EMA=${EMA:-1}
EPOCH_WIN=${EPOCH_WIN:-0-1000000}
if [[ "$EMA" != "0" ]]; then
    EMA_SUF=_ema; FIG_SUF=_ema
else
    EMA_SUF=;     FIG_SUF=
fi

echo
echo "=== Step 1: 3 sweeps over $N_NODES nodes (ranks 0-2 active, rest idle) ==="
echo "EMA mode: $EMA  EPOCH_WIN: $EPOCH_WIN"
echo "(env-overridable: EMA=0 to turn off; EPOCH_WIN=lo-hi to restrict)"
mpirun --map-by ppr:1:node -hostfile "$PBS_NODEFILE" \
    bash "$SCRIPTS_DIR/acceptance_vs_epoch_worker_3d.sh" \
        "$EMA" "$EPOCH_WIN"

echo
echo "=== Step 2: combine per-kappa npz into multi-kappa figure ==="
"$HOME/envs/nenv2/bin/python" analysis/plot_acceptance_vs_epoch_combined.py \
    --inputs \
        analysis/acceptance_vs_epoch_3d_L64_k0.18_sigma2000${EMA_SUF}.npz \
        analysis/acceptance_vs_epoch_3d_L64_k0.1923_sigma2048${EMA_SUF}.npz \
        analysis/acceptance_vs_epoch_3d_L64_k0.2_sigma2760${EMA_SUF}.npz \
    --out ../paper/DM/figures/3Dphi4/acceptance_vs_epoch_3d_L64_multi_kappa${FIG_SUF}.pdf

echo "Done."
