#!/bin/bash
# MSE-equivalent ESS (HMC-equivalent ESS) vs training epoch for 2D phi^4.
# CPU-only analysis on PRE-GENERATED DM samples.
# Sample generation is done separately and dropped into
# {run_dir}/data2/samples_em_steps{N}_epoch=*.npy as (L, L, K) arrays of
# K raw DM configurations.  The Python script also auto-accepts the
# legacy --z2 layout (L, L, 2K) with the second half being the
# elementwise negation of the first half; in that case the redundant
# flipped copies are dropped before analysis.  Observables are the
# Z2-even set {|m|, m^2, m^4, S/V}.
#
# Env overrides:
#   K=0.2705              kappa (default 0.2705)
#   SIGMA=450             sigma_VE suffix on run_dir
#   N_EPOCHS=20           cap on number of log-spaced ckpts; <=0 uses all
#   SDE_STEPS=2000        matches the sample-file naming (samples_em_steps2000_...)
#   N_BOOT=400            bootstrap replicas for N_eff error bars
#   SAMPLES_DIR=...       override the auto-built samples dir
#
# Usage:
#   bash scripts/ess_mse_vs_epoch_and_run.sh
#
# To sweep a different kappa:
#   K=0.28 SIGMA=640 bash scripts/ess_mse_vs_epoch_and_run.sh

set -e

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
PROJ_DIR="$(dirname "$SCRIPTS_DIR")"
cd "$PROJ_DIR"

K=${K:-0.2705}
SIGMA=${SIGMA:-450}
N_EPOCHS=${N_EPOCHS:-20}
SDE_STEPS=${SDE_STEPS:-2000}
N_BOOT=${N_BOOT:-400}
SAMPLES_DIR=${SAMPLES_DIR:-runs/phi4_L128_k${K}_l0.022_ncsnpp_sigma${SIGMA}/data2}
HMC="trainingdata/cfgs_wolff_fahmc_k=${K}_l=0.022_128^2.jld2"
OUT="analysis/results/ess_mse_vs_epoch_L128_k${K}_sigma${SIGMA}_ema.npz"

echo "=== Pre-flight check ==="
[[ -d "$SAMPLES_DIR" ]] || { echo "  MISSING: $SAMPLES_DIR"; exit 1; }
[[ -f "$HMC" ]] || { echo "  MISSING: $HMC"; exit 1; }
N_NPY=$(ls "$SAMPLES_DIR" | grep -cE "^samples_em_steps${SDE_STEPS}_epoch=[0-9]+\.npy$" || true)
echo "  $SAMPLES_DIR  npy files: $N_NPY"
echo "  $HMC  $(stat -c '%s' "$HMC" | awk '{printf "%.1f GB", $1/1e9}')"
if (( N_NPY == 0 )); then
    echo "  -> no samples matched pattern samples_em_steps${SDE_STEPS}_epoch=*.npy"; exit 1
fi

mkdir -p analysis/results

echo
echo "=== Settings ==="
echo "  K=$K  SIGMA=$SIGMA  SDE_STEPS=$SDE_STEPS"
echo "  N_EPOCHS=$N_EPOCHS  N_BOOT=$N_BOOT"
echo "  SAMPLES_DIR=$SAMPLES_DIR"
echo "  OUT=$OUT"
echo

echo "=== Run ==="
"$HOME/envs/nenv2/bin/python" analysis/ess_mse_vs_epoch.py \
    --L 128 --k ${K} --l 0.022 \
    --samples_dir ${SAMPLES_DIR} \
    --sde_steps ${SDE_STEPS} \
    --n_epochs ${N_EPOCHS} \
    --hmc_data ${HMC} \
    --n_boot ${N_BOOT} \
    --out_npz ${OUT}

echo
echo "=== Done ==="
echo "  npz : $OUT"
echo "  pdf : ${OUT%.npz}.pdf"
echo "  png : ${OUT%.npz}.png"
