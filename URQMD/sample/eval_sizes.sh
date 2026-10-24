#!/bin/bash
# Sample and compare a frozen checkpoint at several box sizes, then the loss-bin diagnostic at L=16 and 64.
# Usage: bash sample/eval_sizes.sh <run> <step> [K]
set -e
cd "$(dirname "$0")/.."
RUN=$1; STEP=$2; K=${3:-128}
for L in 8 16 32 48 64; do
    python sample/sample_particle.py $RUN $STEP $L test $K > logs/sample_${RUN}_L$L.log 2>&1
    python analysis/conditional/compare_statistics.py $RUN $STEP $L test $K > logs/compare_${RUN}_L$L.log 2>&1
    echo "== L=$L"; grep -v timestamp logs/compare_${RUN}_L$L.log
done
for L in 16 64; do
    python analysis/conditional/loss_bins.py $RUN $STEP $L test > logs/loss_bins_${RUN}_L$L.log 2>&1
    echo "== loss bins L=$L"; grep -v timestamp logs/loss_bins_${RUN}_L$L.log
done
