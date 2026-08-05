#!/bin/bash
# Serial production probes (one GPU job at a time). Run from weight_scaling/.
set -e
RUNS=../runs
DATA=../trainingdata
CKPT=$RUNS/phi4_Lmulti8-16-32-64_k0.2705_l0.022_ncsnpp/models/epoch=10000.ckpt

# main t-sweep at L=64: three estimators of R(k,t)
python score_propagator.py --ckpt $CKPT --L 64 --probe cov \
  --cfgs "$DATA/cfgs_wolff_fahmc_k=0.2705_l=0.022_64^2.jld2" --n_cfg 2048 \
  --out results/prop_k0.2705_L64_cov.tsv
python score_propagator.py --ckpt $CKPT --L 64 --probe ensemble \
  --cfgs "$DATA/cfgs_wolff_fahmc_k=0.2705_l=0.022_64^2.jld2" --n_cfg 32 \
  --out results/prop_k0.2705_L64_ens.tsv
python score_propagator.py --ckpt $CKPT --L 64 --probe zero \
  --out results/prop_k0.2705_L64_zero.tsv

# FSS sweep: L=128,256 extrapolate beyond the training sizes
for L in 8 16 32 128 256; do
  NC=2048; [ $L -ge 128 ] && NC=512
  python score_propagator.py --ckpt $CKPT --L $L --probe cov \
    --cfgs "$DATA/cfgs_wolff_fahmc_k=0.2705_l=0.022_${L}^2.jld2" --n_cfg $NC \
    --out results/prop_k0.2705_L${L}_cov.tsv
  python score_propagator.py --ckpt $CKPT --L $L --probe zero \
    --out results/prop_k0.2705_L${L}_zero.tsv
done

# exact empirical-mixture score baseline (network error vs non-Gaussianity split)
python exact_score_ref.py --cfgs "$DATA/cfgs_wolff_fahmc_k=0.2705_l=0.022_64^2.jld2" \
  --out results/exact_k0.2705_L64.tsv

# off-critical controls (each run has its own sigma / normalisation in hparams)
for K in 0.26 0.28; do
  python score_propagator.py \
    --ckpt $RUNS/phi4_Lmulti8-16-32-64_k${K}_l0.022_ncsnpp/models/epoch=10000.ckpt \
    --L 64 --probe cov --cfgs "$DATA/cfgs_wolff_fahmc_k=${K}_l=0.022_64^2.jld2" \
    --n_cfg 2048 --out results/prop_k${K}_L64_cov.tsv
done

# training-epoch sweep: how the learned propagator builds up its IR divergence
mkdir -p results/epoch_sweep
for C in $RUNS/phi4_Lmulti8-16-32-64_k0.2705_l0.022_ncsnpp/models/epoch=*.ckpt; do
  E=$(basename $C .ckpt); E=${E#epoch=}
  python score_propagator.py --ckpt $C --L 64 --probe cov \
    --cfgs "$DATA/cfgs_wolff_fahmc_k=0.2705_l=0.022_64^2.jld2" --n_cfg 512 \
    --n_t 9 --out results/epoch_sweep/prop_ep${E}_L64_cov.tsv
done
echo ALL DONE
