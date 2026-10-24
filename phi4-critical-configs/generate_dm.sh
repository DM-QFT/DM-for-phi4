#!/bin/bash
# 生成 HF 数据集中 2D L=16, 32, 64 的 DM 系综，单 GPU 顺序执行，一整批采样不分 batch，log 时间网格。
# compile 用 default，reduce-overhead 会破坏 NCSN++ 上采样的 nearest 插值。
# 2D L=128 与 3D L=64 沿用已有样本，见 make_dm_npy.py。
set -e
cd /work/gw97/w97000/project/DM/2Dphi4
for spec in "16 epoch=9949 " "32 epoch=9999 " "64 10000 --sigma 250"; do
    set -- $spec
    python sample_phi4_sweep.py --L $1 --k 0.2705 --l 0.022 --epochs $2 ${@:3} \
        --num_samples 2048 --n_repeats 1 --sde_steps 2000 --schedule log --compile_mode default --device cuda:0
done
echo "QUEUE DONE"
