# 非守恒分数 (score net 守恒性诊断)

衡量训练好的 score net $s_\theta(x,t)$ 还残留多少**非守恒**成分。真 score $\nabla\log p_t$
是梯度场 (Jacobian $J_{ij}=\partial s_i/\partial x_j$ 对称); 网络直接回归向量场不保证对称。
反对称部分 $A=\tfrac12(J-J^\top)$ 度量非守恒, 指标:

$$\rho^2=\frac{\lVert J-J^\top\rVert_F^2}{4\lVert J\rVert_F^2}=\frac{\lVert A\rVert_F^2}{\lVert J\rVert_F^2}\in[0,1],$$

$\rho=0$ 完全守恒。用 Hutchinson + 自动微分估计 ($Jv$ 双反向, $J^\top v$ 反向), 不显式构造
$d\times d$ 的 $J$。评估点 = HMC 样本前向加噪到 $p_t$ ($x_t=x_0+\text{std}(t)z$), 非反向生成样本。

方法细节见 [`docs/score_nonconservative_check.md`](../../../docs/score_nonconservative_check.md)。

## 脚本

| 脚本 | 作用 |
|---|---|
| `curl_fraction.py` | 单个 checkpoint, 扫 $t$, 输出 $\rho(t)$。`--random_init` 用未训练随机权重给"随机天花板"参照 |
| `curl_fraction_vs_epoch.py` | 一个 run 的多个 log-scale checkpoint, 看 $\rho$ 随 epoch 演化 (配对 RNG) |
| `curl_fraction_baseline.py` | 用真实漂移 $-\partial S/\partial\phi$ 替代 score net, 给"守恒地板"参照 (应 $\approx0$) |

三者共用 `curl_fraction.py` 的核心函数 (`antisym_fraction` / `load_model` / `load_hmc`)。

## 运行 (从 `2Dphi4/` 目录, 用 nenv2 python)

```bash
cd /work/gw97/w97000/project/DM/2Dphi4
PY="$HOME/envs/nenv2/bin/python"
CKPT="runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640/models/epoch=10000.ckpt"
DATA="trainingdata/cfgs_wolff_fahmc_k=0.28_l=0.022_128^2.jld2"

# 1) 单 ckpt 扫 t
"$PY" analysis_clean/non_conservative/curl_fraction.py \
    --checkpoint "$CKPT" --data_path "$DATA" \
    --t_list 0.001,0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0 \
    --num_samples 512 --batch_size 512 --n_hutch 16 --device cuda:0

# 2) 随机天花板参照 (~0.707)
"$PY" analysis_clean/non_conservative/curl_fraction.py \
    --checkpoint "$CKPT" --data_path "$DATA" --random_init \
    --num_samples 512 --batch_size 512 --n_hutch 16 --device cuda:0

# 3) 守恒地板参照 (-dS/dphi, 应 ≈0)
"$PY" analysis_clean/non_conservative/curl_fraction_baseline.py \
    --checkpoint "$CKPT" --data_path "$DATA" \
    --num_samples 512 --batch_size 512 --n_hutch 16 --device cuda:0

# 4) rho 随 epoch
"$PY" analysis_clean/non_conservative/curl_fraction_vs_epoch.py \
    --run_dir runs/phi4_L128_k0.28_l0.022_ncsnpp_sigma640 --data_path "$DATA" \
    --t_list 0.01,0.1,1.0 --max_ckpts 25 \
    --num_samples 256 --batch_size 256 --n_hutch 8 --device cuda:0
```

输出 CSV/PNG 默认写到 `<run_dir>/data/curl_fraction*.{csv,png}`。
路径 (`runs/...`, `trainingdata/...`) 相对 `2Dphi4/`; 脚本本身位置无关 (`Path(__file__).parents[3]` 定位 repo 根)。

## 三参照 (L=128, k=0.28, sigma=640, epoch=10000)

| 参照 | $\rho$ ($t{\to}0$) | $\rho^2$ | 含义 |
|---|---|---|---|
| baseline $-\partial S/\partial\phi$ | $\sim4\times10^{-8}$ | $\sim10^{-15}$ | 守恒地板 (数值噪声) |
| **训练后** | **$\sim8\times10^{-3}$** | **$\sim7\times10^{-5}$** | 实际残留 |
| random init | $\sim0.70$ | $\sim0.50$ | 随机天花板 ($1/\sqrt2$) |

训练后远离随机天花板、贴近守恒地板 (高出地板 ~5 个量级但仍 $\rho\ll0.1$): score net 是
压倒性的梯度场, 非守恒可测量但很小。$\rho$ 随 epoch 单调下降并在小 $t$ 早早平台 (VE 损失权重
$\propto\text{std}(t)^2$ 对小 $t$ 降权, 数据流形上的真 score 训练最弱、残留最多)。
