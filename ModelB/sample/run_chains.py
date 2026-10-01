"""从 t=0 一口气生成每条 DM 链：dt=10 到 t=1000，dt=0.5 到 t=50。

不做续算、不拼接段落。已存在的目标目录会被跳过，删掉它就重跑。

    python sample/run_chains.py             # 全部
    python sample/run_chains.py L64         # 只跑名字含 L64 的
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sample_modelB_chain as chain

ROOT = Path(__file__).resolve().parents[1]
CKPT = {
    "mixed_t10":   ROOT / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/models/epoch=10000.ckpt",
    "mixed_t0p5":  ROOT / "runs/modelBcond_Lmulti8-16-32_k0.34034_l1.0_t0.5_ncsnpp_hmc_noskip/models/epoch=10000.ckpt",
    "hmc_t10":     ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/models/epoch=10000.ckpt",
    "hmc_t0p5":    ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_hmc_noskip/models/epoch=10000.ckpt",
    "corrg_t10":   ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip/models/epoch=10000.ckpt",
    "zlog_t10":    ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip/models/epoch=10000.ckpt",
    # zlog 从 EMA 权重续训 5000 epoch：D4 增广 + 余弦退火 1e-4→1e-5 + EMA 0.99999
    "d4ft_t10":    ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip_d4ft/models/epoch=5000.ckpt",
    "zlog_t0p5":   ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_corrgauss_zlog_noskip/models/epoch=10000.ckpt",
    # Z/m²/M 区间收窄到 HMC 有效范围的高斯训练集（见 run_l32_l1_k0p340340_corrgauss_zhmc_t0p5.jl）
    "zhmc_t0p5":   ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_corrgauss_zhmc_noskip/models/epoch=10000.ckpt",
    # HMC+非平衡轨迹并集的矩形区间、数据加倍 102400、5000 epoch（步数同 zlog）
    "zbox_t0p5":   ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_corrgauss_zbox_noskip/models/epoch=5000.ckpt",
    # 盒 A：Z[0.45,1.15] m²[0,0.8] M±0.8（param_clouds.py 测量），batch 256、1000 epoch 快速测试
    "boxa_t0p5":   ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_corrgauss_boxa_b256_noskip/models/epoch=1000.ckpt",
    "boxa_t10":    ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_boxa_b256_noskip/models/epoch=1000.ckpt",
    # 盒 A 102400 对、batch 128、10000 epoch；wide 版通道 [32,64,128,256]
    "boxa2_t10":      ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_boxa2_b128_noskip/models/epoch=10000.ckpt",
    "boxa2wide_t10":  ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_boxa2_b128_wide_noskip/models/epoch=10000.ckpt",
    # 1024000 对续训（从 boxa2 的 ep10000 出发，batch 256，2000 epoch = 800 万步）
    "boxa3_t10":      ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_boxa3_b256_noskip/models/epoch=2000.ckpt",
}
# 快速测试链：前 512 条轨道、反向 1000 步（其余链 1024 条、2000 步）
QUICK = {"boxa10_hmc_L32_t10", "boxa10_gaussianM_L32_t10"}
L64_BASE = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/samples/hmc_L64_N512_seed2026080201"
L128_INITIAL = ROOT / "results/multiL/hmc_L128_N512_t0.npz"
CORRG = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip/samples"
ZLOG = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip/samples"
D4FT = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip_d4ft/samples"
HMC05 = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_hmc_noskip/samples"
ZLOG05 = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_corrgauss_zlog_noskip/samples"
ZHMC = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_corrgauss_zhmc_noskip/samples"
ZBOX = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_corrgauss_zbox_noskip/samples"
BOXA = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t0.5_ncsnpp_corrgauss_boxa_b256_noskip/samples"
BOXA10 = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_boxa_b256_noskip/samples"
BOXA2 = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_boxa2_b128_noskip/samples"
BOXA2W = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_boxa2_b128_wide_noskip/samples"
BOXA3 = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_boxa3_b256_noskip/samples"

# (名字, checkpoint, t=0 初态, 输出目录, horizon, 目标时间, batch, 首个 seed)
CHAINS = (
    # L32 现存 h5 为 512(2026081100/2026082100)+1536(2026083100/2026084100,
    # 初态 seed 2026080701) 两批合并，细节在文件 attrs["merged_batches"]
    ("mixed_L32_t10",  "mixed_t10",  ROOT / "results/multiL/hmc_L32_N2048_t0.npz",
     ROOT / "results/multiL/t10_L32.h5",  10.0, 1000.0, 2048, 2026081100),
    ("mixed_L64_t10",  "mixed_t10",  L64_BASE / "hmc_ep10000_L64_t0_N512.npz",
     ROOT / "results/multiL/t10_L64.h5",  10.0, 1000.0,  512, 2026081200),
    ("mixed_L128_t10", "mixed_t10",  L128_INITIAL,
     ROOT / "results/multiL/t10_L128.h5", 10.0, 1000.0,  512, 2026081300),
    ("mixed_L32_t0p5", "mixed_t0p5", ROOT / "results/multiL/hmc_L32_N2048_t0.npz",
     ROOT / "results/multiL/t0p5_L32.h5",  0.5,   50.0, 2048, 2026082100),
    ("mixed_L64_t0p5", "mixed_t0p5", L64_BASE / "hmc_ep10000_L64_t0_N512.npz",
     ROOT / "results/multiL/t0p5_L64.h5",  0.5,   50.0,  512, 2026082200),
    ("mixed_L128_t0p5", "mixed_t0p5", L128_INITIAL,
     ROOT / "results/multiL/t0p5_L128.h5", 0.5,   50.0,  512, 2026082300),
    ("single_L64_t10", "hmc_t10",    L64_BASE / "hmc_ep10000_L64_t0_N512.npz",
     L64_BASE / "dm_chain.h5",                          10.0, 1000.0,  512, 2026080300),
    ("hmc_L32_t10",    "hmc_t10",    ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/samples/hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/samples/hmc_chain_ep10000_N1024_seed20260901/chain.h5",
                                                     10.0, 1000.0, 1024, 20260900),
    ("corrg_hmc_L32_t10",      "corrg_t10", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     CORRG / "hmc_chain_ep10000_N1024_seed20260901/chain.h5",       10.0, 1000.0, 1024, 20260900),
    ("corrg_gaussian_L32_t10", "corrg_t10", CORRG / "gaussian_chain_ep10000_N1024_seed20261001/gaussian_ep10000_t0_N1024_seed20260721.npz",
     CORRG / "gaussian_chain_ep10000_N1024_seed20261001/chain.h5",  10.0, 1000.0, 1024, 20261000),
    # zlog 链复用 corrgauss 链的 t=0 初态，modelb_reference 的参照直接可用
    ("zlog_hmc_L32_t10",      "zlog_t10", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     ZLOG / "hmc_chain_ep10000_N1024/chain.h5",       10.0, 2000.0, 1024, 2026091000),
    ("zlog_gaussian_L32_t10", "zlog_t10", CORRG / "gaussian_chain_ep10000_N1024_seed20261001/gaussian_ep10000_t0_N1024_seed20260721.npz",
     ZLOG / "gaussian_chain_ep10000_N1024/chain.h5",  10.0, 2000.0, 1024, 2026092000),
    # 同一组 Gaussian 场，逐轨道 M 换成配对 HMC 的 M，守恒荷分布与 Eq 一致
    ("zlog_gaussianM_L32_t10", "zlog_t10", CORRG / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     ZLOG / "gaussianM_chain_ep10000_N1024/chain.h5", 10.0, 2000.0, 1024, 2026093000),
    # Δt=0.5 单尺寸 HMC 核从 zlog 链的同一初态走 20 步到 t=10, 补 ML4PS Fig.2(c) 的 0<t<10
    ("hmc05_hmc_L32_t0p5",      "hmc_t0p5", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     HMC05 / "hmc_chain_ep10000_N1024/chain.h5",       0.5, 10.0, 1024, 2026096000),
    ("hmc05_gaussian_L32_t0p5", "hmc_t0p5", CORRG / "gaussian_chain_ep10000_N1024_seed20261001/gaussian_ep10000_t0_N1024_seed20260721.npz",
     HMC05 / "gaussian_chain_ep10000_N1024/chain.h5",  0.5, 10.0, 1024, 2026097000),
    ("d4ft_hmc_L32_t10",       "d4ft_t10", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     D4FT / "hmc_chain_ep5000_N1024/chain.h5",        10.0, 2000.0, 1024, 2026094000),
    ("d4ft_gaussianM_L32_t10", "d4ft_t10", CORRG / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     D4FT / "gaussianM_chain_ep5000_N1024/chain.h5",  10.0, 2000.0, 1024, 2026095000),
    # Δt=0.5 zlog 核从 PRL Fig.2 两条链的同一初态走 40 步到 t=20, 补 Fig.2(b) 的 0<t<20
    ("zlog05_hmc_L32_t0p5",       "zlog_t0p5", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     ZLOG05 / "hmc_chain_ep10000_N1024/chain.h5",       0.5, 20.0, 1024, 2026098000),
    ("zlog05_gaussianM_L32_t0p5", "zlog_t0p5", CORRG / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     ZLOG05 / "gaussianM_chain_ep10000_N1024/chain.h5", 0.5, 20.0, 1024, 2026099000),
    # 同一初态、同一步数，检验收窄训练区间是否消除 zlog05 的红外功率流失
    ("zhmc_hmc_L32_t0p5",       "zhmc_t0p5", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     ZHMC / "hmc_chain_ep10000_N1024/chain.h5",       0.5, 20.0, 1024, 2026098000),
    ("zhmc_gaussianM_L32_t0p5", "zhmc_t0p5", CORRG / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     ZHMC / "gaussianM_chain_ep10000_N1024/chain.h5", 0.5, 20.0, 1024, 2026099000),
    ("zbox_hmc_L32_t0p5",       "zbox_t0p5", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     ZBOX / "hmc_chain_ep5000_N1024/chain.h5",       0.5, 20.0, 1024, 2026098000),
    ("zbox_gaussianM_L32_t0p5", "zbox_t0p5", CORRG / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     ZBOX / "gaussianM_chain_ep5000_N1024/chain.h5", 0.5, 20.0, 1024, 2026099000),
    ("boxa_hmc_L32_t0p5",       "boxa_t0p5", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     BOXA / "hmc_chain_ep1000_N1024/chain.h5",       0.5, 20.0, 1024, 2026098000),
    ("boxa_gaussianM_L32_t0p5", "boxa_t0p5", CORRG / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     BOXA / "gaussianM_chain_ep1000_N1024/chain.h5", 0.5, 20.0, 1024, 2026099000),
    ("boxa10_hmc_L32_t10",       "boxa_t10", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     BOXA10 / "hmc_chain_ep1000_N512/chain.h5",       10.0, 1000.0, 512, 2026091000),
    ("boxa10_gaussianM_L32_t10", "boxa_t10", CORRG / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     BOXA10 / "gaussianM_chain_ep1000_N512/chain.h5", 10.0, 1000.0, 512, 2026093000),
    # 正式配方：1024 条轨道、反向 2000 步、到 t=2000，与 zlog 链同设定
    ("boxa2_hmc_L32_t10",           "boxa2_t10",     CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     BOXA2 / "hmc_chain_ep10000_N1024/chain.h5",        10.0, 2000.0, 1024, 2026101000),
    ("boxa2_gaussianM_L32_t10",     "boxa2_t10",     CORRG / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     BOXA2 / "gaussianM_chain_ep10000_N1024/chain.h5",  10.0, 2000.0, 1024, 2026102000),
    ("boxa2wide_hmc_L32_t10",       "boxa2wide_t10", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     BOXA2W / "hmc_chain_ep10000_N1024/chain.h5",       10.0, 2000.0, 1024, 2026103000),
    ("boxa2wide_gaussianM_L32_t10", "boxa2wide_t10", CORRG / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     BOXA2W / "gaussianM_chain_ep10000_N1024/chain.h5", 10.0, 2000.0, 1024, 2026104000),
    ("boxa3_hmc_L32_t10",       "boxa3_t10", CORRG / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     BOXA3 / "hmc_chain_ep2000_N1024/chain.h5",       10.0, 2000.0, 1024, 2026105000),
    ("boxa3_gaussianM_L32_t10", "boxa3_t10", CORRG / "gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz",
     BOXA3 / "gaussianM_chain_ep2000_N1024/chain.h5", 10.0, 2000.0, 1024, 2026106000),
    # ML4PS Fig.2 的 Noneq. 用不配对 M 的 iid 高斯初态
    ("boxa3_gaussian_L32_t10",  "boxa3_t10", CORRG / "gaussian_chain_ep10000_N1024_seed20261001/gaussian_ep10000_t0_N1024_seed20260721.npz",
     BOXA3 / "gaussian_chain_ep2000_N1024/chain.h5",  10.0, 2000.0, 1024, 2026107000),
)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    todo = [c for c in CHAINS if pattern in c[0]]

    missing = [(n, i) for n, _, i, *_ in todo if not i.exists()]
    if missing:
        print("缺 t=0 初态（HMC 的从 results/dynamic_exponent/canonical_l1/can_L32.jld2\n转置得到，Gaussian 的由 default_rng(20260721) 确定性生成）：")
        for n, i in missing:
            print(f"  {n}: {i.relative_to(ROOT)}")
        return

    for name, ck, initial, out, horizon, target, batch, seed0 in todo:
        n_steps = round(target / horizon)
        if out.exists():
            # 文件按 n_steps 预分配，未写入的槽位 times 仍为 0，
            # 所以有效步数要数 times>0，不能用 shape
            with h5py.File(out, "r") as handle:
                done = int((np.asarray(handle["times"]) > 0).sum())
            if done >= n_steps:
                print(f"跳过 {name}: 已有 {done} 步")
                continue
            print(f"{name}: 半成品 {done}/{n_steps} 步，删除后从 t=0 重跑", flush=True)
            out.unlink()
        times = [horizon * (k + 1) for k in range(n_steps)]
        print(f"\n=== {name}  t=0→{target:g}  {n_steps} 步  batch={batch} ===", flush=True)
        chain.TIMES = times
        chain.SEEDS = [seed0 + k + 1 for k in range(n_steps)]
        chain.BATCH = batch
        chain.NUM_STEPS = 1000 if name in QUICK else 2000
        chain.SCHEDULE = "log"
        chain.main([
            "--checkpoint", str(CKPT[ck]),
            "--input", str(initial),
            "--output", str(out),
            "--compile_mode", "reduce-overhead",
        ] + (["--n", str(batch)] if name in QUICK else []))


if __name__ == "__main__":
    main()
