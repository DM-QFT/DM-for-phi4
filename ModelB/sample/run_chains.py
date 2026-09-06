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
    "corrg_t10":   ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip/models/epoch=10000.ckpt",
    "zlog_t10":    ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip/models/epoch=10000.ckpt",
}
L64_BASE = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip/samples/hmc_L64_N512_seed2026080201"
L128_INITIAL = ROOT / "results/multiL/hmc_L128_N512_t0.npz"
CORRG = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip/samples"
ZLOG = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip/samples"

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
        chain.NUM_STEPS = 2000
        chain.SCHEDULE = "log"
        chain.main([
            "--checkpoint", str(CKPT[ck]),
            "--input", str(initial),
            "--output", str(out),
            "--compile_mode", "reduce-overhead",
        ])


if __name__ == "__main__":
    main()
