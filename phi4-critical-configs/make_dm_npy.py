# 把与本数据集同耦合、同 L 训练的扩散模型 (DM) 系综整理到 DM/，轴序改成 (N, L, ..., L) 与 HMC 文件一致。
# 2D L=16..64 由 generate_dm.sh 生成，2D L=128 与 3D L=64 是文章用的已有样本。保留采样器输出的 float32。
import numpy as np
from pathlib import Path

here = Path(__file__).resolve().parent
r2 = here.parent / "2Dphi4" / "runs"
r3 = here.parent / "3Dphi4" / "runs"
files = [
    (r2 / "phi4_L16_k0.2705_l0.022_ncsnpp/data/samples_em_steps2000_epoch=epoch=9949.npy", 2, 16),
    (r2 / "phi4_L32_k0.2705_l0.022_ncsnpp/data/samples_em_steps2000_epoch=epoch=9999.npy", 2, 32),
    (r2 / "phi4_L64_k0.2705_l0.022_ncsnpp_sigma250/data/samples_em_steps2000_epoch=10000.npy", 2, 64),
    (r2 / "phi4_L128_k0.2705_l0.022_ncsnpp_sigma450/data/samples_em_steps2000_epoch=10000.npy", 2, 128),
    (r3 / "phi4_3d_L64_k0.1923_l0.9_ncsnpp_sigma2048/data/samples_em_steps2000_epoch=20000.npy", 3, 64),
]

for src, D, L in files:
    a = np.load(src)  # 采样脚本存成 (L, ..., L, N)
    phi = np.ascontiguousarray(np.moveaxis(a, -1, 0))
    coup = "kappa0.2705_lambda0.022" if D == 2 else "kappa0.1923_lambda0.9"
    f = here / "DM" / f"{D}D" / f"phi4_{D}D_L{L}_{coup}_DM.npy"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, phi)
    back = np.load(f, mmap_mode="r")
    assert back.dtype == np.float32 and back.shape[1:] == (L,) * D and np.array_equal(back, np.moveaxis(a, -1, 0))
    print(f"{f.relative_to(here)}  {phi.shape}  {f.stat().st_size / 1e6:.1f} MB  <- {src.relative_to(here.parent)}",
          flush=True)
    del a, phi, back
