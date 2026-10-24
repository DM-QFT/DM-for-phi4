# 把 Julia 生成的临界点组态从 JLD2 导出为 .npy，回读逐元素核对。物理量表由 observables.py 计算。
# 2D 来自 generate_2D.jl 写在 jld2/ 的文件，3D 来自 3Dphi4/trainingdata/。
import h5py
import numpy as np
from pathlib import Path

here = Path(__file__).resolve().parent
runs = [(2, 0.2705, 0.022, L, here / "jld2" / f"cfgs_wolff_fahmc_k=0.2705_l=0.022_{L}^2.jld2")
        for L in (16, 32, 64, 128)]
runs += [(3, 0.1923, 0.9, L, here.parent / "3Dphi4" / "trainingdata"
          / f"cfgs_wolff_fahmc_k=0.1923_l=0.9_{L}^3{'_thin10' if L == 64 else ''}.jld2")
         for L in (16, 32, 64, 128)]

for D, kappa, lam, L, src in runs:
    with h5py.File(src, "r") as h:
        phi = h["cfgs"][()]
        n = h["n_samples"][()]
    out = here / f"{D}D"
    out.mkdir(exist_ok=True)
    f = out / f"phi4_{D}D_L{L}_kappa{kappa}_lambda{lam}.npy"
    np.save(f, phi)
    back = np.load(f, mmap_mode="r")
    assert back.shape == (n, *(L,) * D) and np.array_equal(back, phi)
    print(f"{f.name}  {phi.shape}  {f.stat().st_size / 1e6:.1f} MB", flush=True)
    del phi, back
