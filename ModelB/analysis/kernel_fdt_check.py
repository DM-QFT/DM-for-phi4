"""平稳态一步核诊断：逐模逐方向比较 DM 核与 SOSRA 的阻尼 c_k 与噪声 D_k。

    python analysis/kernel_fdt_check.py <ckpt> [num_steps]

φ(t) 取 SOSRA 参照链 t=500..950 每 50 一片共 10 片（1 万构型，GPU 约 2.5 min，
精度约 1%），SOSRA 的一步就是链里的 t+10 片，DM 从同一批 φ(t) 走一步。
平衡功率 G_k≈D_k/2c_k，软模 c_k≪1，所以 c_k 的绝对误差直接进入平衡值。
误差按时间片 jackknife。
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sample")]
import sample_modelB_chain as chain

REF = ROOT / "results/modelb_reference/corrgauss_hmc_L32_N1024.h5"
MODES = (("(1,0)", 1, 0), ("(0,1)", 0, 1), ("(1,1)", 1, 1), ("(1,-1)", 1, 31),
         ("(2,0)", 2, 0), ("(0,2)", 0, 2), ("(2,2)", 2, 2), ("(2,-2)", 2, 30),
         ("(3,3)", 3, 3), ("(4,4)", 4, 4))

ckpt = Path(sys.argv[1]).resolve()
num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

with h5py.File(REF, "r") as f:
    t = np.asarray(f["times"])
    idx = [i for i in range(len(t)) if 500 <= t[i] <= 950 and int(t[i]) % 50 == 0]
    phi0 = np.asarray(f["cfgs"][idx], dtype=np.float32).reshape(-1, 32, 32)
    phi_s = np.asarray(f["cfgs"][[i + 1 for i in idx]], dtype=np.float64).reshape(-1, 32, 32)
nslice = len(idx)

tmp = Path(tempfile.mkdtemp())
np.savez(tmp / "phi0.npz", phi_gen=phi0)
chain.TIMES, chain.SEEDS, chain.BATCH = [10.0], [20260911], 1024
chain.NUM_STEPS, chain.SCHEDULE = num_steps, "log"
chain.main(["--checkpoint", str(ckpt), "--input", str(tmp / "phi0.npz"),
            "--output", str(tmp / "dm.h5"), "--compile_mode", "reduce-overhead"])
with h5py.File(tmp / "dm.h5", "r") as f:
    phi_d = np.asarray(f["phi_gen"][0], dtype=np.float64)


def modes(x):
    x = x - x.mean(axis=(1, 2), keepdims=True)
    return np.fft.fft2(x) / 32   # |.|² 即 |φ_k|²/V


def sem(v):
    return v.reshape(nslice, -1).mean(1).std(ddof=1) / np.sqrt(nslice)


f0, fd, fs = modes(phi0.astype(np.float64)), modes(phi_d), modes(phi_s)
print(f"\nckpt={ckpt}\nnum_steps={num_steps}  configs={len(phi0)}  slices={nslice}")
print(" mode     G0     c_dm            c_sosra         D_dm/D_sosra    G_eq dm/sosra")
for name, i, j in MODES:
    a0, ad, as_ = f0[:, i, j], fd[:, i, j], fs[:, i, j]
    g0 = (np.abs(a0) ** 2).mean()
    cd = -np.real(np.conj(a0) * (ad - a0)) / g0
    cs = -np.real(np.conj(a0) * (as_ - a0)) / g0
    Dd, Ds = np.abs(ad - a0) ** 2, np.abs(as_ - a0) ** 2
    geq = (Dd.mean() / (2 * cd.mean() - cd.mean() ** 2)) / (Ds.mean() / (2 * cs.mean() - cs.mean() ** 2))
    print(f"{name:7s} {g0:6.3f}  {cd.mean():.4f}({sem(cd):.4f})  {cs.mean():.4f}({sem(cs):.4f})"
          f"  {Dd.mean() / Ds.mean():.3f}({sem(Dd / Ds.mean()):.3f})     {geq:.3f}")
