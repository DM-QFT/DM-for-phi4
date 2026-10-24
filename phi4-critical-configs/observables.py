# 由 2D/、3D/ 和 DM/ 下的 npy 计算 README 的物理量表，定义与 arXiv:2607.08505 Sec. IV.A 一致。
# 误差为 64 块 jackknife，HMC 块长 160 (2D) 或 40 (3D)，远大于 tau_int。DM 组态彼此独立。
import numpy as np
from pathlib import Path

here = Path(__file__).resolve().parent
runs = [(2, 0.2705, 0.022, f"2D/phi4_2D_L{L}_kappa0.2705_lambda0.022.npy") for L in (16, 32, 64, 128)]
runs += [(3, 0.1923, 0.9, f"3D/phi4_3D_L{L}_kappa0.1923_lambda0.9.npy") for L in (16, 32, 64, 128)]
runs += [(2, 0.2705, 0.022, f"DM/2D/phi4_2D_L{L}_kappa0.2705_lambda0.022_DM.npy") for L in (16, 32, 64, 128)]
runs += [(3, 0.1923, 0.9, "DM/3D/phi4_3D_L64_kappa0.1923_lambda0.9_DM.npy")]
nb = 64
# 文章 App. A 的支撑过滤，阈值取自同 kappa 的 L=64 HMC 参考 (3Dphi4/trainingdata 的 2026-03 链)，
# 即场的全局最小、最大值和 |M| 的最大值。只用于 3D 的 DM 输出。
phi_lo, phi_hi, M_max = -2.257615579368943, 2.313765152525304, 0.22967894815605078


def tau_int(x):
    # Sokal 自动窗口，W 取第一个满足 W >= 5 tau 的滞后
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    C = np.fft.irfft(f * f.conj())[:n] / (n - np.arange(n))
    C /= C[0]
    tau = 0.5
    for W in range(1, n):
        tau += C[W]
        if W >= 5 * tau:
            return tau
    return tau


def obs(c, V):
    # c 的列为每个组态的 phi_x^2 的格点平均、作用量密度 s、磁化 M。
    # chi = V(<M^2> - <|M|>^2)，U_L = 1 - <M^4> / (3 <M^2>^2)
    phi2, s, m = c.T
    m2 = (m**2).mean()
    am = np.abs(m).mean()
    return np.array([am, V * (m2 - am**2), 1 - (m**4).mean() / (3 * m2**2), phi2.mean(), s.mean()])


def fmt(v, e):
    # 误差保留两位有效数字，写成 v(e)
    d = max(0, 1 - int(np.floor(np.log10(e))))
    if d > 0 and round(e * 10**d) >= 100:  # 0.00998 舍入成 100 时少保留一位
        d -= 1
    return f"{v:.{d}f}({round(e * 10**d):d})"


for D, kappa, lam, name in runs:
    phi = np.load(here / name, mmap_mode="r")
    n = len(phi)
    L = phi.shape[1]
    V = L**D
    sp = tuple(range(1, D + 1))
    c = np.empty((n, 3))
    keep = np.ones(n, dtype=bool)
    for i in range(0, n, 64):  # 分块读，3D L=128 整个数组 43 GB
        p = np.asarray(phi[i:i + 64], dtype=np.float64)
        hop = sum(np.roll(p, -1, ax) for ax in sp)
        c[i:i + 64, 0] = (p**2).mean(axis=sp)
        # 作用量 S = sum_x [-2 kappa phi nn + phi^2 + lambda (phi^2 - 1)^2]，按文章惯例略去常数 lambda V
        c[i:i + 64, 1] = (-2 * kappa * p * hop + (1 - 2 * lam) * p**2 + lam * p**4).mean(axis=sp)
        c[i:i + 64, 2] = p.mean(axis=sp)
        keep[i:i + 64] = (p.min(axis=sp) >= phi_lo) & (p.max(axis=sp) <= phi_hi)
    keep &= np.abs(c[:, 2]) <= M_max
    if not (D == 3 and name.startswith("DM/")):
        keep[:] = True
    c = c[keep]
    full = obs(c, V)
    blocks = np.array_split(c, nb)
    jk = np.array([obs(np.concatenate(blocks[:b] + blocks[b + 1:]), V) for b in range(nb)])
    err = np.sqrt((nb - 1) * jk.var(axis=0))
    print(f"{name} | N={len(c)} of {n} | " + " | ".join(fmt(v, e) for v, e in zip(full, err))
          + f" |    tau_int(M)={tau_int(c[:, 2]):.2f}  tau_int(|M|)={tau_int(np.abs(c[:, 2])):.2f}  "
          f"tau_int(S)={tau_int(c[:, 1]):.2f}", flush=True)
