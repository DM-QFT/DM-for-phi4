# 逐构型 MLE 拟合 |φ_q|²/V ~ Exp(1/(Z q̂²+m²))，得到 HMC 平衡态与 gaussianM 非平衡 SOSRA 轨迹
# 各时刻的 (Z, m², M) 云，存 npz 并画分布图，供重新规划高斯训练集。估计噪声用 zhmc 集（真值已知）校准。
#   python analysis/param_clouds.py
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "plots"))
from modelb_palette import ALT, DM, REFERENCE  # noqa: E402

plt.style.use(ROOT.parent / "plots" / "cjp_prl_nature.mplstyle")
L = 32
HMC = ROOT / "trainingdata/modelB_L32_Ne51200_hmc_l1_kappa_0.340340_t0_seed2026080101.jld2"
ZHMC = ROOT / "trainingdata/modelB_L32_Ne51200_corrgausszhmc_l1_kappa_0.340340_t0_seed2026091701.jld2"
G0 = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip/samples/gaussianM_chain_ep10000_N1024/gaussianM_ep10000_t0_N1024.npz"
FINE = ROOT / "results/modelb_reference/corrgauss_gaussianM_L32_N1024_t0p5.h5"
COARSE = ROOT / "results/modelb_reference/corrgauss_gaussianM_L32_N1024.h5"
OUT = ROOT / "results/train_dist"
OUT.mkdir(exist_ok=True)
dev = torch.device("cuda")

q = 4 * np.sin(np.pi * np.arange(L) / L) ** 2
q2 = (q[:, None] + q[None, :]).ravel()
mask = np.ones(L * L, bool); mask[0] = False
cls, inv = np.unique(np.round(q2[mask], 10), return_inverse=True)
n_j = np.bincount(inv).astype(np.float64)
# 线性网格含 Z=0（iid 高斯是平谱），步长 0.005
Zg, m2g = np.arange(0, 1.6001, 0.005), np.arange(0, 1.6001, 0.005)
GZ, GM = np.meshgrid(Zg, m2g, indexing="ij")
g = GZ.ravel()[:, None] * cls[None, :] + GM.ravel()[:, None]
g[g <= 0] = np.nan
const = -(n_j[None, :] * np.log(g)).sum(1)
g_t = torch.tensor(np.nan_to_num(g.T, nan=1.0), dtype=torch.float32, device=dev)
const_t = torch.tensor(np.nan_to_num(const, nan=np.inf), dtype=torch.float32, device=dev)
mask_t, inv_t = torch.tensor(mask, device=dev), torch.tensor(inv, device=dev)


def fit(cfgs):
    """返回 (Z, m², M, 去均值 std, 去均值峭度)，每个形状 (N,)。"""
    out = []
    for s in range(0, len(cfgs), 4096):
        x = torch.tensor(np.asarray(cfgs[s:s + 4096]), dtype=torch.float32, device=dev)
        P = (torch.fft.fft2(x).abs() ** 2 / L**2).reshape(len(x), -1)[:, mask_t]
        S = torch.zeros(len(x), len(cls), device=dev).index_add_(1, inv_t, P)
        i = torch.cat([(const_t[None] + S[a:a + 1024] @ g_t).argmin(1) for a in range(0, len(S), 1024)])
        xd = x.double(); M = xd.mean((1, 2)); d = xd - M[:, None, None]
        v = (d**2).mean((1, 2))
        out.append(np.stack([GZ.ravel()[i.cpu().numpy()], GM.ravel()[i.cpu().numpy()], M.cpu().numpy(),
                             v.sqrt().cpu().numpy(), ((d**4).mean((1, 2)) / v**2).cpu().numpy()]))
    return np.concatenate(out, 1)


clouds = {}
with h5py.File(HMC) as h:
    clouds["hmc"] = fit(h["cfgs"])
with h5py.File(ZHMC) as h:
    est = fit(h["cfgs"][:16384])
    Zt, m2t = h["propagator_Z"][:16384].astype(np.float64), h["propagator_m2"][:16384].astype(np.float64)
clouds["noneq_t0"] = fit(np.load(G0)["phi_gen"])
times = [0.0]
with h5py.File(FINE) as h:
    for i, t in enumerate(h["times"][:]):
        clouds[f"noneq_t{t:g}"] = fit(h["cfgs"][i]); times.append(float(t))
with h5py.File(COARSE) as h:
    for i, t in enumerate(h["times"][:]):
        if t > 20:
            clouds[f"noneq_t{t:g}"] = fit(h["cfgs"][i]); times.append(float(t))
np.savez_compressed(OUT / "clouds.npz", times=np.array(times), fields=np.array(["Z", "m2", "M", "std", "kurt"]),
                    **clouds)

# 估计噪声（zhmc 真值）
dz, dm = np.log10(est[0] / Zt), est[1] - m2t
p = lambda v: "%.3f / %.3f / %.3f" % tuple(np.percentile(v, [2.5, 50, 97.5]))
print(f"估计噪声 (zhmc 真值, n=16384)  log10(Ẑ/Z) 2.5/50/97.5%: {p(dz)}   m̂²−m²: {p(dm)}")
print("  m̂² 相关系数 corr(m̂²−m², Ẑ/Z−1) = %.2f（拟合沿 Z q̂²+m² 的等值线有权衡）" % np.corrcoef(dm, est[0] / Zt - 1)[0, 1])

# 汇总表
hm = clouds["hmc"]
print("\nHMC 51200:  Z %s | m² %s | M %s | std %s | kurt %s" % tuple(p(v) for v in hm))
print("  corr(Z, m²) = %.2f, corr(|M|, Z) = %.2f, corr(|M|, m²) = %.2f" % (
    np.corrcoef(hm[0], hm[1])[0, 1], np.corrcoef(np.abs(hm[2]), hm[0])[0, 1], np.corrcoef(np.abs(hm[2]), hm[1])[0, 1]))
print("  按 |M| 分箱的中位 Z / m² / std:")
for lo, hi in ((0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.9)):
    s = (np.abs(hm[2]) >= lo) & (np.abs(hm[2]) < hi)
    print(f"    |M|∈[{lo:.2f},{hi:.2f}) n={s.sum():5d}  Z {np.median(hm[0][s]):.3f}  m² {np.median(hm[1][s]):.3f}  std {np.median(hm[3][s]):.3f}")
print("\n非平衡轨迹 (1024 条)，各时刻 2.5/50/97.5 百分位:")
for t in (0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000):
    c = clouds[f"noneq_t{t:g}"]
    print(f"  t={t:6g}  Z {p(c[0])}   m² {p(c[1])}   std {p(c[3])}")
pool_early = np.concatenate([clouds[f"noneq_t{t:g}"] for t in times if t < 20], 1)
pool_late = np.concatenate([clouds[f"noneq_t{t:g}"] for t in times if 20 <= t <= 200], 1)
print(f"\n非平衡 0≤t<20 汇总 (n={pool_early.shape[1]}): Z {p(pool_early[0])} | m² {p(pool_early[1])}")
print(f"非平衡 20≤t≤200 汇总 (n={pool_late.shape[1]}): Z {p(pool_late[0])} | m² {p(pool_late[1])}")

# 图：(a)(b)(c) (Z,m²) 二维直方图，(d) 中位数随 t，(e) Z 边缘，(f) m² 边缘
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), layout="constrained")
bins = (np.arange(0, 1.4, 0.02), np.arange(0, 1.4, 0.02))
for ax, (c, title) in zip(axes[0], ((hm, "HMC (Eq.)"), (pool_early, r"Noneq. $0\leq t<20$"), (pool_late, r"Noneq. $20\leq t\leq200$"))):
    ax.hist2d(c[0], c[1], bins=bins, cmap="Greys", density=True, rasterized=True)
    ax.set(xlabel=r"$Z$", ylabel=r"$m^2$", title=title)
    ax.axvspan(0.45, 1.3, ymin=0, ymax=0.3 / 1.4, color=ALT, alpha=0.15, lw=0)  # zhmc 盒
ax = axes[1, 0]
ts = np.array([t for t in times if t > 0])
for k, (lab, color) in enumerate(((r"$Z$", DM), (r"$m^2$", ALT))):
    med = np.array([np.median(clouds[f"noneq_t{t:g}"][k]) for t in ts])
    lo, hi = (np.array([np.percentile(clouds[f"noneq_t{t:g}"][k], pp) for t in ts]) for pp in (16, 84))
    ax.fill_between(ts, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(ts, med, color=color, marker="", label=lab)
    ax.axhline(np.median(hm[k]), color=color, ls="--", lw=0.8)
ax.set(xscale="log", xlim=(0.4, 2500), xlabel=r"$t$", ylabel="Noneq. median (16-84%)")
ax.legend(loc="center right")
for ax, k, lab, b in ((axes[1, 1], 0, r"$Z$", np.arange(0, 1.4, 0.02)), (axes[1, 2], 1, r"$m^2$", np.arange(0, 1.4, 0.02))):
    for c, color, name in ((hm, REFERENCE, "HMC"), (pool_early, DM, r"Noneq. $t<20$"), (pool_late, ALT, r"Noneq. $20$-$200$")):
        ax.hist(c[k], bins=b, density=True, histtype="step", color=color, label=name)
    ax.set(xlabel=lab, ylabel="density")
axes[1, 1].legend(loc="upper right", fontsize=7)
fig.savefig(OUT / "clouds.pdf")
fig.savefig(OUT / "clouds.png", dpi=250)
print(f"saved {OUT}/clouds.npz, clouds.pdf")
