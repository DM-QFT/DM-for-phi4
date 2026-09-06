# 粗化标度分析:L(t)、有效指数、C(r)/S(k) 坍缩、快照。
#   python analysis/plot_coarsening_references.py
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "coarsening"
plt.style.use(ROOT.parent / "plots" / "cjp.mplstyles")


def load(path):
    # JLD2 经 h5py:Julia 列主序,Sk/Cr 读出为 [t, ring m],snaps 为 [t, x, y]
    with h5py.File(path) as f:
        d = {k: f[k][()] for k in ("times", "Lt", "Mdrift", "Sk", "Cr", "snaps")}
        d["L"], d["kappa"] = int(f["L"][()]), float(f["kappa"][()])
    return d


def local_slope(t, L, w=5):
    # ln L 对 ln t 的滑动窗线性拟合斜率 = 1/z_eff
    lt, lL = np.log(t), np.log(L)
    n = len(t) - w + 1
    tc = np.empty(n)
    s = np.empty(n)
    for i in range(n):
        s[i] = np.polyfit(lt[i:i + w], lL[i:i + w], 1)[0]
        tc[i] = np.exp(lt[i:i + w].mean())
    return tc, s


KAPPAS = (0.38, 0.45, 0.55, 0.68, 0.85, 1.0)
KBEST = 0.85

# --- κ 扫描:L(t) 与有效指数(z_eff 面板只画 L<L_box/4 的干净点) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(KAPPAS)))
for k, c in zip(KAPPAS, colors):
    d = load(OUTPUT / f"quench_L128_k{k}.jld2")
    ax1.loglog(d["times"], d["Lt"], "o-", ms=2.5, color=c, label=rf"$\kappa={k}$")
    tc, s = local_slope(d["times"], d["Lt"])
    # L=64/128 对比给出的信任判据:L(t) < 0.2 L_box 内无可见有限尺寸偏差
    ok = np.interp(tc, d["times"], d["Lt"]) < 0.2 * d["L"]
    ax2.semilogx(tc[ok], s[ok], "o-", ms=2.5, color=c)
tg = np.array([200, 5000.0])
ax1.loglog(tg, 1.6 * tg ** (1 / 3), "k--", lw=0.8)
ax1.text(1200, 1.35 * 1200 ** (1 / 3), r"$t^{1/3}$")
ax1.set(xlabel=r"$t$", ylabel=r"$L(t)$")
ax1.legend(frameon=False, fontsize=7)
ax2.axhline(1 / 3, color="k", ls="--", lw=0.8)
ax2.axhline(1 / 4, color="gray", ls=":", lw=0.8)
ax2.text(2.6, 1 / 3 + 0.005, r"$1/3$")
ax2.text(2.6, 1 / 4 + 0.005, r"$1/4$", color="gray")
ax2.set(xlabel=r"$t$", ylabel=r"$\mathrm{d}\ln L/\mathrm{d}\ln t$", ylim=(0.15, 0.45))
fig.tight_layout()
fig.savefig(OUTPUT / "fig_kappa_scan.pdf")

# --- 有限尺寸:κ=0.85 下两个箱子 ---
fig, ax = plt.subplots(figsize=(3.375, 2.6))
for L in (64, 128):
    d = load(OUTPUT / f"quench_L{L}_k{KBEST}.jld2")
    ax.loglog(d["times"], d["Lt"], "o-", ms=2.5, label=rf"$L_{{\rm box}}={L}$")
    ax.axhline(L / 4, color="gray", lw=0.5, ls=":")
ax.loglog(tg, 1.6 * tg ** (1 / 3), "k--", lw=0.8)
ax.set(xlabel=r"$t$", ylabel=r"$L(t)$")
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT / "fig_finite_size.pdf")

# --- 标度坍缩:L=128 的标度窗内时刻 ---
d = load(OUTPUT / f"quench_L128_k{KBEST}.jld2")
Lbox = d["L"]
sel = np.where((d["Lt"] > 6) & (d["Lt"] < Lbox / 4))[0][::2]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
r = np.arange(d["Cr"].shape[1])
k = 2 * np.pi * np.arange(d["Sk"].shape[1]) / Lbox
for i in sel:
    Li = d["Lt"][i]
    ax1.plot(r / Li, d["Cr"][i], "-", lw=0.8, label=rf"$t={d['times'][i]:.0f}$")
    ax2.loglog(k[1:] * Li, d["Sk"][i, 1:] / Li**2, "-", lw=0.8)
ax1.axhline(0, color="k", lw=0.5)
ax1.set(xlabel=r"$r/L(t)$", ylabel=r"$C(r,t)$", xlim=(0, 4))
ax1.legend(frameon=False, fontsize=6)
x = np.array([4, 20.0])
ax2.loglog(x, 1.2 * x**-3.0, "k--", lw=0.8)
ax2.text(8, 0.008, r"$k^{-3}$")
ax2.set(xlabel=r"$k\,L(t)$", ylabel=r"$S(k,t)/L(t)^2$")
fig.tight_layout()
fig.savefig(OUTPUT / "fig_collapse.pdf")

# --- 快照 ---
snap_idx = [np.argmin(np.abs(d["times"] - t)) for t in (10, 100, 1000, d["times"][-1])]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95))
for ax, i in zip(axes, snap_idx):
    ax.imshow(d["snaps"][i].T, cmap="RdBu_r", vmin=-1.8, vmax=1.8,
              origin="lower", interpolation="nearest")
    ax.set(xticks=[], yticks=[], title=rf"$t={d['times'][i]:.0f}$")
fig.tight_layout()
fig.savefig(OUTPUT / "fig_snapshots.png", dpi=300)

# --- 弦长法 vs 第一零点:平均弦长 = 2L²/符号翻转键数(快照仅 1 条系综成员) ---
def chord_L(snaps):
    out = np.empty(len(snaps))
    for i, s in enumerate(np.sign(snaps)):
        flips = (s != np.roll(s, 1, 0)).sum() + (s != np.roll(s, 1, 1)).sum()
        out[i] = 2 * s.size / flips
    return out


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, kk in zip(axes, (0.85, 0.38)):
    d = load(OUTPUT / f"quench_L128_k{kk}.jld2")
    t, Lz = d["times"], d["Lt"]
    Lc = chord_L(d["snaps"])
    win = (t > 200) & (Lz < 0.2 * d["L"])
    ratio = (Lc[win] / Lz[win]).mean()
    ax.loglog(t, Lz, "o-", ms=2.5, label=r"$C(r)$ first zero")
    ax.loglog(t, Lc, "s-", ms=2.5, label="mean chord")
    ax.loglog(t, Lc / ratio, "--", color="gray", lw=0.9,
              label=rf"chord$/{ratio:.2f}$")
    ax.loglog(tg, 1.6 * tg ** (1 / 3), "k--", lw=0.8)
    ax.set(xlabel=r"$t$", ylabel=r"$L(t)$", title=rf"$\kappa={kk}$")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT / "fig_chord_vs_zero.pdf")

for kk in KAPPAS:
    d = load(OUTPUT / f"quench_L128_k{kk}.jld2")
    tc, s = local_slope(d["times"], d["Lt"])
    Lc = np.interp(tc, d["times"], d["Lt"])
    win = (tc > 200) & (Lc < 0.2 * d["L"])
    print(f"kappa={kk}: <dlnL/dlnt>_win={s[win].mean():.3f} +- {s[win].std():.3f}, "
          f"max|M|={d['Mdrift'].max():.2e}")
