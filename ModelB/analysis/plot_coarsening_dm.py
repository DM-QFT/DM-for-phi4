# PRL 正文 Fig.4 的 DM 快照、关联坍缩和两算法的畴尺寸比较,按 cjp_prl_nature.mplstyle,
# legend 与内嵌图字号同 Fig.3。另存 S7:S(k,t)/L² 坍缩(Porod k^-3)。
#   python analysis/plot_coarsening_dm.py
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "coarsening" / "prl"
OUTPUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT.parent / "plots"))
import modelb_palette  # noqa: E402,F401  注册样式表要的 modelb_field 色图
from modelb_palette import DM, GUIDE, REFERENCE, ZERO, time_colors

plt.style.use(ROOT.parent / "plots" / "cjp_prl_nature.mplstyle")

CHAIN = ROOT / (
    "runs/modelBcond_Lmulti8-16-32-64_k0.85_l1.0_t10_ncsnpp_quench_tanhgauss/"
    "samples/quench_chain_ep10000_L128_N32_t1000.h5"
)
TIMES = (100, 160, 250, 400, 630, 1000)
SNAP_TIMES = (10, 100, 1000)


def sign_stats(cfgs):
    L = cfgs.shape[-1]
    psi = np.sign(cfgs)
    S2 = np.abs(np.fft.fft2(psi, axes=(1, 2))) ** 2 / L**2
    C = np.fft.ifft2(S2.mean(0)).real
    C /= C[0, 0]
    d = np.minimum(np.arange(L), L - np.arange(L))
    r = np.round(np.hypot(*np.meshgrid(d, d))).astype(int)
    Cprof = np.bincount(r.ravel(), C.ravel()) / np.bincount(r.ravel())
    # 格点动量 |q| 分壳(壳宽 2π/L):相邻格点间的陡壁给 1/q² 而非 1/k²,
    # 按连续 k 画 Porod 尾在区边界附近上翘、各时刻散开
    k = 2 * np.pi * np.where(np.arange(L) > L // 2, np.arange(L) - L, np.arange(L)) / L
    kx, ky = np.meshgrid(k, k)
    q = np.hypot(2 * np.sin(kx / 2), 2 * np.sin(ky / 2)).ravel()
    m = np.round(q * L / (2 * np.pi)).astype(int)
    cnt = np.bincount(m)
    qm = np.bincount(m, q) / cnt
    Sq = np.stack([np.bincount(m, s.ravel()) for s in S2]) / cnt
    i = np.argmax(Cprof < 0)
    Lt = (i - 1) + Cprof[i - 1] / (Cprof[i - 1] - Cprof[i])
    return Cprof[: L // 2 + 1], qm, Sq.mean(0), Sq.std(0, ddof=1) / np.sqrt(len(Sq)), Lt


with h5py.File(CHAIN) as f:
    tt = f["times"][()]
    idx = [np.argmin(np.abs(tt - t)) for t in TIMES]
    Cs, qs, Ss, Es, Ls = zip(*(sign_stats(f["phi_gen"][i]) for i in idx))
    snaps = [f["phi_gen"][np.argmin(np.abs(tt - t)), 0] for t in SNAP_TIMES]
    t_all = tt
    L_all = np.array([sign_stats(f["phi_gen"][i])[-1] for i in range(len(tt))])
with h5py.File(ROOT / "results/coarsening/quench_L128_k0.85.jld2") as f:
    t_ref, L_ref = f["times"][()], f["Lt"][()]
L_ref_at = np.exp(np.interp(np.log(t_all), np.log(t_ref), np.log(L_ref)))

# 内嵌图里 100 个等间隔 t 的空心圆在对数轴上叠成一条带,只在对数等距的 16 个 t 上画记号
sub = np.unique([np.argmin(np.abs(t_all - t)) for t in np.geomspace(10, 1000, 16)])

# 几何以英寸手工定位(与 Fig.2 同法):主轴框 2.805 x 2.17 in,快照行与主轴左右对齐,色标在左缘
W = 3.375
X0, XR, Y_BOT, AXH, GAP, Y_TOP, G_SNAP = 0.54, 0.03, 0.41, 2.17, 0.11, 0.19, 0.03
AXW = W - X0 - XR
S = (AXW - 2 * G_SNAP) / 3
H = Y_BOT + AXH + GAP + S + Y_TOP
fig = plt.figure(figsize=(W, H), layout="none")


def inch_axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


Y_SNAP = Y_BOT + AXH + GAP
for c, (snap, t) in enumerate(zip(snaps, SNAP_TIMES)):
    axs = inch_axes(X0 + c * (S + G_SNAP), Y_SNAP, S, S)
    im = axs.imshow(snap.T, vmin=-1.7, vmax=1.7, origin="lower", interpolation="nearest",
                    rasterized=True)
    axs.set(xticks=[], yticks=[])
    axs.set_title(rf"$t={t}$", pad=1.0, fontsize=plt.rcParams["font.size"])
    for spine in axs.spines.values():
        spine.set_visible(False)
cax = inch_axes(X0 - 0.11, Y_SNAP, 0.05, S)
cb = fig.colorbar(im, cax=cax, extend="both", ticks=[-1.7, 0, 1.7], extendfrac=0.035)
cax.yaxis.set_ticks_position("left")
cax.set_yticklabels([r"$-1.7$", "$0$", r"$1.7$"])
# 色标字号 8/9 同 Fig.2(a)
cax.tick_params(direction="out", labelsize=8, length=2.0, width=0.6, pad=1.0)
cb.outline.set_linewidth(0.7)
cax.set_title(r"$\phi$", fontsize=9, pad=3)
fig.text(0.02 / W, 1 - 0.02 / H, "(a)", va="top")

ax = inch_axes(X0, Y_BOT, AXW, AXH)
colors = time_colors(len(TIMES))
r = np.arange(len(Cs[0]))
for Cprof, Lt, t, c in zip(Cs, Ls, TIMES, colors):
    ax.plot(r / Lt, Cprof, color=c, marker="", label=rf"$t={t}$")
# 参数块四行高出零线,零线从曲线束过零处附近画起
ax.axhline(0, xmin=0.26, color=ZERO, linestyle=":", zorder=0)
ax.set(xlabel=r"$r/L(t)$", ylabel=r"$C(r,t)$", xlim=(0, 3.2), ylim=(-0.30, 1.02))
# 六条 legend 排两列宽 1.2 in,曲线束与内嵌图之间只有 0.7 in,故单列;8 pt 同 Fig.3
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.215, 0.95))
# 最宽的一行必须压在凹陷(C≈-0.165)之下,其余三行短于曲线束到左轴的间隙
for k, text in enumerate((r"DM, trained at $L=8$-$64$", r"$\lambda=1$", r"$\kappa=0.85$", r"$L=128$")):
    ax.text(0.033, 0.015 + k * 0.18 / AXH, text, transform=ax.transAxes, va="bottom")
ax.text(0.035, 0.98, "(b)", transform=ax.transAxes, va="top")

# 内嵌图字号 7/8 同 Fig.3;三项 legend 放不下,t^{1/3} 像 Fig.3 的 e^{-x} 一样用文字标注
axin = ax.inset_axes([0.605, 0.585, 0.37, 0.38])
axin.loglog(t_ref, L_ref, ":", color=REFERENCE, marker="", label="SOSRA")
axin.loglog(t_all[sub], L_all[sub], "o", color=DM, label="DM")
tg = np.array([10, 1000.0])
axin.loglog(tg, 1.55 * tg ** (1 / 3), "--", color=GUIDE, marker="")
axin.text(450, 7.3, r"$t^{1/3}$", color=GUIDE, fontsize=7, ha="center", va="center")
axin.set(xlim=(9, 1100), ylim=(1.7, 60), yticks=[5, 10, 20, 50], xticklabels=[])
axin.set_ylabel(r"$L(t)$", fontsize=8, labelpad=1)
axin.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
axin.yaxis.set_minor_formatter(plt.NullFormatter())
axin.xaxis.set_minor_formatter(plt.NullFormatter())
axin.tick_params(labelsize=7, pad=1.5)
axin.legend(fontsize=7, loc="upper left")

axr = ax.inset_axes([0.605, 0.40, 0.37, 0.17])
axr.semilogx(t_all[sub], L_all[sub] / L_ref_at[sub], "o", color=DM)
axr.axhline(1, linestyle=":", color=ZERO, zorder=0)
axr.set(xlim=(9, 1100), ylim=(0.90, 1.10), yticks=[0.95, 1.05])
axr.xaxis.set_minor_formatter(plt.NullFormatter())
axr.set_xlabel(r"$t$", fontsize=8, labelpad=0.5)
axr.tick_params(labelsize=7, pad=1.5)
axr.text(0.97, 0.08, r"$L_{\mathrm{DM}}/L_{\mathrm{SOSRA}}$", transform=axr.transAxes,
         fontsize=7, ha="right", va="bottom")

fig.savefig(OUTPUT / "fig_dm_collapse_L128.pdf")
# draft 引用同一张图,重跑自动同步
fig.savefig(ROOT / "draft" / "figures" / "prl" / "collapse_dm_L128.pdf")

# --- S7:S(k,t)/L² 对 qL(t) 坍缩 + Porod ---
# 波数离散,每壳一个符号;六个时刻用样式表的定性色+符号循环,viridis 中段的几个绿分不开
fig2, ax2 = plt.subplots()
for qm, S, E, Lt, t in zip(qs, Ss, Es, Ls, TIMES):
    c = ax2.errorbar(qm[1:] * Lt, S[1:] / Lt**2, E[1:] / Lt**2, linestyle="none", label=rf"$t={t}$")
    c.lines[0].set_zorder(2)  # errorbar 把数据点抬到 2.1 会盖住 Porod 线,放回普通线的 2
# Porod 线画在数据上:斜率固定 -3,幅值由 6<k̂L<40 六个时刻合并的对数最小二乘给出
xs = np.concatenate([qm[1:] * Lt for qm, Lt in zip(qs, Ls)])
ys = np.concatenate([S[1:] / Lt**2 for S, Lt in zip(Ss, Ls)])
sel = (xs > 6) & (xs < 40)
A = np.exp(np.mean(np.log(ys[sel]) + 3 * np.log(xs[sel])))
x = np.array([4, 60.0])
ax2.plot(x, A * x**-3.0, color="k", linestyle="--", marker="")
ax2.text(17, A * 17**-3.0 * 3.5, r"$\hat{k}^{-3}$")
print(f"Porod amplitude A = {A:.3f}  (S/L^2 = A (k̂L)^-3, {sel.sum()} points)")
ax2.set(xscale="log", yscale="log", xlabel=r"$\hat{k}\,L(t)$", ylabel=r"$S(k,t)/L(t)^2$")
ax2.legend(loc="lower left")
ax2.text(0.63, 0.9, r"DM, $L=128$", transform=ax2.transAxes)
fig2.savefig(OUTPUT / "fig_dm_Sk_collapse_L128.pdf")
fig2.savefig(ROOT / "draft" / "figures" / "sk_collapse_dm_L128.pdf")
print("saved Fig.4 and S7 (+draft sync)")
