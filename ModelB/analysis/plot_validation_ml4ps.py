# ML4PS Fig.2: 几何与 PRL validation_L32 逐英寸相同。(b) 最软对角模等时功率 G_1(t), (c) 前三个对角模与初态的
# 关联 ρ_n(t,0), 两个系综同图, 对数时间轴 10~1000。
# Noneq. 用普通 iid 高斯初态（不与 HMC 配对 M）, 与 PRL 图的 gaussianM 不同, 所以全部直接从
# chain.h5 / 参照 h5 算, 不读 chain_data.py 的 npz 归档。PRL 原图和 plot_validation_L32.py 不动。
#   python analysis/plot_validation_ml4ps.py
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT.parent / "plots"
sys.path.insert(0, str(PLOTS))
from modelb_palette import ALT, DM, FIELD_CMAP  # noqa: E402

plt.style.use(PLOTS / "cjp_prl_nature.mplstyle")
FS, FS_BC = 10.5, 9
plt.rcParams.update({"font.size": FS_BC, "axes.labelsize": FS_BC, "xtick.labelsize": FS_BC,
                     "ytick.labelsize": FS_BC, "legend.fontsize": FS_BC})

RUN = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip/samples"
T0 = ROOT / "runs/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_uniformM_noskip/samples"
REF = ROOT / "results/modelb_reference"
PAPER = ROOT / "draft/figures/prl/validation_L32_ml4ps.pdf"
# 初态名, 图例, 颜色, marker, t=0 初态, DM 链, SOSRA 参照
ENSEMBLES = (
    ("hmc", "Eq.", ALT, "o", T0 / "hmc_chain_ep10000_N1024_seed20260901/hmc_ep10000_t0_N1024.npz",
     RUN / "hmc_chain_ep10000_N1024/chain.h5", REF / "corrgauss_hmc_L32_N1024.h5"),
    ("gaussian", "Noneq.", DM, "s",
     T0 / "gaussian_chain_ep10000_N1024_seed20261001/gaussian_ep10000_t0_N1024_seed20260721.npz",
     RUN / "gaussian_chain_ep10000_N1024/chain.h5", REF / "corrgauss_gaussian_L32_N1024.h5"),
)
L = 32
N_MODES = 4  # 对角模 k_n=(2πn/L)(1,1), n=1..4, 与正文动力学标度用的一致
SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
MODE_IDX = [[((n * sy) % L, (n * sx) % L) for sy, sx in SIGNS] for n in range(1, N_MODES + 1)]
T_G = np.arange(0, 2001, 10)  # 链到 t=2000, 200 次迭代
N_RHO = 101  # ρ_n(t,0) 画到 t=1000
SNAP_T = (0, 10, 50, 100, 500, 1000)


def diagonal_modes(x):
    """每构型 φ(k_n) 的 4 个符号模, 形状 (traj, n, sign)；fft/L 使 |·|²=|φ(k)|²/V, 与 chain_data 的 G 同归一。"""
    F = np.fft.fft2(x) / L
    return np.stack([np.stack([F[:, i, j] for i, j in row], 1) for row in MODE_IDX], 1)


def load_series(t0_path, chain_path, ref_path):
    """每条链: 轨道 0 快照 (SNAP_T), G_1 均值/标准误 (T_G), 对角模幅值 (T_G, traj, n, sign)。t=0 两链共用初态。"""
    x0 = np.load(t0_path)["phi_gen"].astype(np.float64)
    out = {}
    for key, path, name in (("dm", chain_path, "phi_gen"), ("modelb", ref_path, "cfgs")):
        snaps, modes = [], []
        with h5py.File(path, "r") as f:
            times = np.asarray(f["times"])
            for tt in T_G:
                if tt == 0:
                    x = x0
                else:
                    i = int(np.argmin(np.abs(times - tt)))
                    assert times[i] == tt
                    x = np.asarray(f[name][i], dtype=np.float64)
                if tt in SNAP_T:
                    snaps.append(x[0].copy())
                modes.append(diagonal_modes(x).astype(np.complex64))
        modes = np.stack(modes)
        p = (modes[:, :, 0].real ** 2 + modes[:, :, 0].imag ** 2).mean(-1).astype(np.float64)  # |φ_1|², 符号模平均
        g1 = np.stack([p.mean(1), p.std(1, ddof=1) / np.sqrt(p.shape[1])], 1)
        out[key] = (np.stack(snaps), g1, modes)
    return out


def rho0(modes):
    """与初态的关联 ρ_n(t,0)=<Re φ_n(t)φ_n(0)*>/sqrt(<|φ_n(t)|²><|φ_n(0)|²>), 非平稳系综也有定义;
    删一轨道 jackknife 误差, 形状 (t, n, 2)。"""
    ntraj = modes.shape[1]
    cross = (modes * modes[:1].conj()).real.mean(-1).astype(np.float64)  # (t, traj, n), 符号模平均
    power = (modes.real ** 2 + modes.imag ** 2).mean(-1).astype(np.float64)
    loo = lambda v: (v.sum(1, keepdims=True) - v) / (ntraj - 1)
    full = cross.mean(1) / np.sqrt(power.mean(1) * power[:1].mean(1))
    jk = loo(cross) / np.sqrt(loo(power) * loo(power[:1]))
    err = np.sqrt((ntraj - 1) / ntraj * ((jk - full[:, None]) ** 2).sum(1))
    return np.stack([full, err], -1)


def panel_label(axis, label):
    axis.text(0.02, 0.98, label, transform=axis.transAxes, ha="left", va="top", fontsize=FS,
              fontweight="bold", zorder=10)


def truth_and_points(axis, t, mb, dm, color, marker, sel):
    """SOSRA 线+带, DM 空心点带误差棒; mb/dm 形状 (nt, 2) = (均值, 误差)。"""
    axis.fill_between(t, mb[:, 0] - mb[:, 1], mb[:, 0] + mb[:, 1], color=color, alpha=0.18, linewidth=0)
    axis.plot(t, mb[:, 0], color=color, linestyle="-", marker="")
    axis.errorbar(t[sel], dm[sel, 0], yerr=dm[sel, 1], color=color, marker=marker, fillstyle="full",
                  markerfacecolor="white", markeredgecolor=color, linestyle="none", markersize=3.5,
                  markeredgewidth=0.9, elinewidth=0.6, capsize=0, zorder=3)


series = {name: load_series(t0_path, chain_path, ref_path)
          for name, _, _, _, t0_path, chain_path, ref_path in ENSEMBLES}

# 几何与 plot_validation_L32.py 逐英寸相同
W, H = 7.0, 2.07
fig = plt.figure(figsize=(W, H), layout="none")
C, G, X0, Y_TITLE, G_PAIR = 0.34, 0.035, 0.22, 0.15, 0.08
Y_BOT, Y_TOP, YLAB = 0.42, 0.15, 0.55


def inch_axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def cell_axes(row, column):
    r = row if row < 2 else row - 1
    y = H - Y_TITLE - (r + 1) * C - r * G - (G_PAIR if r >= 2 else 0)
    return inch_axes(X0 + column * (C + G), y, C, C)


A_RIGHT = X0 + 6 * C + 5 * G
Y_GRID_BOTTOM = H - Y_TITLE - 4 * C - 3 * G - G_PAIR
colorbar_axis = inch_axes(X0 + C + G, Y_GRID_BOTTOM - 0.08, 4 * C + 3 * G, 0.05)
PW = (W - 0.14 - A_RIGHT - 2 * YLAB) / 2
ax_b = inch_axes(A_RIGHT + YLAB, Y_BOT, PW, H - Y_BOT - Y_TOP)
ax_c = inch_axes(A_RIGHT + 2 * YLAB + PW, Y_BOT, PW, H - Y_BOT - Y_TOP)

# (a)
snapshot_rows = (
    (0, "hmc", "modelb", "SOSRA", None),
    (1, "hmc", "dm", "DM", "Eq."),
    (3, "gaussian", "modelb", "SOSRA", None),
    (4, "gaussian", "dm", "DM", "Noneq."),
)
snapshot_vmax = 1.4
for row, ensemble, method, row_label, ensemble_label in snapshot_rows:
    for column, physical_time in enumerate(SNAP_T):
        axis = cell_axes(row, column)
        snapshot_image = axis.imshow(series[ensemble][method][0][column], cmap=FIELD_CMAP,
                                     vmin=-snapshot_vmax, vmax=snapshot_vmax, interpolation="nearest",
                                     rasterized=True)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        if row == 0:
            axis.set_title(fr"$t={physical_time}$" if column == 0 else fr"${physical_time}$",
                           pad=1.0, fontsize=FS)
        if column == 0:
            axis.set_ylabel(row_label, fontsize=8, labelpad=1.0)
            if ensemble_label is not None:
                axis.text(0.04, 0.05, ensemble_label, transform=axis.transAxes, ha="left",
                          va="bottom", fontsize=7.5,
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5})
        if row == 0 and column == 0:
            panel_label(axis, "(a)")

colorbar = fig.colorbar(snapshot_image, cax=colorbar_axis, orientation="horizontal",
                        ticks=(-snapshot_vmax, 0.0, snapshot_vmax), extend="both", extendfrac=0.035)
colorbar.ax.set_xticklabels((fr"${-snapshot_vmax:.1f}$", "$0$", fr"${snapshot_vmax:.1f}$"))
colorbar_ticklabels = colorbar.ax.get_xticklabels()
colorbar_ticklabels[0].set_ha("left")
colorbar_ticklabels[-1].set_ha("right")
colorbar.set_label(r"$\phi$", labelpad=-1, fontsize=9)
colorbar.ax.tick_params(which="both", direction="out", top=False, bottom=True, labelsize=8,
                        length=2.0, width=0.6, pad=1.0)
colorbar.outline.set_linewidth(0.7)

method_handles = [
    Line2D([], [], color="0.35", linestyle="-", marker=""),
    tuple(Line2D([], [], color="0.35", linestyle="none", marker=marker, markerfacecolor="white",
                 markersize=3.5, markeredgewidth=0.9) for marker in ("o", "s")),
]
handler = {tuple: HandlerTuple(ndivide=None, pad=0.35)}

# (b) G_1(t) 对数轴: 颜色=初态系综, 线+带=SOSRA, 空心点=DM。Noneq.(M≈0) 从 1 涨到 ≈26 (t≈300)
# 再回落到 20, Eq. 平在 6.5; 0.7~70 两个数量级, 顶部放参数文字, 方法图例放两条曲线之间 (t>600),
# 系综图例放 Eq. 线下方。
for name, label, color, marker, *_ in ENSEMBLES:
    mb, dm = series[name]["modelb"][1], series[name]["dm"][1]
    truth_and_points(ax_b, T_G, mb, dm, color, marker, slice(2, None, 12))  # 每 120 一个点, 密度同 0-1000 时
    peak = int(np.argmax(mb[:, 0]))
    print(f"{label:7s} G_1 t=0 {mb[0, 0]:.2f}  peak t={T_G[peak]} SOSRA {mb[peak, 0]:.2f}({mb[peak, 1]:.2f})"
          f" DM {dm[peak, 0]:.2f}({dm[peak, 1]:.2f})  t=1000 SOSRA {mb[100, 0]:.2f} DM {dm[100, 0]:.2f}"
          f"  t=2000 SOSRA {mb[-1, 0]:.2f} DM {dm[-1, 0]:.2f}  DM/SOSRA t>=200 {np.mean(dm[20:, 0] / mb[20:, 0]):.3f}")

ensemble_handles = [
    Line2D([], [], color=color, marker=marker, fillstyle="full", markerfacecolor="white", label=label)
    for _, label, color, marker, *_ in ENSEMBLES
]
ax_b.set(xlim=(-30, 2030), ylim=(0.7, 70), yscale="log", xlabel=r"$t$", ylabel=r"$G_1(t)$")
ax_b.set_xticks((0, 1000, 2000))
ax_b.set_yticks((1, 10))
ax_b.set_yticklabels(("$1$", "$10$"))
ax_b.yaxis.labelpad = -0.75
ax_b.tick_params(axis="y", which="major", pad=0.8)
ensemble_legend = ax_b.legend(handles=ensemble_handles, loc="lower right", ncol=1, borderaxespad=0.15)
ax_b.legend(method_handles, ["SOSRA", "DM"], handler_map=handler, loc="upper right",
            bbox_to_anchor=(1.0, 0.70), ncol=2, handlelength=1.0, handletextpad=0.3, columnspacing=0.5,
            borderaxespad=0.15)
ax_b.add_artist(ensemble_legend)
ax_b.text(0.97, 0.98, r"$L=32,\ \lambda=1$" "\n" r"$\kappa=0.34034$", transform=ax_b.transAxes,
          ha="right", va="top", fontsize=FS_BC, linespacing=0.95)
panel_label(ax_b, "(b)")

# (c) 与初态的关联 ρ_n(t,0), n=1,2,3, 两个系综, 颜色=初态系综 (同 (b)), 线型=模, 线+带=SOSRA, 空心点=DM。
# τ_n 从 7 到 190 相差 25 倍, 用对数时间轴 10~1000 才能同图; t=0 的 ρ=1 在对数轴上画不出, 线从 t=10 起。
# 平衡系综等价于 Eq.(6) 的 ρ_n(Δt) (只是统计量少), 非平衡系综给出初态记忆的衰减, 两者可直接对比。
N_SHOW = 3
STYLES = ("-", "--", ":")
# 对数轴上近似等距的 DM 取样点 (t=10..900); n≥2 在 t>100 已归零, 只画到 100
PICK = ((1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90), (1, 2, 3, 5, 8, 10), (1, 2, 4, 6, 10))
for i, (name, label, color, marker, *_) in enumerate(ENSEMBLES):
    r_mb, r_dm = rho0(series[name]["modelb"][2])[:N_RHO], rho0(series[name]["dm"][2])[:N_RHO]
    t = T_G[:N_RHO]
    for n in range(N_SHOW):
        end = N_RHO if n == 0 else 11  # n≥2 到 t=100 已归零, 之后只剩噪声抖动
        mb, dm = r_mb[1:end, n], r_dm[:, n]
        ax_c.fill_between(t[1:end], mb[:, 0] - mb[:, 1], mb[:, 0] + mb[:, 1], color=color, alpha=0.18,
                          linewidth=0)
        ax_c.plot(t[1:end], mb[:, 0], color=color, linestyle=STYLES[n], marker="")
        sel = list(PICK[n])
        ax_c.errorbar(t[sel], dm[sel, 0], yerr=dm[sel, 1], color=color, marker=marker, fillstyle="full",
                      markerfacecolor="white", markeredgecolor=color, linestyle="none", markersize=3.5,
                      markeredgewidth=0.9, elinewidth=0.6, capsize=0, zorder=3)
    for n in range(N_MODES):
        tau = lambda r: 10 * np.interp(np.exp(-1), r[::-1, n, 0], np.arange(N_RHO)[::-1])
        print(f"{label:7s} n={n + 1}: rho0(10) SOSRA {r_mb[1, n, 0]:.3f}({r_mb[1, n, 1]:.3f}) DM {r_dm[1, n, 0]:.3f}"
              f"({r_dm[1, n, 1]:.3f})  rho0(100) SOSRA {r_mb[10, n, 0]:.3f} DM {r_dm[10, n, 0]:.3f}"
              f"  tau_1/e SOSRA {tau(r_mb):.0f} DM {tau(r_dm):.0f}"
              f"  max|DM-SOSRA| {np.abs(r_dm[:, n, 0] - r_mb[:, n, 0]).max():.3f}")

ax_c.set(xlim=(8, 1250), ylim=(-0.08, 1.04), xscale="log", xlabel=r"$t$", ylabel=r"$\rho_n(t,0)$")
ax_c.set_xticks((10, 100, 1000))
ax_c.set_xticklabels(("$10$", "$100$", "$1000$"))
ax_c.set_yticks((0, 0.5, 1))
ax_c.yaxis.labelpad = -0.75
ax_c.tick_params(axis="y", which="major", pad=0.8)
# 右上角 t>300, ρ>0.5 是空的 (Eq. n=1 在 t=300 已到 0.28); 只放模图例, SOSRA/DM 的编码沿用 (b)
ax_c.legend([Line2D([], [], color="0.35", linestyle=s, marker="") for s in STYLES],
            [fr"$n={n + 1}$" for n in range(N_SHOW)], loc="upper right", ncol=1,
            handlelength=1.6, handletextpad=0.3, labelspacing=0.15, borderaxespad=0.15)
panel_label(ax_c, "(c)")

fig.savefig(PAPER, transparent=True)
print(f"saved {PAPER}")
