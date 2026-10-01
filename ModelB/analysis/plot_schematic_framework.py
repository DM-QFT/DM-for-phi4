# PRL 正文 Fig.1：学习到的转移核示意图。场构型取自 L=128 的 DM 粗化链第 0 条轨迹：
# 上排 t=0,100,110,1000；下排是真实增量 φ_110−φ_100 加高斯噪声，模拟反向扩散中间态。
#   python analysis/plot_schematic_framework.py
from pathlib import Path
import sys

import h5py
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "plots"))
from modelb_palette import DM, FIELD_CMAP, REFERENCE, ZERO  # noqa: E402

plt.style.use(ROOT.parent / "plots" / "cjp_prl_nature.mplstyle")
plt.rcParams.update({"text.usetex": True, "figure.constrained_layout.use": False})
OUT = ROOT / "draft" / "figures" / "prl" / "schematic.pdf"
RUN = ROOT / "runs/modelBcond_Lmulti8-16-32-64_k0.85_l1.0_t10_ncsnpp_quench_tanhgauss/samples"

with h5py.File(RUN / "quench_chain_ep10000_L128_N32_t1000.h5") as f:
    tt = f["times"][()]
    phi = {t: f["phi_gen"][np.argmin(np.abs(tt - t)), 0] for t in (100, 110, 1000)}
phi[0] = np.load(RUN / "quench_t0_L128_N32.npz")["phi_gen"][0]
dphi = phi[110] - phi[100]
# 示意用噪声幅度，单位 std(Δφ)。训练所用 σ=190 的 VE 边际标准差在 s=0.3 已是增量的十倍，
# 按真实 s 画则中间两格只剩纯噪声，看不出逐步去噪。
NOISE = {1.0: 7.0, 0.7: 2.3, 0.3: 0.8, 0.0: 0.0}
rng = np.random.default_rng(0)
dphi_s = {s: dphi + a * dphi.std() * rng.standard_normal(dphi.shape) for s, a in NOISE.items()}

# 布局单位为英寸，坐标轴铺满整图，1 数据单位 = 1 英寸
W, H, S = 3.375, 2.10, 0.48
X_TOP = (0.02, 0.955, 1.94, 2.875)  # ⊕ 夹在第 2、3 格之间，上排间距比下排略宽
X_BOT = (0.14, 1.012, 1.883, 2.755)
Y_TOP, Y_BOT = 1.43, 0.29
LW = 1.0                          # 连线、箭头、⊕ 统一线宽（pt）
HEAD_L, HEAD_W = 5 / 72, 2 / 72   # 箭头长与半宽
TINT = 1 - 0.18 * (1 - np.array(mc.to_rgb(DM)))  # 学习部件的浅蓝底色，不透明以盖住虚线
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(0, H))
ax.set_axis_off()


def panel(x, y, field, vmax, label, below=False):
    a = fig.add_axes([x / W, y / H, S / W, S / H])
    a.imshow(field.T, cmap=FIELD_CMAP, vmin=-vmax, vmax=vmax, origin="lower",
             interpolation="nearest")
    a.set(xticks=[], yticks=[])
    for sp in a.spines.values():
        sp.set(color="#666666", linewidth=0.5)
    ax.text(x + S / 2, y - 0.035 if below else y + S + 0.035, label, ha="center",
            va="top" if below else "bottom", fontsize=8, color=REFERENCE)


def line(a, b, color):
    # style 的 prop_cycle 带 marker，必须显式关掉
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=LW, marker="", solid_capstyle="butt")


def head(tip, u, color):
    # 实心三角、不描边，箭尖才锐利；返回底边中点供连线对接
    tip, u = np.asarray(tip), np.asarray(u)
    n = np.array([-u[1], u[0]])
    base = tip - HEAD_L * u
    ax.add_patch(Polygon([tip, base + HEAD_W * n, base - HEAD_W * n], closed=True,
                         facecolor=color, edgecolor="none"))
    return base


def arrow(p0, p1, color, dots=False):
    p0, p1 = np.asarray(p0), np.asarray(p1)
    u = (p1 - p0) / np.hypot(*(p1 - p0))
    base = head(p1, u, color)
    if not dots:
        return line(p0, base, color)
    mid = (p0 + p1) / 2
    line(p0, mid - 6 / 72 * u, color)
    line(mid + 6 / 72 * u, base, color)
    for k in (-3.2, 0, 3.2):
        ax.add_patch(Circle(mid + k / 72 * u, 0.55 / 72, color=color, lw=0))


for x, t, lab in zip(X_TOP, (0, 100, 110, 1000),
                     (r"$\phi_0$", r"$\phi_t$", r"$\phi_{t+\delta t}$", r"$\phi_{t+n\delta t}$")):
    # 初态振幅只有 0.1，按 ±1.8 画是一片白，改按 3.5 std 显示噪声结构
    panel(x, Y_TOP, phi[t], 3.5 * phi[0].std() if t == 0 else 1.8, lab)
for x, s, lab in zip(X_BOT, (1.0, 0.7, 0.3, 0.0),
                     (r"$\Delta\phi_{s=1}\sim\mathcal{N}$", r"$\Delta\phi_{s=0.7}$",
                      r"$\Delta\phi_{s=0.3}$", r"$\Delta\phi_{s=0}$")):
    panel(x, Y_BOT, dphi_s[s], 3 * dphi.std(), lab, below=True)

# 上排物理轨迹 φ_t ⊕ Δφ → φ_{t+δt}
yc, xc, r = Y_TOP + S / 2, W / 2, 6.1 / 72
arrow((X_TOP[0] + S, yc), (X_TOP[1] - 0.015, yc), REFERENCE, dots=True)
arrow((X_TOP[1] + S, yc), (xc - r - 0.015, yc), REFERENCE)
arrow((xc + r, yc), (X_TOP[2] - 0.015, yc), REFERENCE)
arrow((X_TOP[2] + S, yc), (X_TOP[3] - 0.015, yc), REFERENCE, dots=True)
ax.add_patch(Circle((xc, yc), r, facecolor="white", edgecolor=REFERENCE, lw=LW))
line((xc - 0.55 * r, yc), (xc + 0.55 * r, yc), REFERENCE)
line((xc, yc - 0.55 * r), (xc, yc + 0.55 * r), REFERENCE)

# 条件 φ_t 送入 score network（蝴蝶结）
bx, by = X_TOP[1] + S / 2, 0.98
arrow((bx, Y_TOP), (bx, by + 0.11 + 0.02), REFERENCE)
for sgn in (-1, 1):
    ax.add_patch(Polygon([(bx + sgn * 0.17, by - 0.11), (bx + sgn * 0.17, by + 0.11),
                          (bx + sgn * 0.016, by + 0.04), (bx + sgn * 0.016, by - 0.04)],
                         facecolor=TINT, edgecolor=DM, lw=0.7))
ax.text(bx + 0.17 + 0.065, by, "score network\n" r"$s_\theta(\Delta\phi_s;\phi_t,s)$",
        ha="left", va="center", fontsize=8, color=DM, linespacing=1.3)

# 下排反向扩散 s: 1 → 0，最后一格回馈到 ⊕
yb = Y_BOT + S / 2
for i in range(3):
    arrow((X_BOT[i] + S, yb), (X_BOT[i + 1] - 0.015, yb), DM, dots=True)
x1, yh, rc = X_BOT[3] + S / 2, 1.31, 0.06
base = head((xc, yc - r - 0.014), (0, 1), DM)
loop = MPath([(x1, Y_BOT + S), (x1, yh - rc), (x1, yh), (x1 - rc, yh), (xc + rc, yh), (xc, yh),
              (xc, yh + rc), base],
             [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
              MPath.CURVE3, MPath.CURVE3, MPath.LINETO])
ax.add_patch(PathPatch(loop, fill=False, edgecolor=DM, lw=LW, capstyle="butt"))
ax.add_patch(FancyBboxPatch((0.029, 0.056), 3.317, 1.148, boxstyle="round,pad=0,rounding_size=0.06",
                            facecolor="none", edgecolor=ZERO, lw=0.9, linestyle="--", zorder=0))
# 角标：左上角与虚线框圆角同心（外扩 0.5pt 盖住线宽），其余三角只微圆，虚线从直边干净穿出
pad, R, rr = 0.04, 0.067, 0.014
x0, y1 = 0.029 - 0.007, 1.204 + 0.007
lab = ax.text(x0 + pad, y1 - pad, r"$p_\theta(\Delta\phi\mid\phi_t)$", ha="left", va="top",
              fontsize=8, color=DM)
ext = lab.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
x1, y0 = x0 + ext.width + 2 * pad, y1 - ext.height - 2 * pad
badge = MPath([(x0 + R, y1), (x1 - rr, y1), (x1, y1), (x1, y1 - rr), (x1, y0 + rr), (x1, y0),
               (x1 - rr, y0), (x0 + rr, y0), (x0, y0), (x0, y0 + rr), (x0, y1 - R), (x0, y1),
               (x0 + R, y1)],
              [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3,
               MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3,
               MPath.CURVE3])
ax.add_patch(PathPatch(badge, facecolor=TINT, edgecolor="none", zorder=2))

fig.savefig(OUT)
print(OUT)
