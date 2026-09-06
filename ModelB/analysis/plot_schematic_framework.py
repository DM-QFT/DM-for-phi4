# PRL 正文 Fig.1：学习到的转移核示意图。场构型取自 L=128 的 DM 粗化链第 0 条轨迹：
# 上排 t=0,100,110,1000；下排是真实增量 φ_110−φ_100 加高斯噪声，模拟反向扩散中间态。
#   python analysis/plot_schematic_framework.py
from pathlib import Path
import sys

import h5py
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
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
W, H, S = 3.375, 2.3, 0.48
X_TOP = (0.02, 0.955, 1.94, 2.875)  # ⊕ 夹在第 2、3 格之间，上排间距比下排略宽
X_BOT = (0.14, 1.012, 1.883, 2.755)
Y_TOP, Y_BOT = 1.66, 0.40
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(0, H))
ax.set_axis_off()


def panel(x, y, field, vmax, label):
    a = fig.add_axes([x / W, y / H, S / W, S / H])
    a.imshow(field.T, cmap=FIELD_CMAP, vmin=-vmax, vmax=vmax, origin="lower",
             interpolation="nearest")
    a.set(xticks=[], yticks=[])
    for sp in a.spines.values():
        sp.set(color="#666666", linewidth=0.5)
    ax.text(x + S / 2, y + S + 0.035, label, ha="center", va="bottom", fontsize=8,
            color=REFERENCE)


def arrow(p0, p1, color, lw=1.0, dots=False):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0))
    if dots:
        ax.text((p0[0] + p1[0]) / 2, p0[1], r"$\cdots$", ha="center", va="center", fontsize=8,
                color=color, bbox=dict(facecolor="white", edgecolor="none", pad=0.7))


for x, t, lab in zip(X_TOP, (0, 100, 110, 1000),
                     (r"$\phi_0$", r"$\phi_t$", r"$\phi_{t+\Delta t}$", r"$\phi_{t+n\Delta t}$")):
    # 初态振幅只有 0.1，按 ±1.8 画是一片白，改按 3.5 std 显示噪声结构
    panel(x, Y_TOP, phi[t], 3.5 * phi[0].std() if t == 0 else 1.8, lab)
for x, s, lab in zip(X_BOT, (1.0, 0.7, 0.3, 0.0),
                     (r"$\Delta\phi_{s=1}\sim\mathcal{N}$", r"$\Delta\phi_{s=0.7}$",
                      r"$\Delta\phi_{s=0.3}$", r"$\Delta\phi_{s=0}$")):
    panel(x, Y_BOT, dphi_s[s], 3 * dphi.std(), lab)

# 上排物理轨迹：φ_t ⊕ Δφ → φ_{t+Δt}
yc, xc, r = Y_TOP + S / 2, W / 2, 6.1 / 72
arrow((X_TOP[0] + S, yc), (X_TOP[1] - 0.015, yc), REFERENCE, dots=True)
arrow((X_TOP[1] + S, yc), (xc - r - 0.015, yc), REFERENCE)
arrow((xc + r, yc), (X_TOP[2] - 0.015, yc), REFERENCE)
arrow((X_TOP[2] + S, yc), (X_TOP[3] - 0.015, yc), REFERENCE, dots=True)
ax.add_patch(Circle((xc, yc), r, facecolor="white", edgecolor=REFERENCE, lw=0.8))
ax.text(xc, yc, "+", ha="center", va="center", fontsize=9, color=REFERENCE)

# 条件 φ_t 送入 score network（蝴蝶结）
bx, by = X_TOP[1] + S / 2, 1.21
arrow((bx, Y_TOP), (bx, by + 0.11 + 0.02), REFERENCE, lw=0.9)
for sgn in (-1, 1):
    ax.add_patch(Polygon([(bx + sgn * 0.17, by - 0.11), (bx + sgn * 0.17, by + 0.11),
                          (bx + sgn * 0.016, by + 0.04), (bx + sgn * 0.016, by - 0.04)],
                         facecolor=mc.to_rgba(DM, 0.18), edgecolor=DM, lw=0.7))
ax.text(bx + 0.17 + 0.065, by, r"score network $s_\theta(\Delta\phi_s;\phi_t,s)$",
        ha="left", va="center", fontsize=7, color=DM)

# 下排反向扩散 s: 1 → 0，最后一格回馈到 ⊕
yb = Y_BOT + S / 2
for i in range(3):
    arrow((X_BOT[i] + S, yb), (X_BOT[i + 1] - 0.015, yb), DM, dots=True)
x1, yh, rc = 3.19, 1.54, 0.06
loop = MPath([(x1, Y_BOT + S), (x1, yh - rc), (x1, yh), (x1 - rc, yh), (xc + rc, yh), (xc, yh),
              (xc, yh + rc), (xc, yc - r - 0.014)],
             [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
              MPath.CURVE3, MPath.CURVE3, MPath.LINETO])
ax.add_patch(FancyArrowPatch(path=loop, arrowstyle="-|>", mutation_scale=7, color=DM, lw=0.9))
ax.add_patch(FancyBboxPatch((0.029, 0.056), 3.317, 1.378, boxstyle="round,pad=0,rounding_size=0.06",
                            facecolor="none", edgecolor=ZERO, lw=0.9, linestyle="--"))
ax.text(3.265, 0.125, r"$p_\theta(\Delta\phi\mid\phi_t)$", ha="right", va="bottom", fontsize=7.5,
        color=DM)

fig.savefig(OUT)
print(OUT)
