# Model-B 稿件图片修改指南

## 目标与适用范围

本文档面向当前稿件 *Generative Modeling of Real-Time Stochastic Dynamics* 的主文 Fig. 1–3 和 Supplemental Fig. S1–S6。整体风格采用 **PRL-first** 的设计，同时吸收 Nature 和 Nature Physics 常见的可读性原则。

核心目标如下。

1. 在所有图片中建立统一且可复用的视觉语言。
2. 让颜色直接对应物理对象、mode 或有序参数。
3. 让线型、marker 和填充方式对应数值方法或数据来源。
4. 保证图片在 PRL 最终印刷尺寸下仍清晰可读。
5. 保证黑白打印和常见色觉缺陷条件下仍可区分主要数据。
6. 减少 legend、重复文字和装饰性元素，使视觉重点落在物理结论上。

配套文件如下。

- `plots/cjp_prl_nature.mplstyle`
- `plots/modelb_palette.py`

建议后续所有绘图脚本均加载这两个文件，不再在各个脚本中单独定义颜色、字号和线宽。

---

## 1. 总体设计原则

### 1.1 颜色与物理语义

全篇采用以下优先级。

- **颜色表示物理对象、mode、ensemble 或有序变量**
- **线型表示数值方法或理论关系**
- **marker 表示离散数据来源**
- **marker 的 filled/open 状态表示第二个二元变量，例如时间步长**

不要在不同 figure 中让同一种颜色表示不同的物理量。特别是 Fig. 1(d)、Fig. 2 和 Fig. S2 中的 mode index $n$，必须始终采用同一套颜色和 marker。

### 1.2 最终尺寸优先

绘图时直接采用最终发表尺寸，不要先画大图再在 LaTeX 中大幅缩小。

| 类型 | 推荐宽度 | 推荐高度 |
|---|---:|---:|
| PRL 单栏普通图 | 3.375 in | 2.2–2.8 in |
| PRL 单栏纵向组合图 | 3.375 in | 3.8–4.5 in |
| PRL 双栏图 | 7.0 in | 3.0–3.6 in |
| SI 双面板图 | 7.0 in | 2.7–3.3 in |

在 LaTeX 中采用最终宽度插入。

```tex
\includegraphics[width=\columnwidth]{figure.pdf}
```

双栏图使用 `figure*` 并设置为 `\textwidth`。不要依赖 `scale=...` 调整最终字号。

---

## 2. 统一配色

### 2.1 方法和参考结果

| 物理含义 | 颜色 | Hex | 推荐视觉编码 |
|---|---|---:|---|
| Direct Model B / reference | charcoal | `#262626` | solid line 或 filled circle |
| Diffusion model | blue | `#0072B2` | open square 或 open circle |
| 第三个方法，例如 DPM-Solver-v3 | vermillion | `#D55E00` | open triangle 或 dashed line |
| 理论 scaling law | medium gray | `#7A7A7A` | dashed line |
| $y=0$ 或 ratio $=1$ | light gray | `#B0B0B0` | thin dashed line |

只有当 panel 中颜色不需要表示 mode、ensemble 或 physical time 时，才使用这套 method colors。

### 2.2 Mode colors

以下映射在 Fig. 1(d)、Fig. 2 和 Fig. S2 中保持不变。

| Mode | Color | Hex | Marker |
|---|---|---:|---|
| $n=1$ | vermillion | `#D55E00` | `o` |
| $n=2$ | blue | `#0072B2` | `s` |
| $n=3$ | reddish purple | `#CC79A7` | `D` |
| $n=4$ | bluish green | `#009E73` | `^` |
| $n=5$ | orange | `#E69F00` | `v` |
| $n=6$ | sky blue | `#56B4E9` | `P` |

对于同一个 $n$，Model B 和 DM 使用同一种颜色。方法通过线型和 marker 区分。

### 2.3 Physical-time colors

物理时间 $t$ 是有序变量，应使用 sequential colormap。统一采用截断后的 `viridis`。

```python
colors = time_colors(n_times)
```

`time_colors` 默认使用 `viridis` 的 $0.12$ 到 $0.82$ 区间，避免过暗的紫色和过亮的黄色。

按时间从早到晚，颜色应单调变化。不要把不同时间当作无序类别随机分配颜色。

### 2.4 Field configurations

带正负号的标量场采用 centered diverging colormap。

```python
FIELD_CMAP = blue -> near-white -> muted red
```

具体端点为

- negative field: `#3B6FB6`
- zero: `#F4F4F2`
- positive field: `#C64A45`

同一组 snapshots 必须共享对称 normalization。

```python
vmax = max(np.max(np.abs(phi)) for phi in snapshot_list)

ax.imshow(
    phi,
    cmap=FIELD_CMAP,
    vmin=-vmax,
    vmax=vmax,
    interpolation="nearest",
)
```

不要对每张 snapshot 单独自动设置 `vmin` 和 `vmax`。否则颜色变化无法直接表示 field amplitude 的真实变化。

---

## 3. 字体、线宽和坐标轴

统一使用 `cjp_prl_nature.mplstyle` 中的设置。

| 元素 | 推荐值 |
|---|---:|
| 正文字号 | 9 pt |
| tick label | 8 pt |
| legend | 7.5–8 pt |
| 主曲线线宽 | 1.2 pt |
| 理论 guide 线宽 | 0.9–1.0 pt |
| marker size | 4.0 pt |
| marker edge width | 1.0 pt |
| axes spine | 0.7 pt |
| major tick | 0.7 pt |
| minor tick | 0.55 pt |
| error-bar cap | 1.5 pt |

推荐保留以下风格。

- ticks 向内
- top ticks 和 right ticks 开启
- minor ticks 开启
- 不使用背景网格
- legend 无边框
- 坐标轴标题使用数学符号，不使用冗长文字
- panel 内的参数说明尽量放在空白区域，而不是增加独立标题

Panel labels 统一放在左上角。

```python
ax.text(
    0.02, 0.98, "(a)",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=9,
    fontweight="bold",
)
```

所有 panel label 的相对位置必须一致。

---

## 4. 线型和 marker 语法

### 4.1 同一 observable 比较 Model B 与 DM

推荐采用

- Model B: solid line，必要时加浅色 error band
- DM: open marker，通常不连接，或使用较细的同色线
- 相同物理对象使用相同颜色

示例。

```python
ax.plot(
    x_ref, y_ref,
    color=color,
    lw=1.2,
    ls="-",
)

ax.fill_between(
    x_ref,
    y_ref - err_ref,
    y_ref + err_ref,
    color=color,
    alpha=0.15,
    linewidth=0,
)

ax.errorbar(
    x_dm, y_dm,
    yerr=err_dm,
    color=color,
    marker=marker,
    mfc="white",
    mec=color,
    ls="none",
    ms=4,
)
```

### 4.2 三种方法比较

推荐采用

- Model B: charcoal, solid, circle
- NCSN++: blue, dashed or open square
- DPM-Solver-v3: vermillion, dash-dotted or open triangle

颜色和 marker 必须同时区分方法，避免只依赖颜色。

### 4.3 两个时间步长比较

在 Fig. S4 这类问题中，颜色只表示 method。

- $\Delta t=0.5$: filled marker + solid line
- $\Delta t=10$: open marker + dashed line

不要为每一个 `method × time step` 组合设置独立颜色。

---

## 5. 主文逐图修改方案

## 5.1 Figure 1

### 当前物理内容

Fig. 1 同时包含

- Model B 和 DM 的 configuration snapshots
- $G_1(t)$
- conservation drift
- 前四个 diagonal modes 的 $\rho_n(\Delta t)$

这张图承担方法有效性的总体展示，信息密度最高。建议作为主文唯一的双栏图。

### 推荐尺寸

```python
figsize = (7.0, 3.2)
```

若 caption 后仍显拥挤，可增加到 $7.0\times3.4$ in，不建议继续增加高度。

### 推荐排版

```text
┌──────────────────────────┬──────────────────────┐
│ (a) snapshots            │ (b) G1(t)            │
│ Model B  t1  t2  t3      │                      │
│ DM       t1  t2  t3      │                      │
├────────────────┬─────────┴──────────────────────┤
│ (c) conservation drift   │ (d) rho_n(Delta t)   │
│ narrow panel             │ wide panel           │
└──────────────────────────┴──────────────────────┘
```

Panel (c) 的信息结构简单，可以较窄。Panel (d) 包含四个 modes、logarithmic axis 和 error bars，应分配最大的横向空间。

### Panel (a)

1. 采用 $2\times3$ snapshots。
2. 所有 snapshots 为相同的正方形尺寸。
3. 上方仅标记 $t=10,100,1000$。
4. 左侧标记 `Model B` 和 `DM`。
5. 去除 ticks、spines 和各自的小标题。
6. 六张图共享同一个 centered diverging normalization。
7. 两行 snapshots 使用独立噪声时，caption 中说明即可，图内不增加说明文字。
8. 若 field range 在两种 dynamics 中相同，不加 colorbar。若必须加 colorbar，使用整组共享的一个窄 colorbar。

### Panel (b)

这里同时存在两个 ensemble 和两个方法。

- 颜色表示 ensemble
- 线型和 marker 表示 dynamics

推荐映射。

- HMC: vermillion
- i.i.d. Gaussian: blue
- Model B: solid line + band
- DM: open marker

建议使用两个简短 legend，分别解释 ensemble colors 和 dynamics styles。也可以通过 caption 解释 dynamics styles，只保留 ensemble legend。

参数

```text
L = 32
kappa = 0.34034
lambda = 1
```

放在 panel 内右上角，并使用两行或三行紧凑排版。

### Panel (c)

这里只有 method 这一主要类别，因此采用 method colors。

- Model B: charcoal solid line
- DM: blue open circles
- 浮点误差尺度用浅灰 horizontal guide
- 使用 logarithmic $y$ axis
- 避免在曲线末端直接写红色 `DM` 和灰色 `Model B`
- legend 放在无数据区域

### Panel (d)

颜色表示 mode，方法表示为

- Model B: solid line
- DM: open marker

同一个 mode 的两种方法必须使用同一种颜色。不要让 `Model B = black` 和 `DM = blue`，否则读者难以追踪四个 relaxation modes。

建议将 legend 分成两个部分。

1. mode legend，显示 $n$、$\hat k_n$、颜色和 marker
2. method legend，显示 solid line 和 open marker

method legend 可以放在 caption 中，以进一步减小 panel 内信息密度。

---

## 5.2 Figure 2

### 推荐尺寸

```python
figsize = (3.375, 2.55)
```

保持单栏。

### 主图

- $n=1,\ldots,4$ 使用固定 mode colors。
- Model B 使用 solid line。
- DM 使用相同颜色的 open markers。
- $e^{-x}$ 使用 `GUIDE` gray dashed line。
- 不使用彩色 fit line 覆盖主曲线。
- 若 DM fit 与 Model B fit 几乎重合，降低 fit line 的视觉权重，避免遮挡数据。
- mode legend 保持一列，放在右下或右中空白区。

### Inset

Inset 研究的是 dispersion relation，而非 mode identity，因此此处改用 method colors。

- Model B: charcoal filled circles
- DM: blue open squares
- 两条 fit 使用各自 method color
- 理论 slope 或 guide 使用 gray dashed line
- 直接在 inset 中标注
  \[
  z_{\rm DM}=...,\qquad z_{\rm Model\,B}=...
  \]
- 删除重复 legend，或仅保留两项极简 legend

Inset 建议占主轴宽度的 $35\%$ 到 $40\%$，避免过小。

---

## 5.3 Figure 3

### 推荐尺寸

```python
figsize = (3.375, 4.0)
```

保留单栏纵向结构。

### Panel (a)

- 三张 snapshots 使用与 Fig. 1(a) 相同的 field colormap。
- 三张图共享 symmetric `vmin/vmax`。
- 去除坐标刻度和边框。
- 时间标签置于每张图上方。
- snapshot 间距保持一致。
- 不使用饱和度过高的红蓝色。

### Panel (b)

- 六条 $C(r/L(t),t)$ 曲线按时间使用 sequential `viridis`。
- 图例中按时间从早到晚排序。
- 线宽保持一致。
- $C=0$ guide 可使用浅灰细线。
- 主要观察量是 scaling collapse，不需要为每条曲线使用不同 marker。

### Inset

- DM $L(t)$ 使用 blue points 或 blue solid line。
- $t^{1/3}$ 使用 charcoal 或 gray dashed line。
- 不再单独使用 orange 表示理论线。
- 使用 log-log axes。
- 将理论关系直接写为 $t^{1/3}$，避免冗长 legend。

---

## 6. Supplemental Material 逐图修改方案

## 6.1 Figure S1

比较三种方法时统一使用

- Model B: charcoal circle + solid line
- NCSN++: blue square + dashed line
- DPM-Solver-v3: vermillion triangle + dash-dotted line

Power-law fits 使用同色但较细的线，或者全部使用 gray 并在旁边标注 $z$。如果三条 fit 几乎重合，推荐只绘制一条灰色 slope guide，然后在 legend 中给出三个 $z$。

Legend 文本应缩短，例如

```text
Model B, z = ...
NCSN++, z = ...
DPM-v3, z = ...
```

避免在图内重复完整采样步数说明，详细信息留给 caption。

## 6.2 Figure S2

与 Fig. 1(d) 和 Fig. 2 完全复用 mode colors 和 mode markers。

- data points 使用 mode colors
- analytic guide 使用 gray dashed
- sum-rule fit 使用 charcoal dash-dotted
- $y=1$ horizontal guide 使用 light gray dotted
- 低 $1/x$ 区域 error bars 较大时，适度降低 marker size，避免重叠

## 6.3 Figure S3

保持 $2\times4$ 排版，但减少重复信息。

- 第一列显示 $y$ labels，其余列不重复
- 下排显示 $x$ labels，上排不重复
- panel label 和 $(n,\hat k)$ 使用统一位置
- 整张图共享一个 method legend
- 颜色表示 method
- step size 使用 line style 或 marker fill
- 不在每个 panel 中重复 legend

建议将两种 step 的语法统一为

- step 10: open marker + dashed line
- step 0.5: filled marker + solid line

## 6.4 Figure S4

这是 Supplemental 中优先级最高的修改对象。

当前图中每一个 `time-step × dynamics` 组合使用独立颜色，视觉变量过多。改为以下二维编码。

### 颜色表示 method

| Method | Color | Marker |
|---|---|---|
| Model B | charcoal | circle |
| single-$L$ | blue | square |
| multi-$L$ | vermillion | triangle |

### 填充和线型表示 step

| Step | Marker | Line |
|---|---|---|
| $\Delta t=0.5$ | filled | solid |
| $\Delta t=10$ | open | dashed |

两块 panel 共享 legend。可以分别设置一个 method legend 和一个 step legend。

## 6.5 Figure S5

- 左图 Model B 使用 charcoal solid line。
- 左图 DM 使用 blue open circles。
- 不使用 orange 作为 DM 的唯一标识。
- 右图 ratio 数据使用 blue open circles。
- ratio $=1$ 使用 light gray dashed line。
- 两个 panel 的高度、坐标轴和字体完全一致。
- 将 sample size 保留在 caption 中，legend 中只保留方法名。

## 6.6 Figure S6

与 Fig. 3(b) 使用完全一致的 `time_colors`。

- 时间从早到晚使用 sequential colors。
- Porod law $k^{-3}$ 使用 gray dashed line。
- 理论 guide 旁直接标记 $k^{-3}$。
- 不使用黑色数据曲线，因为黑色已保留给 reference method。
- legend 按时间升序排列。
- 若曲线密集，可将 legend 设为两列。

---

## 7. Matplotlib 使用方式

### 7.1 加载统一 style

```python
import matplotlib.pyplot as plt

plt.style.use("cjp_prl_nature.mplstyle")
```

若脚本不在 style 文件所在目录，使用绝对路径。

```python
plt.style.use("/path/to/cjp_prl_nature.mplstyle")
```

### 7.2 加载 palette

```python
from modelb_palette import (
    REFERENCE,
    DM,
    ALT,
    GUIDE,
    ZERO,
    MODE_COLORS,
    MODE_MARKERS,
    FIELD_CMAP,
    time_colors,
)
```

### 7.3 Mode comparison 示例

```python
for i, n in enumerate(mode_indices):
    color = MODE_COLORS[i]
    marker = MODE_MARKERS[i]

    ax.plot(
        x_ref[n],
        rho_ref[n],
        color=color,
        lw=1.2,
        ls="-",
    )

    ax.errorbar(
        x_dm[n],
        rho_dm[n],
        yerr=rho_dm_err[n],
        color=color,
        marker=marker,
        mfc="white",
        mec=color,
        ls="none",
        ms=4,
        label=rf"$n={n}$",
    )
```

### 7.4 Ordered-time curves 示例

```python
times = [100, 160, 250, 400, 630, 1000]
colors = time_colors(len(times))

for t, color in zip(times, colors):
    ax.plot(
        r_over_L[t],
        correlation[t],
        color=color,
        lw=1.2,
        label=rf"$t={t}$",
    )
```

### 7.5 Shared field normalization 示例

```python
snapshots = [phi_t10, phi_t100, phi_t1000]
vmax = max(float(abs(phi).max()) for phi in snapshots)

for ax, phi in zip(axes, snapshots):
    ax.imshow(
        phi,
        cmap=FIELD_CMAP,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
```

### 7.6 自定义 method legend

```python
from matplotlib.lines import Line2D

method_handles = [
    Line2D(
        [0], [0],
        color=REFERENCE,
        lw=1.2,
        ls="-",
        label="Model B",
    ),
    Line2D(
        [0], [0],
        color=REFERENCE,
        marker="o",
        mfc="white",
        mec=REFERENCE,
        lw=0,
        label="DM",
    ),
]

ax.legend(handles=method_handles, loc="best")
```

当颜色用于 mode 时，method legend 建议使用 neutral charcoal，强调线型和 marker 语法，而不是再次引入 method colors。

---

## 8. Error bars 和 uncertainty bands

1. Reference trajectory 的连续曲线适合使用浅色 uncertainty band。
2. DM 的离散预测适合使用 open markers 和 error bars。
3. Band alpha 建议为 `0.12–0.18`。
4. Band 不要绘制边框。
5. Error bar 与数据点使用同色。
6. 误差棒密集时可降低绘制频率，但必须在 caption 中说明。
7. 不要为了视觉平滑而省略具有物理意义的统计误差。
8. Log scale 上必须检查误差下界是否为负。必要时使用 asymmetric errors 或只绘制允许的区间。

---

## 9. Legend 和文字标注

### 9.1 Legend 的优先级

优先采用以下顺序。

1. 直接标注简单理论线
2. 一个共享 legend
3. 分离 method legend 和 mode/ensemble legend
4. 最后才考虑在每个 panel 中重复 legend

### 9.2 Legend 文本

使用短标签。

推荐

```text
Model B
DM
single-L
multi-L
DPM-v3
```

避免在 legend 中放入完整句子、所有参数和 sample size。参数写在 panel 空白区，sample size 写在 caption。

### 9.3 Panel 内参数

统一采用左上或右上角的小型 text block。

```python
ax.text(
    0.97, 0.95,
    r"$L=32$" "\n"
    r"$\kappa=0.34034,\ \lambda=1$",
    transform=ax.transAxes,
    ha="right",
    va="top",
)
```

所有 figure 中相同参数的格式保持一致。

---

## 10. 输出格式

推荐保存为 vector PDF。

```python
fig.savefig(
    "fig1.pdf",
    dpi=600,
    facecolor="white",
)
```

原则如下。

- curves、markers、axes 和文字保持 vector。
- `imshow` field snapshots 可以 rasterized。
- PDF 中字体使用 TrueType embedding。
- 不要截屏后转成 PDF。
- 不要将所有 panel 先保存为 PNG 再拼接。
- 不要使用 JPEG。
- 除非必须，不使用透明背景。
- 不建议额外使用 `bbox_inches="tight"`，因为它可能改变最终画布尺寸和 panel 间距。当前 style 已固定标准 bounding box。
- 输出后在最终物理尺寸下检查，而不是只在放大状态下检查。

---

## 11. 推荐修改顺序

### 第一阶段

1. 在所有绘图脚本中加载统一 style。
2. 删除脚本内部零散的 color lists。
3. 导入 `modelb_palette.py`。
4. 固定 mode、method、time 和 field 的视觉语义。

### 第二阶段

1. 首先重排 Fig. 1。
2. 统一 Fig. 1(d)、Fig. 2 和 Fig. S2 的 mode colors。
3. 统一 Fig. 3 和 Fig. S6 的 time colors。
4. 重构 Fig. S4 的二维视觉编码。

### 第三阶段

1. 调整所有 legends。
2. 统一 panel labels。
3. 删除重复参数文字。
4. 检查 error bars 和 bands。
5. 检查所有 snapshots 的 normalization。

### 第四阶段

1. 以最终宽度嵌入 manuscript。
2. 打印或按 100% 比例查看 PDF。
3. 检查黑白可读性。
4. 检查线型和 marker 在缩小后是否仍可区分。
5. 检查 caption 是否准确说明视觉编码。

---

## 12. 最终检查清单

### 全局

- [ ] 所有 figure 均加载同一 `.mplstyle`
- [ ] 同一物理量在不同 figure 中使用相同颜色
- [ ] 颜色、线型、marker 的物理语义没有冲突
- [ ] 主要结论不依赖颜色才能识别
- [ ] 字号在最终印刷尺寸下可读
- [ ] axes、ticks、line widths 和 marker sizes 全篇一致
- [ ] 所有 panel label 位置一致
- [ ] 无背景网格和装饰性阴影
- [ ] 理论 guide 统一为 gray
- [ ] error bands 足够浅，不遮挡数据
- [ ] legend 无边框且不覆盖关键数据
- [ ] 参数和 sample size 没有在图内过度重复

### Field snapshots

- [ ] 使用 centered diverging colormap
- [ ] 同组 snapshots 使用相同 symmetric `vmin/vmax`
- [ ] snapshots 的尺寸和 aspect ratio 一致
- [ ] snapshots 不显示无意义的 ticks
- [ ] 颜色变化可解释为 field amplitude 的变化

### Mode plots

- [ ] $n=1,\ldots,4$ 的颜色映射在 Fig. 1(d)、Fig. 2 和 Fig. S2 中一致
- [ ] Model B 与 DM 对同一 mode 使用相同颜色
- [ ] 方法由 line style 和 marker 区分
- [ ] mode legend 与 method legend 没有混淆

### Time-dependent plots

- [ ] 时间使用 sequential colormap
- [ ] legend 按时间升序排列
- [ ] 早晚时间的颜色顺序在 Fig. 3 和 Fig. S6 中一致
- [ ] 理论 growth law 或 Porod law 使用 neutral gray

### 输出

- [ ] 保存为 PDF
- [ ] 文字和曲线保持 vector
- [ ] raster snapshots 的分辨率足够
- [ ] PDF 字体正确嵌入
- [ ] 图片在 `\columnwidth` 或 `\textwidth` 下直接使用
- [ ] 未在 LaTeX 中再次大幅缩放
- [ ] 无标签被裁切
- [ ] 黑白打印后仍可区分主要数据
- [ ] 最终 PDF 中的图和 caption 对应正确

---

## 13. 最优先修改项

按照对主文视觉质量和物理叙事的影响排序，建议优先处理以下四项。

1. 将 Fig. 1 改为双栏并重新分配四个 panels 的空间。
2. 在 Fig. 1(d)、Fig. 2 和 Fig. S2 中固定 mode colors。
3. 在 Fig. 3 和 Fig. S6 中用 sequential colormap 表示 physical time。
4. 重构 Fig. S4，使颜色表示 method，marker fill 和 line style 表示 $\Delta t$。

完成这四项后，整篇稿件的视觉语言将基本统一。其余工作主要是字号、legend、panel spacing 和局部标注的精修。
