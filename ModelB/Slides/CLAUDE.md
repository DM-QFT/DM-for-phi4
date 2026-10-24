不要根据tex编译器的输出修改，应该根据页面实际情况修改

用户说"第 N 页"指页脚编号 N/总数，不是 PDF 页码。标题页、section page 和 overlay 都会让两者错开。

## 内容

Slides 只讲物理、想法、概念。不放优化器、EMA、bf16、compile、epoch 数、参数量、求解器步数、bootstrap 数、标定常数，也不放 HMC 与求解器的引用。每页标题写它回答的物理问题。要点分 built-in（周期性、平移对称、守恒律投影）、architecture（U-Net、skip）、learned 三栏。"local: convolutions only" 这种说法不用。

记号最简。不用 `\widehat`、`\widetilde`。记号跟 draft 一致。kernel 的时间间隔是 δt，ρ_n 的自变量是 time separation Δt。演化后的场写 φ_{δt}，不写 φ_{10}。不用 "Laboratory" 称格点模型。非序参量的守恒荷不用 M。引用别人的报告写 "topic: \alert{Name's talk}"，不放脚注。没有别人报告可引的 seminar 不写 "this talk:" 标注，直接写内容。

## 模板

参照 `ERG2026_PINNfRG.tex`。metropolis，class 选项 `[11pt,t]`，xelatex 编译两遍，页底引用用 `\refs{}`。有四个坑。

- `\begin{frame}{Title}` 后紧跟 `{...}` 会被吞成 subtitle，正文消失。
- class 选项 t 下 columns 要写 `[T]`。
- 带参数的 `\newcommand` 放在 frame 外。
- `\onslide` 状态会从左栏漏到右栏，右栏开头用 `\onslide<1->` 复位。
