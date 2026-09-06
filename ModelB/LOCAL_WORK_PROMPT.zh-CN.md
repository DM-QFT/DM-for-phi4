# ModelB 本地作图与文章修改提示词

请把本文件交给本地 Codex 或其他编程助手，并要求它在处理本项目时遵守以下内容。

## 当前任务

你正在处理二维守恒序参量动力学 Model B 及其条件扩散模型代理。

当前工作的重点只有两项。

1. 修改 draft/modelb.tex 和 draft/references.bib。
2. 修改并重画论文正文 Fig. 1、Fig. 2、Fig. 3、Fig. 4。

除非用户明确要求，不运行训练、扩散模型生成、Model B 数值积分或新的大规模分析。不要尝试补下载缺失的训练集、检查点和完整生成链。

压缩包是为本地写作准备的精简副本。它没有 Git 历史，也不包含全部原始数据。磁盘中的当前文件就是工作基准。不要因为没有 .git 而覆盖用户已经修改的内容。

## 项目位置

不要假定 Miyabi 上的绝对路径。先从当前目录定位项目根目录。项目根目录包含 README.md、draft、analysis、results、runs 和 trainingdata。

解压后的目录应保持下面的关系。

~~~text
解压目录
├── ModelB
└── plots
    ├── cjp.mplstyles
    ├── cjp_prl_nature.mplstyle
    └── modelb_palette.py
~~~

plots 必须与 ModelB 并列。现有作图脚本按照这个相对位置读取论文样式。

## 工作原则

- 优先做完成当前修改所需的最小改动。
- 使用现有函数、数组和脚本。不要新增 class、配置系统、通用命令行接口或平行分析流程。
- 不要为了本地兼容而改变物理定义、拟合窗口、随机种子、样本选择或误差估计。
- 用户只要求版式调整时，不要改变数值结果。
- 当前 PRL 图使用 plots/cjp_prl_nature.mplstyle 和 plots/modelb_palette.py。旧图仍可使用 plots/cjp.mplstyles。
- APS 单栏图宽为 3.375 inch。通栏图宽为 7.0 inch。
- 图在 LaTeX 中按 columnwidth 插入，不再额外缩放字体。
- 正文优先使用直接的物理语言。避免含糊的机器学习术语。
- 修改前先读相关脚本和输入文件。修改后只运行对应图，不运行全部分析入口。
- 不创建长期保留的一次性脚本。若确实需要临时文件，验证后删除。

## 论文图的唯一工作目录

当前论文图统一放在 draft/figures/prl。draft/figures 根目录中曾经存在的同名旧版本已经移到 draft/figures/archive_20260826_root_duplicates。不要从归档目录插图。

所有论文图都由现有 Python 代码生成。不要解析或重写已有 PDF。

- analysis/plot_schematic_framework.py 生成流程示意图。
- analysis/plot_validation_L32.py 生成长链验证图。
- analysis/dm_vs_sosra/plots_prl.py 生成临界标度、有效衰减率和 L 等于 64 的色散图。色散图使用 plot_reverse_nfe_tau 和完整缓存。
- analysis/plot_coarsening_dm.py 生成粗化标度和结构因子图。
- analysis/compare_multil_reverse.py 生成一步转移和 L 等于 32 的长链关联图。

## 本地 Python 环境

作图需要 Python 以及 numpy、scipy、matplotlib、h5py。

先检查现有环境。

~~~bash
cd ModelB
python -c "import numpy, scipy, matplotlib, h5py"
~~~

如果没有可用环境，可以建立项目内虚拟环境。

~~~bash
cd ModelB
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib h5py
~~~

Windows PowerShell 使用下面的激活命令。

~~~powershell
.venv\Scripts\Activate.ps1
~~~

Matplotlib 样式关闭了 text.usetex。重画 PDF 不要求安装 LaTeX。编译论文仍需要包含 revtex4-2 的 TeX 环境。

完整的图形规范和逐图修改说明位于 analysis/figure_revision_guide.md。

## 正文 Fig. 1

图文件为 draft/figures/prl/schematic.pdf。

作图脚本为 analysis/plot_schematic_framework.py。脚本读取归档的场快照并直接生成最终图，不经过 PowerPoint。

## 正文 Fig. 2

图文件为 draft/figures/prl/validation_L32.pdf。

作图脚本为 analysis/plot_validation_L32.py。

主要输入位于下面的目录。

~~~text
results/dm_vs_modelb/modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_corrgauss_zlog_noskip
~~~

其中 hmc_chain.npz 和 gaussian_chain.npz 已经保存图中使用的时间序列、误差和快照。TSV 文件保存对角模、功率和观测量结果。不需要原始 HDF5 生成链。

从 ModelB 根目录运行。

~~~bash
python analysis/plot_validation_L32.py
~~~

脚本会更新下面的文件。

~~~text
draft/figures/prl/validation_L32.pdf
~~~

## 正文 Fig. 3

图文件为 draft/figures/prl/collapse_dm_L32.pdf。

分析和作图函数位于下面三个文件。

~~~text
analysis/dm_vs_sosra/core.py
analysis/dm_vs_sosra/run_all.py
analysis/dm_vs_sosra/plots_prl.py
~~~

本地包已经保存三份逐轨道低模缓存。

~~~text
results/multiL/analysis/dm_vs_sosra/cache/L32_sosra.npz
results/multiL/analysis/dm_vs_sosra/cache/L32_dm_t10.npz
results/multiL/analysis/dm_vs_sosra/cache/L32_dm_t0p5.npz
~~~

缓存包含前四个对角模、磁化和作用量密度。误差分析、bootstrap、弛豫时间和标度指数仍可从逐轨道数据重新计算。HMC 参照初态也已经保留。

不要直接运行完整的 analysis/dm_vs_sosra/plots_prl.py。它的主程序还会请求未打包的 L64 和反向采样数据。

在 zsh 或 bash 中，从 ModelB 根目录只重画正文 Fig. 3。

~~~bash
python - <<'PY'
from pathlib import Path
import sys

module_dir = Path("analysis/dm_vs_sosra").resolve()
sys.path.insert(0, str(module_dir))

import plots_prl
import run_all

result = run_all.analyze(32)
output = Path("draft/figures/prl")
plots_prl.plot_collapse(32, "dm", result, output_dir=output)
print(output / "collapse_dm_L32.pdf")
PY
~~~

参考数值应保持为下面的结果。

~~~text
z_DM = 3.792(19)
z_Model B = 3.764(36)
~~~

如果这些数值在纯版式修改后发生变化，停止并检查代码和数据选择。

## 正文 Fig. 4

图文件为 draft/figures/prl/collapse_dm_L128.pdf。

作图脚本为 analysis/plot_coarsening_dm.py。

原始 L 等于 128 的粗化链已经保留。

~~~text
runs/modelBcond_Lmulti8-16-32-64_k0.85_l1.0_t10_ncsnpp_quench_tanhgauss/samples/quench_chain_ep10000_L128_N32_t1000.h5
~~~

从 ModelB 根目录运行。

~~~bash
python analysis/plot_coarsening_dm.py
~~~

脚本会更新正文 Fig. 4，也会更新补充材料中的结构因子图。

~~~text
draft/figures/prl/collapse_dm_L128.pdf
draft/figures/prl/sk_collapse_dm_L128.pdf
~~~

## 修改和编译文章

正文入口为 draft/modelb.tex。参考文献数据库为 draft/references.bib。现有 PDF 为 draft/modelb.pdf。

如果本地安装了 latexmk，可以运行下面的命令。

~~~bash
cd draft
latexmk -pdf modelb.tex
~~~

如果只有 pdflatex 和 bibtex，可以运行下面的顺序。

~~~bash
cd draft
pdflatex modelb.tex
bibtex modelb
pdflatex modelb.tex
pdflatex modelb.tex
~~~

缺少 TeX 宏包时，优先安装缺失宏包。不要为了让编译通过而删除正文中的宏包、公式或引用。

修改正文时保持符号和图注一致。正文四幅图的对应关系如下。

1. Fig. 1 对应 schematic。
2. Fig. 2 对应 validation_L32。
3. Fig. 3 对应 collapse_dm_L32。
4. Fig. 4 对应 collapse_dm_L128。

## 每次修改后的检查

完成任务前执行与改动相称的检查。

1. 确认目标 PDF 已重新生成且文件大小非零。
2. 确认脚本没有读取缺失的大文件。
3. 若只改版式，确认论文中的关键数值没有变化。
4. 若修改 modelb.tex，至少完成一次 LaTeX 编译并检查未解析引用。
5. 报告修改了哪些文件、运行了哪些命令、生成了哪些 PDF。
6. 不删除压缩包中已有的数据和图片。

当用户提出新的修改要求时，先判断它属于文章、Fig. 1、Fig. 2、Fig. 3 或 Fig. 4。只打开该任务直接需要的文件，然后完成修改和验证。
