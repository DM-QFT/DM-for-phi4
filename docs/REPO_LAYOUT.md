# DM 仓库目录规范

本文取代 `docs/PROJECT_STRUCTURE.md`（那份只描述现状，不构成约束）。
本文是**约束**：新建任何文件前先对照第 4 节的清单。

---

## 0. 现状诊断（2026-08-05 实测）

| 问题 | 量化 |
|---|---|
| core dump 堆在根目录 | 9 个文件，**34 GB** |
| 集群 stdout 日志散落各处 | 32 个 `*.o[0-9]*` |
| 根目录混放共享代码与垃圾 | 8 个 `.py` + 9 个 core + 2 个 `.o` + `texput.log` |
| 三套项目布局并存 | `ModelB`（最规整）/ `4Dphi4`（接近）/ `2Dphi4`,`3Dphi4`（脚本堆在顶层，各 20+ 个） |
| 分析代码与产物同目录 | `3Dphi4/analysis/` 里 `.py` 和 `.npz`/`.pdf` 混放；`2Dphi4/analysis` 与 `analysis_clean` 并存且重复 |
| 命名不统一 | `trainingdata` vs `training_data`；目录名带空格 `physical observable` |
| 论文源分散 5 处 | `draft/`, `overleaf/`, `ModelB/draft/`, `PIKDM/draft/`, `betaconfig/draft/` |
| 归档分散 4 处 | `archive/`, `ModelB_obsolete_20260718/`, `docs/archive/`, `PIKDM/archive/` |

**迁移成本约束（决定了规范怎么定）**

- 173 个文件硬编码 `trainingdata`，223 个文件硬编码 `runs/`。
- `sys.path.append` 出现 100+ 次，其中 `parents[1]` / `parents[2]` / `"../.."` 这类**对目录深度敏感**的写法占多数。把一个脚本上移或下移一层就会静默 import 错模块。

所以本规范**就多数、不就理想**：凡是已被大量硬编码的名字（`runs/`, `trainingdata/`, `analysis/`, `results/`）一律定为标准名，反过来改少数派。分析脚本一律停在 `<项目>/analysis/<主题>/` 这一深度（即 `parents[2]` = 仓库根），不再新增层级。

---

## 1. 核心原则：三分法

每个文件只属于三类之一，**三类不得同目录**。

| 类别 | 特征 | 入库 | 位置 |
|---|---|---|---|
| **代码** | 人写的、小的、可读的 | 是 | `analysis/`, `train/`… 只放 `.py` `.jl` `.sh` `.md` |
| **产物** | 机器生成、大、可重跑复现 | 否 | `runs/`, `results/`, `logs/`, `trainingdata/` |
| **交付** | 进论文的图和表 | 是 | `paper/figures/`, `paper/tables/` |

一条判据：**这个文件删掉后能不能用一条命令重新生成？** 能，就是产物，进 `runs/` 或 `results/`，写进 `.gitignore`。不能，就是代码或交付，入库。

推论（这几条是实际最常被违反的）：

- 分析脚本目录里不许出现 `.npz` `.pdf` `.png` `.csv`。产物写到对应的 `results/<主题>/`。
- `runs/` 里不许出现 `.py`。
- 论文图不是从 `runs/` 里直接引用，而是由 `analysis/` 脚本导出到 `results/`，再挑选进 `paper/figures/`。

---

## 2. 顶层布局

```
DM/
├── CLAUDE.md  AGENTS.md  README.md  requirements.txt  .gitignore
├── common/              # 全项目共享的模型代码（原根目录散落的 .py）
│   ├── data.py  diffusion_lightning.py  edm_lightning.py
│   ├── networks.py  networks_nd.py  networks_edm2.py  mp_layers.py
│   └── phi4_action.py
├── 2Dphi4/  3Dphi4/  4Dphi4/  ModelB/        # 物理项目，布局见第 3 节
├── PIKDM/  Yukawa/  betaconfig/  RGDM/       # 同上
├── paper/               # 所有论文与幻灯片源
│   ├── DM/              # 主论文（原 draft/）
│   ├── ModelB/  PIKDM/  betaconfig/
│   └── overleaf/        # Overleaf 独立 git checkout，不入主库
├── docs/                # 方法笔记、本规范。只放 .md
├── references/          # 第三方实现与文献，不入库
├── tools/               # 跨项目小工具（原根 scripts/, julia/, mathematica/）
└── attic/               # 唯一的归档区，见第 5 节
```

**`common/` 的处理方式**：根目录的共享模块必须搬走，但 100+ 处 `sys.path` 计数依赖它们在仓库根可见。做法是保留根目录的一行 shim（`from common.networks_nd import *`），或在 `common/` 建立后于根放同名符号链接。二选一，不要两个都做。这是全套迁移里唯一需要碰代码的部分，见第 6 节 Tier 3。

---

## 3. 项目内统一布局

每个物理项目一律长这样。没有的目录就不建，但**不允许换名**。

```
<项目>/
├── README.md            # 必需。入口命令、当前参数、结论状态
├── train/               # 训练入口 train_*.py
├── sample/              # 采样入口 sample_*.py
├── solver/              # 物理求解器（Julia/MC）与数据生成
├── analysis/            # 分析代码，按主题分子目录。只放代码
│   ├── propagator/      #   analysis/<主题>/ 与 results/<主题>/ 一一镜像
│   ├── calibration/
│   └── ...
├── tests/               # 正确性验证（CLAUDE.md 明确要求保留的那一类）
├── jobs/                # .pbs 与配套 *_worker.sh
├── trainingdata/        # 训练输入数据          （产物，忽略）
├── runs/                # 训练与采样输出         （产物，忽略）
├── results/             # 分析产物，镜像 analysis （产物，忽略，白名单例外）
└── logs/                # 集群 stdout/stderr    （产物，忽略）
```

**`runs/<run 名>/` 内部固定为：**

```
runs/<run名>/
├── models/       # *.ckpt
├── data/         # 该 run 生成的样本
└── meta.json     # 训练参数快照（可选，但强烈建议）
```

**镜像规则**：`analysis/propagator/export_prop_to_dat.py` 的输出一律落在
`results/propagator/`。看到 `results/` 下某个主题目录，就知道去
`analysis/` 下同名目录找生成它的代码。这条替代掉现在
`2Dphi4/analysis/dat/`、`3Dphi4/analysis/dat/`、`2Dphi4/acceptance_rate/dat/`
三处各行其是的 `dat` 目录。

### 3.1 三条硬规则（子目录混乱的真正病根）

实测下来，子目录乱不是因为层数多，而是因为下面三件事没约束。

**规则 A — 保留目录名全项目唯一，只许出现在指定位置。**

`results` `data` `dat` `figures` `logs` `models` `samples` 这些名字语义太通用，一旦允许在任意深度出现，就没人找得到东西。实测 `2Dphi4` 里 `results` 出现了 **5 次**，深度各不相同：

```
2Dphi4/results/                                  ← 项目级
2Dphi4/analysis/results/                         ← 又一个
2Dphi4/acceptance_rate/results/                  ← 又一个
2Dphi4/weight_scaling/results/                   ← 又一个
2Dphi4/analysis_clean/score_quality/results/dat/ ← 还嵌了一层 dat
```

规范：这些名字的**合法位置只有一个**，出现在别处即为违规。

| 目录名 | 唯一合法位置 | 含义 |
|---|---|---|
| `results/` | `<项目>/results/` | 分析产物根。**只此一处** |
| `runs/` | `<项目>/runs/` | 训练输出根 |
| `logs/` | `<项目>/logs/` 或 `<项目>/runs/<run名>/logs/` | 前者是集群 stdout/stderr，后者是该 run 自己的训练日志 |
| `trainingdata/` | `<项目>/trainingdata/` | 训练输入 |
| `models/` | `<项目>/runs/<run名>/models/` | checkpoint |
| `samples/` | `<项目>/runs/<run名>/samples/` | 该 run 生成的构型 |
| `figures/` | `paper/<论文>/figures/` | 只有论文图用这个名 |
| `dat/` | **废止** | 一律并入 `results/<主题>/` |
| `data/` | **废止** | 输入用 `trainingdata/`，输出用 `samples/` |

`data/` 与 `dat/` 是重灾区，语义完全靠上下文猜。现在 `runs/<run>/data/` 存的是采样构型，改叫 `samples/`；`analysis/dat/` 存的是导出表格，改叫 `results/<主题>/`。

**规则 A 附则 — 项目自有的库模块留在项目根，不下沉。**

`train/` `sample/` `analysis/` `solver/` `tests/` 装的是**入口脚本**，被人直接调用。项目自己的库模块（4Dphi4 的 `networks_4d.py`、`networks_nd.py`、`diffusion_lightning.py`、`phi4_action.py`，以及 `Project.toml`、`requirements.txt`）不属于任何一类，留在项目根。

理由是所有入口脚本靠 `sys.path.insert(0, 项目根)` 找它们。库模块一下沉，每个脚本都要多算一层，而入口脚本本来就已经因为进 `train/` 多算了一层。

同理，`analysis/<主题>/` 的主题子目录**只在分属多个主题时才建**。像 betaconfig 那样整个项目就一个主题、5 个脚本，直接平铺在 `analysis/` 下，不要为了对称硬造一层。

**规则 A 附则二 — 场数据的磁盘布局全库统一为 `[..., ensemble, x, y]`，时间序列合并成单个文件。**

场数组在磁盘上只有一种下标顺序：`[ensemble, x, y]`，时间序列前面再加一维成 `[time, ensemble, x, y]`。读取端**不许按路径名猜转置**。

这条不是洁癖。ModelB 里两族 Model B 参照曾经一族是 `[ens,y,x]`、另一族是 `[ens,x,y]`，读取端靠 `"physical_reference_t200" in path.parts` 区分。x↔y 是格点对称，所以标量观测量完全看不出差别（实测差 1e-15），只有对角模分解会错——这个 bug 因此长期没被发现。

Julia 侧是列主序，写盘时必须 `permutedims(phi_gen, (3, 2, 1))`，h5py 读出来才直接是 `[ensemble, x, y]`。写成 `(2, 3, 1)` 会得到 `[ensemble, y, x]`，正是上面那个 bug 的来源。改动这类代码后用往返测试验证一次：造一个 `f[e,x,y] = 100e + 10x + y` 的小数组写盘再读，检查 `h5[1,2,3] == 234`。

**时间序列不要逐时刻存盘。** 一条链、一组参照就是一个文件，时间作为第一维，另存 `times` 数组。ModelB 曾经每个时间点一个 npz 加一个 json sidecar，1491 个文件 6.8 GB；合并后 28 个文件 4.8 GB（省下的是可由相邻差还原的 `dphi`）。

格式用 HDF5 不用 npz：分析按时间点循环取，npz 每次访问成员都要整体解压，HDF5 按时间分块可以惰性切片。

**规则 B — 人写的目录最多 3 层，机器生成的不限。**

```
<项目>/analysis/propagator/          ← 3 层，合法，到此为止
<项目>/analysis/propagator/julia/    ← 4 层，违规
```

这不是洁癖。100+ 处 `sys.path.append(parents[N])` 对深度敏感，第 4 层一进来就得改 import。

例外是**机器生成的树**，规范不管它们的内部深度：`runs/`、`trainingdata/`、`lightning_logs/`、`references/`。实测 `3Dphi4` 的 252 个 ≥4 层目录**全部**来自 `runs/*/lightning_logs/version_N/`，`ModelB` 最深的 7 层来自 `references/*/.git/`。也就是说，真正归人管的部分现在其实已经在 3 层以内，规则 B 是把它钉死，不是要你去削 runs。

同一主题下如果确实需要区分语言（Python 与 Julia 并存），用文件名后缀而不是子目录：
`analysis/observables/hmc_table.py` 与 `analysis/observables/hmc_table.jl` 平级放。

**规则 C — `runs/<run名>/` 内部只许有这五项，分析产物一律不进 run 目录。**

```
runs/<run名>/
├── models/          # *.ckpt
├── samples/         # 该 run 生成的构型
├── logs/            # 该 run 的训练日志
├── lightning_logs/  # 框架自建，不管
└── meta.json        # 训练参数快照（强烈建议）
```

现在 run 目录里混着大量分析产物，这些**全部**该搬到 `results/<主题>/<run名>/`：

```
ModelB/runs/<run>/t200_ep10000_analysis/          → results/t200/<run>/
ModelB/runs/<run>/critical_correlation_diagnosis/ → results/correlation/<run>/
ModelB/runs/<run>/modelb_reference_dt0p5/         → results/reference/<run>/
2Dphi4/runs/<run>/mathematica_data/               → results/<主题>/<run>/
2Dphi4/runs/<run>/data_mh_tail/                   → results/mh_tail/<run>/
```

判据很干净：**训练脚本写的东西留在 run 里，分析脚本写的东西出去。** 这样删掉整个 `results/` 重跑分析不会碰到 checkpoint，反过来重训也不会冲掉分析结论。

### 3.2 `results/` 内部结构

```
results/
└── <主题>/              # 与 analysis/<主题>/ 同名，一一对应
    └── <run名>/         # 可选。产物跨多个 run 时才分，否则直接平铺
        ├── *.dat *.csv *.tsv *.json
        └── *.pdf *.png
```

最多两层。第二层只在"同一分析跑了多个 run"时才建，且必须用完整 run 名，不许简写。

**主题词表**（`analysis/<主题>` 与 `results/<主题>` 共用，现有的词已经不错，此处固化。新增主题前先确认不能归入已有的）：

`propagator` `observables` `calibration` `acceptance` `score_quality`
`ess` `cumulants` `non_conservative` `edm` `sigma_ablation` `weight_scaling`

**取代关系**（这些是现存目录，按此归位，不是新建）：

| 现在 | 归到 |
|---|---|
| `2Dphi4/analysis_clean/` | 成为唯一的 `analysis/`，旧 `analysis/` 进 `attic/` |
| `2Dphi4/acceptance_rate/`, `weight_scaling/`, `validation/` | `analysis/<同名主题>/` + `results/<同名主题>/` |
| `2Dphi4/*.py`（顶层 20+ 个） | 按前缀进 `train/` `sample/`，其余进 `analysis/` |
| `2Dphi4/scripts/`, `3Dphi4/scripts/` | `.pbs`/`.sh` 进 `jobs/`，`.py` 进 `analysis/` 或 `tools/` |
| `ModelB/training_data/` | 改名 `trainingdata/`（只 7 个文件引用） |
| `ModelB/checks/`, `ModelB/validation/` | 合并为 `tests/` |
| `ModelB/data_generation/` + `ModelB/solver/` | 合并为 `solver/` |
| `ModelB/draft/`, `ModelB/slides/` | 已完成 → `paper/ModelB/` |
| `ModelB/references/` | 根 `references/ModelB/` |
| 各项目 `julia/`, `mathematica/` | 保留在项目内，属于 `solver/` 或 `analysis/` |
| `2Dphi4/analysis/results/`, `acceptance_rate/results/`, `weight_scaling/results/`, `analysis_clean/score_quality/results/` | 全部并入唯一的 `2Dphi4/results/<主题>/`（规则 A） |
| `2Dphi4/analysis/dat/`, `acceptance_rate/dat/`, `3Dphi4/analysis/dat/` | 并入 `results/<主题>/`，`dat` 名废止 |
| `2Dphi4/analysis_clean/score_quality/test/` | `tests/score_quality/` |
| `2Dphi4/analysis_clean/physical observable/julia/` | 升到 `analysis/observables/`，文件名带 `.jl` 区分（规则 B） |
| `2Dphi4/mathematica/img/` | `results/<主题>/` |
| `runs/<run>/data/` | 改名 `runs/<run>/samples/` |
| `runs/<run>/` 内的分析产物目录 | `results/<主题>/<run名>/`（规则 C，清单见 3.1） |
| `2Dphi4/analysis/compare_kappa_scaling_L128/` | 产物目录混进代码区 → `results/propagator/` |

---

## 4. 命名规范

**目录名**：`snake_case`，**不含空格**。（现存违例只有一个：`2Dphi4/analysis_clean/physical observable` → `observables`。）

大写字母允许，且不该消灭：`L128`、`Lmulti8-16-32-64`、`ESS`、`HMC`、`N2048` 是物理符号和公认缩写，和 CLAUDE.md「变量名用物理符号」是同一条原则。审计脚本只查空格，不查大小写。

**脚本名**：`<动词>_<对象>[_<限定>].py`。动词只用这几个，一个脚本一件事：

`train_` `sample_` `analyze_` `compare_` `export_` `plot_` `check_` `calibrate_` `run_`

**run 目录名**：字段顺序固定，缺省字段直接省略，不留占位。

```
<模型>[_<维度>]_L<尺寸>_k<kappa>_l<lambda>[_t<演化时间>]_<网络>[_sigma<σ>][_<标签>]
```

现有命名已经符合，本条是把它固化：

```
phi4_L128_k0.28_l0.022_ncsnpp_sigma640
phi4_3d_Lmulti4-8-16-32_k0.18_l0.9_ncsnpp_sigma200
modelBcond_L32_k0.34034_l1.0_t10_ncsnpp_hmc_noskip
```

多尺寸写 `Lmulti8-16-32-64`。小数点保留（`k0.2705`），文件名里需要避开小数点时统一写 `dt0p002` 这种 `p` 替换，两种写法不要混用在同一层。

**产物文件名**：跟着生成它的 run 名走，便于反查。
`prop_<run名>_ep<epoch>.dat`、`acceptance_vs_epoch_<run名>.pdf`。

**日期**：一律 ISO `YYYY-MM-DD`。归档目录写 `attic/2026-07-18_ModelB_obsolete/`，不写 `ModelB_obsolete_20260718`。

**集群日志**：PBS 脚本必须写 `#PBS -o logs/` 和 `#PBS -e logs/`，并在脚本头加 `ulimit -c 0`。这两条直接消灭本文第 0 节的前两行问题。

---

## 5. 归档与删除

`attic/` 是全库唯一归档区，布局 `attic/<日期>_<原路径>/`。进入 `attic/` 的东西默认不入库、不维护、随时可删。

判据：

- **删**：core dump、`*.o[0-9]*`、`texput.log`、`__pycache__`、可一条命令重跑的中间产物。不进 `attic`，直接删。
- **归档**：曾经产出过结论、但现在不再跑的代码和数据。进 `attic/`。
- **留**：论文引用到的，或 README 里作为入口列出的。

现存的 `archive/`、`ModelB_obsolete_20260718/`、`docs/archive/`、`PIKDM/archive/` 全部并入 `attic/`。

---

## 6. 迁移分级

按风险从低到高分四级。每级独立可停，不要跨级混做。

**Tier 0 — 已完成（2026-08-05）。零风险，纯删除，回收 34 GB**
9 个 core dump、32 个 `*.o[0-9]*`、3 个 `texput.log`、21 个 `__pycache__`。全部未被 git 追踪，`git status` 条目数删前删后一致。

**Tier 1 — 已完成（2026-08-05）。低风险，需改路径引用**
建 `attic/`、`paper/`、`tools/` 并把归档类、论文类、根级零散工具搬进去。

原以为零风险，实际不是。执行时发现三点：

- `draft/` 被 18 个文件共 28 行以 `../draft/figures/` 形式引用，已随移动改写为 `paper/DM/`。
- `ModelB/draft` 是 gitlink（mode 160000，无 `.gitmodules`），`betaconfig/draft` 是 Overleaf clone，`PIKDM/draft` 与 `4Dphi4` 也是独立嵌套仓库。这些必须 `git mv`（或整目录 `mv`）搬，不能拆开，搬完要复查各自 HEAD 未变。
- `ModelB_obsolete_20260718` 有 50 个文件被主库追踪。`git mv` 保留追踪，`.gitignore` 加 `/attic/` 只挡新增，不会把它们变成未追踪。这是刻意的：是否取消追踪属于另一个决定。

**教训**：凡是要移动目录，先跑一遍
`grep -rn --include="*.py" --include="*.sh" --include="*.jl" "<目录名>/" .`
和 `find . -maxdepth 3 -name .git`，再动手。

**Tier 2（低风险，需 grep 核对）**
项目内目录归位：`analysis_clean` → `analysis`，`training_data` → `trainingdata`，`scripts` → `jobs`，空格目录改名。每一步先 `grep -rl "<旧名>"` 列出引用文件，改完再 grep 一次确认为零。

**Tier 3（有风险，必须逐个跑通验证）**
根共享模块搬进 `common/`。100+ 处 `sys.path.append(parents[N])` 对深度敏感，且 `sys.path.append("..")` 这种写法在改变 CWD 后行为不同。做法：先加 `common/` 与根 shim 并存，确认全部脚本能 import，再删 shim。**这一级建议单独一个 commit，并在合并前至少跑一个 train 和一个 analysis 脚本。**

分析脚本的深度在本规范下不变（都是 `<项目>/analysis/<主题>/`），所以 Tier 2 不触发 `parents[N]` 问题。这是刻意设计的。

---

## 7. 新建文件或目录前的检查清单

1. 它是代码、产物、还是交付？三类不同目录（第 1 节）。
2. 目录名撞上保留名了吗？`results` `data` `dat` `figures` `logs` `models` `samples` 只许在唯一合法位置出现（规则 A）。
3. 建完是第几层？人写的目录不超过 3 层（规则 B）。想建第 4 层时，先问能不能用文件名后缀代替。
4. 如果是分析脚本，主题在词表里吗（3.2）？对应的 `results/<同名>/` 呢？
5. 如果输出会落进 `runs/<run名>/`，它是训练脚本写的吗？不是就该去 `results/`（规则 C）。
6. 文件名的动词前缀在第 4 节列表里吗？
7. 如果是 PBS 脚本，写了 `-o logs/` 和 `ulimit -c 0` 吗？
8. 这件事已经有脚本在做了吗？（AGENTS.md 的复用规则：能改参数复用的，不许另起一套。）

---

## 8. 逐目录整理流程

一次只整理一个项目，做完验证再开下一个。每个项目按下面六步走，顺序不要换：**先建骨架、再搬代码、最后搬产物**。产物最多、引用最少，放最后搬风险最低。

**第 1 步 — 审计。** 在项目目录下跑第 8.1 节的脚本，得到违规清单。先看，不动手。

**第 2 步 — 建骨架。** `mkdir -p analysis train sample solver tests jobs results logs`。只建这个项目真正会用到的，空目录不留。

**第 3 步 — 搬代码。** 顶层散落的 `train_*.py` → `train/`，`sample_*.py` → `sample/`，`.pbs`/`*_worker.sh` → `jobs/`，其余 `.py` → `analysis/<主题>/`。

> 这一步唯一的风险是 `sys.path` 深度。顶层脚本从 `<项目>/` 移到 `<项目>/train/` 会**加深一层**，脚本里的 `sys.path.append("..")` 要改成 `"../.."`，`parents[1]` 要改成 `parents[2]`。搬之前先
> `grep -n "sys.path" <文件>`，逐个对着改。这是整个整理过程中最容易静默出错的地方——import 错模块不报错，只是行为变了。

**第 4 步 — 归并保留名目录。** 把该项目里所有多余的 `results/`、`dat/` 合并到唯一的 `<项目>/results/<主题>/`。合并前先确认没有同名文件冲突。

**第 5 步 — 清 run 目录。** 把 `runs/<run名>/` 里非训练产出的子目录搬到 `results/<主题>/<run名>/`，`data/` 改名 `samples/`。

**第 6 步 — 复验。** 重跑第 1 步的审计脚本，应无输出。然后**实际跑通一个分析脚本和一个训练脚本的 `--help`**，确认 import 没断。写 README 记录本次归位的对照表。

### 8.1 审计脚本

放在 `tools/audit_layout.sh`，用法 `bash tools/audit_layout.sh 2Dphi4`。

```bash
#!/bin/bash
# 检查一个项目是否符合 docs/REPO_LAYOUT.md。无输出即合规。
P="${1:?用法: audit_layout.sh <项目目录>}"
# runs/ trainingdata/ lightning_logs/ references/ 非本人维护，规范不管其内部
EXCLUDE='-path */runs/* -o -path */trainingdata/* -o -path */lightning_logs/* -o -path */references/* -o -path */.git/*'

echo "== 规则 A: 保留名出现在非法位置 =="
find "$P" \( $EXCLUDE \) -prune -o -type d \
  \( -name results -o -name dat -o -name data -o -name figures -o -name logs \) -print \
  | grep -vE "^$P/(results|logs)$"

echo "== 规则 B: 人写的目录超过 3 层 =="
find "$P" \( $EXCLUDE -o -path "$P/results/*" \) -prune -o -type d -print \
  | awk -F/ 'NF>3'

echo "== 规则 C: run 目录内的非法子目录 =="
find "$P/runs" -mindepth 2 -maxdepth 2 -type d 2>/dev/null \
  | grep -vE '/(models|samples|logs|lightning_logs)$'

# 只查空格。大写不查: L128、Lmulti、ESS、HMC 是物理符号和缩写，本就该保留
echo "== 目录名含空格 =="
find "$P" \( $EXCLUDE \) -prune -o -type d -print | grep ' '

echo "== 代码目录里混入产物 =="
find "$P"/{analysis,train,sample,solver,tests} -type f \
  \( -name '*.npz' -o -name '*.npy' -o -name '*.pdf' -o -name '*.png' -o -name '*.dat' -o -name '*.csv' \) 2>/dev/null
```

完整可执行版本在 `tools/audit_layout.sh`，上面是它的内容摘录。

### 8.2 建议的整理顺序

按违规量从少到多，先在小项目上把流程跑顺：

| 顺序 | 项目 | 状况 |
|---|---|---|
| 1 | `betaconfig` | 最深才 2 层，只需分 `analysis/` 与 `results/` |
| 2 | `4Dphi4` | 已接近规范，主要是 `trainingdata`/`scripts` 归位 |
| 3 | `Yukawa`, `RGDM`, `PIKDM` | 体量小 |
| 4 | `ModelB` | 结构最好但保留名冲突多，run 目录里塞了大量分析产物 |
| 5 | `2Dphi4` | `results` 出现 5 次，顶层 20+ 散落脚本 |
| 6 | `3Dphi4` | 同上，且 `analysis/` 里代码与 `.npz`/`.pdf` 混放最严重 |

---

## 9. 新建项目的模板

新开一个物理课题时，从这个骨架起步，不要从复制旧项目起步（复制旧项目就是把旧项目的历史包袱一起复制过来）。

```bash
P=NewProject
mkdir -p $P/{analysis,solver,tests,jobs,results,logs}
cat > $P/README.md <<'EOF'
# <项目名>

一句话说明这个项目在算什么物理。

## 入口

| 做什么 | 命令 |
|---|---|
| 生成数据 | `julia solver/run.jl` |
| 训练 | `python train/train_xxx.py --L 32 --k 0.28` |
| 采样 | `python sample/sample_xxx.py --ckpt runs/<run名>/models/epoch=10000.ckpt` |

## 参数

kappa / lambda / sigma 的当前取值与选取依据。

## 状态

已确认的结论，以及正在查的问题。
EOF
```

README 是必需的，且只写这三段。它的作用是让三个月后的自己知道从哪个命令开始，不是写文档。

新项目从第一天就遵守规则 A/B/C，代价接近零。等到积累了两千个文件再整理，代价就是本文第 6 节那四级迁移。
