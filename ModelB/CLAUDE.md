以下规则针对论文正文，代码文件和笔记不受约束。

规则按优先级排列。冲突时以靠前的为准。

---

## 0. 不要碰的部分

在做任何风格清理前，先排除以下区域。

- 数学环境（`$...$`、`\begin{equation}` 等）
- `\cite{}`、`\ref{}`、`\label{}` 的参数
- `.bib` 文件
- 人名连接符与数字区间
---

## 1. 语域
主文本先讲物理。做了什么、为什么这么做、结果说明了什么。

技术实现、超参数、网络结构、训练细节一律放 End Matter 或 Supplemental Material。主文本里提到方法时，用一句物理语言概括它做的事，不展开机制。

不要用 ML 圈内的行话。需要用到技术概念时解释它，不要只报名字。

这是物理论文，不要写成技术报告。
---

## 2. 标点

正文中不用破折号、冒号、分号。

以下位置允许保留冒号。图表标题的编号格式、`\item` 的标签、表格表头、参考文献格式

---

## 3. 词汇

写完后逐个搜索下表，每一个都问"删掉会损失信息吗"。多数情况下不会。

**尽量不用**
delve, intricate, pivotal, crucial, comprehensive, robust（除非指统计意义上的稳健）, nuanced, leverage（动词）, underscore, showcase, realm, landscape, testament, myriad, seamless, foster, harness, shed light on, it is worth noting that, plays a crucial role in, in the realm of

**连接词**
Moreover / Furthermore / Additionally / Notably 在一篇里合计不超过三次。多数情况下直接开始下一句即可，逻辑关系应该由内容承担而不是由连接词标注。
---

## 4. 句式

- 不要用反转句式（"不是 X，而是 Y"、"X 不仅仅是 Y"）。
尽量不要出现下面的写法
> not only ... but also ...

> not merely ... but rather ...

> rather than simply ...

> while X, Y ...

> Although X, it is important to note that Y ...
这些结构容易制造人为对比。很多情况下，两个事实并不存在逻辑冲突。

- 三元并列（"fast, robust, and interpretable"）全文最多两次
- 段落最后一句不要总是总结句。有的段落说完就停
- 主动语态优先。被动语态只在动作主体不重要时使用
- 句长要有起伏。允许出现五个字的短句
- 避免高频使用空泛学术套话
---

## 5. 结构

- 不写路标句。删掉"In this section we present…"和"本节将介绍……"，直接进入内容
- Introduction 不套"领域重要 / 已有工作不足 / 我们做了什么"三段等长模板。按实际逻辑分配篇幅
- 结论不复述摘要。结论应包含摘要里没有的判断，比如这个结果对什么问题有影响、什么地方还不确定
- 主文本少用 bullet。需要论证的地方用连贯散文，列表会切碎推理链条
- 不用"首先……其次……最后"标注不需要编号的内容
- 避免每个段落长度近似，每句话结构完整，缺少自然的信息层级。这种均匀性使文章看起来流畅，却可能削弱重点。物理论文中的段落长度应由论证需要决定。关键结果可以单独成句，技术限定可以更短，核心推导可以更长。
- 尽量避免每段都以概括句开始并以意义句结束

典型结构如下

> We first introduce ...

> Specifically, ...

> This demonstrates ...

> These results highlight ...

这种结构偶尔使用没有问题。每段都采用同一模板时，文章会呈现机械感。


---

## 6. 内容纪律

这一层不是风格，是审稿人会直接质疑的地方。以下每条都不要替我猜，缺信息就明确问我。

**引用**

- 不要凭印象生成 `\cite` key。不确定就写 `\cite{TODO:描述}` 让我补
- 不用综述代替原始文献
- 采用inspire标准格式

**数字**

- 不写"显著改善"，写具体数值和不确定度
- 每个误差都要能说清来源（统计、系统、拟合）
- 有效位数和不确定度匹配。不写 `0.4271(3)` 除非确实做到了那个精度
- 不要生成我没提供的数值。

**符号与量纲**

- 同一物理量全文一个记号，下标含义不漂移
- 单位约定一致，自然单位和物理单位不混用

**论证**
- 结果到结论之间不跳步，中间的物理论证要写出来
- 对比不要人为对称。
- 允许有立场。不要平衡到读完不知道作者主张什么

**语义**
- 文章的正文需要防止语义重复
---

## 7. 本文已定的叙事、记号与措辞

**叙事与结构。** 目标 PRL。主线依次是自治系统，时齐 Markov kernel 由成对数据唯一确定且复合外推严格成立（误差只来自学习），Hohenberg–Halperin，Model B，VE-SDE，训练族外初态看相分离，粗化标度，临界标度，multi-L 延拓。Sec II 独立讲 kernel 与复合。结果部分不加标题和 subsection。End Matter 放胶体 Fig.5 和 wall-time 表。SM 在 `\bibliography` 之后，编号带 S 前缀。摘要不写数值。PRL 摘要上限 600 字符含空格，改摘要要数字符。2026-09-26 合作者重写后的版本是 1060 字符。

**记号。** 尽量与 arXiv:2607.08505 统一。configuration 或 field，不叫 state。Markovian。Δφ 叫 increment。η∼N(0,𝟙)。扩散时间 s，物理时间 t，弛豫时间 τ_n。直接积分统一叫 SOSRA，"Model B" 只指物理模型。不用 rollout，说迭代生成。"lag" 只零星用，ρ_n(Δt) 的自变量写 time separation Δt、grid points、retained range。

**数字。** 数字保持数字形式（About $1200$ tracks、$60$ min）。精简时每个数值要么整个删掉改成定性陈述，要么原样保留，不写成 twelve hundred。速度主张写 "five to a hundred times faster than general-purpose GPU SDE implementations and as fast as a hand-optimized SOSRA integrator"。尺寸迁移写 "twice the largest training size"。

**批注与文献。** 合作者的批注（`\jy`、`\YT` 等）不许删，经确认后才可以随改写移除。bib 用缩写期刊名。apsrev4-2 的 JHEP 条目要 volume=年份、number=期号。Fukushima:2024oij 只有两位作者，INSPIRE 多出的 Hirono 是录入错误。

Git 流程、各图的脚本、计算侧的经验在 `README.md`。