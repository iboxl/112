# CIM 加速器延迟解析模型

## 概述

本文档描述 MIP 求解器（`utils/SolverTSS.py`）中使用的延迟递归解析模型。该模型对 CIM（Compute-in-Memory）加速器上嵌套循环数据流的执行延迟进行闭式近似，用于在映射优化中替代周期级模拟器（`Simulator/Simulax.py`）作为快速评估手段。

模型由六层递推组成，从底层的 per-memory 传输代价向上递推到总执行延迟。每层具有明确的硬件对应关系。

---

## 符号定义

| 符号 | 定义 |
|------|------|
| $\lambda \in \{I, W, O\}$ | 操作数：输入（Input）、权重（Weight）、输出/部分和（Output） |
| $m$ | Memory 层级索引（DRAM, Global Buffer, Output Buffer, Input Buffer, OReg, IReg, Macro） |
| $i = 0, 1, \ldots, N{-}1$ | 时间循环层级（0 = 最外层，$N{-}1$ = 最内层） |
| $F(i)$ | Level $i$ 的循环因子（该层循环迭代次数，$\geq 2$） |
| $\text{dbl}(i, \lambda)$ | Level $i$ 操作数 $\lambda$ 是否启用双缓冲（1=是，0=否） |
| $\text{xMem}(i, \lambda)$ | Level $i$ 操作数 $\lambda$ 是否发生数据搬移（1=搬移，0=驻留/stationary） |
| $c(\lambda)$ | 读写系数：$c(O) = 2$（读 psum + 写回），$c(I) = c(W) = 1$（只读） |
| $\text{mem}(i, \lambda)$ | Level $i$ 为操作数 $\lambda$ 分配的 memory 层级 |
| $t_{\text{MAC}}$ | 单次 MAC 运算的计算延迟（时钟周期数） |

---

## 第零层（L0）：Per-Memory 传输代价

$$
\text{rawTrans}(m, \lambda) = \frac{\text{tileVolume}(m, \lambda) \times \text{precision}(m, \lambda)}{\text{bandwidth}(m)}
$$

$$
\text{Trans}(m, \lambda) = \max\!\big(\text{rawTrans}(m, \lambda),\ 1\ \text{cycle}\big)
$$

**硬件含义**：tile 数据在 memory $m$ 的总线上传输所需时钟周期数。`tileVolume` 是该层 memory 存储的一个 tile 的元素数量（由空间展开和时间循环决定）；`precision` 是每个元素的位宽；`bandwidth` 是总线每周期传输的位数。

**ceil 下界**：硬件总线按时钟周期计数，不存在亚周期传输。任何非零数据搬移至少占用 1 个时钟周期。

### Reg bypass 惩罚

$$
\text{bypass}(m, \lambda) = \big[m = \text{innermostMem}(\lambda)\big] \wedge \big[\text{Reg 被 bypass}\big]
$$

$$
\text{EffTrans}(m, \lambda) = \text{Trans}(m, \lambda) + \text{bypass}(m, \lambda) \times 1\ \text{cycle}
$$

**硬件含义**：CIM 阵列采用位串行计算，数据需要经过 Reg 层做串转并。当最内层循环因子为 1（无法显式映射到 Reg）时，Reg 被"bypass"，但数据仍需经过该路径，每次传输额外增加 1 个时钟周期。

将 bypass 惩罚嵌入 `EffTrans` 而非逐层判断，使所有映射到同一 memory 的层级自动继承该开销——因为 bypass 是 memory 路径的物理属性，与循环层级无关。

---

## 第一层（L1）：Per-Level 传输延迟

$$
\text{Transfer}(i, \lambda) = \text{xMem}(i, \lambda) \times \text{EffTrans}\!\big(\text{mem}(i, \lambda),\ \lambda\big)
$$

**含义**：level $i$ 每次循环迭代对操作数 $\lambda$ 的传输开销。

- $\text{xMem} = 1$：该层发生数据搬移（operand 的 memory 分配在此层变化），需要传输。
- $\text{xMem} = 0$：数据驻留（stationary），无传输开销。此时 $\text{Transfer} = 0$。

---

## 第二层（L2）：Body — 循环体执行时间

$$
\text{Body}(i) = \max_{\forall \lambda}\!
\begin{cases}
\text{Process}(i{+}1, \lambda) - \text{dbl}(i{+}1,\lambda) \times \text{Transfer}(i{+}1,\lambda) & \lambda \in \{I, W\} \\[4pt]
\text{Process}(i{+}1, \lambda) & \lambda = O
\end{cases}
$$

**硬件含义**：level $i$ 每次迭代的循环体执行时间，即子级（level $i{+}1$）完整执行的墙钟时间。

- **I/W 双缓冲扣减**：当 $\lambda \in \{I,W\}$ 在子级双缓冲时，其首迭代传输可与父级前一迭代的尾部执行流水线重叠。从 Body 中扣除该传输开销，避免重复计算。
- **O 不扣减**（保守近似）：O 的首迭代需要 pre-read 部分和累加器，该传输在关键路径上无法被重叠。保守地不扣减。

---

## 第三层（L3）：Critical — 单次稳态迭代瓶颈

$$
\text{Critical}(i) = \max\!
\begin{cases}
\text{Body}(i) \\[4pt]
c(\lambda) \times \text{Transfer}(i, \lambda) & \forall \lambda \\[4pt]
c(\lambda) \times \text{Transfer}(i, \lambda) + \text{Body}(i) & \forall \lambda\ \text{where}\ \neg\text{dbl}(i, \lambda)
\end{cases}
$$

**硬件含义**：level $i$ 每次稳态迭代（非首迭代）的瓶颈耗时。

- **双缓冲操作数**：第 $k$ 次迭代的传输与第 $k{-}1$ 次迭代的子级执行在时间上重叠。瓶颈为 $\max(\text{Transfer}, \text{Body})$。
- **单缓冲操作数**：传输完成后才能启动子级执行（端口独占），传输与子级串行。瓶颈为 $\text{Transfer} + \text{Body}$。

$c(O) = 2$ 体现 O 的读写双向传输（读入旧 psum + 写回新 psum）。

---

## 第四层（L4）：Process — 总执行时间

$$
\text{Process}(i, \lambda) = c(\lambda) \times \text{Transfer}(i, \lambda) + \text{Process}(i{+}1, \lambda) + \big(F(i) - 1\big) \times \text{Critical}(i)
$$

$$
\text{Process}(N, \lambda) = t_{\text{MAC}} \qquad \text{（基例：最内层 MAC 计算延迟）}
$$

**含义**：level $i$ 为操作数 $\lambda$ 执行全部 $F(i)$ 次迭代的总墙钟时间。

- **首迭代**（$c \times \text{Transfer} + \text{Process}_{i+1}$）：传输数据后启动子级执行，二者串行。
- **后续迭代**（$(F{-}1) \times \text{Critical}$）：每次迭代耗时由 Critical 决定（传输与子级执行根据缓冲模式并行或串行）。

**Hierarchical Decay 削减约束**：

$$
\text{Process}(i, \lambda) \geq 2 \times \text{Process}(i{+}1, \lambda)
$$

物理含义：任何循环因子 $F \geq 2$ 时，父级至少是子级的 2 倍。用于加速求解器收敛。

---

## 第五层（L5）：MaxStartup — 跨操作数流水线启动

$$
\text{BootstrapRead}(\lambda) = \frac{\text{totalSize}(\lambda) \times \text{precision}_{\text{DRAM}}}{\text{bandwidth}_{\text{DRAM}}}
$$

$$
\text{MaxStartup} = \max_{\forall \lambda}\!\Big\{\text{BootstrapRead}(\lambda) \times \mathbb{1}[\lambda\text{ 首层不在 DRAM}] + \sum_{i} \text{Transfer}(i, \lambda)\Big\}
$$

**硬件含义**：首次 MAC 执行前的流水线填充时间。需要：

1. **引导读入**（Bootstrap）：将 $\lambda$ 的完整数据从 DRAM 搬入首层片上 memory。若 $\lambda$ 直接映射到 DRAM（不经过片上 buffer），则无需引导。
2. **级联预读**（$\Sigma$ Transfer）：数据到达首层 memory 后，逐级 tile-by-tile 预读到最内层。

MaxStartup 取三个操作数中**最慢**者的启动时间——因为首次 MAC 必须等待 I、W、O 全部就绪。

---

## 第六层（L6）：总延迟

$$
\forall \lambda:\quad \text{Latency} \geq \text{MaxStartup} - \sum_{i}\text{Transfer}(i, \lambda) + \text{Process}(0, \lambda) + \text{BootstrapWrite}(O)
$$

**含义**：

- $\text{MaxStartup}$ 是所有操作数共同承受的启动等待。
- $-\sum \text{Transfer}(\lambda)$：扣除 $\lambda$ 自身的级联预读时间（已包含在 $\text{Process}$ 中，避免重复计算）。
- $\text{Process}(0, \lambda)$：$\lambda$ 的完整嵌套循环执行时间。
- $\text{BootstrapWrite}(O)$：仅 O 操作数需要在计算完成后将结果写回 DRAM。

总延迟取所有操作数中的**最大值**（$\geq$ 约束在最小化目标下等价于 max）。

---

## 与 Simulator 的对应关系

| 模型层级 | Simulator 对应代码 | 关键差异 |
|---------|-------------------|---------|
| L0 Trans | `math.ceil(tileSize*prec/bw)` + bypass +1 | 完全对齐 |
| L1 Transfer | `Cons[op] if xMem else 0` | 完全对齐 |
| L2 Body | 隐式（子级递归调用的墙钟时间） | Simulator 逐迭代追踪 timer 差值；模型用闭式 max 近似 |
| L3 Critical | `stall = max(0, cur-nxt)` (dbl) / `Cons[op]` (single) | Simulator 逐迭代动态计算 stall；模型取稳态瓶颈 |
| L4 Process | 递归 `loopExecution` 的累计 timer 推进 | Simulator 逐迭代累积；模型用 首迭代+$(F{-}1)\times$Critical 闭式 |
| L5 MaxStartup | `timer` 初始化阶段的 bootstrap 搬入 | 完全对齐 |
| L6 Latency | `max(timer[mem,op])` | 完全对齐 |

**准确率验证**（8 个 benchmark case）：

| Case | 类型 | 准确率 |
|------|------|--------|
| ResNet Layer 1 | 标准卷积 | 98-99% |
| ResNet Layer 12 | 标准卷积 | 95% |
| ResNet Layer 15 | 标准卷积 | 96% |
| ResNet Layer 17 | 标准卷积 | 98% |
| MobileNetV2 Conv_1 | Depthwise | 100% |
| MobileNetV2 Conv_10 | Depthwise | 100% |
| MobileNetV2 Conv_40 | Depthwise | 98% |
| MobileNetV2 Conv_43 | Depthwise | 99% |

---

## 代码定位

延迟模型实现位于 `utils/SolverTSS.py`，核心代码块约 150 行，结构如下：

| 代码位置 | 模型层级 | 关键变量 |
|---------|---------|---------|
| L0 块 | transLatency, effectiveTransLatency, transfer | `transLatency[m,op]`, `effectiveTransLatency[m,op]`, `transfer[i,op]` |
| L1 块 | latency_Transfer | `latency_Transfer[i,op]` — xMem 门控后的 per-level 传输 |
| L2 块 | latency_Body | `latency_Body[i]` — 循环体时间（含 dbl_overlap 扣减） |
| L3 块 | latency_Critical | `latency_Critical[i]` — 稳态迭代瓶颈 |
| L4 块 | latency_Process | `latency_Process[i,op]` — 总执行时间 |
| L5-L6 块 | latency_MaxStartup, res_latency | `latency_BootstrapRead`, `latency_SumTransfer`, `latency_MaxStartup` |

---

## 模型局限性与保守近似

1. **Body 对 O 不做 dbl_overlap 扣减**：O 的首迭代 pre-read 在关键路径上，保守地不扣减。可能导致约 1-2% 的高估。

2. **Critical 为稳态近似**：模型假设非首迭代的每次耗时均为 Critical（稳态瓶颈）。实际上 Simulator 中每次迭代的 stall 可能因 timer 状态差异而不同。对小循环因子（F=2-3）的近似误差较大，但对大因子趋近精确。

3. **Trans 的 ceil 近似**：`max(rawTrans, 1)` 仅保证下界为 1 cycle，对 1 < rawTrans < 2 的情况（如 1.125 cycles，硬件 ceil 到 2），模型仍使用连续值 1.125。此差异对小 tile 的标准卷积有 ~1% 影响。
