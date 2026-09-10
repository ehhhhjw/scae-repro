# SCAE —— 论文实验可复现性代码包

**What Process Evaluation of Coding Agents Actually Measures: Action, Task, and Step Are Three Different Levels**（ICLR 2027 投稿）

本仓库是该论文的完整实验代码。
（完整产物约 17GB）——只有跑出论文全部数字所需的代码、执行步骤与口径说明。

---

## 论文一句话

过程评测把三件不同的事当成了一件：预测 agent 下一步做什么（**action**）、任务结果还有多不确定
（**task**）、某一步是否真的改变了结果（**step**）。三者要求的**条件化集合**不同，因此"过程分数"
到底在测什么取决于它条件于什么，而不是取决于谁来打分。论文用 SCAE（一个从 agent 执行的结构因果
模型导出的、基于重放的估计器）把第三件事形式化，并以它为量具去测前两件。

在 SWE-bench-Verified 的 499 个文件定位 episode 上，三层呈现不对称结论：

| 层级 | 结论 | 关键数字 |
|---|---|---|
| **action** | 强，但由 agent 自己的**执行来源**驱动，而非代码依赖图 | 首访步 top-3：来源 0.326 vs 图 0.058 |
| **task** | 执行不确定性住在**任务**层，不在步层 | 实例 ICC 0.640（置换零 0.196）；步位、工具类型均不显著 |
| **step** | 在可行重放预算下**无法认证** | 190 个效应，BH-FDR(q=0.10) 后存活 **0** 个 |
| **judge** | full-trace judge 有系统性**碰撞偏差** | 归责位置位移 **+0.537**，sign test p = 1.1×10⁻¹⁶ |

结论：full-trace judge 度量的更像是"下游信息塑造的语义相关性"，而非"经认证的因果贡献"。

---

## 快速开始

```bash
# 1) 依赖（纯 CPU，无 GPU）
pip install -r requirements.txt

# 2) 配置
cp env.example.sh env.sh && vim env.sh && source env.sh

# 3) 自检
python3 setup/check_env.py            # 离线分析档
python3 setup/check_env.py --collect  # 数据采集档（额外查 codex CLI / API key）

# 4) 不需要任何数据就能跑的一项：论文 Appendix D.7 的 UQ 三分解自检
python3 -m pilot.analyze.uq_selftest
```

第 4 步应当输出（本包已验证复现）：

```
参数恢复 corr(w_true, MAP) = 0.9935      论文：0.994
支撑内概率 MAE = 0.022                    论文：0.022
预测熵 0.4638 = 0.3075 + 0.1458 + 0.0124  论文：0.464 = 0.308/0.146/0.012
epistemic 膨胀 0.0124 -> 0.1002 (x8.08)   论文：8.1 倍
```

完整执行步骤（含算力/时间/API 预算）见 **[REPRODUCE.md](REPRODUCE.md)**。

---

## 目录结构

```
scae-repro/
├── README.md                  本文件
├── REPRODUCE.md               ★ 逐步执行步骤 + 预算 + 分层复现策略
├── requirements.txt           numpy（可选 datasets）
├── env.example.sh             全部环境变量
├── setup/
│   ├── check_env.py           环境自检（两档：离线分析 / 数据采集）
│   ├── prepare_dataset.py     从 SWE-bench-Verified 构建 instance_summary.csv + 仓库快照
│   └── stage_attr_shards.py   归因阶段的分片准备
├── scripts/
│   ├── common.sh              公共前导（钉死环境噪声、校验前置）
│   ├── 00_smoke.sh            阶段0 可行性验证 V1–V5 + 重放一致性
│   ├── 01_locate.sh           阶段1 499 实例文件定位（全部结论的语料底座）
│   ├── 02_attribution.sh      阶段2 前缀重放归因（最贵：~12.7h）
│   ├── 03_judge_crossmodel.sh 阶段3 judge 跨模型复制（Table 2）
│   ├── 04_analysis_offline.sh 阶段4 ★ 全部离线分析，零 API
│   ├── 05_upolicy.sh          阶段5 U_policy 的 K=4 重复 rollout（D.7）
│   └── run_all.sh             全流程串起来
├── pilot/                     ★ 代码本体（包名保持 pilot，见下）
│   ├── harness/               agent 执行层：app-server 驱动、rollout 改写、前缀重放
│   ├── attribution/           因果估计层：v_t / Δ_t / 混合估计器 / judge 基线
│   ├── uq/                    不确定性层：贝叶斯 logistic + Laplace + 三分解
│   ├── channels/              信道模型：工具信道、代码图信道、命令解析
│   ├── graph/                 AST 索引与图探针（纯 stdlib）
│   ├── probes/                任务定义：实例加载、prompt、结果判定 Y
│   ├── smoke/                 可行性与 positivity 探针
│   ├── analyze/               各 RQ 的离线分析（+ legacy/ 已取代脚本）
│   └── legacy_pipelines/      原始编排脚本，仅作溯源参考（已被 scripts/ 取代）
└── docs/
    ├── PAPER_MAP.md           ★ 论文每个数字 → 脚本 → 输出 JSON 字段
    ├── SCRIPT_INVENTORY.md    全部脚本的状态与取代关系
    ├── PITFALLS.md            ★ 会静默毁掉结果的坑（先读这个）
    └── METHOD_NOTES.md        估计器与口径的实现说明
```

### 为什么包名还叫 `pilot`

论文 Appendix E.1 记载的复现命令形如 `python3 -m pilot.analyze.entropy_scan`。
保留包名可以让论文里的命令**原样执行**，也避免重写 70 个文件的 import。

---

## 三条复现路线

按你手上有什么选：

| 路线 | 需要 | 耗时 | 能复现什么 |
|---|---|---|---|
| **A 自检** | 只要 python + numpy | 秒级 | Appendix D.7 的 UQ 三分解（合成数据，有已知真值） |
| **B 离线分析** | 阶段 1–3 的录制产物 | 几分钟 | 论文**全部**表格与统计量 |
| **C 全量重跑** | codex CLI + API key + 数据集 + 30h+ | 30h+ | 从零重跑，含 agent 执行 |

**本包不携带路线 B 需要的录制数据。** 若你要走 B，需自行准备一份 `pilot/runs/`
（结构见 [docs/PAPER_MAP.md](docs/PAPER_MAP.md) 的"产物结构"一节），或先跑路线 C 生成。

---

## 实验设置（论文 Appendix C.2，代码中逐项对齐）

| 项 | 值 | 位置 |
|---|---|---|
| 被测 agent | Codex CLI，JSON-RPC app-server，`capabilities.experimentalApi=true` | `pilot/harness/appserver.py` |
| 沙箱 | `read-only`，`approvalPolicy=never`，并验证审批请求数为 0 | `pilot/harness/runner.py` |
| 背后模型 | `gpt-5.5-0424-global`，单轮超时 900s | `SCAE_MODEL` |
| 重放次数 m | **8**（RQ4 步级功效档）/ 3（RQ5 judge 档） | `SCAE_M_POWER` / `SCAE_M_JUDGE` |
| 估值点 | 每 episode **5 个等距点** | `--window-points 5` |
| 熵阈值 τ_H | **0**（最宽松：重放里出现 ≥2 个不同动作即走观测分支） | `causal.hybrid_delta_at` |
| 结果变量 | **recall over F\***（干预敏感度 2.505，优于 F₁ 2.165、二值 1.643） | `--outcome recall` |
| prefix-only judge | 固定 **60%** 前缀，隐藏结果 Y | `--prefix-frac 0.6` |
| 归因样本 | 只跑多文件实例（单文件几乎必成功，Δ≡0） | `--multifile-only` |
| 三个 judge | gpt-5.5-0424-global / qwen3.6-plus / qwen3-coder-plus | `SCAE_JUDGE_*` |
| 多重比较 | BH-FDR，q = 0.10 | 各分析脚本内 |

---

## 已验证的复现结果

以下由本包代码实跑得到，与论文逐位对齐（详见 [docs/PAPER_MAP.md](docs/PAPER_MAP.md)）：

- **Appendix D.7** UQ 三分解自检：0.9935 / 0.022 / 0.308+0.146+0.012 / ×8.08
- **§5.5 + D.4** 复用结构：工具复用率 0.7912、对象 0.4159、置换零标准差 0.00e+00、
  工具紧邻重复 1.02×(p=0.511)、对象紧邻重复 1.83×(p~1e-141)、
  高频 lift 1.6965 CI[1.63,1.76]、近期 lift 1.739 CI[1.67,1.81]、n=487 episode
- **§5.7 + D.10 碰撞偏差（论文头条结果）**：位移 **0.5368**（论文 +0.537）、
  D₁ 分歧率 **97.1%**、judge top-1 落在 gold-provenance **0.900 CI[0.829,0.971]**、
  全步基线 **0.4131**（论文 0.413）、judge 落在估值窗内 **32.9%**（论文 D.10: 32.9%）、70 episode
- **D.7 U_policy**：把原始 K=4 结果喂给本包估计器，输出 **0.0489**（论文 0.049）逐字段相同
- **数据集接口**：`setup/prepare_dataset.py` 生成的 `instance_summary.csv` 与论文实际使用的
  文件解析后**逐字段完全一致**（500 行 × 4 字段，71 个多文件实例）

---

## 语料与那个 500 vs 499

`instance_summary.csv` 有 500 个实例，论文报告 **499 个 episode**。差额是
`django__django-15278`：数据集只分发了 499 个仓库快照，缺它。`setup/check_env.py` 与
`scripts/01_locate.sh` 都会把这类实例显式报出来，不会静默丢弃。

语料由 `django` 主导（46%），因此凡是要拟合预测器的地方都用
**leave-one-repository-out**，共形集用按仓库分层的 Mondrian 校准。

---

## 重要限制（论文 §5.8 / Appendix E.3）

复现前请知道这些结论**本身就是负结果**，不是代码没跑对：

1. **步级参照弱**：190 个效应中 0 个通过 BH-FDR。论文最强的归因结论是关于 judge 判决**如何移动**
   的机制断言，而不是关于它选中的步**是否正确**。
2. **任务范围窄**：只做文件定位——因为这是可重放、前缀可重构、结果可核验的最小现实环境。
   代码编辑或开放式工具使用的过程结构可能不同。
3. **三分解未在真轨迹上验证**：只在合成生成过程上验证（D.7），真轨迹上仅 U_policy 被直接估计。
4. **图信道只覆盖 46.2%**：只有这个比例的 agent 操作对象能干净地解析到依赖图节点。
5. **构图依赖外部工具**：`pilot/runs/graphs/` 由两个不随本包分发的上游工具生成，
   详见 [REPRODUCE.md](REPRODUCE.md) 的"图信道的外部依赖"。

---

## 引用与许可

论文处于 ICLR 2027 匿名评审阶段，引用信息待接收后补充。
代码仅用于学术复现；被分析的仓库来自 SWE-bench-Verified，各自遵循其原始许可。
所有 agent 执行都在只读沙箱中进行，不修改任何被测仓库。
