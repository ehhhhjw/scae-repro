# 复现执行步骤

本文件是本仓库的操作手册。按顺序读到你要走的那条路线即可。

> **先读 [docs/PITFALLS.md](docs/PITFALLS.md)。** 那里列的坑（尤其是 gold patch 泄题）
> 会在不报任何错的情况下毁掉全部结果，而且毁掉的方式是**让数字变好看**。

---

## 0. 选一条路线

| 路线 | 前置 | 耗时 | 产出 |
|---|---|---|---|
| **A 自检** | python 3.10+、numpy | ~10 秒 | Appendix D.7（合成数据，有真值） |
| **B 离线分析** | 一份 `pilot/runs/` 录制产物 | ~5–30 分钟 | 论文全部表格与统计量 |
| **C 全量重跑** | codex CLI + API key + 数据集 + 30h+ | 30h+ | 从零重跑，含 agent 执行 |

本包**不携带录制数据**（完整 17GB）。路线 B 需你自备产物，路线 C 从头生成。

---

## 1. 环境与数据集

### 1.1 依赖

```bash
python3 --version          # 需 >= 3.10（代码用了 X | Y 类型标注）
pip install -r requirements.txt
```

只有 `numpy`。**没有** scipy / sklearn / torch —— 估计器与检验（IRLS + Laplace、
置换检验、bootstrap、Spearman、BH-FDR）都是手写的，这样复现不受版本漂移影响。

### 1.2 配置

```bash
cp env.example.sh env.sh
vim env.sh          # 至少填 IDEALAB_API_KEY（仅采集需要）与数据集路径
source env.sh
```

### 1.3 数据集

论文语料是 SWE-bench-Verified 的 500 个实例。代码需要两样东西：

1. `instance_summary.csv` —— 字段 `instance_id, base_commit, problem_statement, patch_file`，
   其中 `patch_file` 是 gold patch 触及的文件路径（换行分隔），**就是论文的 F\***
2. `projects/<instance_id>/` —— 每实例一个 git 仓库快照，HEAD 停在 `base_commit`

```bash
# 只生成 CSV（约 900KB，秒级）
python3 setup/prepare_dataset.py --from-hf --skip-clone
# 或用本地 SWE-bench-Verified JSON
python3 setup/prepare_dataset.py --from-json /path/to/instance_questions.json

# 克隆仓库快照（几十 GB，数小时；路线 C 必需，路线 A/B 不需要）
python3 setup/prepare_dataset.py --from-hf --clone --workers 4

# 已有快照？只体检是否停在 base_commit（强烈建议跑）
python3 setup/prepare_dataset.py --verify-only
```

`--verify-only` 会报出 `dirty` / `wrong_commit` 的实例。**任何一个非 ok 都意味着泄题风险**，
因为分发的快照常常叠着 gold patch（= 已修复状态）。

### 1.4 自检

```bash
python3 setup/check_env.py            # 离线分析档
python3 setup/check_env.py --collect  # 采集档：额外查 git / codex CLI / API key
```

---

## 2. 路线 A：自检（不需要任何数据）

```bash
python3 -m pilot.analyze.uq_selftest
```

复现论文 Appendix D.7 —— 在**已知生成过程**的合成数据上验证三分解：

| 量 | 本包实测 | 论文 |
|---|---|---|
| 参数恢复 corr(w_true, MAP) | 0.9935 | 0.994 |
| 支撑内概率 MAE | 0.022 | 0.022 |
| 预测熵分解 | 0.4638 = 0.3075 + 0.1458 + 0.0124 | 0.464 = 0.308/0.146/0.012 |
| epistemic 支撑外膨胀 | 0.0124 → 0.1002（×8.08） | ×8.1 |

这项验证的**边界**要说清楚：它证明三分解的机器在真值已知时工作正常，
**不**证明三个分量在真实轨迹上可识别。论文 D.7 明确写了后者未建立。

---

## 3. 路线 C：全量重跑

四个阶段。每个阶段的脚本都可单独重跑，且都会校验前置产物。

### 阶段 0：可行性验证（~10–30 分钟）

```bash
bash scripts/00_smoke.sh
```

验掉五个决定方法可行性的未知量，并产出论文 §5.2 报告的两项前置验证：

- **重放一致性 1.00**（100 个动作逐字节相同）→ 假设 3「可重放环境」
- **压缩事件 0 次** → 假设 1「上下文可观测」

同时用 `probe_v5` 测动作熵 = **positivity 的直接测量**。论文的关键发现之一是
positivity 是**局部**性质：50% 的探测点有动作随机性，且与步位的秩相关只有 +0.072
（**不**是单调衰减——这是预注册里没被支持的一项，论文如实报告了）。

> 任一判据不通过就**不要往下走**。重放不一致意味着环境没钉死
> （见 `pilot/config/env_pin.sh`），此后所有因果估计都不可信。

### 阶段 1：全量文件定位（并行 6 路 ~10–20 小时）

```bash
bash scripts/01_locate.sh
# 先小样本验链路：
SCAE_LIMIT=10 bash scripts/01_locate.sh
```

产出全部结论的语料底座：

- `pilot/runs/full_locate/results.jsonl` —— 每实例的 Y / recall / n_steps / 预测文件集
- `pilot/runs/full_locate/shard*/rollouts/*.jsonl` —— 完整轨迹（前缀重放的输入）

脚本结束会打印按仓库的定位成功率，对照论文 Table 7（总体 **93.99%**）。

**泄题闸门**：每个实例跑前强制 `git reset --hard <base_commit>` + `clean -fdq`
（`pilot/probes/task.py::reset_repo`）。这是最后一道闸，不要绕过。

### 阶段 2：前缀重放归因（~12.7 小时，最贵）

```bash
bash scripts/02_attribution.sh          # 两档都跑
bash scripts/02_attribution.sh judge    # 只跑 RQ5 档
bash scripts/02_attribution.sh power    # 只跑 RQ4 档
```

跑两档，因为论文用它们回答两个不同问题：

| 档 | 规模 | m | 用途 |
|---|---|---|---|
| `full_attr` | 70 episode / 10 仓库 / 19 个失败 | 3 | RQ5 judge 对比（D.10 / D.11） |
| `v4prime` | 38 episode → **190 估值点** | **8** | RQ4 步级功效（§5.6） |

论文 §5.2 的「1901 continuations over 12.7 hours」就是 `v4prime` 这一档。

脚本会先调 `setup/stage_attr_shards.py` 把定位结果摆成 `parallel --cmd attr` 期望的形状
（原始实验里这步是手工做的，这里固定下来）。结束时打印 episode 数、估值点数、
degenerate 率，可直接对照论文（`full_attr` 档 degenerate 应为 **22.9% = 16/70**，
Δ 恰为 0 的估值点应占 **54.7%**）。

### 阶段 3：judge 跨模型复制（~11 小时，受限流支配）

```bash
bash scripts/03_judge_crossmodel.sh
```

复现论文 Table 2。核心操纵：**模型与 rubric 完全不变，只改 judge 看得到的信息集**。

- `full-trace`：条件于 前缀 + 后继步骤 + 实现的结果 Y ⇒ 条件于 A_t 的后代 ⇒ 碰撞偏差
- `prefix-only`：条件于固定 60% 前缀，结果 Y 隐藏

若 full-trace judge 在逼近一个前缀条件化的步局部量，这个操纵**不应**移动归责位置。
实测位移 **+0.537**，sign test p = 1.1×10⁻¹⁶，三个 judge 全为正。

> **限流**：默认 10 次调用 / 60 分钟窗口，25 episode × 2 条件 = 50 次调用 ≈ 5.5 小时/模型。
> 跑不完不用重头再来：`x9_judge_crossmodel` 支持 `--analyze-only`，在已有 JSONL 上重算。

### 阶段 5：U_policy 的 K=4 重复 rollout（~2–4 小时）

```bash
bash scripts/05_upolicy.sh
```

三分解 `H[Y|q,G] = U_task + U_policy + U_model` 里，只有 `U_policy` 能从真轨迹**直接**估计：
固定同一个 q 重复跑 K 次，取各实例结果方差的均值。论文报告 **0.049**（23 个 K=4 实例）。

主流水线用 K=1（500 × 4 太贵），所以这一档单独跑一个小子集。

> **复现口径的诚实说明**：论文实际用的那 23 个实例 id 是一份数据产物，**无法从代码推导**。
> 脚本用一个写死的抽样规则重选（定位成功 + 步数 ≥ 5 + 按仓库分层轮转），
> 因此你的 `U_policy` 会接近但不必等于 0.0489。有原始列表时用 `SCAE_UPOLICY_IDS` 指过去。
> 已验证：把原始 `upolicy/results.jsonl` 喂给本包的 `t3_uq_decomposition`，
> 输出与论文产物**逐字段相同**（0.0489）——即估计器实现无偏差，差异只来自实例抽样。

### 阶段 4：离线分析（几分钟，零 API）

```bash
bash scripts/04_analysis_offline.sh            # 全跑
bash scripts/04_analysis_offline.sh rq1        # 单组：rq1|rq2|rq3|rq4|rq5|report|selftest
```

论文的每个数字都由这里的某个脚本写进某个 JSON。逐个映射见
[docs/PAPER_MAP.md](docs/PAPER_MAP.md)。缺上游产物时脚本会**显式报跳过**，
不会静默出一个残缺的结论。

---

## 4. 路线 B：只做离线分析

已有一份录制产物时：

```bash
export SCAE_RUNS=/path/to/existing/runs
bash scripts/04_analysis_offline.sh
```

`pilot/runs/` 需要的最小结构：

```
runs/
├── full_locate/
│   ├── results.jsonl              定位结果（合并）
│   └── shard*/rollouts/*.jsonl    轨迹（RQ1–RQ3 全靠它）
├── full_attr/
│   ├── attribution.jsonl          70 episode，m=3
│   └── shard*/attribution.jsonl
├── v4prime/
│   ├── attribution.jsonl          38 episode，m=8，含 ys_t 明细
│   └── shard*/attribution.jsonl
├── v5_scan/*/v5_report.json       熵探针（C.4 / Table 6）
├── graphs/<instance_id>/          构图产物（RQ2，71 个实例）
└── tool_model_official.json       工具信道（由阶段 4 的 rq1 组生成）
```

体量参考：`full_locate` 的 `rollouts` 约 85MB，各 `attribution.jsonl` 合计约 300KB，
派生 JSON 约 3.6MB。真正占地方的是 fork 中间产物（`attr_work/`）与 `graphs/`。
只做离线分析**不需要** `attr_work/`。

---

## 5. 图信道的外部依赖（RQ2 的已知缺口）

`pilot/runs/graphs/<instance_id>/` 由 `pilot/channels/graph_builder.py` 生成，
它调用两个**不随本包分发**的上游工具：

```
tree-sitter-fold/AST_Tree_Graph      →  entities.json, dependency_graph.json
graph_leidan_fold/rag-python-tools   →  cluster_entities.json, cluster_communities.json
```

因此 RQ2 的完整复现需要这两个工具。缺失时的处理：

- `scripts/04_analysis_offline.sh rq2` 会显式提示缺 `graphs/`，并退化为只报可解析子集
- `pilot/graph/ast_index.py` 是**纯 stdlib** 的 AST 索引（符号定义点、import 边、图距离），
  可作为依赖图的替代来源，但它产出的图与论文用的不是同一个，数字不可直接比对

这个缺口不影响 RQ1 的 provenance 半边：论文 §5.3 的结论「执行来源击败代码结构」
在丢掉代码图后样本反而从 509 个决策点扩到 7616 个（Appendix D.2），
因为 provenance 只需要字符串归一化。

---

## 6. 预算总表

| 阶段 | 时间 | API | 磁盘 |
|---|---|---|---|
| 0 冒烟 | 10–30 min | 少量 | < 1GB |
| 1 定位 | 10–20 h（6 路并行） | ~500 长会话 | ~1.4GB + 轨迹 |
| 2 归因 | ~12.7 h | ~3000 次续跑 | ~8.6GB（含 fork 中间产物） |
| 3 judge | ~11 h（限流支配） | ~96 次 | < 100MB |
| 4 分析 | 几分钟 | **0** | ~4MB |

并行度受 API 限流与磁盘 IO 双重约束。`SCAE_WORKERS=6` 是论文实际用的值：
按 repo 分片，保证同一仓库目录同一时刻只被一个 worker reset。

---

## 7. 复现结果如何核对

论文声明每个数字都可由 CPU 脚本重新生成。核对方式：

1. 跑 `scripts/04_analysis_offline.sh`，产物 JSON 落在 `$SCAE_RUNS/`
2. 按 [docs/PAPER_MAP.md](docs/PAPER_MAP.md) 逐项比对 JSON 字段与论文数字
3. 注意随机性来源：bootstrap CI 与置换检验会有末位波动；
   点估计与 p 值的量级应当稳定

**已知会有偏差的地方**（论文如实记录）：

- 归因估计带蒙特卡洛误差，会把回归系数向 0 衰减（Appendix E.1）
- `v_t` 在等距点而非每步估计，故 Σ Δ_t 不精确等于 v_T − v_0，论文因此**不**把可加性
  当证据来报（Appendix E.1）
- 熵落在粗格上（4 次重放时非零可取值只有 0.56 / 1.04 / 1.39 nats，Table 6）

---

## 8. 出问题时

| 症状 | 多半是 |
|---|---|
| 定位成功率明显高于 93.99% | **gold patch 泄题**，仓库没 reset 干净 → `--verify-only` |
| 重放一致性 < 1.00 | 环境没钉死 → 确认 `source pilot/config/env_pin.sh` 生效 |
| 出现非零压缩事件 | 假设 1 在该轨迹失效，该轨迹归因不可直接采用 |
| judge 阶段卡住 | 限流。用 `--analyze-only` 就已有数据出结论 |
| 分析脚本报缺文件 | 上游阶段没跑完；脚本会指名缺哪个产物 |
| 查询信道 top-3 异常低 | 已知的 `shlex` 解析缺陷（26.6% 命令段），见 Appendix D.5 |

更多见 [docs/PITFALLS.md](docs/PITFALLS.md)。
