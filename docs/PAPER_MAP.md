# 论文数字 → 脚本 → 产物映射

论文每个报告的数字都由某个 CPU 脚本写进某个 JSON。本表给出对应关系，
用于核对复现结果。`$RUNS` = `pilot/runs`（或 `SCAE_RUNS`）。

标注说明：**✅ 已实跑验证** = 本包代码实际跑出该数字并与论文对齐；
其余为按代码路径与论文文本建立的映射（需要对应录制产物才能核对）。

---

## 正文主张

### §5.2 实验设置与前置验证

| 论文数字 | 脚本 | 产物 |
|---|---|---|
| 499 episode / 12 仓库 | `scripts/01_locate.sh` | `$RUNS/full_locate/results.jsonl` |
| 定位成功率 93.99%（Table 7） | `pilot.analyze.full_report` | `$RUNS/analysis.json` |
| 重放一致 1.00 / 100 动作 | `pilot.analyze.replay_check` | `$RUNS/replay_check.json` |
| 压缩事件 0 次 | `rollout.compaction_lines()`，由上式带出 | `$RUNS/replay_check.json` |
| recall 可动比例 98.6% vs 二值 73.2% | `pilot.analyze.x8_outcome_sensitivity` | `$RUNS/x8_outcome_sensitivity.json` |
| 190 前缀 × 8 续跑 / 38 实例 / 12.7h | `scripts/02_attribution.sh power` | `$RUNS/v4prime/attribution.jsonl` |

> **500 vs 499**：CSV 有 500 行，`django__django-15278` 缺仓库快照，故语料为 499。

### §5.3 RQ1 什么预测下一个动作（Table 1）

所有预测器共享同一分母：realized 对象落在候选集外的点记为 miss 而非丢弃。

| 论文数字 | 脚本 | 产物字段 |
|---|---|---|
| 首访步 341 个决策点 | `pilot.analyze.v3g_regime` | `$RUNS/v3g_regime.json` |
| 来源(last, 位置排序) top-3 **0.326** CI[0.276,0.375] | `pilot.analyze.v3j_trace_graph` | `$RUNS/v3j_trace_graph.json` |
| 来源(recency 加权) top-3 0.308，top-10 0.499 | 同上 | 同上 |
| 代码图（强化版）top-3 **0.058** CI[0.039,0.078] | `pilot.analyze.v3i_graph_probe` | `$RUNS/v3i_graph_probe.json` |
| 仓库频次（边缘）top-3 0.072 | `pilot.analyze.v3g_regime` | 同上 |
| 访问历史频次 覆盖 0.000（首访步按构造为空） | 同上 | 同上 |
| 配对提升 top-3 +0.136 CI[+0.102,+0.173] | `pilot.analyze.v3j_trace_graph` | 同上 |
| 有效候选数 exp(H)：来源 **3.46** vs 图 1719.13 | 同上 / `v3i_graph_probe` | 同上 |
| 候选效率 0.689 的首访目标已在早前输出中出现 | `pilot.analyze.v3j_trace_graph` | 同上 |
| handler 27.3% top-3 0.582 / path 51.1% 0.185 / query 21.6% **0.029** | `pilot.analyze.x1c_tool_routed` | `$RUNS/x1c_tool_routed.json` |
| 工具×对象互信息 I = **0.159** nats，p = 5×10⁻⁴ | `pilot.analyze.x1b_joint_independence` | `$RUNS/x1b_joint_independence.json` |
| 路由 top-1 0.109 → 0.150；exp(H) 39.93 → 12.85（Table 10） | `pilot.analyze.x1c_tool_routed` | 同上 |
| 搜索后 top-3 0.469 ECE 0.013；读取后 ECE 0.160（Table 11） | `pilot.analyze.x1b_joint_independence` | 同上 |

### §5.4 RQ2 代码依赖图有什么用（Figure 7）

零假设是**期内置换**：保持访问过的多重集不变，只打乱顺序。

| 论文数字 | 脚本 | 产物字段 |
|---|---|---|
| 一跳提升 **0.94×**、三跳 1.02×、同社区 1.01× | `pilot.analyze.v3i_graph_probe` | `$RUNS/v3i_graph_probe.json` |
| 距离 0（即时自重复）提升 1.47× | 同上 | 同上 |
| 访问集内部边密度 **0.269** vs 零假设 0.0087（**31×**） | 同上 | 同上 |
| 未见 gold 文件秩分位 0.049 CI[0.021,0.084] | 同上 | 同上 |
| 65% gold 文件在排序时已被访问过（泄漏控制） | 同上 | 同上 |
| 46.2% 的 agent 对象能解析到图节点 | 同上 | 同上 |
| 原始一跳率 0.284 vs 置换期望 0.301（D.3） | 同上 | 同上 |

### §5.5 RQ3 哪些复用模式是真的 ✅ 已实跑验证

| 论文数字 | 本包实测 | 产物字段 |
|---|---|---|
| 工具复用率 79.1% | **0.7912** CI[0.7815,0.8005] | `$RUNS/v3h_reuse.json` |
| 对象复用率 41.6% | **0.4159** CI[0.4001,0.4313] | 同上 |
| 复用率的置换零标准差**恰为 0** | **0.00e+00** | 同上 |
| 对象紧邻重复 1.83×，p ~ 10⁻¹⁴¹ | **1.83×**，p ~ 1e-141 | 同上 |
| 工具自重复 1.02×，p = 0.51（**不显著**） | **1.02×**，p = **0.511** | 同上 |
| 工具信道互信息 1.13×，p ~ 10⁻²² | **1.13×**，p ~ 1e-22 | 同上 |
| 访问计数偏好 lift 1.70× CI[1.63,1.76] | **1.6965** CI[1.6335,1.7618] | 同上 |
| recency 秩偏好 lift 1.74× CI[1.67,1.81] | **1.739** CI[1.6702,1.809] | 同上 |
| 487 episode | **n=487 ep**（载入 491） | 同上 |

脚本：`pilot.analyze.v3h_reuse --runs-root $RUNS/full_locate`

### §5.6 RQ4 执行不确定性住在哪一层（Figure 8）

| 论文数字 | 脚本 | 产物字段 |
|---|---|---|
| 步级 ρ = **−0.088** CI[−0.244,+0.062] | `pilot.analyze.x4_uexec_corr` | `$RUNS/x4_uexec_corr.json` |
| 实例级 ρ = −0.219 | `pilot.analyze.x4c_instance_level` | `$RUNS/x4c_instance_level.json` |
| 正控制 ρ = **−0.425** CI[−0.695,−0.102] | `pilot.analyze.x4_uexec_corr` | 同上 |
| 实例 ICC **0.640** vs 置换零 0.196，p = 5×10⁻⁴ | `pilot.analyze.x4b_uexec_validity` | `$RUNS/x4b_uexec_validity.json` |
| 步位 ρ = −0.131；工具类型 p = 0.65（均不显著） | 同上 | 同上 |
| **190 个效应中 5.3% 达未校正 p<0.05，BH-FDR 后存活 0 个** | `pilot.analyze.x5_step_power` | `$RUNS/x5_step_power.json` |
| m=8 时中位重放方差为 0；degenerate 22.9%（16/70） | `pilot.analyze.process_uncertainty` | `$RUNS/process_uncertainty.json` |
| 阈值扫描比值 3.36 → 1.71（D.6 的伪例） | `pilot.analyze.x6b_effect_criterion` | `$RUNS/x6b_effect_criterion.json` |
| 被否证的 5× 原始结果 | `pilot.analyze.x6_budget_alloc` | `$RUNS/x6_budget_alloc.json` |
| U^exec 的 Bernoulli 近似（C.3 的负控制） | `pilot.analyze.v4_structural_vs_exec` | `$RUNS/v4_structural_vs_exec.json` |

### §5.7 RQ5 观测性归因在测什么（Figure 4 / Table 2 / Table 12）

| 论文数字 | 脚本 | 产物字段 |
|---|---|---|
| 碰撞位移 **+0.537** CI[+0.459,+0.610]，sign test p = 1.1×10⁻¹⁶ | `pilot.analyze.attribution_compare` | `$RUNS/d1d3_report.json` |
| 59/70 episode 判决后移；0.214 → 0.751 | 同上 | 同上 |
| judge top-1 落在 gold-provenance 步 **0.900** vs 基线 0.413 | `pilot.analyze.v3k_trace_causal` | `$RUNS/v3k_trace_causal.json` |
| 类内偏晚 0.746 vs 均匀零 0.5 CI[+0.172,+0.311] | `pilot.analyze.x2_judge_mechanism` | `$RUNS/x2_judge_mechanism.json` |
| 零 LLM 最佳规则「最后一个 gold-provenance 步」0.243 vs 类内 0.139（1.7×，Table 12） | 同上 | 同上 |
| 三 judge 位移 +0.533 / +0.434 / +0.394（Table 2） | `pilot.analyze.x9_judge_crossmodel` | `$RUNS/x9_{general,coder}.json` |
| judge 间异质性：位移 p = 0.181（无）；富集 p = 0.044（有） | `pilot.analyze.x9b_crossmodel_pool` | `$RUNS/x9b_crossmodel_pool.json` |
| D₁ 分歧率 97.1%（**论文已退役该判据**，见 D.12） | `pilot.analyze.attribution_compare` | `$RUNS/d1d3_report.json` |
| 干预基线 top-1 平均位置 0.333 | 同上 | 同上 |

---

## 附录

| 附录 | 内容 | 脚本 | 产物 |
|---|---|---|---|
| **C.4** | 熵在 10 个探测点：50% 非零，ρ = +0.072（Table 6 / Figure 5） | `pilot.analyze.entropy_scan` + `pilot.smoke.probe_v5` | `$RUNS/v5_scan/entropy_scan_summary.json` |
| **C.6** | 结果变量筛选：recall 2.505 > F₁ 2.165 > 二值 1.643 | `pilot.analyze.x8_outcome_sensitivity` | `$RUNS/x8_outcome_sensitivity.json` |
| **D.1** | 按仓库定位成功率（Table 7） | `pilot.analyze.full_report` | `$RUNS/analysis.json` |
| **D.2** | 每 episode 提及 50.5 文件、实访 6.5（转化 18.0%）；7616 决策点 | `pilot.analyze.v3j_trace_graph` / `x1c_tool_routed` | `$RUNS/v3j_trace_graph.json` |
| **D.5** | `shlex` 在 26.6% 命令段失败；75.3% query 对象受影响 | `pilot.analyze.x1d_query_channel` | `$RUNS/x1d_query_channel.json` |
| **D.6** | 二值判据制造的伪例 + α ∈ {0.01,0.05,0.10,0.20} 扫描 | `pilot.analyze.x6b_effect_criterion` | `$RUNS/x6b_effect_criterion.json` |
| **D.7** ✅ | 三分解合成自检：0.994 / 0.022 / 0.308-0.146-0.012 / ×8.1 | `pilot.analyze.uq_selftest` | `$RUNS/uq_selftest.json` |
| **D.8** | v₀ 的 LORO AUROC **0.481**（at chance）、ECE 0.015、Brier 0.058、AURC 0.058 | `pilot.analyze.full_report` | `$RUNS/analysis.json` |
| **D.10** | 归因样本 70 episode / 10 仓库 / 19 失败；judge 落窗 32.9%（n=23） | `pilot.analyze.v3k_trace_causal` | `$RUNS/v3k_trace_causal.json` |
| **D.11** | \|N\|=54 非退化 episode；54.7% 估值点 Δ 恰为 0 | `pilot.analyze.estimability` | `$RUNS/estimability.json` |
| **D.12** | 为何退役 D₁ | `pilot.analyze.attribution_compare` | `$RUNS/d1d3_report.json` |
| **E.2** | 预注册记录与结果对照（Table 9） | 各脚本汇总 | `$RUNS/synthesis.json` |

### 论文 Appendix E.1 明确记载的命令

```bash
python3 -m pilot.analyze.uq_selftest      # D.7 三分解自检
python3 -m pilot.analyze.entropy_scan     # C.4 熵结构
python3 -m pilot.analyze.full_report      # 定位与预测表
```

（另有两个画图脚本与两个论文校验脚本，按要求不含在本包内。）

---

## 产物结构

```
$RUNS/
├── full_locate/
│   ├── results.jsonl                  ★ 定位结果（RQ1–RQ3 + v₀ 语料）
│   └── shard*/rollouts/*.jsonl        ★ 轨迹
├── full_attr/
│   ├── attribution.jsonl              ★ 70 episode，m=3（RQ5）
│   └── shard*/{attribution.jsonl,attr_work/}
├── v4prime/
│   ├── attribution.jsonl              ★ 38 episode，m=8，含 ys_t 明细（RQ4）
│   └── shard*/
├── v5_scan/<instance>/v5_report.json  熵探针
├── graphs/<instance>/                 构图产物（RQ2，71 实例，外部工具生成）
├── tool_model_official.json           工具信道（阶段 4 rq1 组生成）
└── *.json                             各分析脚本的输出（约 3.6MB）
```

`attribution.jsonl` 每行一个 episode，关键字段：

| 字段 | 含义 |
|---|---|
| `Y`, `recall`, `n_gold`, `missed_files` | 结果变量（论文用 `recall`） |
| `n_steps` | 轨迹长度 T（用于归一化位置 t/T） |
| `causal_v0` | 执行前预测 v₀ = E[Y \| q, G] |
| `causal_segments[]` | 每个估值点：`t`, `delta`(Δ_t), `v_t`, `v_prev`, `pos`, `action` |
| `causal_segments[].ys_t` | m 次重放的逐次结果（**m=8 档才有**，是 U^exec 的来源） |
| `causal_top1_step/pos/delta` | 因果 top-1 归责步 |
| `judge_full_top1_step/pos` | full-trace judge 的归责（碰撞偏差臂） |
| `judge_prefix_top1_step/pos` | prefix-only judge 的归责（对照臂） |
| `degenerate` | 该 episode 是否所有 Δ 恒为 0 |

碰撞位移的定义即 `pos(judge_full_top1) − pos(judge_prefix_top1)` 的期均值。
