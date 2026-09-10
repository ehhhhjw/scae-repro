# 脚本清单与状态

全部代码按"论文是否引用"分成 **core** 与 **legacy** 两类。
legacy 的脚本被保留是因为论文的论证链条需要它们（多数是"被后续分析否证/取代"的那一环），
但它们不产出终稿数字。

产物路径中 `$RUNS` = `pilot/runs`（或 `SCAE_RUNS`）。

---

## 执行层 `pilot/harness/`

| 文件 | 作用 | 被谁调用 |
|---|---|---|
| `appserver.py` | codex app-server 的 JSON-RPC/stdio 客户端；`thread/start`、`thread/fork`、`turn/start`、三路消息解复用；`PINNED_ENV` | 全部采集脚本 |
| `rollout.py` | rollout JSONL 的读/写/截断/改写 —— **前缀重放的底层原语**。`truncate_before_step`、`substitute_action`、`mask_output_lines`、`action_bucket`、`assert_no_dangling_calls`、`compaction_lines` | 全部 |
| `runner.py` | 批量文件定位；写 `config.toml`、强制 reset、解析 `<answer>`、断点续跑 | `parallel.py` |
| `parallel.py` | 按 repo 分片并行；locate 与 attr 两种模式；合并各 shard JSONL | `scripts/01`、`scripts/02` |
| `action_exec.py` | 只读替代动作的**真实执行**（干预分支的 o' 来源）；白名单 + 禁用集 | `attribution/causal.py` |

## 因果估计层 `pilot/attribution/`

| 文件 | 作用 |
|---|---|
| `causal.py` | 核心估计器：`estimate_v`（m 次重放估 v_t）、`causal_attribution`（Δ_t = v_t − v_{t−1}）、`estimate_policy_at`（估 π̂_t）、`hybrid_delta_at`（**τ_H 分支：观测重放 vs 软干预**） |
| `judge.py` | full-trace / prefix-only LLM judge 基线（被攻击的对照臂） |
| `run_attribution.py` | 归因编排：等距估值点、三种结果口径（binary/per_file/recall） |

## 不确定性层 `pilot/uq/`

| 文件 | 作用 | 状态 |
|---|---|---|
| `predictor.py` | 贝叶斯 logistic（IRLS + Laplace 后验）、三分解、ECE/Brier/NLL/AUROC/AURC | core |
| `features.py` | v₀ 的**执行前**特征（7 协变量，**不读轨迹**，否则泄漏执行期信息） | core |
| `nextstep.py` | 下一步动作预测（8 特征 + LORO） | 早期版本，已被 `analyze/x1c` 取代 |
| `anytime.py` | anytime AUROC(t)/ECE(t) 曲线 | 未进入终稿 |

## 信道层 `pilot/channels/`

| 文件 | 作用 | 备注 |
|---|---|---|
| `parsed_command.py` | codex `ParsedCommand` 的 Python 复刻，把 shell 命令分类成 (kind, path/query) | **动作语义层的定义**；已知 `shlex` 缺陷见 PITFALLS #6 |
| `tool_model.py` | 工具信道 Markov 模型（M0–M3），产出 H_tool | 关键实现细节：`exec_command` 占 8159/8200 次调用，**语义工具是 shell 命令头而非名义 API** |
| `code_channel.py` | 代码依赖图信道 U^obj，三层结构先验（W=0.62/0.30/0.08） | 需 `$RUNS/graphs/` |
| `graph_builder.py` | 构图编排 | **依赖两个不随包分发的外部工具**，见 REPRODUCE.md §5 |

## 任务层 `pilot/probes/` 与 `pilot/graph/`

| 文件 | 作用 |
|---|---|
| `probes/task.py` | **数据集接口的唯一真相来源**：实例加载、prompt、结果判定 Y、`reset_repo`（防泄题） |
| `graph/ast_index.py` | 纯 stdlib AST 索引（符号定义点、import 边、图距离），无 tree-sitter 依赖 |
| `graph/ast_probe.py` | AST 歧义特征是否区分成败（Mann-Whitney U） |
| `graph/step_semantics.py` | 步的语义分类（E5）与相邻步 AST 距离分布（E3） |

## 可行性探针 `pilot/smoke/`

| 文件 | 作用 |
|---|---|
| `smoke.py` | S0 的 V1–V4：fork 可行性、只读工具面、无审批请求、`Continue.` 语义 |
| `probe_v5.py` | 动作熵探针 = **positivity 的直接测量**（论文 C.4 / Figure 5） |

---

## 分析层 `pilot/analyze/` —— core

按研究问题分组。全部零 API（除标注者）。

### RQ1 下一动作预测（§5.3 / Table 1、10、11）

| 脚本 | 作用 | 产物 |
|---|---|---|
| `v3_nextstep_dual.py` | **共享 episode 载入库**（被下面 7 个脚本 import，不要移动） | — |
| `v3g_regime.py` | 共享分母的 509 决策点 + 首访/重访双 regime 分解 | `v3g_regime.json` |
| `v3j_trace_graph.py` | **执行来源图预测（Table 1 主行）** | `v3j_trace_graph.json` |
| `v3b_distribution_quality.py` | top-k 曲线 / MRR / ECE / NLL / exp(H) | `v3b_distribution.json` |
| `x1_joint_l1.py` | H_tool 分布与分层 | `x1_joint_l1.json` |
| `x1b_joint_independence.py` | 工具×对象条件独立性（I = 0.159 nats）+ 分层可信度（Table 11） | `x1b_joint_independence.json` |
| `x1c_tool_routed.py` | 按预测工具路由候选集（Table 10），样本从 71 扩到 ~496 实例 | `x1c_tool_routed.json` |
| `x1d_query_channel.py` | 查询信道诊断 + `shlex` 缺陷量化（D.5） | `x1d_query_channel.json` |

### RQ2 代码依赖图（§5.4 / Figure 7）

| 脚本 | 作用 | 产物 |
|---|---|---|
| `v3i_graph_probe.py` | 多角度图论证：邻接 vs 置换零、k 跳覆盖、gold 排序、访问集聚类 | `v3i_graph_probe.json` |

### RQ3 复用模式（§5.5 / D.4）

| 脚本 | 作用 | 产物 |
|---|---|---|
| `v3h_reuse.py` | 先证复用**率**是置换不变量（零信息），再检验时序**结构** | `v3h_reuse.json` |

### RQ4 不确定性层级（§5.6 / Figure 8 / D.6、D.7）

| 脚本 | 作用 | 产物 |
|---|---|---|
| `x4_uexec_corr.py` | corr(H_struct, U^exec)；偏相关控制 v_t | `x4_uexec_corr.json` |
| `x4b_uexec_validity.py` | U^exec 有无**非机械**结构（残差化 ICC 是关键） | `x4b_uexec_validity.json` |
| `x4c_instance_level.py` | 实例层聚合复核 | `x4c_instance_level.json` |
| `x5_step_power.py` | **步级功效 + BH-FDR（论文核心负结果）** | `x5_step_power.json` |
| `x6_budget_alloc.py` | 熵引导的预算分配（报出 5×，随后被 x6b 否证） | `x6_budget_alloc.json` |
| `x6b_effect_criterion.py` | 证明上面的 5× 是二值阈值伪例（α 扫描 3.36→1.71） | `x6b_effect_criterion.json` |
| `v4_structural_vs_exec.py` | Bernoulli 近似 U^exec = 4v(1−v)；论文 C.3 保留为**负控制** | `v4_structural_vs_exec.json` |
| `process_uncertainty.py` | 熵轨迹、脆弱成功、收敛点 t*、degenerate 率 | `process_uncertainty.json` |
| `estimability.py` | 三态记账 INESTIMABLE / NO_EFFECT / EFFECT | `estimability.json` |
| `x8_outcome_sensitivity.py` | 六个结果变量的干预敏感度筛选（C.6） | `x8_outcome_sensitivity.json` |
| `uq_selftest.py` | **三分解合成自检（D.7，唯一正面验证）** | `uq_selftest.json` |
| `entropy_scan.py` | 熵结构汇总（C.4 / Table 6 / Figure 5） | `entropy_scan_summary.json` |

### RQ5 归因与碰撞偏差（§5.7 / Table 2、12 / Figure 4）

| 脚本 | 作用 | 产物 |
|---|---|---|
| `attribution_compare.py` | D₁ / D₃ 与**碰撞隔离**（+0.537） | `d1d3_report.json` |
| `v3k_trace_causal.py` | \|Δ_t\| 是否集中在 gold-provenance；judge/causal top-1 落点 | `v3k_trace_causal.json` |
| `x2_judge_mechanism.py` | 零 LLM 结构规则能否复现 judge（Table 12）+ 选择空间校正 | `x2_judge_mechanism.json` |
| `judge_http.py` | `/chat/completions` 直连（qwen 的流式与 `/responses` 不兼容）**需 API key** | — |
| `x9_judge_crossmodel.py` | 跨模型复制（Table 2）**需 API key**，支持 `--analyze-only` | `x9_*.json` |
| `x9b_crossmodel_pool.py` | 汇总：位移应同质、富集应异质 | `x9b_crossmodel_pool.json` |

### 基础设施

| 脚本 | 作用 | 产物 |
|---|---|---|
| `full_report.py` | 论文表格总汇（定位 Table 7、v₀ LORO D.8） | `analysis.json` |
| `x_synthesis.py` | 聚合各 JSON 供论文宏定义使用 | `synthesis.json` |
| `replay_check.py` | 重放一致性（假设 3 的实证验证） | `replay_check.json` |
| `recover_v5_report.py` | 崩溃恢复工具：从残留 fork 重建 `v5_report.json` | — |

---

## 分析层 `pilot/analyze/legacy/` —— 已取代/未进终稿

保留原因：论文的论证是"先报一个结果，再用更严的检验否证/取代它"。
删掉这些会让读者无法核对被否证的那一环。

| 脚本 | 原用途 | 为何 legacy |
|---|---|---|
| `v3f_ablation.py` | L1/L2/L3 三层结构先验消融 | 被 `v3g_regime` + `v3i_graph_probe` 取代 |
| `m_robustness.py` | 论证 m=3 够用（m3 vs m8 的 Spearman 一致性） | 终稿正文口径是 **m=8**，该论证不再需要 |
| `x7_fragile_l1.py` | 脆弱成功 ↔ H_tool 关联 | 未进终稿；相关结论由 `process_uncertainty` 承担 |
| `x3_prov_mask.py` | provenance 掩蔽的 do-calculus 实验（3 臂 PSE）**需 API key** | 论文未报告 PSE；该设计被判定在当前预算下不可行 |
| `repair_runner.py` | 四臂修复实验 E1（原 `attribution/repair.py`）**需 API key** | 论文未报告 RepairSuccess@1 |
| `repair_analysis.py` | 上者的分析端 | 同上 |

调用方式变为 `python3 -m pilot.analyze.legacy.<name>`。
`x3_prov_mask` 的 `plan` / `analyze` 两个模式零 API，只有 `run` 模式要 key。

---

---

## 编排入口 `scripts/` 与 `setup/`

| 脚本 | 阶段 | 需要 API | 耗时 |
|---|---|---|---|
| `setup/check_env.py` | 前置自检（两档） | 否 | 秒 |
| `setup/prepare_dataset.py` | 构建 CSV + 仓库快照 | 否 | 秒 ~ 数小时 |
| `setup/stage_attr_shards.py` | 归因分片准备 | 否 | 秒 |
| `scripts/00_smoke.sh` | 可行性 V1–V5 + 重放一致性 | **是** | 10–30 min |
| `scripts/01_locate.sh` | 499 实例定位 | **是** | 10–20 h |
| `scripts/02_attribution.sh` | 前缀重放归因（两档） | **是** | ~12.7 h |
| `scripts/03_judge_crossmodel.sh` | judge 跨模型（Table 2） | **是** | ~11 h（限流） |
| `scripts/05_upolicy.sh` | U_policy 的 K=4 重复（D.7） | **是** | 2–4 h |
| `scripts/04_analysis_offline.sh` | 全部离线分析 | **否** | 几分钟 |
| `scripts/common.sh` | 公共前导（环境钉死 + 前置校验） | — | — |
| `scripts/run_all.sh` | 全流程 | **是** | 30h+ |

`pilot/legacy_pipelines/` 存放原始编排脚本（`run_pilot.sh`、`run_remaining.sh`），
仅作溯源参考——它们引用不随包分发的数据产物，直接运行会失败。
对应关系见该目录的 README。

---

## 依赖关系里的一个陷阱

`v3_nextstep_dual.py` 名字像是"早期探索"，但它是 **7 个 core 脚本共同 import 的
episode 载入库**（`v3b`、`v3g`、`v3h`、`v3i`、`x1b`、`x1c` 以及 legacy 的 `v3f`）。
移动或删除它会一次性打断 RQ1–RQ3 全部分析。

其余关键 import 边：

```
v3j_trace_graph  ← v3k, x1b, x1c, x1d, x4b, x5, x6b, x8
v3k_trace_causal ← x2, x4b, x4c, x5, x6b, x8, x9, x9b
v3g_regime       ← v3j, x1b, x1c, x1d
```

改这三个文件的公共函数签名会波及一大片下游。
