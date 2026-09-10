# `pilot/` —— SCAE 代码本体

本目录是论文的实现代码。包名保持为 `pilot`，因此论文 Appendix E.1 记载的命令
（如 `python3 -m pilot.analyze.entropy_scan`）可原样执行。

顶层文档见 `../README.md`（总览）与 `../REPRODUCE.md`（执行步骤）。

## 模块划分

```
config/env_pin.sh        环境噪声钉死（U_env pinning / CRN），与 appserver.PINNED_ENV 等价
harness/appserver.py     codex app-server 的 JSON-RPC/stdio 客户端（experimentalApi）
harness/rollout.py       rollout JSONL 的读写/截断/动作改写 —— 前缀重放的底层原语
harness/runner.py        批量文件定位（S1），每实例强制 reset 到 base_commit
harness/parallel.py      按 repo 分片并行，合并各 shard 的 JSONL
harness/action_exec.py   只读替代动作的真实执行（干预分支的 o' 来源）
probes/task.py           实例加载、prompt、结果判定 Y、仓库前置检查
attribution/causal.py    v_t 估计、Δ_t、混合估计器（hybrid_delta_at，τ_H 分支）
attribution/judge.py     full-trace / prefix-only LLM judge 基线
attribution/run_attribution.py  归因编排（等距估值点）
uq/predictor.py          贝叶斯 logistic + Laplace 后验 + 三分解 + 校准指标
uq/features.py           v_0 的执行前特征（7 协变量，不读轨迹）
channels/parsed_command.py  codex ParsedCommand 的 Python 复刻（动作语义层）
channels/code_channel.py    代码依赖图信道 U^obj
channels/tool_model.py      工具信道 Markov 模型 U^tool
graph/ast_index.py       纯 stdlib AST 索引（无 tree-sitter 依赖）
smoke/smoke.py           S0 可行性验证（V1–V4）
smoke/probe_v5.py        动作熵探针（positivity 测量）
analyze/                 各 RQ 的离线分析（见 ../docs/SCRIPT_INVENTORY.md）
analyze/legacy/          已被取代/未进入终稿的探索性脚本
```

## 依赖

Python 3.10+ 标准库 + `numpy`（仅部分 analyze 脚本）。**无 GPU。**
数据采集阶段另需已登录的 `codex` CLI 与 API key。

## 数据路径

不携带数据。通过环境变量指向本地 SWE-bench-Verified 快照：

```bash
export SCAE_CSV=/path/to/instance_summary.csv
export SCAE_PROJECTS=/path/to/swebench_verified_projects/projects
```

未设置时默认读 `<repo>/dataset/instance_summary.csv` 与 `<repo>/dataset/projects`。

## S0 的五个可行性判据

正式实验前先跑 `../scripts/00_smoke.sh`。这五项决定方法是否可行：

| 项 | 问题 | 判据 | 不通过的后果 |
|---|---|---|---|
| **V1a** | `capabilities.experimentalApi=true` 能否开出 `Thread.path` 与按 path fork | `thread.path` 非空 | 前缀重放降级为 turn 级 → 需重设计 |
| **V2** | 工具面能否收窄到只读定位 | 无 `apply_patch/write/edit` 类调用 | 动作空间不离散 |
| **V2b** | 有没有收到服务端反向请求（审批） | 数量为 0 | `approvalPolicy=never` / `sandbox=read-only` 没生效 |
| **V3** | fork 自造 rollout 最少需要哪些 item 类型 | 至少一个 ablation 变体 fork 成功 | 阻塞，必须解决 |
| **V4** | fork 后用 `Continue.` 是接着做还是重启 | 前 k−1 步动作桶逐一相同，且总步数 ≥ k | 所有因果估计失效 |
| **V5** | 同一前缀重复 fork 的动作熵 | 至少 2 个不同动作（H > 0） | positivity 局部失效 → 走干预分支 |

V5 若 H ≈ 0，说明策略在该前缀上近乎确定，定理 1 中「相对于策略自身基线的效应」项退化。
这不是实现 bug 而是论文的核心发现之一：正文 §4.3 的混合估计器（式 8）正是为此设计，
低熵点改走**软干预**分支，并在报告中标注该估计量是相对于扰动策略 π̃ 而非原策略 π。

## 关键实现约定（改代码时不要破坏）

对应论文 Appendix C.1「two implementation invariants are load-bearing and easy to violate」：

1. **续跑语 `CONTINUE_PROMPT` 必须在所有 arm 中逐字一致。** 它进入上下文，因此是估计量定义的
   一部分；任何 arm 间差异都会变成一个未申明的干预，污染 Δ_t。
2. **`substitute_action` 的 `new_output` 必须是真正执行 `new_args` 得到的结果。** 填编造输出就不是
   忠实的 do(A_t = a')，而是在干预观测——两者因果语义完全不同。
3. **截断后必须过 `assert_no_dangling_calls`。** 悬空 tool call 会被模型侧拒绝。
   两个截断算子按构造安全，但改写路径需要显式检查。
4. **跑任务前 `check_repo` 必须通过**（工作区干净 + HEAD == base_commit）。否则 agent 会读到
   已修复代码 = 泄题，且重放不可比。详见 `../docs/PITFALLS.md`。
5. **压缩事件不要静默忽略。** `compaction_lines()` 非空即意味着假设 1（上下文可观测）在该轨迹上
   失效，可识别性被破坏。论文 §5.2 报告的是 0 压缩事件；若你的运行出现非零，该轨迹的归因结论
   不可直接采用。
