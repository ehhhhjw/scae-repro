# 会静默毁掉复现结果的坑

按危险程度排序。**排在前面的几个不会报错，只会让数字变好看**——这是最难发现的一类失败。

---

## 1. ⚠️ 数据集工作区叠着 gold patch（泄题）

**症状**：定位成功率明显高于论文的 93.99%，且没有任何报错。

**原因**：分发的 SWE-bench 仓库快照，工作区常常处于**已修复**状态（gold patch 已应用），
有的还带着 `procbench-*` 之类的 stash。定位任务必须跑在 buggy 的 `base_commit` 上，
否则 agent 直接从自己的工作树里读到答案。

**本包的三道闸**：

1. `setup/prepare_dataset.py --verify-only` —— 全量体检 HEAD 与工作区是否干净
2. `setup/check_env.py` —— 抽查 20 个实例的 `base_commit`
3. `pilot/probes/task.py::reset_repo` —— 每个实例跑之前强制
   `checkout -f` + `reset --hard` + `clean -fdq`

```python
def reset_repo(inst):
    _git(repo, "checkout", "-f", inst.base_commit)
    _git(repo, "reset", "--hard", inst.base_commit)
    _git(repo, "clean", "-fdq")
    return check_repo(inst)   # 断言：工作区干净 且 HEAD == base_commit
```

**不要为了省时间绕过 reset。** 论文 Appendix E.1 专门记了这一点：
"同一条流水线跑在分发时的快照上，测的是一个能从自己工作树里读到答案的 agent，
并且会在没有任何可见症状的情况下报出虚高的定位准确率。"

---

## 2. ⚠️ 环境噪声没钉死 → 重放不可比

**症状**：`replay_check` 的一致率 < 1.00，或同一前缀重放出的轨迹莫名发散。

**原因**：locale / 时区 / 终端宽度 / 颜色 / hash seed / git pager 都会进入工具输出，
从而进入上下文，从而改变策略。论文假设 3（可重放环境）要求这些全部固定。

**做法**：任何入口都要先 `source pilot/config/env_pin.sh`（`scripts/common.sh` 已代劳），
其 Python 等价物是 `pilot/harness/appserver.py` 的 `PINNED_ENV`。

```bash
LC_ALL=C  LANG=C  TZ=UTC  NO_COLOR=1  TERM=dumb  COLUMNS=200  LINES=50
PYTHONHASHSEED=0  GIT_PAGER=cat  PAGER=cat  RIPGREP_CONFIG_PATH=/dev/null  CI=1
```

**改一处必须改两处**：`env_pin.sh` 与 `PINNED_ENV` 不一致，会让 shell 起的进程与
app-server 起的进程看到不同环境，这种偏差极难定位。

---

## 3. ⚠️ 续跑语在不同 arm 间不一致

**症状**：Δ_t 出现无法解释的系统性偏移。

**原因**：`CONTINUE_PROMPT`（"Continue."）进入上下文，因此**是估计量定义的一部分**。
任何 arm 间的字面差异都变成一个未申报的干预。

论文 Appendix C.1 把这列为两个"load-bearing 且容易违反"的实现不变量之一：
续跑 prompt 必须在所有 arm 间**逐字节相同**。

---

## 4. ⚠️ 编造替代动作的输出

**症状**：干预分支的 Δ 看起来很干净，但不可复制。

**原因**：`substitute_action(k, tool, args, output)` 的 `output` 必须是**真正执行** `args`
得到的结果（`pilot/harness/action_exec.py::exec_readonly_action`）。
填一个看起来合理的输出，就不再是 `do(A_t = a')`，而是在干预**观测**——
两者是不同的因果量。

这是 Appendix C.1 的第二个不变量。Algorithm 1 第 16 行写的是
"**execute** a' for real in the reset repository ▷ never a fabricated output"。

---

## 5. `rg` 在 shell 里可能是函数而不是二进制

**症状**：`action_exec` 的只读白名单校验通过，但实际执行行为与 agent 当时不同。

**原因**：`rg`/`grep` 可能被 shell 函数或 alias 覆盖。`action_exec.py` 的白名单按命令头
匹配（`READONLY_CMDS`），但真正执行走 `shell=True`。

**做法**：`env_pin.sh` 里 `RIPGREP_CONFIG_PATH=/dev/null` 消掉 ripgrep 的用户配置；
执行时用 `PINNED_ENV` 而非继承的交互式 shell 环境。

---

## 6. `shlex` 在 26.6% 的命令段上失败（论文已披露的缺陷）

**症状**：查询信道 top-3 异常低（0.029）。

**原因**：命令解析用 `shlex` 恢复搜索模式，遇到不平衡引号（如 `grep "a\b"`）会抛错，
然后退回按空白切分，保留引号字符并截断。**75.3% 的 query 对象受影响**，
其中 249 个塌缩成单个片段 `q: "def`。

**论文的处理**：不静默修掉，而是重新实现一个不依赖 `shlex` 的解析器重跑 1661 个 query 步，
结果把数字往**下**修（0.067 → 0.031）——即缺陷让 query 看起来比实际**更可预测**，
所以它偏保守，不威胁"query 是瓶颈"的结论。见 Appendix D.5 与
`pilot.analyze.x1d_query_channel`。

**复现时**：不要"顺手修好" `parsed_command.py` 就直接和论文数字比对，
两者口径会不同。

---

## 7. 上下文压缩会破坏可识别性（不是变难估，是不可识别）

**症状**：`compaction_lines()` 返回非空。

**原因**：压缩把 h_t 映射成 φ(h_t) 且不可逆。被丢掉的上下文成为**未观测混杂**，
论文假设 1 失效，Proposition 1 的证明第 1 步不成立。

**论文报告的是 0 次压缩事件**（§5.2）。若你的运行出现非零：
该轨迹的归因结论**不可直接采用**，不是"精度差一点"而是"该量不可识别"。

---

## 8. 归因阶段需要预先摆好分片目录

**症状**：`parallel --cmd attr` 什么都不做就退出。

**原因**：attr 模式不自己挑实例，它要求 `--out` 下已存在 `shard*/`，
每个 shard 内有 `results.jsonl` 与 `rollouts/`。原始实验里这步是手工做的。

**做法**：用 `setup/stage_attr_shards.py`（`scripts/02_attribution.sh` 已自动调用）。

---

## 9. 单文件实例做归因说不出话

**症状**：Δ_t 几乎全为 0，归因结论空洞。

**原因**：以二值 Y 为结果时，单目标实例几乎必然被解决，多数替代动作不改变指示函数，
于是 Δ_t ≡ 0。论文因此把归因限制在**多文件子集**（`--multifile-only`，
`parallel.py` 已写死），并把结果变量换成 **recall over F\***
（Appendix C.6：干预敏感度 2.505 vs 二值 1.643）。

即便如此仍有 **22.9% 的 episode 完全退化**、**54.7% 的估值点 Δ 恰为 0** ——
这本身是论文的结论（RQ4），不是 bug。

---

## 10. 答案要从 rollout 里解析，不是从最终文本

**症状**：定位结果偏低或缺失。

**原因**：`<answer>` 块更可靠的来源是 rollout JSONL 而非会话末尾的文本。
`pilot/probes/task.py` 同时提供 `parse_answer(text)` 与 `answer_from_rollout(items)`，
后者是主路径。

---

## 11. 判定口径有三个，别混用

论文主口径是 **`gold ⊆ answer`（recall == 1）**，同时报两个敏感性口径：

| 口径 | 含义 | 数值 |
|---|---|---|
| `recall == 1`（主） | F\* ⊆ F̂，允许多报 | **93.99%** |
| 严格集合相等 | F\* == F̂ | 15.23% |
| 过滤测试文件后相等 | 丢掉测试路径再比 | ≈ 主口径 |

93.99% 与 15.23% 的巨大差距**全部由 agent 额外点名测试文件造成**（gold patch 不含测试）。
论文因此把严格相等当作"agent 命名习惯"的诊断量，而非定位能力的度量。

---

## 12. 截断后必须检查悬空 tool call

**症状**：fork 出去的 rollout 被模型端拒绝。

**原因**：截断可能留下有 `function_call` 而没有对应 `function_call_output` 的悬空调用。
`truncate_before_step` / `truncate_including_step` 按构造安全，但任何**改写**路径
（如 `substitute_action`、`mask_output_lines`）之后都要过
`rollout.assert_no_dangling_calls()`。

---

## 13. 磁盘：完整产物约 17GB

| 目录 | 体量 | 离线分析是否需要 |
|---|---|---|
| `v4prime/` | 5.3GB | 只需其中的 `attribution.jsonl`（140KB） |
| `graphs/` | 3.4GB | RQ2 需要 |
| `full_attr/` | 3.3GB | 只需 `attribution.jsonl`（156KB） |
| `m8/` | 1.9GB | 否（m 稳健性检查，已移入 legacy） |
| `full_locate/` | 1.4GB | **需要 `rollouts/`（85MB）与 `results.jsonl`（240KB）** |

绝大部分体量是 fork 的中间产物（`attr_work/`），**离线分析不需要**。
只做路线 B 的话，约 90MB + `graphs/` 就够。

---

## 14. 统计口径：约定先定，不能看完结果再挑

论文 Appendix C.5 把统计约定固定下来，复现时不要替换：

- 置信区间 = episode 级或 instance 级 **bootstrap**
- 零假设 = **期内置换** 或 **闭式期望**，绝不用采样出来的"随机基线"
- 排序指标用**随机打破平局的闭式期望**（计数式打分器平局极多，任意打破就是未申报的设计选择）
- 不同候选集大小的预测器要放在**共同分母**上比（落在候选集外记 miss，
  否则弃权就能买到准确率）
- 多重比较 **BH-FDR，q = 0.10**，并区分探索性与验证性发现
- Wilcoxon 在 n < 6 时拒绝给 p 值，而不是给一个不可靠的

---

## 15. 论文有四项预注册没成立，别当成 bug 去"修"

Appendix E.2 的 Table 9 如实记录（复现时会重现这些"未达成"）：

| 预注册 | 结果 |
|---|---|
| 熵随步位**单调衰减**（ρ ≤ −0.2） | **不支持**：ρ = +0.072，非单调（"确定–随机–确定"） |
| v₀ 的 LORO AUROC ≥ 0.60 | **未达**：0.481（at chance），作为负结果报告 |
| degenerate 率 30–40% | **优于预期**：22.9% |
| D₁ top-1 分歧率落在 [40%,75%] | **判据退役**：观测 97.1%，但零假设本身也在 97.1% 附近 ⇒ 判据本身不成立 |

三个方向性预测（D₃ 位移、碰撞隔离、U_model 膨胀）都在预注册区间**之上**成立。
