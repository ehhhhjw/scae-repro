#!/usr/bin/env bash
# 阶段 4：全部离线分析 —— 零 API、零 GPU、CPU 秒级到分钟级
#
# 本阶段只读阶段 1–3 的录制产物，不再调用任何模型。论文的每个数字都由这里的某个
# 脚本写进一个 JSON。按研究问题分组，与论文 §5.3–§5.7 一一对应。
# 逐个数字到脚本的映射见 docs/PAPER_MAP.md。
#
# 用法：
#   bash scripts/04_analysis_offline.sh          # 全跑
#   bash scripts/04_analysis_offline.sh rq1      # 只跑某一组：rq1|rq2|rq3|rq4|rq5|report|selftest
set -euo pipefail
source "$(dirname "$0")/common.sh"

GROUP="${1:-all}"
LOCATE="$RUNS/full_locate"
ATTR="$RUNS/full_attr"
V4P="$RUNS/v4prime"

log "阶段 4：离线分析（组=$GROUP）"
banner

FAILED=(); SKIPPED=()

# 跑一个分析脚本；缺输入就跳过而不是让整个阶段崩掉
run() {
  local label="$1"; shift
  printf '\n  \033[1m→ %s\033[0m\n' "$label"
  printf '    $ %s\n' "python3 $*"
  if python3 "$@"; then
    return 0
  else
    printf '    \033[31m失败\033[0m（多半是缺输入产物；见上方报错）\n'
    FAILED+=("$label")
    return 0
  fi
}

need() {  # need <路径> <标签> —— 输入不存在则登记跳过
  if [ ! -e "$1" ]; then
    printf '\n  \033[33m跳过 %s\033[0m —— 缺 %s\n' "$2" "$1"
    SKIPPED+=("$2 (缺 $1)")
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------- 自检（无需数据）
if [ "$GROUP" = "all" ] || [ "$GROUP" = "selftest" ]; then
  log "UQ 自检（合成数据，不需要任何录制产物）"
  # 论文 Appendix D.7：三分解在已知生成过程上参数恢复相关 0.994，
  # U_model 在支撑外膨胀 8.1 倍。这是对三分解唯一的正面验证。
  run "D.7 UQ 三分解自检" -m pilot.analyze.uq_selftest --out "$RUNS/uq_selftest.json"
fi

# ---------------------------------------------------------------- RQ1 下一动作预测
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rq1" ]; then
  log "RQ1：什么预测下一个动作？（论文 §5.3 / Table 1 / Table 10 / Table 11）"
  if need "$LOCATE" "RQ1"; then
    # 共享分母的 509 决策点 + 首访/重访双 regime 分解 → Table 1 的分母口径
    run "V3g 共享分母与 regime 分解" -m pilot.analyze.v3g_regime \
      --runs-root "$LOCATE" --out "$RUNS/v3g_regime.json"
    # 执行来源（provenance）预测器族：uniform / 频次 / recency / last-output → Table 1 主行
    run "V3j 执行来源图预测（Table 1 核心）" -m pilot.analyze.v3j_trace_graph \
      --runs-root "$LOCATE" --out "$RUNS/v3j_trace_graph.json"
    # 分布质量：top-k 曲线 / MRR / ECE / NLL / 有效候选数 exp(H)
    run "V3b 预测分布质量" -m pilot.analyze.v3b_distribution_quality \
      --runs-root "$LOCATE" --out "$RUNS/v3b_distribution.json"
    # 工具信道熵 H_tool，供 RQ4 的 x4/x6 复用
    run "工具信道 Markov 模型" -m pilot.channels.tool_model \
      --runs-root "$LOCATE" --out "$RUNS/tool_model_official.json"
    run "X1 联合 L1 分布" -m pilot.analyze.x1_joint_l1 \
      --runs-root "$LOCATE" --tool-model "$RUNS/tool_model_official.json" \
      --prov-results "$RUNS/v3j_trace_graph.json" --out "$RUNS/x1_joint_l1.json"
    # 工具与对象类型的依赖 I=0.159 nats（p=5e-4）⇒ 因子化 P(tool)P(obj) 是误设
    run "X1b 工具/对象条件独立性检验（Table 11）" -m pilot.analyze.x1b_joint_independence \
      --runs-root "$LOCATE" --out "$RUNS/x1b_joint_independence.json"
    # 按预测工具路由候选集 vs 因子化共享候选集 → Table 10
    run "X1c 工具路由候选集（Table 10）" -m pilot.analyze.x1c_tool_routed \
      --runs-root "$LOCATE" --out "$RUNS/x1c_tool_routed.json"
    # 查询信道：top-3=0.029 是真上界还是 shlex 解析缺陷（26.6% 命令段）→ Appendix D.5
    run "X1d 查询信道与解析缺陷（D.5）" -m pilot.analyze.x1d_query_channel \
      --runs-root "$LOCATE" --out "$RUNS/x1d_query_channel.json"
  fi
fi

# ---------------------------------------------------------------- RQ2 代码依赖图
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rq2" ]; then
  log "RQ2：代码依赖图到底有什么用？（论文 §5.4 / Figure 7 / D.3）"
  if need "$LOCATE" "RQ2"; then
    # 图的两个角色：不描述步间移动（一跳提升 0.94×），但描述工作区域（内部边密度 31×）
    # 需要 pilot/runs/graphs/<instance_id>/ 的构图产物，见 REPRODUCE.md「图信道」一节
    if [ ! -d "$RUNS/graphs" ]; then
      printf '\n  \033[33m注意\033[0m RQ2 需要 %s（构图产物，71 个实例）。\n' "$RUNS/graphs"
      printf '    构图依赖两个外部工具，未随复现包分发；见 REPRODUCE.md「图信道的外部依赖」。\n'
      printf '    缺失时下面的脚本会退化为只报能解析的子集。\n'
      SKIPPED+=("RQ2 图信道完整性 (缺 $RUNS/graphs)")
    fi
    run "V3i 图探针（Figure 7 / §5.4）" -m pilot.analyze.v3i_graph_probe \
      --runs-root "$LOCATE" --out "$RUNS/v3i_graph_probe.json"
  fi
fi

# ---------------------------------------------------------------- RQ3 复用模式
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rq3" ]; then
  log "RQ3：哪些复用模式是真的？（论文 §5.5 / D.4）"
  if need "$LOCATE" "RQ3"; then
    # 复用「率」是置换不变量（零方差零信息），复用「模式」才有信息：
    # 连续步重复 1.83×（p~1e-141）、访问计数 1.70×、recency 秩 1.74×；
    # 工具自重复 1.02×（p=0.51，不显著）⇒ 民间说法的工具那一半是假的
    run "V3h 复用结构置换检验" -m pilot.analyze.v3h_reuse \
      --runs-root "$LOCATE" --out "$RUNS/v3h_reuse.json"
  fi
fi

# ---------------------------------------------------------------- RQ4 不确定性层级
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rq4" ]; then
  log "RQ4：执行不确定性住在哪一层？（论文 §5.6 / Figure 8 / D.6 / D.7）"
  # 零成本动作熵是否能代理重放方差 → ρ=−0.088（步级）/−0.219（实例级）/−0.425（正控制）
  if need "$V4P" "RQ4 步级"; then
    run "X4 结构熵 vs 重放方差" -m pilot.analyze.x4_uexec_corr \
      --v4prime-dir "$V4P" --tool-model "$RUNS/tool_model_official.json" \
      --runs-root "$LOCATE" --out "$RUNS/x4_uexec_corr.json"
    # U^exec 有无非机械结构：实例 ICC=0.640（置换零 0.196）、步位 ρ=−0.131、工具类型 p=0.65
    run "X4b U^exec 结构效度（Figure 8a）" -m pilot.analyze.x4b_uexec_validity \
      --v4prime "$V4P/attribution.jsonl" --runs-root "$LOCATE" \
      --out "$RUNS/x4b_uexec_validity.json"
    run "X4c 实例层复核" -m pilot.analyze.x4c_instance_level \
      --v4prime "$V4P/attribution.jsonl" --runs-root "$LOCATE" \
      --out "$RUNS/x4c_instance_level.json"
    # 论文的核心负结果：190 个效应中 5.3% 达到未校正 p<0.05，BH-FDR(q=0.10) 后 0 个存活
    run "X5 步级统计功效（§5.6 核心负结果）" -m pilot.analyze.x5_step_power \
      --v4prime-dir "$V4P" --runs-root "$LOCATE" --out "$RUNS/x5_step_power.json"
    # X6 报的 5× 与 X6b 对它的否证：阈值扫描显示 α 放宽时比值从 3.36 单调衰到 1.71
    run "X6 预算分配（被 X6b 否证的那个结果）" -m pilot.analyze.x6_budget_alloc \
      --v4prime-dir "$V4P" --tool-model "$RUNS/tool_model_official.json" \
      --out "$RUNS/x6_budget_alloc.json"
    run "X6b 二值判据的伪例（D.6）" -m pilot.analyze.x6b_effect_criterion \
      --v4prime "$V4P/attribution.jsonl" --runs-root "$LOCATE" \
      --out "$RUNS/x6b_effect_criterion.json"
    # 被取代的 Bernoulli 近似 U^exec=4v(1−v)，论文 C.3 把它当负控制保留
    run "V4 结构 vs 执行（负控制，C.3）" -m pilot.analyze.v4_structural_vs_exec \
      --attr "$V4P/attribution.jsonl" --locate "$LOCATE/results.jsonl" \
      --runs-root "$LOCATE" --tool-ent "$RUNS/tool_model_official.json" \
      --out "$RUNS/v4_structural_vs_exec.json"
  fi
  if need "$ATTR/attribution.jsonl" "RQ4 过程不确定性"; then
    # degenerate 率 22.9%、脆弱成功、收敛点 t*
    run "过程不确定性（degenerate 率）" -m pilot.analyze.process_uncertainty \
      --attr "$ATTR/attribution.jsonl" --out "$RUNS/process_uncertainty.json"
    # 三态记账：INESTIMABLE / NO_EFFECT / EFFECT
    run "可估性三态记账" -m pilot.analyze.estimability \
      --attr "$ATTR/attribution.jsonl" --locate "$LOCATE/results.jsonl" \
      --runs-root "$LOCATE" --out "$RUNS/estimability.json"
  fi
  # 结果变量选择：recall 的干预敏感度 2.505 > F1 2.165 > 二值成功 1.643（Appendix C.6）
  if need "$LOCATE" "C.6 结果变量筛选"; then
    run "X8 结果变量敏感性（C.6）" -m pilot.analyze.x8_outcome_sensitivity \
      --runs-root "$LOCATE" --out "$RUNS/x8_outcome_sensitivity.json"
  fi
  # positivity 的局部性：42.9%/50% 探测点有动作随机性，与步位的秩相关 +0.072（非单调）
  if need "$RUNS/v5_scan" "C.4 熵结构"; then
    run "C.4 熵扫描（Table 6 / Figure 5）" -m pilot.analyze.entropy_scan \
      --scan-dir "$RUNS/v5_scan"
  fi
fi

# ---------------------------------------------------------------- RQ5 归因与碰撞偏差
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rq5" ]; then
  log "RQ5：观测性归因到底在测什么？（论文 §5.7 / Table 2 / Table 12 / Figure 4）"
  if need "$ATTR/attribution.jsonl" "RQ5"; then
    # D1 分歧率（已被论文退役，见 D.12）与 D3 位置位移 +0.537（碰撞隔离的核心量）
    run "D1/D3 与碰撞隔离（Figure 4a）" -m pilot.analyze.attribution_compare \
      --attr "$ATTR/attribution.jsonl" --out "$RUNS/d1d3_report.json"
    # |Δ_t| 是否集中在 gold-provenance 步；judge/causal top-1 落在该类的比率
    run "V3k 轨迹因果与 gold-provenance" -m pilot.analyze.v3k_trace_causal \
      --attr-dir "$ATTR" --runs-root "$LOCATE" --out "$RUNS/v3k_trace_causal.json"
    # 零 LLM 结构规则能否复现 judge 的选择 → Table 12，最好的规则只比猜类内好 1.7×
    run "X2 judge 机制与零 LLM 规则（Table 12）" -m pilot.analyze.x2_judge_mechanism \
      --attr-dir "$ATTR" --runs-root "$LOCATE" --out "$RUNS/x2_judge_mechanism.json"
  fi
fi

# ---------------------------------------------------------------- 汇总报告
if [ "$GROUP" = "all" ] || [ "$GROUP" = "report" ]; then
  log "汇总报告（论文 Appendix E.1 记载的两个命令）"
  if need "$LOCATE/results.jsonl" "汇总"; then
    # 定位表（Table 7）+ v_0 的 LORO 校准与预测（Table 8 / D.8）
    run "全量报告（Table 7 / D.8）" -m pilot.analyze.full_report \
      --locate "$LOCATE/results.jsonl" --attr "$ATTR/attribution.jsonl" \
      --scan "$RUNS/v5_scan" "$RUNS/v5" --out "$RUNS/analysis.json"
  fi
  # 把各 JSON 聚成论文宏定义用的单一文件
  run "结论聚合" -m pilot.analyze.x_synthesis --out "$RUNS/synthesis.json"
fi

# ---------------------------------------------------------------- 结语
log "阶段 4 完成"
if [ ${#SKIPPED[@]} -gt 0 ]; then
  printf '\n\033[33m跳过 %d 项（缺输入）：\033[0m\n' "${#SKIPPED[@]}"
  for s in "${SKIPPED[@]}"; do printf '  - %s\n' "$s"; done
fi
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '\n\033[31m失败 %d 项：\033[0m\n' "${#FAILED[@]}"
  for f in "${FAILED[@]}"; do printf '  - %s\n' "$f"; done
  printf '\n失败通常意味着上游产物缺失或不完整，而不是分析逻辑坏了。\n'
fi
info "产物 JSON 都在 $RUNS/ 下。论文每个数字对应哪个 JSON 字段，见 docs/PAPER_MAP.md"
[ ${#FAILED[@]} -eq 0 ] || exit 1
