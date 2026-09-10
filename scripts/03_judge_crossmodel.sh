#!/usr/bin/env bash
# 阶段 3：judge 跨模型复制 —— 论文 Table 2
#
# 核心操纵：模型与 rubric 完全不变，只改 judge 能看到的信息集
#   full-trace  条件于  前缀 + 后继步骤 + 实现的结果 Y   → 条件于 A_t 的后代 ⇒ 碰撞偏差
#   prefix-only 条件于  固定 60% 前缀，结果 Y 隐藏
# 若 full-trace judge 在逼近一个前缀条件化的步局部量，这个操纵不应移动归责位置。
# 论文实测位移 +0.537（sign test p = 1.1e-16），三个 judge 全为正。
#
# 三个 judge（论文 Table 2 / Appendix C.2）：
#   gpt-5.5-0424-global  原始（由阶段 2 的 run_attribution 顺带产出，不在本脚本内）
#   qwen3.6-plus         通用模型
#   qwen3-coder-plus     代码专用模型
# 后两个走 /chat/completions 直连（pilot/analyze/judge_http.py），因为它们的
# 流式实现与 codex 用的 /responses 端点不兼容。
#
# ⚠️ 限流：默认 10 次调用 / 60 分钟窗口。25 episode × 2 条件 = 50 次调用 ≈ 5.5 小时/模型。
#    论文实际用 48 次调用/模型，最终在 23 个共同 episode 上比较。
set -euo pipefail
source "$(dirname "$0")/common.sh"

ATTR="$RUNS/full_attr"
LOCATE="$RUNS/full_locate"
LIMIT="${SCAE_JUDGE_LIMIT:-25}"

log "阶段 3：judge 跨模型复制（论文 Table 2）"
banner
require_api_key
[ -d "$ATTR" ] || die "找不到 $ATTR，先跑 scripts/02_attribution.sh"

run_judge() {
  local model="$1" tag="$2"
  log "judge = $model"
  # --analyze-only 可在已有 JSONL 上重算，不再花 API
  python3 -m pilot.analyze.x9_judge_crossmodel \
    --attr-dir "$ATTR" --runs-root "$LOCATE" \
    --model "$model" \
    --out "$RUNS/x9_${tag}.jsonl" --report "$RUNS/x9_${tag}.json" \
    --limit "$LIMIT" --prefix-frac 0.6 \
    --max-calls "${SCAE_JUDGE_MAX_CALLS:-10}" \
    --window-min "${SCAE_JUDGE_WINDOW_MIN:-60}" \
    || info "⚠️ $model 未跑完（多半是限流/超时）。可加 --analyze-only 就已有数据出结论。"
}

run_judge "${SCAE_JUDGE_GENERAL:-qwen3.6-plus}"     "general"
run_judge "${SCAE_JUDGE_CODER:-qwen3-coder-plus}"   "coder"

# 汇总：位移的同质性检验（机制应跨 judge 一致）与富集的异质性检验（内容不应一致）
log "跨模型汇总（论文 Table 2 的两个检验）"
JSONLS=()
for f in "$RUNS"/x9_general.jsonl "$RUNS"/x9_coder.jsonl "$RUNS"/x9_judge_crossmodel.jsonl; do
  [ -s "$f" ] && JSONLS+=("$f")
done
[ ${#JSONLS[@]} -gt 0 ] || die "没有任何 x9 JSONL 产物，无法汇总"
info "参与汇总：${JSONLS[*]}"
python3 -m pilot.analyze.x9b_crossmodel_pool \
  --jsonl "${JSONLS[@]}" \
  --attr-dir "$ATTR" --runs-root "$LOCATE" \
  --out "$RUNS/x9b_crossmodel_pool.json"

python3 - "$RUNS/x9b_crossmodel_pool.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("\n  论文 Table 2 对照：")
print("    位移应在每个 judge 上为正，且 judge 间无可检出异质性（p = 0.181）")
print("    gold-provenance 富集只在 2/3 个 judge 上显著，异质性显著（p = 0.044）")
print("    → 机制的批评可跨 scorer 迁移，内容的批评不行")
print("\n  " + json.dumps(r, ensure_ascii=False)[:700])
PY

log "阶段 3 完成"
info "下一步：bash scripts/04_analysis_offline.sh"
