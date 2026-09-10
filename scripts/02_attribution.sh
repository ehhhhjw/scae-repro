#!/usr/bin/env bash
# 阶段 2：前缀重放归因 —— 论文最贵的一步
#
# 跑两个样本，因为论文用它们回答两个不同的问题：
#
#   full_attr  70 episode / 10 仓库 / 19 个失败 episode，m=3
#              → RQ5 的 judge 对比（full-trace vs prefix-only），Appendix D.10 / D.11
#
#   v4prime    38 episode，m=8，5 个等距估值点 = 190 个估值点
#              → RQ4 的步级功效（§5.6：190 个效应中 0 个通过 BH-FDR）
#              论文 §5.2 记的「1901 continuations over 12.7 hours」就是这一档
#
# 论文 Appendix C.2 的设置（本脚本逐项对齐）：
#   m = 8 replays / 估值点      5 个等距估值点 / episode
#   结果变量 = recall over F*    τ_H = 0（观测分支只要重放里出现 ≥2 个不同动作就成立）
#   prefix-only judge = 固定 60% 前缀（--prefix-frac 0.6，由 parallel.py 写死）
#   只跑多文件实例（--multifile-only）—— 单文件实例几乎必成功，Δ_t ≡ 0，归因说不出话
#   单轮超时 900s
#
# 耗时：v4prime 档并行 6 路约 12.7 小时（论文实测）。full_attr 档约 6–10 小时。
set -euo pipefail
source "$(dirname "$0")/common.sh"

LOCATE="$RUNS/full_locate"
STAGE="${1:-both}"   # both | judge | power

log "阶段 2：前缀重放归因"
banner
info "定位输入   $LOCATE"
info "本次跑     $STAGE"
require_codex; require_api_key; require_dataset
[ -d "$LOCATE" ] || die "找不到 $LOCATE，先跑 scripts/01_locate.sh"

run_arm() {
  local name="$1" out="$2" m="$3" limit="$4" desc="$5"
  log "$name：$desc"
  info "m=$m   估值点=5   结果变量=recall   输出=$out"

  # 分片：把定位结果与 rollouts 摆成 parallel --cmd attr 期望的形状
  local stage_args=(--locate "$LOCATE" --out "$out" --workers "$WORKERS" --force)
  [ "$limit" != "0" ] && stage_args+=(--limit "$limit")
  python3 setup/stage_attr_shards.py "${stage_args[@]}"

  # 归因：m 次重放估 v_t，Δ_t = v_t − v_{t−1}；同时跑 full-trace / prefix-only judge
  python3 -m pilot.harness.parallel \
    --cmd attr --workers "$WORKERS" --out "$out" \
    --model "$MODEL" --model-provider "$PROVIDER" \
    --m "$m" --window-points 5 --outcome recall \
    --turn-timeout "$TURN_TIMEOUT"

  python3 - "$out" "$m" <<'PY'
import json, sys, glob
out, m = sys.argv[1], int(sys.argv[2])
recs = []
for p in sorted(glob.glob(f"{out}/shard*/attribution.jsonl")) or [f"{out}/attribution.jsonl"]:
    try:
        recs += [json.loads(l) for l in open(p) if l.strip()]
    except FileNotFoundError:
        pass
if not recs:
    print("  ⚠️ 没产出任何归因记录，检查 shard*/attr.log")
    sys.exit(0)
pts = sum(len(r.get("causal_segments") or []) for r in recs)
degen = sum(1 for r in recs if r.get("degenerate"))
fails = sum(1 for r in recs if not r.get("Y"))
repos = {r.get("project") for r in recs}
zero = sum(1 for r in recs for s in (r.get("causal_segments") or []) if (s.get("delta") or 0) == 0)
print(f"\n  episode {len(recs)}   仓库 {len(repos)}   失败 episode {fails}")
print(f"  估值点   {pts}   （m={m} → 约 {pts*m} 次续跑）")
print(f"  degenerate（Δ≡0）episode  {degen}/{len(recs)} = {100*degen/len(recs):.1f}%")
print(f"  Δ 恰为 0 的估值点          {zero}/{pts} = {100*zero/pts:.1f}%" if pts else "")
print("  论文对照：full_attr 档 70 episode / 10 仓库 / 19 失败 / degenerate 22.9%(16/70) / Δ=0 占 54.7%")
print("            v4prime 档 38 episode / 190 估值点 / m=8")
PY
}

# RQ5 judge 档：m=3 足够（judge 对比不依赖 v_t 的方差精度）
if [ "$STAGE" = "both" ] || [ "$STAGE" = "judge" ]; then
  run_arm "full_attr" "$RUNS/full_attr" "${SCAE_M_JUDGE:-3}" "0" \
    "RQ5 judge 对比样本（论文 D.10 的 70 episode）"
fi

# RQ4 步级功效档：m=8 是论文正文的口径
if [ "$STAGE" = "both" ] || [ "$STAGE" = "power" ]; then
  run_arm "v4prime" "$RUNS/v4prime" "${SCAE_M_POWER:-8}" "${SCAE_N_POWER:-38}" \
    "RQ4 步级功效样本（论文 §5.6 的 190 估值点）"
fi

log "阶段 2 完成"
info "产物：$RUNS/full_attr/attribution.jsonl   $RUNS/v4prime/attribution.jsonl"
info "下一步：bash scripts/03_judge_crossmodel.sh（跨模型复制）或 scripts/04_analysis_offline.sh"
