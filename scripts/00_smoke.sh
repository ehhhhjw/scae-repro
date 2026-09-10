#!/usr/bin/env bash
# 阶段 0：可行性验证（S0）+ positivity 探针（V5）
#
# 目的不是产出论文数字，而是验掉五个决定方法可行性的未知量。论文 §5.2 报告的两项
# 前置验证就在这里产生：
#   - 重放一致性 1.00（100 个动作逐字节相同）→ 假设 3
#   - 压缩事件 0 次                          → 假设 1
# 任一项不成立，后续所有因果估计都不可信，必须先解决再往下走。
#
# 耗时：约 10–30 分钟（单实例）。API 调用量小。
set -euo pipefail
source "$(dirname "$0")/common.sh"

INSTANCE="${SCAE_SMOKE_INSTANCE:-astropy__astropy-12907}"

log "阶段 0：可行性验证（S0）"
banner
info "探针实例   $INSTANCE"
require_codex; require_api_key; require_dataset

# --- V1–V4：链路验证（fork / 只读工具面 / 无审批请求 / Continue 语义）---
log "V1–V4：app-server 链路与前缀重放语义"
python3 pilot/smoke/smoke.py \
  --instance "$INSTANCE" \
  --model "$MODEL" --model-provider "$PROVIDER" --codex-bin "$CODEX_BIN" \
  --out "$RUNS/smoke" --skip-v5 \
  --turn-timeout "$TURN_TIMEOUT"

REPORT="$RUNS/smoke/$INSTANCE/report.json"
[ -f "$REPORT" ] || die "smoke 未产出 $REPORT"
python3 - "$REPORT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("\n  S0 判据：")
def show(k, v):
    ok = v.get("pass") if isinstance(v, dict) else None
    mark = {True: "通过", False: "未通过", None: "—"}[ok]
    print(f"    {k:22s} {mark}")
for k, v in r.items():
    if isinstance(v, dict) and "pass" in v:
        show(k, v)
verdict = r.get("verdict") or r.get("go")
print(f"\n  总判定：{verdict}")
PY

# --- V5：动作熵 = positivity 的直接测量（论文 §4.3 / Appendix C.4）---
# τ_H = 0：只要重放里出现 ≥2 个不同动作就走观测分支，否则走干预分支。
log "V5：动作熵探针（positivity 测量）"
ROLLOUT="$RUNS/smoke/$INSTANCE/base.rollout.jsonl"
[ -f "$ROLLOUT" ] || die "找不到 $ROLLOUT"
python3 -m pilot.smoke.probe_v5 \
  --rollout "$ROLLOUT" --instance "$INSTANCE" \
  --model "$MODEL" --model-provider "$PROVIDER" --codex-bin "$CODEX_BIN" \
  --fork-steps 2 4 6 --samples "${SCAE_V5_SAMPLES:-6}" \
  --out "$RUNS/v5" \
  --turn-timeout "$TURN_TIMEOUT"

# --- 重放一致性：论文 §5.2 的 1.00 / 100 动作 ---
log "重放一致性检查（假设 3 的实证验证）"
python3 -m pilot.analyze.replay_check \
  --rollouts "$ROLLOUT" \
  --repeats 3 --max-actions 120 \
  --out "$RUNS/replay_check.json"
python3 - "$RUNS/replay_check.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"\n  {json.dumps(r, ensure_ascii=False)[:400]}")
print("\n  论文 §5.2 口径：重放一致率应为 1.00，且压缩事件为 0。")
print("  若一致率 < 1，说明环境未钉死（见 pilot/config/env_pin.sh），不要继续。")
PY

log "阶段 0 完成"
info "产物：$RUNS/smoke/  $RUNS/v5/  $RUNS/replay_check.json"
info "下一步：bash scripts/01_locate.sh"
