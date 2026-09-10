#!/usr/bin/env bash
# 全流程一键复现：阶段 0 → 1 → 2 → 3 → 4
#
# ⚠️ 这会花掉 30+ 小时真实 agent 执行时间与相当数量的 API 调用。
#    先读 REPRODUCE.md 确认预算，或用 SCAE_LIMIT 先跑小样本验证链路：
#
#      SCAE_LIMIT=10 bash scripts/run_all.sh
#
# 只想复现论文的分析与统计（不重跑 agent）：直接
#      bash scripts/04_analysis_offline.sh
# 但它需要阶段 1–3 的录制产物；复现包不携带这些数据。
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "SCAE 全流程复现"
banner
info "预计耗时   阶段1 ~10-20h + 阶段2 ~12.7h + 阶段3 ~11h + 阶段5 ~2-4h + 阶段4 几分钟"
[ -n "${SCAE_LIMIT:-}" ] && info "限流       SCAE_LIMIT=$SCAE_LIMIT（小样本冒烟）"

if [ "${SCAE_YES:-0}" != "1" ]; then
  printf '\n继续？这会消耗大量 API 预算。输入 yes 确认：'
  read -r ans
  [ "$ans" = "yes" ] || { echo "已取消。"; exit 0; }
fi

bash "$REPO_ROOT/scripts/00_smoke.sh"
bash "$REPO_ROOT/scripts/01_locate.sh"
bash "$REPO_ROOT/scripts/02_attribution.sh" both
bash "$REPO_ROOT/scripts/03_judge_crossmodel.sh" || info "阶段 3 未跑完（限流常见），继续分析"
bash "$REPO_ROOT/scripts/05_upolicy.sh" || info "阶段 5 未跑完，U_policy 将标为不可估"
bash "$REPO_ROOT/scripts/04_analysis_offline.sh"

log "全流程完成"
info "产物在 $RUNS/，论文数字映射见 docs/PAPER_MAP.md"
