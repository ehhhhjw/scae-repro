#!/usr/bin/env bash
# One-shot full-pipeline reproduction: stage 0 → 1 → 2 → 3 → 4
#
# ⚠️ This will burn 30+ hours of real agent execution time and a substantial number of API calls.
#    Read REPRODUCE.md first to confirm the budget, or use SCAE_LIMIT to check the pipeline on a small sample:
#
#      SCAE_LIMIT=10 bash scripts/run_all.sh
#
# If you only want to reproduce the paper's analyses and statistics (without re-running the agent), just run
#      bash scripts/04_analysis_offline.sh
# But it needs the recorded artifacts of stages 1-3; the reproduction package does not carry that data.
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "SCAE full-pipeline reproduction"
banner
info "estimated time  stage1 ~10-20h + stage2 ~12.7h + stage3 ~11h + stage5 ~2-4h + stage4 a few minutes"
[ -n "${SCAE_LIMIT:-}" ] && info "throttle        SCAE_LIMIT=$SCAE_LIMIT (small-sample smoke)"

if [ "${SCAE_YES:-0}" != "1" ]; then
  printf '\nContinue? This will consume a large API budget. Type yes to confirm: '
  read -r ans
  [ "$ans" = "yes" ] || { echo "Cancelled."; exit 0; }
fi

bash "$REPO_ROOT/scripts/00_smoke.sh"
bash "$REPO_ROOT/scripts/01_locate.sh"
bash "$REPO_ROOT/scripts/02_attribution.sh" both
bash "$REPO_ROOT/scripts/03_judge_crossmodel.sh" || info "stage 3 did not finish (rate limiting is common), continuing to analysis"
bash "$REPO_ROOT/scripts/05_upolicy.sh" || info "stage 5 did not finish, U_policy will be marked inestimable"
bash "$REPO_ROOT/scripts/04_analysis_offline.sh"

log "full pipeline complete"
info "artifacts are in $RUNS/; for the mapping to the paper's numbers see docs/PAPER_MAP.md"
