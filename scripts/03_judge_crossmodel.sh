#!/usr/bin/env bash
# Stage 3: judge cross-model replication -- paper Table 2
#
# The core manipulation: model and rubric stay exactly the same, only the information set the judge sees changes
#   full-trace   conditions on  prefix + successor steps + realized outcome Y   → conditions on descendants of A_t ⇒ collider bias
#   prefix-only  conditions on  fixed 60% prefix, outcome Y hidden
# If the full-trace judge were approximating a prefix-conditional step-local quantity, this manipulation should not move where blame lands.
# The paper measures a displacement of +0.537 (sign test p = 1.1e-16), positive for all three judges.
#
# The three judges (paper Table 2 / Appendix C.2):
#   gpt-5.5-0424-global  original (produced as a side effect of run_attribution in stage 2, not in this script)
#   qwen3.6-plus         general-purpose model
#   qwen3-coder-plus     code-specialized model
# The latter two go through /chat/completions directly (pilot/analyze/judge_http.py), because their
# streaming implementation is incompatible with the /responses endpoint codex uses.
#
# ⚠️ Rate limit: 10 calls per 60-minute window by default. 25 episodes × 2 conditions = 50 calls ≈ 5.5 hours per model.
#    The paper actually used 48 calls per model and finally compared on the 23 shared episodes.
set -euo pipefail
source "$(dirname "$0")/common.sh"

ATTR="$RUNS/full_attr"
LOCATE="$RUNS/full_locate"
LIMIT="${SCAE_JUDGE_LIMIT:-25}"

log "Stage 3: judge cross-model replication (paper Table 2)"
banner
require_api_key
[ -d "$ATTR" ] || die "cannot find $ATTR, run scripts/02_attribution.sh first"

run_judge() {
  local model="$1" tag="$2"
  log "judge = $model"
  # --analyze-only recomputes on existing JSONL without spending more API budget
  python3 -m pilot.analyze.x9_judge_crossmodel \
    --attr-dir "$ATTR" --runs-root "$LOCATE" \
    --model "$model" \
    --out "$RUNS/x9_${tag}.jsonl" --report "$RUNS/x9_${tag}.json" \
    --limit "$LIMIT" --prefix-frac 0.6 \
    --max-calls "${SCAE_JUDGE_MAX_CALLS:-10}" \
    --window-min "${SCAE_JUDGE_WINDOW_MIN:-60}" \
    || info "⚠️ $model did not finish (most likely rate limiting / timeout). Add --analyze-only to draw conclusions from the data you already have."
}

run_judge "${SCAE_JUDGE_GENERAL:-qwen3.6-plus}"     "general"
run_judge "${SCAE_JUDGE_CODER:-qwen3-coder-plus}"   "coder"

# Summary: homogeneity test of the displacement (the mechanism should be consistent across judges) and heterogeneity test of the enrichment (the content should not be)
log "cross-model summary (the two tests of paper Table 2)"
JSONLS=()
for f in "$RUNS"/x9_general.jsonl "$RUNS"/x9_coder.jsonl "$RUNS"/x9_judge_crossmodel.jsonl; do
  [ -s "$f" ] && JSONLS+=("$f")
done
[ ${#JSONLS[@]} -gt 0 ] || die "no x9 JSONL artifacts at all, cannot summarize"
info "included in summary: ${JSONLS[*]}"
python3 -m pilot.analyze.x9b_crossmodel_pool \
  --jsonl "${JSONLS[@]}" \
  --attr-dir "$ATTR" --runs-root "$LOCATE" \
  --out "$RUNS/x9b_crossmodel_pool.json"

python3 - "$RUNS/x9b_crossmodel_pool.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("\n  Paper Table 2 reference:")
print("    the displacement should be positive for every judge, with no detectable heterogeneity across judges (p = 0.181)")
print("    gold-provenance enrichment is significant for only 2/3 judges, and the heterogeneity is significant (p = 0.044)")
print("    → criticism of the mechanism transfers across scorers, criticism of the content does not")
print("\n  " + json.dumps(r, ensure_ascii=False)[:700])
PY

log "Stage 3 complete"
info "next: bash scripts/04_analysis_offline.sh"
