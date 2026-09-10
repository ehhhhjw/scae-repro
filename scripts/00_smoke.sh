#!/usr/bin/env bash
# Stage 0: feasibility verification (S0) + positivity probe (V5)
#
# The goal is not to produce the paper's numbers, but to kill off five unknowns that decide whether
# the method is feasible at all. The two prerequisite checks reported in paper §5.2 are produced here:
#   - replay consistency 1.00 (100 actions byte-for-byte identical) → Assumption 3
#   - compaction events 0                                          → Assumption 1
# If either fails, every downstream causal estimate is untrustworthy; fix it before going further.
#
# Runtime: about 10-30 minutes (single instance). Small API volume.
set -euo pipefail
source "$(dirname "$0")/common.sh"

INSTANCE="${SCAE_SMOKE_INSTANCE:-astropy__astropy-12907}"

log "Stage 0: feasibility verification (S0)"
banner
info "probe instance   $INSTANCE"
require_codex; require_api_key; require_dataset

# --- V1-V4: pipeline verification (fork / read-only tool surface / no approval requests / Continue semantics) ---
log "V1-V4: app-server pipeline and prefix replay semantics"
python3 pilot/smoke/smoke.py \
  --instance "$INSTANCE" \
  --model "$MODEL" --model-provider "$PROVIDER" --codex-bin "$CODEX_BIN" \
  --out "$RUNS/smoke" --skip-v5 \
  --turn-timeout "$TURN_TIMEOUT"

REPORT="$RUNS/smoke/$INSTANCE/report.json"
[ -f "$REPORT" ] || die "smoke did not produce $REPORT"
python3 - "$REPORT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("\n  S0 criteria:")
def show(k, v):
    ok = v.get("pass") if isinstance(v, dict) else None
    mark = {True: "pass", False: "fail", None: "-"}[ok]
    print(f"    {k:22s} {mark}")
for k, v in r.items():
    if isinstance(v, dict) and "pass" in v:
        show(k, v)
verdict = r.get("verdict") or r.get("go")
print(f"\n  overall verdict: {verdict}")
PY

# --- V5: action entropy = direct measurement of positivity (paper §4.3 / Appendix C.4) ---
# τ_H = 0: take the observational branch as long as the replays contain ≥2 distinct actions, otherwise the intervention branch.
log "V5: action entropy probe (positivity measurement)"
ROLLOUT="$RUNS/smoke/$INSTANCE/base.rollout.jsonl"
[ -f "$ROLLOUT" ] || die "cannot find $ROLLOUT"
python3 -m pilot.smoke.probe_v5 \
  --rollout "$ROLLOUT" --instance "$INSTANCE" \
  --model "$MODEL" --model-provider "$PROVIDER" --codex-bin "$CODEX_BIN" \
  --fork-steps 2 4 6 --samples "${SCAE_V5_SAMPLES:-6}" \
  --out "$RUNS/v5" \
  --turn-timeout "$TURN_TIMEOUT"

# --- replay consistency: the 1.00 / 100 actions of paper §5.2 ---
log "replay consistency check (empirical verification of Assumption 3)"
python3 -m pilot.analyze.replay_check \
  --rollouts "$ROLLOUT" \
  --repeats 3 --max-actions 120 \
  --out "$RUNS/replay_check.json"
python3 - "$RUNS/replay_check.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"\n  {json.dumps(r, ensure_ascii=False)[:400]}")
print("\n  Paper §5.2 definition: replay consistency rate should be 1.00, and compaction events 0.")
print("  If the rate is < 1, the environment is not pinned down (see pilot/config/env_pin.sh); do not continue.")
PY

log "Stage 0 complete"
info "artifacts: $RUNS/smoke/  $RUNS/v5/  $RUNS/replay_check.json"
info "next: bash scripts/01_locate.sh"
