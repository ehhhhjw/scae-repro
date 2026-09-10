#!/usr/bin/env bash
# Stage 2: prefix replay attribution -- the most expensive step in the paper
#
# Two samples are run, because the paper uses them to answer two different questions:
#
#   full_attr  70 episodes / 10 repositories / 19 failed episodes, m=3
#              → the RQ5 judge comparison (full-trace vs prefix-only), Appendix D.10 / D.11
#
#   v4prime    38 episodes, m=8, 5 equally spaced valuation points = 190 valuation points
#              → the RQ4 step-level power analysis (§5.6: 0 of 190 effects survive BH-FDR)
#              The "1901 continuations over 12.7 hours" recorded in paper §5.2 is this arm
#
# The setup from paper Appendix C.2 (this script matches it item by item):
#   m = 8 replays / valuation point    5 equally spaced valuation points / episode
#   outcome variable = recall over F*  τ_H = 0 (the observational branch holds as long as the replays contain ≥2 distinct actions)
#   prefix-only judge = fixed 60% prefix (--prefix-frac 0.6, hardcoded by parallel.py)
#   multi-file instances only (--multifile-only) -- single-file instances nearly always succeed, Δ_t ≡ 0, so attribution has nothing to say
#   per-turn timeout 900s
#
# Runtime: the v4prime arm takes about 12.7 hours with 6-way parallelism (as measured in the paper). The full_attr arm about 6-10 hours.
set -euo pipefail
source "$(dirname "$0")/common.sh"

LOCATE="$RUNS/full_locate"
STAGE="${1:-both}"   # both | judge | power

log "Stage 2: prefix replay attribution"
banner
info "localize input  $LOCATE"
info "running         $STAGE"
require_codex; require_api_key; require_dataset
[ -d "$LOCATE" ] || die "cannot find $LOCATE, run scripts/01_locate.sh first"

run_arm() {
  local name="$1" out="$2" m="$3" limit="$4" desc="$5"
  log "$name: $desc"
  info "m=$m   valuation points=5   outcome variable=recall   output=$out"

  # Sharding: lay out the localization results and rollouts in the shape parallel --cmd attr expects
  local stage_args=(--locate "$LOCATE" --out "$out" --workers "$WORKERS" --force)
  [ "$limit" != "0" ] && stage_args+=(--limit "$limit")
  python3 setup/stage_attr_shards.py "${stage_args[@]}"

  # Attribution: m replays estimate v_t, Δ_t = v_t − v_{t−1}; the full-trace / prefix-only judges run alongside
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
    print("  ⚠️ no attribution records produced at all, check shard*/attr.log")
    sys.exit(0)
pts = sum(len(r.get("causal_segments") or []) for r in recs)
degen = sum(1 for r in recs if r.get("degenerate"))
fails = sum(1 for r in recs if not r.get("Y"))
repos = {r.get("project") for r in recs}
zero = sum(1 for r in recs for s in (r.get("causal_segments") or []) if (s.get("delta") or 0) == 0)
print(f"\n  episodes {len(recs)}   repositories {len(repos)}   failed episodes {fails}")
print(f"  valuation points   {pts}   (m={m} → about {pts*m} continuations)")
print(f"  degenerate (Δ≡0) episodes           {degen}/{len(recs)} = {100*degen/len(recs):.1f}%")
print(f"  valuation points with Δ exactly 0   {zero}/{pts} = {100*zero/pts:.1f}%" if pts else "")
print("  paper reference: full_attr arm 70 episodes / 10 repositories / 19 failures / degenerate 22.9%(16/70) / Δ=0 accounts for 54.7%")
print("                   v4prime arm 38 episodes / 190 valuation points / m=8")
PY
}

# RQ5 judge arm: m=3 is enough (the judge comparison does not depend on the variance precision of v_t)
if [ "$STAGE" = "both" ] || [ "$STAGE" = "judge" ]; then
  run_arm "full_attr" "$RUNS/full_attr" "${SCAE_M_JUDGE:-3}" "0" \
    "RQ5 judge comparison sample (the 70 episodes of paper D.10)"
fi

# RQ4 step-level power arm: m=8 is the definition used in the paper's main text
if [ "$STAGE" = "both" ] || [ "$STAGE" = "power" ]; then
  run_arm "v4prime" "$RUNS/v4prime" "${SCAE_M_POWER:-8}" "${SCAE_N_POWER:-38}" \
    "RQ4 step-level power sample (the 190 valuation points of paper §5.6)"
fi

log "Stage 2 complete"
info "artifacts: $RUNS/full_attr/attribution.jsonl   $RUNS/v4prime/attribution.jsonl"
info "next: bash scripts/03_judge_crossmodel.sh (cross-model replication) or scripts/04_analysis_offline.sh"
