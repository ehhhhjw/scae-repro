#!/usr/bin/env bash
# Stage 1: full-set file localization (S1) -- the corpus foundation for every conclusion in the paper
#
# Runs file localization once for each of the 500 SWE-bench-Verified instances, producing:
#   - results.jsonl        per-instance Y / recall / n_steps / predicted file set (RQ1-RQ3 + v_0 corpus)
#   - rollouts/*.jsonl     full execution traces (the input to prefix replay, required by stage 2)
#
# Paper definitions (§5.2 / Table 7): 499 episodes, 12 repositories, localization success rate 93.99%.
# The 500 → 499 gap is django__django-15278, which lacks a repository snapshot (the dataset only has 499 projects directories).
#
# ⚠️ Answer-leakage gate: every instance is forcibly git reset back to base_commit before it runs.
#    The workspaces shipped with the dataset have the gold patch applied (= already-fixed state);
#    without the reset you are letting the agent read the answer.
#    This step is guaranteed by pilot/probes/task.py::reset_repo -- do not bypass it. See docs/PITFALLS.md.
#
# Runtime: about 10-20 hours with 6-way parallelism. API volume ~500 long sessions.
set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT="$RUNS/full_locate"
K="${SCAE_K:-1}"
LIMIT="${SCAE_LIMIT:-0}"

log "Stage 1: full-set file localization"
banner
info "output            $OUT"
info "repeats/instance  K=$K"
[ "$LIMIT" != "0" ] && info "debug throttle    only the first $LIMIT instances"
require_codex; require_api_key; require_dataset

log "Launching $WORKERS-way parallel localization (sharded by repo, so no repository directory is reset concurrently)"
EXTRA=()
[ "$LIMIT" != "0" ] && EXTRA+=(--limit "$LIMIT")
python3 -m pilot.harness.parallel \
  --cmd locate --workers "$WORKERS" --out "$OUT" \
  --model "$MODEL" --model-provider "$PROVIDER" \
  --k "$K" --turn-timeout "$TURN_TIMEOUT" "${EXTRA[@]}"

log "Localization results summary"
python3 - "$OUT" <<'PY'
import json, sys, collections
from pathlib import Path
out = Path(sys.argv[1])
files = [out / "results.jsonl"] + sorted(out.glob("shard*/results.jsonl"))
uniq = {}
for f in files:
    if not f.exists():
        continue
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        i = r["instance_id"]
        if i not in uniq or (r.get("ok") and not uniq[i].get("ok")):
            uniq[i] = r
ok = [r for r in uniq.values() if r.get("ok")]
bad = [r for r in uniq.values() if not r.get("ok")]
print(f"  unique instances  {len(uniq)}")
print(f"  executed ok       {len(ok)}")
if bad:
    print(f"  execution failed  {len(bad)}:")
    for r in bad[:10]:
        print(f"    {r['instance_id']}: {str(r.get('error'))[:110]}")

# The definition behind paper Table 7: F* ⊆ F̂ (recall == 1)
hit = [r for r in ok if (r.get("recall") or 0) >= 1.0]
if ok:
    print(f"\n  localization success rate {len(hit)}/{len(ok)} = {100*len(hit)/len(ok):.2f}%   (paper Table 7: 93.99%)")
per = collections.defaultdict(lambda: [0, 0])
for r in ok:
    p = r.get("project") or r["instance_id"].split("__")[0]
    per[p][1] += 1
    if (r.get("recall") or 0) >= 1.0:
        per[p][0] += 1
print("\n  by repository (compare against paper Table 7):")
for p, (h, n) in sorted(per.items(), key=lambda kv: -kv[1][1]):
    print(f"    {p:16s} n={n:4d}  {100*h/n:6.1f}%")
PY

log "Stage 1 complete"
info "artifacts: $OUT/results.jsonl  +  $OUT/shard*/rollouts/*.jsonl"
info "next: bash scripts/02_attribution.sh   (or run 04 first for the purely offline RQ1-RQ3 analysis)"
