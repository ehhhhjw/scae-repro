#!/usr/bin/env bash
# Stage 5: K=4 repeated rollouts for U_policy -- the only component in paper Appendix D.7 estimated directly on real traces
#
# In the three-way decomposition H[Y|q,G] = U_task + U_policy + U_model, only U_policy can be estimated
# directly from real data: fix the same q, run K rollouts, and take the mean of the per-instance outcome variances.
#   U_policy = mean_i Var(Y_i^{(1..K)})
# The paper reports **0.049**, from 23 instances with K=4 repeated rollouts.
#
# The main pipeline (stage 1) uses K=1, because 500 instances × K=4 is too expensive. So this arm runs a small subset separately.
#
# ⚠️ Honest note on reproduction definitions
# The 23 instance ids the paper actually used are a data artifact (runs/upolicy_ids.txt) and
# **cannot be derived from the code**. This script re-selects them with an explicitly written sampling rule:
# stratified sampling by repository among instances that localized successfully and have enough steps. So the
# U_policy you get will be close to but not necessarily equal to 0.0489. If you have the original id list,
# point SCAE_UPOLICY_IDS at it.
#
# Runtime: 23 instances × K=4 = 92 localizations, about 2-4 hours with 6-way parallelism.
set -euo pipefail
source "$(dirname "$0")/common.sh"

LOCATE="$RUNS/full_locate"
OUT="$RUNS/upolicy"
N="${SCAE_UPOLICY_N:-23}"
K="${SCAE_UPOLICY_K:-4}"
IDS_FILE="${SCAE_UPOLICY_IDS:-$RUNS/upolicy_ids.txt}"

log "Stage 5: U_policy (K=$K repeated rollouts)"
banner
require_codex; require_api_key; require_dataset
[ -d "$LOCATE" ] || die "cannot find $LOCATE, run scripts/01_locate.sh first"

# --- pick instances ---
if [ -s "$IDS_FILE" ]; then
  log "using the existing id list: $IDS_FILE"
else
  log "stratified sampling of $N instances by repository (the rule is written in the python below, and is reproducible)"
  python3 - "$LOCATE" "$N" "$IDS_FILE" <<'PY'
import json, sys, collections
from pathlib import Path
locate, n, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])

# Read the localization results and deduplicate
uniq = {}
for f in [locate / "results.jsonl"] + sorted(locate.glob("shard*/results.jsonl")):
    if not f.exists():
        continue
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            uniq.setdefault(r["instance_id"], r)

# Sampling rule (deterministic, no random numbers):
#   1) keep only instances that executed successfully
#   2) n_steps >= 5 -- traces that are too short have nothing to vary between repeats, variance is always 0, budget wasted
#   3) round-robin across repositories, lexicographic by instance_id within a repository, so django does not eat the whole sample
ok = [r for r in uniq.values() if r.get("ok") and (r.get("n_steps") or 0) >= 5]
by_repo = collections.defaultdict(list)
for r in ok:
    by_repo[r.get("project") or r["instance_id"].split("__")[0]].append(r["instance_id"])
for k in by_repo:
    by_repo[k].sort()

picked, repos = [], sorted(by_repo, key=lambda k: (-len(by_repo[k]), k))
i = 0
while len(picked) < n and any(by_repo.values()):
    rp = repos[i % len(repos)]
    if by_repo[rp]:
        picked.append(by_repo[rp].pop(0))
    i += 1

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(picked) + "\n", encoding="utf-8")
print(f"  {len(ok)} usable instances, picked {len(picked)} → {out}")
print("  repository distribution:", dict(collections.Counter(p.split('__')[0] for p in picked)))
PY
fi

IDS="$(tr '\n' ' ' < "$IDS_FILE")"
[ -n "${IDS// /}" ] || die "$IDS_FILE is empty"

# --- run K repeats ---
log "running K=$K times for each of $(wc -w <<< "$IDS") instances"
mkdir -p "$OUT"
python3 -m pilot.harness.runner \
  --model "$MODEL" --model-provider "$PROVIDER" --codex-bin "$CODEX_BIN" \
  --k "$K" --instances $IDS \
  --out "$OUT" --turn-timeout "$TURN_TIMEOUT" --progress-every 5

# --- estimate U_policy ---
log "estimating U_policy"
python3 - "$OUT/results.jsonl" "$RUNS/upolicy_t3.json" <<'PY'
import json, sys
from pathlib import Path
# common.sh has already added the repository root to PYTHONPATH
from pilot.analyze.full_report import t3_uq_decomposition

res = Path(sys.argv[1])
recs = [json.loads(l) for l in res.read_text(encoding="utf-8").splitlines() if l.strip()]
out = t3_uq_decomposition(recs)
Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n  " + json.dumps(out, ensure_ascii=False))
print(f"\n  → {sys.argv[2]}")
print("  Paper Appendix D.7: U_policy = 0.049, from 23 instances with K=4")
print("  (a different instance set ⇒ a different number; this is sampling variation, not an implementation difference)")
PY

log "Stage 5 complete"
info "artifacts: $OUT/results.jsonl  and  $RUNS/upolicy_t3.json"
info "this JSON is read by pilot.analyze.x_synthesis into the final summary"
