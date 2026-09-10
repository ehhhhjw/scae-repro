#!/usr/bin/env bash
# The remaining experiment pipeline after full-set localization finishes -- sequential, no manual intervention needed.
#   1) merge the full-set localization results + first-pass analysis (T1/T2/T3/T5)
#   2) the K=4 repeated experiment for U_policy (23 instances) → completes the UQ three-way decomposition
#   3) full-set attribution (multi-file tasks, 6-way parallel) → D1/D3 with statistical power
#   4) final analysis → analysis.json
set -uo pipefail
cd "$(dirname "$0")/.."
export IDEALAB_API_KEY="${IDEALAB_API_KEY:?need IDEALAB_API_KEY}"
MODEL="${SCAE_MODEL:-gpt-5.5-0424-global}"
PROV="${SCAE_PROVIDER:-idealab}"
FL=pilot/runs/full_locate

echo "########## [1/4] merge the full-set localization results ##########"
cat $FL/shard*/results.jsonl > $FL/results.jsonl 2>/dev/null
echo "merged $(wc -l < $FL/results.jsonl) records"
python3 -m pilot.analyze.full_report --locate $FL/results.jsonl \
  --attr $FL/attribution.jsonl --out pilot/runs/analysis_locate.json || true

echo "########## [2/4] U_policy: 23 instances x K=4 ##########"
mkdir -p pilot/runs/upolicy
python3 -m pilot.harness.runner --model "$MODEL" --model-provider "$PROV" \
  --k 4 --instances $(cat pilot/runs/upolicy_ids.txt | tr '\n' ' ') \
  --out pilot/runs/upolicy --turn-timeout 900 --progress-every 5 || true

echo "########## [3/4] full-set attribution (multi-file tasks, 6-way parallel) ##########"
python3 - <<'PYEOF'
import json, os, glob
from pathlib import Path
FL="pilot/runs/full_locate"
recs=[json.loads(l) for l in open(f"{FL}/results.jsonl") if l.strip()]
multi=[r for r in recs if r.get("ok") and r.get("n_gold",1)>=2]
print(f"{len(multi)} multi-file tasks")
W=6
root=Path("pilot/runs/full_attr"); root.mkdir(parents=True, exist_ok=True)
for i in range(W):
    d=root/f"shard{i}"; d.mkdir(exist_ok=True)
    part=multi[i::W]
    with open(d/"results.jsonl","w") as f:
        for r in part: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    rl=d/"rollouts"
    if not rl.exists():
        rl.mkdir()
        for src in glob.glob(f"{FL}/shard*/rollouts/*.jsonl"):
            dst=rl/os.path.basename(src)
            if not dst.exists():
                try: os.symlink(os.path.abspath(src), dst)
                except OSError: pass
    print(f"  shard{i}: {len(part)} tasks")
PYEOF
python3 -m pilot.harness.parallel --cmd attr --workers 6 \
  --out pilot/runs/full_attr --model "$MODEL" --model-provider "$PROV" \
  --m 3 --window-points 5 --outcome recall --turn-timeout 900 || true

echo "########## [4/4] final full-set analysis ##########"
python3 -m pilot.analyze.full_report \
  --locate $FL/results.jsonl \
  --attr pilot/runs/full_attr/attribution.jsonl \
  --scan pilot/runs/v5_scan pilot/runs/v5 \
  --out pilot/runs/analysis.json
echo "all done -> pilot/runs/analysis.json"
