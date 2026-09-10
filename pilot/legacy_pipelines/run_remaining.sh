#!/usr/bin/env bash
# 全量定位完成后的剩余实验流水线 —— 顺序执行，无需人工介入。
#   1) 合并全量定位结果 + 首轮分析（T1/T2/T3/T5）
#   2) U_policy 的 K=4 重复实验（23 实例）→ 补齐 UQ 三分解
#   3) 全量归因（多文件任务，6 并行）→ 有统计功效的 D1/D3
#   4) 最终分析 → analysis.json
set -uo pipefail
cd "$(dirname "$0")/.."
export IDEALAB_API_KEY="${IDEALAB_API_KEY:?need IDEALAB_API_KEY}"
MODEL="${SCAE_MODEL:-gpt-5.5-0424-global}"
PROV="${SCAE_PROVIDER:-idealab}"
FL=pilot/runs/full_locate

echo "########## [1/4] 合并全量定位结果 ##########"
cat $FL/shard*/results.jsonl > $FL/results.jsonl 2>/dev/null
echo "合并 $(wc -l < $FL/results.jsonl) 条"
python3 -m pilot.analyze.full_report --locate $FL/results.jsonl \
  --attr $FL/attribution.jsonl --out pilot/runs/analysis_locate.json || true

echo "########## [2/4] U_policy: 23 实例 x K=4 ##########"
mkdir -p pilot/runs/upolicy
python3 -m pilot.harness.runner --model "$MODEL" --model-provider "$PROV" \
  --k 4 --instances $(cat pilot/runs/upolicy_ids.txt | tr '\n' ' ') \
  --out pilot/runs/upolicy --turn-timeout 900 --progress-every 5 || true

echo "########## [3/4] 全量归因（多文件任务，6 并行）##########"
python3 - <<'PYEOF'
import json, os, glob
from pathlib import Path
FL="pilot/runs/full_locate"
recs=[json.loads(l) for l in open(f"{FL}/results.jsonl") if l.strip()]
multi=[r for r in recs if r.get("ok") and r.get("n_gold",1)>=2]
print(f"多文件任务 {len(multi)} 个")
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
    print(f"  shard{i}: {len(part)} 任务")
PYEOF
python3 -m pilot.harness.parallel --cmd attr --workers 6 \
  --out pilot/runs/full_attr --model "$MODEL" --model-provider "$PROV" \
  --m 3 --window-points 5 --outcome recall --turn-timeout 900 || true

echo "########## [4/4] 最终全量分析 ##########"
python3 -m pilot.analyze.full_report \
  --locate $FL/results.jsonl \
  --attr pilot/runs/full_attr/attribution.jsonl \
  --scan pilot/runs/v5_scan pilot/runs/v5 \
  --out pilot/runs/analysis.json
echo "全部完成 -> pilot/runs/analysis.json"
