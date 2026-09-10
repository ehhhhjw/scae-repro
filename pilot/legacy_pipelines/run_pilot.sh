#!/usr/bin/env bash
# SCAE Pilot 总控 —— 一键复现全部 pilot 结论。
# 依赖：codex 已登录；IDEALAB_API_KEY 环境变量。
set -euo pipefail
cd "$(dirname "$0")/.."

export IDEALAB_API_KEY="${IDEALAB_API_KEY:?请先 export IDEALAB_API_KEY=...}"
MODEL="${SCAE_MODEL:-gpt-5.5-0424-global}"
PROVIDER="${SCAE_PROVIDER:-idealab}"
RUNS="pilot/runs"

echo "########## Stage 0: Smoke（V1-V4 链路 + 单任务定位）##########"
python3 pilot/smoke/smoke.py --instance astropy__astropy-12907 \
  --model "$MODEL" --model-provider "$PROVIDER" --out "$RUNS/smoke" --skip-v5 || true

echo "########## Stage V5: 单点故障（positivity 动作熵）##########"
python3 -m pilot.smoke.probe_v5 \
  --rollout "$RUNS/smoke/astropy__astropy-12907/base.rollout.jsonl" \
  --instance astropy__astropy-12907 --model "$MODEL" --model-provider "$PROVIDER" \
  --fork-steps 2 4 6 --samples 6 --out "$RUNS/v5" || true

echo "########## Stage S1: 批量定位（成功率分布 + v0 语料）##########"
# 默认 40 分层样本；全量传 SCAE_FULL=1
if [ "${SCAE_FULL:-0}" = "1" ]; then
  python3 -m pilot.harness.runner --model "$MODEL" --model-provider "$PROVIDER" \
    --k "${SCAE_K:-1}" --out "$RUNS/s1_full"
  S1DIR="$RUNS/s1_full"
else
  python3 -m pilot.harness.runner --model "$MODEL" --model-provider "$PROVIDER" \
    --k 1 --instances $(cat "$RUNS/sample40.txt" | tr '\n' ' ') --out "$RUNS/s1_sample40"
  S1DIR="$RUNS/s1_sample40"
fi

echo "########## Stage V5-scan: 熵衰减普遍性 ##########"
for iid in $(python3 -c "
import json,os
seen=set()
for l in open('$S1DIR/results.jsonl'):
    r=json.loads(l)
    if r.get('ok') and r.get('n_steps',0)>=12 and r['project'] not in seen:
        print(r['instance_id']); seen.add(r['project'])
" | head -4); do
  ns=$(python3 -c "import json;print([json.loads(l)['n_steps'] for l in open('$S1DIR/results.jsonl') if json.loads(l).get('instance_id')=='$iid'][0])")
  pts=$(python3 -c "n=$ns;print(' '.join(str(max(2,round(1+i*(n-1)/4))) for i in range(5)))")
  python3 -m pilot.smoke.probe_v5 --rollout "$S1DIR/rollouts/${iid}__k0.jsonl" \
    --instance "$iid" --model "$MODEL" --model-provider "$PROVIDER" \
    --fork-steps $pts --samples 4 --out "$RUNS/v5_scan" || true
done
python3 -m pilot.analyze.entropy_scan --scan-dir "$RUNS/v5_scan" \
  --extra "$RUNS/v5/astropy__astropy-12907/v5_report.json" || true

echo "########## Stage C3: 归因对比（D1/D3）##########"
python3 -m pilot.attribution.run_attribution --runs-dir "$S1DIR" \
  --model "$MODEL" --model-provider "$PROVIDER" --m 4 --window-points 6 || true
python3 -m pilot.analyze.attribution_compare --attr "$S1DIR/attribution.jsonl" || true

echo "########## 完成。产物在 $RUNS 下 ##########"
