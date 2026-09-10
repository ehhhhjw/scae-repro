#!/usr/bin/env bash
# 阶段 1：全量文件定位（S1）—— 论文全部结论的语料底座
#
# 对 SWE-bench-Verified 的 500 个实例各跑一次文件定位，产出：
#   - results.jsonl        每实例的 Y / recall / n_steps / 预测文件集（RQ1–RQ3 + v_0 语料）
#   - rollouts/*.jsonl     完整执行轨迹（前缀重放的输入，阶段 2 必需）
#
# 论文口径（§5.2 / Table 7）：499 episode、12 个仓库、定位成功率 93.99%。
# 500 → 499 的差额是 django__django-15278 缺仓库快照（数据集只有 499 个 projects 目录）。
#
# ⚠️ 泄题闸门：每个实例跑之前会强制 git reset 回 base_commit。
#    数据集分发的工作区叠着 gold patch（= 已修复状态），不 reset 就是让 agent 读答案。
#    这一步由 pilot/probes/task.py::reset_repo 保证，不要绕过。见 docs/PITFALLS.md。
#
# 耗时：并行 6 路约 10–20 小时。API 调用 ~500 次长会话。
set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT="$RUNS/full_locate"
K="${SCAE_K:-1}"
LIMIT="${SCAE_LIMIT:-0}"

log "阶段 1：全量文件定位"
banner
info "输出       $OUT"
info "每实例重复 K=$K"
[ "$LIMIT" != "0" ] && info "调试限流   只跑前 $LIMIT 个实例"
require_codex; require_api_key; require_dataset

log "启动 $WORKERS 路并行定位（按 repo 分片，同一仓库目录不会被并发 reset）"
EXTRA=()
[ "$LIMIT" != "0" ] && EXTRA+=(--limit "$LIMIT")
python3 -m pilot.harness.parallel \
  --cmd locate --workers "$WORKERS" --out "$OUT" \
  --model "$MODEL" --model-provider "$PROVIDER" \
  --k "$K" --turn-timeout "$TURN_TIMEOUT" "${EXTRA[@]}"

log "定位结果汇总"
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
print(f"  唯一实例   {len(uniq)}")
print(f"  成功执行   {len(ok)}")
if bad:
    print(f"  执行失败   {len(bad)}:")
    for r in bad[:10]:
        print(f"    {r['instance_id']}: {str(r.get('error'))[:110]}")

# 论文 Table 7 的口径：F* ⊆ F̂（recall == 1）
hit = [r for r in ok if (r.get("recall") or 0) >= 1.0]
if ok:
    print(f"\n  定位成功率 {len(hit)}/{len(ok)} = {100*len(hit)/len(ok):.2f}%   （论文 Table 7: 93.99%）")
per = collections.defaultdict(lambda: [0, 0])
for r in ok:
    p = r.get("project") or r["instance_id"].split("__")[0]
    per[p][1] += 1
    if (r.get("recall") or 0) >= 1.0:
        per[p][0] += 1
print("\n  按仓库（对照论文 Table 7）：")
for p, (h, n) in sorted(per.items(), key=lambda kv: -kv[1][1]):
    print(f"    {p:16s} n={n:4d}  {100*h/n:6.1f}%")
PY

log "阶段 1 完成"
info "产物：$OUT/results.jsonl  +  $OUT/shard*/rollouts/*.jsonl"
info "下一步：bash scripts/02_attribution.sh   （或先跑 04 做纯离线的 RQ1–RQ3 分析）"
