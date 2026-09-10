#!/usr/bin/env bash
# 阶段 5：U_policy 的 K=4 重复 rollout —— 论文 Appendix D.7 唯一在真轨迹上直接估计的分量
#
# 三分解 H[Y|q,G] = U_task + U_policy + U_model 里，只有 U_policy 能从真实数据直接估：
# 固定同一个 q，重复跑 K 次 rollout，取各实例结果方差的均值。
#   U_policy = mean_i Var(Y_i^{(1..K)})
# 论文报告 **0.049**，来自 23 个带 K=4 重复 rollout 的实例。
#
# 主流水线（阶段 1）用 K=1，因为 500 实例 × K=4 代价过高。所以这一档单独跑一个小子集。
#
# ⚠️ 复现口径的诚实说明
# 论文实际用的那 23 个实例 id 是一份数据产物（runs/upolicy_ids.txt），
# **无法从代码推导出来**。本脚本用一个明确写下来的抽样规则重新选：
# 在定位成功且步数足够的实例里按仓库分层取样。因此你得到的 U_policy
# 会接近但不必等于 0.0489。若你手上有原始 id 列表，用 SCAE_UPOLICY_IDS 指过去。
#
# 耗时：23 实例 × K=4 = 92 次定位，并行 6 路约 2–4 小时。
set -euo pipefail
source "$(dirname "$0")/common.sh"

LOCATE="$RUNS/full_locate"
OUT="$RUNS/upolicy"
N="${SCAE_UPOLICY_N:-23}"
K="${SCAE_UPOLICY_K:-4}"
IDS_FILE="${SCAE_UPOLICY_IDS:-$RUNS/upolicy_ids.txt}"

log "阶段 5：U_policy（K=$K 重复 rollout）"
banner
require_codex; require_api_key; require_dataset
[ -d "$LOCATE" ] || die "找不到 $LOCATE，先跑 scripts/01_locate.sh"

# --- 选实例 ---
if [ -s "$IDS_FILE" ]; then
  log "使用已有 id 列表：$IDS_FILE"
else
  log "按仓库分层抽 $N 个实例（规则写在下面的 python 里，可复现）"
  python3 - "$LOCATE" "$N" "$IDS_FILE" <<'PY'
import json, sys, collections
from pathlib import Path
locate, n, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])

# 读定位结果并去重
uniq = {}
for f in [locate / "results.jsonl"] + sorted(locate.glob("shard*/results.jsonl")):
    if not f.exists():
        continue
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            uniq.setdefault(r["instance_id"], r)

# 抽样规则（确定性，不用随机数）：
#   1) 只取执行成功的实例
#   2) 步数 >= 5 —— 太短的轨迹重复之间没什么可变的，方差恒 0，浪费预算
#   3) 按仓库分层轮转，仓库内按 instance_id 字典序，保证 django 不吃掉整个样本
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
print(f"  可用实例 {len(ok)}，抽出 {len(picked)} 个 → {out}")
print("  仓库分布：", dict(collections.Counter(p.split('__')[0] for p in picked)))
PY
fi

IDS="$(tr '\n' ' ' < "$IDS_FILE")"
[ -n "${IDS// /}" ] || die "$IDS_FILE 为空"

# --- 跑 K 次重复 ---
log "对 $(wc -w <<< "$IDS") 个实例各跑 K=$K 次"
mkdir -p "$OUT"
python3 -m pilot.harness.runner \
  --model "$MODEL" --model-provider "$PROVIDER" --codex-bin "$CODEX_BIN" \
  --k "$K" --instances $IDS \
  --out "$OUT" --turn-timeout "$TURN_TIMEOUT" --progress-every 5

# --- 估 U_policy ---
log "估计 U_policy"
python3 - "$OUT/results.jsonl" "$RUNS/upolicy_t3.json" <<'PY'
import json, sys
from pathlib import Path
# common.sh 已把仓库根加进 PYTHONPATH
from pilot.analyze.full_report import t3_uq_decomposition

res = Path(sys.argv[1])
recs = [json.loads(l) for l in res.read_text(encoding="utf-8").splitlines() if l.strip()]
out = t3_uq_decomposition(recs)
Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n  " + json.dumps(out, ensure_ascii=False))
print(f"\n  → {sys.argv[2]}")
print("  论文 Appendix D.7：U_policy = 0.049，来自 23 个 K=4 实例")
print("  （实例集合不同 ⇒ 数值会有差异；这是抽样差异，不是实现差异）")
PY

log "阶段 5 完成"
info "产物：$OUT/results.jsonl  与  $RUNS/upolicy_t3.json"
info "该 JSON 会被 pilot.analyze.x_synthesis 读入最终汇总"
