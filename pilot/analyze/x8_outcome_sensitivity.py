"""X8 前半：为干预实验挑选**最敏感的 outcome 口径**。零 API。

为什么必须先做（它是 X3 的前置）
--------------------------------
主口径 Y=(gold⊆pred) 的基线率 94% —— 天花板效应使任何干预的效应量被压缩：
若 94% 的实例本来就成功，掩蔽一条溯源边最多只能让 6% 的实例"从成功变失败"，
而剩下 94% 里的变化被二值化抹平。**所以在 X3 花 11.7h 之前，必须先确认口径能承载效应。**

选择标准（关键：不是"基线率接近 50%"就好）
------------------------------------------
干预实验要检测的是**均值差**，其可检出性取决于

    效应量 / 标准误 ∝ (可移动的空间) / sd

因此三个指标一起看：
  1. **基线均值**离天花板/地板有多远（可移动空间 = min(mean, 1−mean)）
  2. **标准差**（越大越容易被噪声掩盖，但也说明有区分度）
  3. **有效动态范围**：在 gold 溯源被移除时该口径**理论上**能变多少
     用一个"反事实上界"近似：把 pred 中的 gold 文件全部移除后该口径的值
     → 这给出"若掩蔽完全奏效"时的口径值，与基线之差即**可检出的最大效应**

候选口径
--------
  Y            gold ⊆ pred（主口径，二值）
  Y_exact      严格集合相等（二值）
  Y_filtered   过滤 test/docs 后相等或包含（二值）
  recall       连续 [0,1]
  precision    连续（惩罚过报）
  f1           连续
  n_steps      效率（步数，连续，无上界）

判据（预注册）
--------------
| 结果 | 走向 |
|---|---|
| 存在口径其"最大可检出效应/sd" 比主口径高 ≥50% | **X3 改用该口径为主口径**，recall 作次要 |
| 各口径相当 | 沿用 recall（连续，已优于二值 Y），无需改动 |
| 所有口径的最大可检出效应都 <0.1 | 口径不是瓶颈 → 问题在任务本身太易，需换更难子集 |
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                  # noqa: E402
from pilot.probes import task as T                      # noqa: E402
from pilot.analyze.v3j_trace_graph import boot_ci       # noqa: E402
from pilot.analyze.v3k_trace_causal import gold_relation  # noqa: E402

METRICS = ("Y", "Y_exact", "Y_filtered", "recall", "precision", "f1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x8_outcome_sensitivity.json")
    ap.add_argument("--multifile-only", action="store_true",
                    help="只看多文件实例（X3 的实际总体）")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    rows = []
    for p in sorted(Path(args.runs_root).glob("shard*/rollouts/*.jsonl")):
        iid = p.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None or not inst.gold:
            continue
        if args.multifile_only and len(inst.gold) < 2:
            continue
        try:
            items = R.load(p)
        except (OSError, ValueError):
            continue
        ans = T.answer_from_rollout(items)
        oc = T.outcome(ans, inst.gold)
        if not oc.get("parsed"):
            continue
        # 反事实：**只**丢掉在 t*（first_gold 步）首次暴露的那些 gold 文件。
        # 首版把全部 gold 都移除，导致每个口径都归零、"最大效应"退化为基线均值，
        # 于是判据变成了 mean/sd（只会选出均值高方差小的口径），不是敏感性。
        st = R.steps(items)
        rel = gold_relation(st, set(inst.gold))
        # 找 first_gold 步，并取出该步首次暴露的 gold 集合
        lost: set[str] = set()
        gold_seen: set[str] = set()
        for s in st:
            txt = s.output_text or ""
            ment = {gp for gp in inst.gold
                    if gp in txt or gp.split("/")[-1] in txt}
            first = ment - gold_seen
            gold_seen |= ment
            if first:
                lost = first            # 第一个有 first_gold 的步
                break
        if not lost:
            continue
        cf_pred = frozenset((ans or frozenset()) - lost)
        cf = T.outcome(cf_pred, inst.gold)
        rows.append({
            "instance_id": iid, "n_gold": len(inst.gold),
            "n_pred": len(ans) if ans else 0,
            "n_lost_if_masked": len(lost),
            "n_steps": len(st),
            **{m: oc[m] for m in METRICS},
            **{f"cf_{m}": cf[m] for m in METRICS},
        })

    if not rows:
        print("无样本")
        return 1
    print(f"载入 {len(rows)} 个实例"
          f"（{'仅多文件' if args.multifile_only else '全部'}）")

    res: dict = {"n_instances": len(rows),
                 "multifile_only": bool(args.multifile_only),
                 "metrics": {}}
    n = len(rows)
    for m in METRICS:
        vals = [float(r[m]) for r in rows]
        cfs = [float(r[f"cf_{m}"]) for r in rows]
        # **配对差**：每个 instance 上"丢掉 t* 暴露的 gold"会让该口径掉多少
        diffs = [a - b for a, b in zip(vals, cfs)]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if n > 1 else 0.0
        d_mean = statistics.mean(diffs)
        d_sd = statistics.stdev(diffs) if n > 1 else 0.0
        # 配对设计的可检出性 ∝ mean(Δ)/sd(Δ)·sqrt(n)（配对 t 统计量）
        tstat = (d_mean / d_sd * (n ** 0.5)) if d_sd > 1e-9 else None
        res["metrics"][m] = {
            "mean": round(mean, 4), "sd": round(sd, 4),
            "headroom_to_ceiling_or_floor": round(min(mean, 1.0 - mean), 4),
            "counterfactual_mean": round(statistics.mean(cfs), 4),
            "paired_effect_mean": round(d_mean, 4),
            "paired_effect_sd": round(d_sd, 4),
            "paired_t_stat": (round(tstat, 2) if tstat is not None else None),
            "effect_over_sd": (round(d_mean / d_sd, 3) if d_sd > 1e-9 else None),
            "frac_instances_moved": round(
                sum(1 for d in diffs if abs(d) > 1e-9) / n, 4),
            "ci_mean": boot_ci(vals)["ci95"],
        }
    # 步数（无上界，单独处理）
    ns = [float(r["n_steps"]) for r in rows]
    res["n_steps"] = {"mean": round(statistics.mean(ns), 2),
                      "sd": round(statistics.stdev(ns), 2) if len(ns) > 1 else 0.0,
                      "cv": (round(statistics.stdev(ns) / statistics.mean(ns), 3)
                             if len(ns) > 1 and statistics.mean(ns) > 0 else None),
                      "note": "效率口径：无天花板，但与 gold 可发现性的关系间接"}

    # 裁决
    base = res["metrics"]["recall"]
    best = max((m for m in METRICS
                if res["metrics"][m]["effect_over_sd"] is not None),
               key=lambda m: res["metrics"][m]["effect_over_sd"])
    b_eos = res["metrics"][best]["effect_over_sd"]
    r_eos = base["effect_over_sd"]
    gain = (b_eos / r_eos - 1.0) if (r_eos and r_eos > 0) else None
    res["VERDICT"] = {
        "best_metric": best, "best_effect_over_sd": b_eos,
        "recall_effect_over_sd": r_eos,
        "relative_gain_over_recall": (round(gain, 3) if gain is not None else None),
        "decision": (
            f"改用 {best} 为 X3 主口径（效应/sd 比 recall 高 {gain:.0%}）"
            if gain is not None and gain >= 0.5 else
            "沿用 recall 为 X3 主口径（其他口径无实质增益）"),
        "all_effects_tiny": bool(all(
            res["metrics"][m]["paired_effect_mean"] < 0.1 for m in METRICS)),
    }

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== 各 outcome 口径的干预敏感性 ===")
    print(f"  反事实 = 只丢掉 t*(first_gold 步) 首次暴露的 gold 文件（掩蔽的真实后果）")
    print(f"{'口径':<12}{'基线':>8}{'反事实':>9}{'配对效应':>10}"
          f"{'效应sd':>9}{'效应/sd':>9}{'配对t':>8}{'移动占比':>9}")
    for m in METRICS:
        d = res["metrics"][m]
        eos, ts = d["effect_over_sd"], d["paired_t_stat"]
        print(f"{m:<12}{d['mean']:>8.4f}{d['counterfactual_mean']:>9.4f}"
              f"{d['paired_effect_mean']:>10.4f}{d['paired_effect_sd']:>9.4f}"
              f"{(f'{eos:.3f}' if eos is not None else '—'):>9}"
              f"{(f'{ts:.1f}' if ts is not None else '—'):>8}"
              f"{d['frac_instances_moved']:>9.4f}")
    n = res["n_steps"]
    print(f"\n  步数：mean {n['mean']} sd {n['sd']} cv {n['cv']}  ({n['note']})")
    v = res["VERDICT"]
    print(f"\n=== 裁决 ===")
    print(f"  最敏感口径 **{v['best_metric']}**（效应/sd = {v['best_effect_over_sd']}）"
          f"   recall = {v['recall_effect_over_sd']}")
    print(f"  {v['decision']}")
    if v["all_effects_tiny"]:
        print(f"  ⚠️ 所有口径最大效应 <0.1 → 口径不是瓶颈，需换更难子集")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
