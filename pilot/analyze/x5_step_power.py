"""X5（零 API，V4′ 后）：步级功效建立与 V3k 因果归因重做。

动机
----
V3k 的功效检验显示 m=8 fork-replay 的 |Δ_t| 对位置等协变量无可检出结构
（Spearman rho +0.019 n.s.），所以"因果与溯源无关"这一零结果不可发表。

V4′ 持久化了每个估值点的 ys_t（原始 m 次 recall），使我们可以：
  1. 对每个估值点做 Δ_t ≠ 0 的显著性检验（bootstrap / t 检验）
  2. 仅在「显著点」上重做 V3k 的溯源–因果对照

三档判据
--------
| 显著点 ≥ 30% 且溯源有效应  | G4 解除，V3k 因果侧给正面结论                         |
| 显著点 ≥ 30% 但溯源仍无关  | 「语义相关 ≠ 因果相关」可被断言                        |
| 显著点 < 10%               | m=8 不足；靠 X3 的靶向干预（已预置），放弃全扫描      |

统计
----
- 单估值点：bootstrap t 检验（ys_t vs ys_prev，零假设 Δ=0）
- 多重比较：BH-FDR q=0.10（每 instance 内独立检验）
- 溯源–因果对比：仅在显著点上的 |Δ| by gold_provenance class
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                          # noqa: E402
from pilot.probes import task as T                              # noqa: E402
from pilot.analyze.v3k_trace_causal import gold_relation, boot_ci, paired_ci  # noqa: E402

SEED = 20260814


def bootstrap_test_delta(ys_t: list[float], ys_prev: list[float],
                         b: int = 2000) -> dict:
    """bootstrap t 检验：H0: E[ys_t] - E[ys_prev] = 0。
    返回 p_approx = 2 × min(tail_prob)（双侧近似）。
    """
    if len(ys_t) < 2 or len(ys_prev) < 2:
        return {"note": "样本不足"}
    obs_diff = statistics.mean(ys_t) - statistics.mean(ys_prev)
    # Bootstrap 零分布：把两组混合后重采样（置换近似）
    combined = ys_t + ys_prev
    n1, n2 = len(ys_t), len(ys_prev)
    rng = random.Random(SEED)
    null = []
    for _ in range(b):
        samp = [combined[rng.randrange(len(combined))] for _ in range(len(combined))]
        null.append(statistics.mean(samp[:n1]) - statistics.mean(samp[n1:]))
    null.sort()
    p_right = sum(1 for x in null if x >= abs(obs_diff)) / b
    p_left = sum(1 for x in null if x <= -abs(obs_diff)) / b
    p_approx = p_right + p_left
    lo = null[int(0.025 * b)]
    hi = null[int(0.975 * b)]
    return {"obs_diff": round(obs_diff, 4),
            "ci95_null": [round(lo, 4), round(hi, 4)],
            "p_approx": round(min(p_approx, 1.0), 4),
            "var_ys_t": round(statistics.variance(ys_t), 6),
            "significant_p05": bool(p_approx < 0.05)}


def bh_fdr(pvals: list[float], q: float = 0.10) -> list[bool]:
    n = len(pvals)
    if n == 0:
        return []
    ranked = sorted(range(n), key=lambda i: pvals[i])
    reject = [False] * n
    last = -1
    for rank, idx in enumerate(ranked):
        if pvals[idx] <= (rank + 1) * q / n:
            last = rank
    for rank, idx in enumerate(ranked):
        if rank <= last:
            reject[idx] = True
    return reject


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4prime-dir", default="pilot/runs/v4prime")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x5_step_power.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    roll = {p.stem.split("__k")[0]: p
            for p in Path(args.runs_root).glob("shard*/rollouts/*.jsonl")}

    # ---- 1. 载入 V4′ ----
    recs: list[dict] = []
    v4p = Path(args.v4prime_dir)
    for f in sorted(v4p.glob("shard*/attribution.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    print(f"V4′ 归因记录: {len(recs)}")

    if not recs:
        print("⚠️  V4′ 归因尚无数据。请等 V4′ 完成后重跑本脚本。")
        return 1

    # ---- 2. 逐步显著性检验 ----
    all_tests: list[dict] = []    # {iid, t, p, sig_p05, sig_bh, delta, |delta|}
    by_inst: dict[str, list[dict]] = defaultdict(list)

    for r in recs:
        iid = r["instance_id"]
        segs = r.get("causal_segments") or []
        for seg in segs:
            t = seg.get("t")
            ys_t = seg.get("ys_t")
            ys_prev = seg.get("ys_prev")
            if not ys_t or not ys_prev or t is None:
                continue
            test = bootstrap_test_delta(ys_t, ys_prev)
            if "note" in test:
                continue
            row = {"iid": iid, "t": t, "p": test["p_approx"],
                   "sig_p05": test["significant_p05"],
                   "obs_diff": test["obs_diff"],
                   "abs_diff": abs(test["obs_diff"]),
                   "var_ys_t": test["var_ys_t"],
                   "delta": seg.get("delta", 0.0)}
            all_tests.append(row)
            by_inst[iid].append(row)

    n_tests = len(all_tests)
    if n_tests == 0:
        print("⚠️  没有有效的 ys_t/ys_prev 数据")
        return 1

    # BH-FDR 全局校正
    pvals = [r["p"] for r in all_tests]
    bh_reject = bh_fdr(pvals, q=0.10)
    for r, sig in zip(all_tests, bh_reject):
        r["sig_bh"] = sig

    # ---- 3. 功效摘要 ----
    n_sig_p05 = sum(1 for r in all_tests if r["sig_p05"])
    n_sig_bh = sum(1 for r in all_tests if r["sig_bh"])
    power_summary = {
        "n_tests": n_tests,
        "n_significant_p05": n_sig_p05,
        "frac_significant_p05": round(n_sig_p05 / n_tests, 4),
        "n_significant_bh_q10": n_sig_bh,
        "frac_significant_bh_q10": round(n_sig_bh / n_tests, 4),
        "abs_diff_mean_all": round(statistics.mean(r["abs_diff"] for r in all_tests), 4),
        "abs_diff_mean_sig": (round(statistics.mean(r["abs_diff"] for r in all_tests
                                                    if r["sig_bh"]), 4) if n_sig_bh > 0 else None),
        "var_ys_t_mean": round(statistics.mean(r["var_ys_t"] for r in all_tests), 6),
    }

    # ---- 4. 在「显著点」上重做 V3k：|Δ| 按溯源类别 ----
    # 只在 sig_bh=True 的步上做分析
    sig_tests = [r for r in all_tests if r["sig_bh"]]
    print(f"BH-FDR 显著点: {len(sig_tests)}/{n_tests}")

    abs_by_cls_sig: dict[str, list[float]] = defaultdict(list)
    abs_by_cls_all: dict[str, list[float]] = defaultdict(list)
    pair_gold_sig, pair_none_sig = [], []
    gold_top1_hit, gold_top1_chance = [], []
    cls_counts_sig: Counter = Counter()

    for r in recs:
        iid = r["instance_id"]
        inst = insts.get(iid)
        segs = r.get("causal_segments") or []
        if inst is None or not inst.gold or iid not in roll or not segs:
            continue
        try:
            steps = R.steps(R.load(roll[iid]))
        except (OSError, ValueError):
            continue
        from pilot.analyze.v3j_trace_graph import PATH_RE
        rel = gold_relation(steps, set(inst.gold))

        sig_set = {row["t"] for row in by_inst.get(iid, []) if row["sig_bh"]}
        pts_all = [(s["t"], abs(float(s.get("delta") or 0.0))) for s in segs if s.get("t") in rel]
        pts_sig = [(t, d) for t, d in pts_all if t in sig_set]

        # 全体点
        for t, d in pts_all:
            k = ("first_gold" if rel[t]["first_gold"] else
                 "visit_gold" if rel[t]["visit_gold"] else
                 "mention_gold" if rel[t]["mention_gold"] else "none")
            abs_by_cls_all[k].append(d)

        # 显著点
        if pts_sig:
            g_sig = [d for t, d in pts_sig if rel[t].get("any_gold")]
            n_sig = [d for t, d in pts_sig if not rel[t].get("any_gold")]
            if g_sig and n_sig:
                pair_gold_sig.append(statistics.mean(g_sig))
                pair_none_sig.append(statistics.mean(n_sig))
            for t, d in pts_sig:
                k = ("first_gold" if rel[t]["first_gold"] else
                     "visit_gold" if rel[t]["visit_gold"] else
                     "mention_gold" if rel[t]["mention_gold"] else "none")
                abs_by_cls_sig[k].append(d)
                cls_counts_sig[k] += 1

    # ---- 5. top-1 归因在显著点上 ----
    for r in recs:
        iid = r["instance_id"]
        inst = insts.get(iid)
        segs = r.get("causal_segments") or []
        if inst is None or not inst.gold or iid not in roll or not segs:
            continue
        try:
            steps = R.steps(R.load(roll[iid]))
        except (OSError, ValueError):
            continue
        from pilot.analyze.v3j_trace_graph import PATH_RE
        rel = gold_relation(steps, set(inst.gold))
        sig_set = {row["t"] for row in by_inst.get(iid, []) if row["sig_bh"]}
        if not sig_set:
            continue
        # top-1 = 最大 |Δ| 的显著点
        pts = [(s["t"], abs(float(s.get("delta") or 0.0)))
               for s in segs if s.get("t") in rel and s["t"] in sig_set]
        if not pts:
            continue
        t1 = max(pts, key=lambda x: x[1])[0]
        share = sum(1 for t, _ in pts if rel[t].get("any_gold")) / len(pts)
        gold_top1_hit.append(1.0 if rel[t1].get("any_gold") else 0.0)
        gold_top1_chance.append(share)

    res = {
        "n_instances": len(recs),
        "power_summary": power_summary,
        "step_class_counts_all": {k: len(v) for k, v in abs_by_cls_all.items()},
        "abs_delta_by_class_ALL": {k: boot_ci(v) for k, v in abs_by_cls_all.items()},
        "abs_delta_by_class_SIG_BH": {k: boot_ci(v) for k, v in abs_by_cls_sig.items()},
        "sig_counts_by_class": dict(cls_counts_sig),
        "within_instance_paired_SIG": {
            "gold_provenance_steps": boot_ci(pair_gold_sig),
            "non_gold_steps": boot_ci(pair_none_sig),
            "diff": paired_ci(pair_gold_sig, pair_none_sig),
            "n_instances": len(pair_gold_sig),
        },
        "causal_top1_SIG_on_gold_prov": {
            "observed": boot_ci(gold_top1_hit),
            "closed_form_chance": boot_ci(gold_top1_chance),
            "diff": paired_ci(gold_top1_hit, gold_top1_chance),
        },
    }

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== X5：步级功效 + V3k 重做（显著点） ===")
    pw = res["power_summary"]
    print(f"总测试点: {pw['n_tests']}")
    print(f"显著（p<0.05）: {pw['n_significant_p05']} ({pw['frac_significant_p05']:.1%})")
    print(f"显著（BH q=0.10）: {pw['n_significant_bh_q10']} ({pw['frac_significant_bh_q10']:.1%})")
    print()
    print("全体点 |Δ| by class:")
    for k, v in res["abs_delta_by_class_ALL"].items():
        if "note" not in v:
            print(f"  {k:15s}: {v['mean']:.4f} CI{v['ci95']} n={v['n']}")
    print()
    print("显著点 |Δ| by class:")
    for k, v in res["abs_delta_by_class_SIG_BH"].items():
        if "note" not in v:
            print(f"  {k:15s}: {v['mean']:.4f} CI{v['ci95']} n={v['n']}")
    wp = res["within_instance_paired_SIG"]
    if "note" not in wp["diff"]:
        print(f"\n显著点 instance 配对: gold={wp['gold_provenance_steps']['mean']:.4f} "
              f"none={wp['non_gold_steps']['mean']:.4f} "
              f"diff={wp['diff']['diff']:+.4f} CI{wp['diff']['ci95']} "
              f"{'显著' if wp['diff']['significant'] else '不显著'}")
    ct = res["causal_top1_SIG_on_gold_prov"]
    if "note" not in ct["observed"]:
        print(f"\n显著点 top-1 落在 gold 溯源步: obs={ct['observed']['mean']:.3f} "
              f"chance={ct['closed_form_chance']['mean']:.3f}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
