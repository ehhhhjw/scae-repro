"""E5 + E3：用 AST 给关键步赋予语义类型，并检验 A5 的经验基础。零 API 成本。

E5 成功 case 的关键步是什么样的步？
---------------------------------
已知 P4：成功 case 的 top-1 正向步 Δ=+0.223（CI 不含 0，显著）。
但"显著"没回答"那是什么步"。用 AST 把 top-1 步分类，回答"成功是怎么发生的"。

E3 A5（扰动可迁移）是否有经验支撑？
----------------------------------
归因用的替代动作若取"AST 图一跳邻居"，前提是策略本身的相邻动作也多在一跳内。
统计真实 trace 中相邻两步目标的图距离分布即可检验：
  若多数 ≤1 → 一跳邻居确实是策略的自然候选，A5 有经验支撑
  若普遍很大 → A5 不成立，须如实报告并改用其他替代动作来源
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.graph.ast_index import build_index                 # noqa: E402
from pilot.harness import rollout as R                        # noqa: E402
from pilot.probes import task as T                            # noqa: E402


def target_of(step, repo) -> str:
    _, tgt = R.action_bucket(step, repo)
    return tgt if tgt and tgt != "-" else ""


def classify_step(steps, k: int, gold: frozenset[str], repo) -> list[str]:
    """给第 k 步（1-based）打语义标签。可多标签。"""
    labels = []
    tgt = target_of(steps[k - 1], repo)
    prior = {target_of(s, repo) for s in steps[: k - 1]}
    gold_dirs = {g.rsplit("/", 1)[0] for g in gold if "/" in g}
    prior_dirs = {p.rsplit("/", 1)[0] for p in prior if "/" in p}

    if not tgt:
        labels.append("pure_search")
        return labels
    if tgt in gold:
        labels.append("first_touch_gold_file" if tgt not in prior
                      else "revisit_gold_file")
    d = tgt.rsplit("/", 1)[0] if "/" in tgt else "."
    if d in gold_dirs and d not in prior_dirs:
        labels.append("first_touch_gold_dir")
    if "test" in tgt.lower():
        labels.append("test_file")
    if not labels:
        labels.append("other_file")
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="pilot/runs/full_attr/attribution.jsonl")
    ap.add_argument("--locate", default="pilot/runs/full_locate/results.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/step_semantics.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    loc = {json.loads(l)["instance_id"]: json.loads(l)
           for l in Path(args.locate).read_text().splitlines() if l.strip()}
    attr = [json.loads(l) for l in Path(args.attr).read_text().splitlines() if l.strip()]
    attr = [r for r in attr if not r.get("error") and r.get("causal_top1_step")]

    e5 = {"succeeded": Counter(), "failed": Counter()}
    e5_n = {"succeeded": 0, "failed": 0}
    hops_adj: list[int] = []          # 相邻两步的图距离
    hops_unreach = 0
    hops_total = 0
    per_case = []

    for i, r in enumerate(attr, 1):
        inst = insts.get(r["instance_id"])
        lrec = loc.get(r["instance_id"])
        if inst is None or lrec is None or not lrec.get("rollout"):
            continue
        cands = list(Path(args.runs_root).glob(f"shard*/{lrec['rollout']}"))
        if not cands:
            continue
        try:
            steps = R.steps(R.load(cands[0]))
            idx = build_index(inst.repo)
        except (OSError, ValueError, RecursionError):
            continue

        k = r["causal_top1_step"]
        if k > len(steps):
            continue
        grp = "succeeded" if r.get("Y") == 1 else "failed"
        labs = classify_step(steps, k, inst.gold, inst.repo)
        for L in labs:
            e5[grp][L] += 1
        e5_n[grp] += 1
        per_case.append({"instance_id": r["instance_id"], "Y": r.get("Y"),
                         "top1_step": k, "labels": labs,
                         "top1_delta": r.get("causal_top1_delta")})

        # E3：相邻两步的 AST 图距离
        prev = ""
        for s in steps:
            tgt = target_of(s, inst.repo)
            if prev and tgt and prev.endswith(".py") and tgt.endswith(".py"):
                hops_total += 1
                d = idx.graph_distance(prev, tgt, max_hops=3)
                if d is None:
                    hops_unreach += 1
                else:
                    hops_adj.append(d)
            if tgt:
                prev = tgt
        if i % 10 == 0:
            print(f"  已处理 {i}/{len(attr)}", flush=True)

    dist = Counter(hops_adj)
    le1 = sum(c for d, c in dist.items() if d <= 1)
    res = {
        "E5_top1_step_types": {
            g: {"n_cases": e5_n[g],
                "label_counts": dict(e5[g]),
                "label_rates": {k2: round(v / e5_n[g], 4)
                                for k2, v in e5[g].items()} if e5_n[g] else {}}
            for g in ("succeeded", "failed")
        },
        "E3_adjacent_step_graph_distance": {
            "n_pairs_py_to_py": hops_total,
            "n_reachable_within_3": len(hops_adj),
            "n_unreachable": hops_unreach,
            "distance_counts": dict(sorted(dist.items())),
            "frac_within_1_hop_among_reachable": (round(le1 / len(hops_adj), 4)
                                                  if hops_adj else None),
            "frac_reachable": (round(len(hops_adj) / hops_total, 4)
                               if hops_total else None),
            "median_distance": (statistics.median(hops_adj) if hops_adj else None),
            "A5_interpretation": None,
        },
        "per_case": per_case,
    }
    f1 = res["E3_adjacent_step_graph_distance"]["frac_within_1_hop_among_reachable"]
    fr = res["E3_adjacent_step_graph_distance"]["frac_reachable"]
    res["E3_adjacent_step_graph_distance"]["A5_interpretation"] = (
        "无 py-py 相邻对，不可判定" if not hops_total else
        f"可达对中 {f1:.0%} 在一跳内，但仅 {fr:.0%} 的相邻对在 3 跳内可达；"
        + ("A5 有经验支撑" if (f1 or 0) >= 0.5 and (fr or 0) >= 0.5
           else "A5 缺经验支撑：多数相邻动作在依赖图上并不邻近，"
                "'一跳邻居'不是策略的自然候选，需改用其他替代动作来源")
    )

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print("\n=== E5 top-1 关键步的语义类型 ===")
    for g in ("succeeded", "failed"):
        d = res["E5_top1_step_types"][g]
        print(f"  {g} (n={d['n_cases']}):")
        for k2, v in sorted(d["label_rates"].items(), key=lambda x: -x[1]):
            print(f"     {k2:<24} {v:>6.1%}  ({d['label_counts'][k2]})")
    e3 = res["E3_adjacent_step_graph_distance"]
    print(f"\n=== E3 相邻步的 AST 图距离（A5 检验）===")
    print(f"  py→py 相邻对 {e3['n_pairs_py_to_py']}，3跳内可达 {e3['n_reachable_within_3']}"
          f"（{e3['frac_reachable']}），不可达 {e3['n_unreachable']}")
    print(f"  距离分布 {e3['distance_counts']}  中位 {e3['median_distance']}")
    print(f"  → {e3['A5_interpretation']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
