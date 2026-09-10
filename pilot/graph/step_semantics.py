"""E5 + E3: assign semantic types to critical steps via AST, and test A5's empirical basis. Zero API cost.

E5 What kind of step is the critical step of a successful case?
---------------------------------------------------------------
Known from P4: the top-1 positive step of successful cases has Δ=+0.223 (CI excludes 0, significant).
But "significant" does not answer "what step was that". Classify top-1 steps with AST to answer
"how success happened".

E3 Does A5 (perturbation transferability) have empirical support?
-----------------------------------------------------------------
If attribution takes the substitute action to be an "AST one-hop neighbour", the premise is that the
policy's own adjacent actions also mostly stay within one hop.
Testing it only needs the graph-distance distribution over adjacent-step targets in real traces:
  if most are ≤1 → one-hop neighbours really are the policy's natural candidates, A5 has empirical support
  if generally large → A5 fails, must be reported faithfully and another substitute-action source used
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
    """Assign semantic labels to step k (1-based). Multi-label allowed."""
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
    hops_adj: list[int] = []          # graph distance between adjacent steps
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

        # E3: AST graph distance between adjacent steps
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
            print(f"  processed {i}/{len(attr)}", flush=True)

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
        "no py-py adjacent pairs, undecidable" if not hops_total else
        f"{f1:.0%} of reachable pairs are within one hop, but only {fr:.0%} of adjacent pairs are reachable within 3 hops; "
        + ("A5 has empirical support" if (f1 or 0) >= 0.5 and (fr or 0) >= 0.5
           else "A5 lacks empirical support: most adjacent actions are not adjacent in the dependency graph, "
                "'one-hop neighbour' is not the policy's natural candidate, another substitute-action source is needed")
    )

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print("\n=== E5 semantic types of the top-1 critical step ===")
    for g in ("succeeded", "failed"):
        d = res["E5_top1_step_types"][g]
        print(f"  {g} (n={d['n_cases']}):")
        for k2, v in sorted(d["label_rates"].items(), key=lambda x: -x[1]):
            print(f"     {k2:<24} {v:>6.1%}  ({d['label_counts'][k2]})")
    e3 = res["E3_adjacent_step_graph_distance"]
    print(f"\n=== E3 AST graph distance of adjacent steps (A5 test) ===")
    print(f"  py→py adjacent pairs {e3['n_pairs_py_to_py']}, reachable within 3 hops {e3['n_reachable_within_3']}"
          f" ({e3['frac_reachable']}), unreachable {e3['n_unreachable']}")
    print(f"  distance distribution {e3['distance_counts']}  median {e3['median_distance']}")
    print(f"  → {e3['A5_interpretation']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
