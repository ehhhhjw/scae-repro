"""Discriminativeness test for AST features -- verify before investing.

Before building a full index over 499 repositories (an estimated 20+ minutes), answer one
question on a subset first:
**can AST semantic features separate successful from failed localization tasks?**

If not, the AST route is just as useless as the earlier surface behavioral features and is not
worth the full investment.
If yes, then the three prior prediction failures really did stem from "missing semantics" rather
than "the task is unpredictable".

Test design
-----------
Take all failure cases (30) + an equal number of stratified-sampled success cases; for each instance:
  1. extract candidate symbols from the issue text (CamelCase / snake_case identifiers)
  2. measure the ambiguity of those symbols with the AST index (number of definition sites,
     max name collisions, number of files hit, directory spread)
  3. additionally measure the structural position of the gold files in the dependency graph
     (in-degree, out-degree, leaf or not)
     -- note: gold structural features are **for explanatory analysis only** and must not enter
     the predictive features (they would leak)
  4. compare the distributions of the success group vs the failure group, report Mann-Whitney U
     and the effect size

Strict separation of two feature classes
----------------------------------------
  predictive (execution-time available): computed only from issue text + repository AST, no gold
  explanatory only (post-hoc only): involves the structural position of F*, used to understand the
  failure mechanism, never enters the model
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.graph.ast_index import build_index, summarize     # noqa: E402
from pilot.probes import task as T                           # noqa: E402

_CAMEL = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}\b")
_SNAKE = re.compile(r"\b[a-z_][a-z0-9_]{3,}\b")
_STOP = {"self", "none", "true", "false", "return", "import", "class", "print",
         "value", "error", "result", "should", "would", "there", "which",
         "python", "django", "test", "tests", "code", "line", "file", "with",
         "this", "that", "from", "have", "when", "then", "will", "using"}


def issue_symbols(text: str, limit: int = 25) -> list[str]:
    """Extract candidate symbols from the issue text. Text only, no repository, no gold."""
    cams = _CAMEL.findall(text or "")
    snakes = [w for w in _SNAKE.findall(text or "") if w not in _STOP]
    cnt = Counter(cams + snakes)
    return [w for w, _ in cnt.most_common(limit)]


def mann_whitney_u(a: list[float], b: list[float]) -> dict:
    """Mann-Whitney U (normal approximation) + rank-biserial r effect size. No scipy dependency."""
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    na, nb = len(a), len(b)
    if na < 5 or nb < 5:
        return {"n_a": na, "n_b": nb, "note": "insufficient samples"}
    allv = [(v, 0) for v in a] + [(v, 1) for v in b]
    allv.sort(key=lambda x: x[0])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    ra = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    ua = ra - na * (na + 1) / 2
    mu = na * nb / 2
    sd = math.sqrt(na * nb * (na + nb + 1) / 12)
    z = (ua - mu) / sd if sd else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    r_rb = 2 * ua / (na * nb) - 1          # rank-biserial, range [-1,1]
    return {"n_a": na, "n_b": nb,
            "median_a": round(statistics.median(a), 3),
            "median_b": round(statistics.median(b), 3),
            "U": round(ua, 1), "z": round(z, 3), "p_two_sided": round(p, 4),
            "rank_biserial_r": round(r_rb, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="pilot/runs/full_locate/results.jsonl")
    ap.add_argument("--n-success", type=int, default=40,
                    help="how many success cases to sample as control")
    ap.add_argument("--out", default="pilot/runs/ast_probe.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    recs = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    ok = [r for r in recs if r.get("ok")]
    fails = [r for r in ok if r.get("Y") == 0]
    succs = [r for r in ok if r.get("Y") == 1]
    # stratified sampling of success controls: match the failure group on repo and gold count as far as possible
    key = lambda r: (r.get("project"), min(r.get("n_gold", 1), 3))          # noqa: E731
    want = Counter(key(r) for r in fails)
    picked, pool = [], sorted(succs, key=lambda r: r["instance_id"])
    for k, need in want.items():
        cands = [r for r in pool if key(r) == k and r not in picked]
        picked += cands[: need * 2]
    for r in pool:
        if len(picked) >= args.n_success:
            break
        if r not in picked:
            picked.append(r)
    picked = picked[: args.n_success]

    target = fails + picked
    print(f"test sample: fail {len(fails)} + success {len(picked)} = {len(target)}")

    rows = []
    for i, r in enumerate(target, 1):
        inst = insts.get(r["instance_id"])
        if inst is None or not inst.repo.is_dir():
            continue
        try:
            idx = build_index(inst.repo)
        except (OSError, RecursionError):
            continue
        syms = issue_symbols(inst.problem_statement)
        amb = idx.ambiguity(syms)
        # explanatory only: position of gold in the dependency graph
        gold_in = gold_out = 0
        gold_found = 0
        for g in inst.gold:
            if g in idx.file_syms or g in idx.edges:
                gold_found += 1
            gold_out += len(idx.edges.get(g, set()))
            gold_in += sum(1 for f, outs in idx.edges.items() if g in outs)
        rows.append({
            "instance_id": r["instance_id"], "project": r.get("project"),
            "Y": r.get("Y"), "n_gold": r.get("n_gold"),
            "n_issue_syms": len(syms),
            # --- predictive (no gold) ---
            "n_defsites": amb["n_defsites"],
            "max_fanout": amb["max_fanout"],
            "n_files_hit": amb["n_files_hit"],
            "spread": amb["spread"],
            "repo_files": idx.n_files,
            "hit_ratio": round(amb["n_files_hit"] / max(1, idx.n_files), 5),
            # --- explanatory only (contains gold, must not enter the model) ---
            "gold_indeg": gold_in, "gold_outdeg": gold_out,
            "gold_indexed": gold_found,
        })
        if i % 10 == 0:
            print(f"  processed {i}/{len(target)}", flush=True)

    A = [x for x in rows if x["Y"] == 0]     # failure
    B = [x for x in rows if x["Y"] == 1]     # success
    tests = {}
    for f in ("n_defsites", "max_fanout", "n_files_hit", "spread", "hit_ratio",
              "repo_files", "gold_indeg", "gold_outdeg"):
        tests[f] = mann_whitney_u([x[f] for x in A], [x[f] for x in B])

    res = {"n_rows": len(rows), "n_fail": len(A), "n_success": len(B),
           "predictive_features": ["n_defsites", "max_fanout", "n_files_hit",
                                   "spread", "hit_ratio", "repo_files"],
           "explanatory_only": ["gold_indeg", "gold_outdeg"],
           "tests": tests, "rows": rows}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== AST feature discriminativeness (fail {len(A)} vs success {len(B)}) ===")
    print(f"{'feature':<16}{'fail med':>10}{'succ med':>10}{'z':>8}{'p':>9}{'r':>8}  category")
    for f, t in tests.items():
        if "note" in t:
            print(f"  {f:<14} {t['note']}")
            continue
        cat = "predictive" if f in res["predictive_features"] else "explanatory only"
        star = " *" if t["p_two_sided"] < 0.05 else ""
        print(f"{f:<16}{t['median_a']:>10}{t['median_b']:>10}"
              f"{t['z']:>8}{t['p_two_sided']:>9}{t['rank_biserial_r']:>8}  {cat}{star}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
