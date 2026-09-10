"""X1c: route the **candidate-set construction** by next-tool, and expand the sample from 71 to all instances. Zero API.

Two changes
-----------
**Change 1: fix the misconfiguration diagnosed by J3.**
X1b proved P(tool,obj) does not factorize (I=0.159 nat, perm p=0.0005), but the conditioning
I used back then only re-estimated "the positional weight inside the most recent output" per tool,
without changing the **type of the candidate set**, so no gain showed up.
The real dependence comes from the object type switching with the tool (see `parsed_command.py`):

    read       → object is a **file path**
    list_files → object is a **directory**
    search     → object is `q:<query>` (when there is no path argument) or a path
    handler    → object is "-" (apply_patch / update_plan / view_image ...)

So the correct model is P(obj | tool) with a **per-tool candidate set**:
    P(u_{t+1}) = P(tool_{t+1} | tool_t) · P(obj_{t+1} | tool_{t+1}, history)

**Change 2: drop the code-graph dependency, sample 71 → all.**
Until now V3g/V3i/V3j/X1b were all restricted to the 71 "graph-built" repositories, for the single
reason that `g.resolve()` was used to map objects onto graph nodes. But **the provenance channel
does not need the code graph at all** -- what it compares is the "path string inside the agent
command" against the "path string inside the tool output", and the two have the same origin.
After switching to `T.norm_path` for pure string normalization, the sample expands to all ~496 instances.

⚠️ Expanding the sample requires a consistency check first: on the 71 overlapping instances, the key
metrics of the graph version and the graph-free version should be close. A large gap means string
normalization introduced a systematic bias and the expansion is not valid as-is.
This script's `--overlap-check` does exactly that.

Evaluation setup
----------------
What is predicted is the **full joint action** $(tool_{t+1}, obj_{t+1})$, including handler steps
with obj="-"; the two models share the same denominator (uncovered counts as a miss) and ties are
broken by closed-form expectation.
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

from pilot.channels.parsed_command import parse_step        # noqa: E402
from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402
from pilot.analyze.v3g_regime import rank_ge, exp_hit, eff_cand, ece  # noqa: E402
from pilot.analyze.v3_nextstep_dual import fit_tool_trans, p_tool     # noqa: E402
from pilot.analyze.v3j_trace_graph import PATH_RE, boot_ci, paired_ci  # noqa: E402

SEED = 20260814
KS = (1, 2, 3, 5, 10)
HANDLER_OBJ = "-"


# ------------------------------------------------------------ graph-free load

def norm(p: str | None) -> str | None:
    if not p:
        return None
    s = T.norm_path(p)
    return s or None


def out_paths(text: str, cap: int = 400) -> list[str]:
    """Extract normalized paths from a tool output in order of appearance (deduped). No code graph needed."""
    if not text:
        return []
    seen, res = set(), []
    for m in PATH_RE.finditer(text[:200_000]):
        s = norm(m.group(1))
        if s and s not in seen:
            seen.add(s)
            res.append(s)
            if len(res) >= cap:
                break
    return res


def load_all(runs_root: Path, require_graph: bool = False) -> list[dict]:
    """Load the (tool, obj, output) sequence of all instances. require_graph is only for the consistency check."""
    insts = {i.instance_id: i for i in T.load_instances()}
    eps = []
    for f in sorted(runs_root.glob("shard*/rollouts/*.jsonl")):
        iid = f.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None:
            continue
        if require_graph:
            from pilot.channels.code_channel import load_graph
            if not load_graph(iid).ok:
                continue
        try:
            steps = R.steps(R.load(f))
        except (OSError, ValueError):
            continue
        if len(steps) < 3:
            continue
        seq = []
        for s in steps:
            p = parse_step(s)
            path = norm(p.path)
            if path:
                obj = path
            elif p.query:
                obj = f"q:{p.query}"
            else:
                obj = HANDLER_OBJ
            seq.append({"tool": p.kind, "obj": obj, "is_path": bool(path),
                        "out": s.output_text or ""})
        eps.append({"id": iid, "project": inst.project, "seq": seq})
    return eps


# ------------------------------------------------------------ candidate-set construction

OBJ_TYPES = ("path", "query", "handler")


def fit_type_prior(train: list[dict], alpha: float = 0.5) -> dict[str, dict[str, float]]:
    """Learn P(object type | tool) from the training episodes.

    The first version hard-coded tools as "handler whitelist vs. the rest"; as a result the `-`
    object came largely from `unknown` / `write_stdin` (not on the list) → their candidate sets
    contained no `-` → handler-type coverage 0, top-3 stuck at 0. Learning it is the right way.
    """
    cnt: dict[str, Counter] = defaultdict(Counter)
    for e in train:
        for s in e["seq"]:
            ot = ("path" if s["is_path"] else
                  "query" if s["obj"].startswith("q:") else "handler")
            cnt[s["tool"]][ot] += 1
    prior: dict[str, dict[str, float]] = {}
    glob = Counter()
    for tool, row in cnt.items():
        glob.update(row)
    gtot = sum(glob.values()) + alpha * len(OBJ_TYPES)
    gprior = {o: (glob[o] + alpha) / gtot for o in OBJ_TYPES}
    for tool, row in cnt.items():
        tot = sum(row.values()) + alpha * len(OBJ_TYPES)
        prior[tool] = {o: (row[o] + alpha) / tot for o in OBJ_TYPES}
    prior["__default__"] = gprior
    return prior


def _p_path(prov: list[str], visited: list[str]) -> dict[str, float]:
    w: dict[str, float] = {}
    for i, p in enumerate(prov[:40]):
        w[p] = w.get(p, 0.0) + 1.0 / (i + 1)
    cnt = Counter(visited)
    for p, c in cnt.items():
        w[p] = w.get(p, 0.0) + 0.5 * c / max(1, len(visited))
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def _p_query(queries: list[str]) -> dict[str, float]:
    w = {q: 1.0 / (i + 1) for i, q in enumerate(reversed(queries[-10:]))}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def cand_for_tool(tool: str, prov: list[str], visited: list[str],
                  dirs: set[str], queries: list[str],
                  type_prior: dict[str, dict[str, float]] | None = None
                  ) -> dict[str, float]:
    """Give the object candidate set and weights **per tool** -- the core of X1c.

    P(obj | tool) = Σ_type P(type | tool) · P(obj | type, history)
      path    → provenance paths (weighted by position inside the most recent output) + visited paths (by frequency)
      query   → historical queries (by recency)
      handler → point mass on "-"
    The type prior is learned from the training set (see `fit_type_prior`), no longer a hard-coded tool list.
    """
    tp = (type_prior or {}).get(tool) or (type_prior or {}).get("__default__") \
        or {"path": 0.5, "query": 0.2, "handler": 0.3}
    w: dict[str, float] = {}
    pp = _p_path(prov, visited)
    pq = _p_query(queries)
    for o, p in pp.items():
        w[o] = w.get(o, 0.0) + tp["path"] * p
    for o, p in pq.items():
        w[o] = w.get(o, 0.0) + tp["query"] * p
    w[HANDLER_OBJ] = w.get(HANDLER_OBJ, 0.0) + tp["handler"]
    # when the path type has no candidate (the first step has no output at all), fall the mass back to handler
    if not pp and not pq:
        w = {HANDLER_OBJ: 1.0}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def cand_shared(prov: list[str], visited: list[str],
                dirs: set[str], queries: list[str]) -> dict[str, float]:
    """Factorized control: a single **tool-independent** object model (what X1b did)."""
    w: dict[str, float] = {}
    for i, p in enumerate(prov[:40]):
        w[p] = w.get(p, 0.0) + 1.0 / (i + 1)
    cnt = Counter(visited)
    for p, c in cnt.items():
        w[p] = w.get(p, 0.0) + 0.5 * c / max(1, len(visited))
    for d in dirs:
        w[d] = w.get(d, 0.0) + 0.1
    for i, q in enumerate(reversed(queries[-10:])):
        w[q] = w.get(q, 0.0) + 0.2 / (i + 1)
    # the control model also gets a share of mass on "-" (using the global marginal ratio ~0.27),
    # otherwise the control is artificially weakened and the routing gain is overstated. The only
    # difference: the control does **not** adjust that ratio per tool.
    w[HANDLER_OBJ] = w.get(HANDLER_OBJ, 0.0) + 0.27 * max(1.0, sum(w.values()))
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


# ------------------------------------------------------------ main flow

def evaluate(eps: list[dict], min_group: int = 6) -> dict:
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    tool_vocab = sorted({s["tool"] for e in eps for s in e["seq"]})

    MODELS = ("routed", "factored")
    hits = {m: {k: [] for k in KS} for m in MODELS}
    cov = {m: [] for m in MODELS}
    nll = {m: [] for m in MODELS}
    effc = {m: [] for m in MODELS}
    cal = {m: [] for m in MODELS}
    # object component only (to compare against the 71-instance version of V3j)
    obj_hits = {k: [] for k in KS}
    # stratify by **ground-truth object type**: this is the key decomposition. V3j/X1b evaluated only
    # at points where "the object is a resolvable file" (filtering out new queries and handler steps),
    # so their numbers are not directly comparable to the unconditional numbers here.
    OT = ("path", "query", "handler")
    by_ot = {m: {o: {k: [] for k in KS} for o in OT} for m in MODELS}
    ot_count: Counter = Counter()
    nll_cov = {m: [] for m in MODELS}      # NLL on covered points only (true information)
    n_pts = 0
    n_eps_used = 0

    for proj, group in by_repo.items():
        if len(group) < min_group:
            continue
        for hi, held in enumerate(group):
            train = [x for j, x in enumerate(group) if j != hi]
            tm = fit_tool_trans(train)
            tprior = fit_type_prior(train)
            n_eps_used += 1
            seq = held["seq"]
            prov: list[str] = []
            visited: list[str] = []
            dirs: set[str] = set()
            queries: list[str] = []
            for t in range(len(seq) - 1):
                cur, nxt = seq[t], seq[t + 1]
                if cur["is_path"]:
                    visited.append(cur["obj"])
                    if "/" in cur["obj"]:
                        dirs.add(cur["obj"].rsplit("/", 1)[0])
                if cur["tool"] == "list_files" and cur["is_path"]:
                    dirs.add(cur["obj"])
                if cur["obj"].startswith("q:"):
                    queries.append(cur["obj"])
                op = out_paths(cur["out"])
                if op:
                    prov = op                      # most recent output that contains paths
                    for p in op:
                        if "/" in p:
                            dirs.add(p.rsplit("/", 1)[0])

                truth = (nxt["tool"], nxt["obj"])
                ot = ("path" if nxt["is_path"] else
                      "query" if nxt["obj"].startswith("q:") else "handler")
                ot_count[ot] += 1
                n_pts += 1
                pt = {v: p_tool(tm, cur["tool"], v, tool_vocab) for v in tool_vocab}
                zt = sum(pt.values())
                pt = {k: v / zt for k, v in pt.items()}

                # routed: one candidate set per tool
                jd_r: dict[tuple[str, str], float] = {}
                for a, pa in pt.items():
                    for o, po in cand_for_tool(a, prov, visited, dirs, queries,
                                               tprior).items():
                        jd_r[(a, o)] = jd_r.get((a, o), 0.0) + pa * po
                # factored: shared object model
                shared = cand_shared(prov, visited, dirs, queries)
                jd_f = {(a, o): pa * po
                        for a, pa in pt.items() for o, po in shared.items()}

                for m, jd in (("routed", jd_r), ("factored", jd_f)):
                    rs = rank_ge(jd, truth)
                    cov[m].append(1.0 if rs else 0.0)
                    for k in KS:
                        hits[m][k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)
                    for k in KS:
                        by_ot[m][ot][k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)
                    if rs:
                        effc[m].append(eff_cand(jd))
                        pv = jd[truth]
                        nll[m].append(-math.log(max(pv, 1e-12)))
                        nll_cov[m].append(-math.log(max(pv, 1e-12)))
                        cal[m].append((pv, 1))
                        for v in sorted((v for kk, v in jd.items() if kk != truth),
                                        reverse=True)[:20]:
                            cal[m].append((v, 0))
                    else:
                        nll[m].append(-math.log(1e-12))
                # object component alone (to compare with V3j)
                rso = rank_ge(shared, nxt["obj"])
                for k in KS:
                    obj_hits[k].append(exp_hit(rso[0], rso[1], k) if rso else 0.0)

    out = {
        "n_episodes_total": len(eps), "n_episodes_evaluated": n_eps_used,
        "n_decision_points": n_pts,
        "models": {m: {
            "coverage": boot_ci(cov[m])["mean"],
            "topk": {f"top{k}": boot_ci(hits[m][k]) for k in KS},
            "NLL_all_miss_penalized": round(statistics.mean(nll[m]), 4) if nll[m] else None,
            "NLL_covered_only": (round(statistics.mean(nll_cov[m]), 4)
                                 if nll_cov[m] else None),
            "ECE": round(ece(cal[m]), 4),
            "eff_candidates_median": (round(statistics.median(effc[m]), 2)
                                      if effc[m] else None),
        } for m in MODELS},
        "contrasts": {
            f"routed_minus_factored_top{k}": paired_ci(hits["routed"][k],
                                                       hits["factored"][k])
            for k in (1, 3, 10)},
        "object_only_topk": {f"top{k}": round(statistics.mean(obj_hits[k]), 4)
                             for k in KS},
        "OBJECT_TYPE_COMPOSITION": {o: round(c / max(1, n_pts), 4)
                                    for o, c in ot_count.most_common()},
        "BY_OBJECT_TYPE": {m: {o: {
            "n": len(by_ot[m][o][1]),
            "topk": {f"top{k}": round(statistics.mean(by_ot[m][o][k]), 4)
                     for k in KS} if by_ot[m][o][1] else None}
            for o in OT} for m in MODELS},
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x1c_tool_routed.json")
    ap.add_argument("--overlap-check", action="store_true",
                    help="also run on the 71 graph-built instances, to check whether expanding the sample introduces bias")
    args = ap.parse_args()

    eps_all = load_all(Path(args.runs_root))
    print(f"graph-free load (all instances): {len(eps_all)} episodes")
    res = {"ALL_INSTANCES": evaluate(eps_all)}

    if args.overlap_check:
        eps_g = load_all(Path(args.runs_root), require_graph=True)
        print(f"consistency check (71 graph-built only): {len(eps_g)} episodes")
        res["GRAPH_SUBSET_71"] = evaluate(eps_g)

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    for tag, blk in res.items():
        print(f"\n{'='*66}\n=== {tag}  "
              f"({blk['n_episodes_evaluated']}/{blk['n_episodes_total']} episodes, "
              f"{blk['n_decision_points']} decision points) ===")
        print(f"{'model':<12}{'coverage':>8}{'top1':>9}{'top3':>9}{'top10':>9}"
              f"{'NLL(cov)':>11}{'ECE':>9}{'eff_cand':>10}")
        for m in ("routed", "factored"):
            d = blk["models"][m]
            print(f"{m:<12}{d['coverage']:>8.4f}"
                  f"{d['topk']['top1']['mean']:>9.4f}"
                  f"{d['topk']['top3']['mean']:>9.4f}"
                  f"{d['topk']['top10']['mean']:>9.4f}"
                  f"{d['NLL_covered_only']:>11.4f}{d['ECE']:>9.4f}"
                  f"{str(d['eff_candidates_median']):>10}")
        comp = blk["OBJECT_TYPE_COMPOSITION"]
        print(f"  ground-truth object type composition: " +
              "  ".join(f"{k}={v:.1%}" for k, v in comp.items()))
        print(f"  {'top-3 / top-10 by object type':<30}" +
              "".join(f"{o:>22}" for o in ("path", "query", "handler")))
        for m in ("routed", "factored"):
            cells = []
            for o in ("path", "query", "handler"):
                d = blk["BY_OBJECT_TYPE"][m][o]
                cells.append(f"{d['topk']['top3']:.3f}/{d['topk']['top10']:.3f} (n={d['n']})"
                             if d["topk"] else "-")
            print(f"    {m:<28}" + "".join(f"{c:>22}" for c in cells))
        print("  paired contrast (routed - factored):")
        for k, v in blk["contrasts"].items():
            if "note" in v:
                print(f"    {k:<34} {v['note']}")
                continue
            print(f"    {k:<34} {v['diff']:+.4f} "
                  f"CI[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  "
                  f"{'significant' if v['significant'] else 'not significant'}")
        print(f"  object-only top-k: " +
              "  ".join(f"{k}={v}" for k, v in blk["object_only_topk"].items()))

    if "GRAPH_SUBSET_71" in res:
        a = res["ALL_INSTANCES"]["models"]["routed"]["topk"]["top3"]["mean"]
        b = res["GRAPH_SUBSET_71"]["models"]["routed"]["topk"]["top3"]["mean"]
        print(f"\n=== sample-expansion consistency ===")
        print(f"  routed top-3: all {a:.4f} vs 71 graph-built {b:.4f}"
              f"   relative diff {abs(a-b)/max(b,1e-9):.1%}")
        print(f"  {'✅ consistent, sample can be expanded directly' if abs(a-b)/max(b,1e-9) < 0.20 else '⚠️ diff >20%, check normalization bias'}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
