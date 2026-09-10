"""V3j: treat the trace itself as a graph -- prediction and attribution on the execution provenance graph. Zero API.

Motivation
----------
All earlier experiments treated the trace as a **sequence** (tool_t → tool_{t+1}) and pinned the structural
information on the **code dependency graph**. But the trace is itself a graph: the **output** of step t is the
**input evidence** of step t'. V3i already showed the code graph does not characterize step-to-step transitions
(one-hop rate lift 0.94 on adjacent steps), while a hypothesis much closer to the mechanism was never tested:

    the agent does not walk to "a neighbor in the code", it walks to **something it just saw in some tool output**.

grep returns `astropy/io/ascii/rst.py:35:class RST(...)`, and the next step reads that file.
This is a **provenance edge** s_t → s_{t'}, unrelated to code dependency. It happens to cover exactly the 67% of
first-visit steps V3g declared "unpredictable" -- a first-visit object has never been **visited**, yet it has very
likely already been **mentioned**.

Provenance graph definition
---------------------------
nodes = steps s_1..s_n (+ terminal answer node A)
edges:
  P provenance  s_t → s_{t'}  if the object operated on by s_{t'} **appears literally in** s_t's output (t<t')
  S sequential  s_t → s_{t+1}
  A answer      s_t → A       if s_t's object appears in the final answer

Experiments in this script
--------------------------
  G0 provenance coverage: what fraction of first-visit objects were **mentioned earlier**? (bounds the hypothesis)
  G1 provenance predictors: uniform / recency / freq weighting, evaluated on a point set **pointwise identical
     to V3g's**, and compared pairwise against marginal and code-graph multi-source proximity (V3i's C1)
  G2 routing fix: pi·history + (1−pi)·provenance(unvisited), against V3g's pi·history + (1−pi)·code graph
  G3 structural quantities of the provenance graph: in-degree/out-degree/whether an answer ancestor, as
     candidate per-step UQ quantities
     (their correlation with U^exec waits on V4′'s real replay variance once V4′ is done)

Criterion
---------
If G0 coverage is high and G1 significantly beats both marginal and C1 on the F regime,
then the negative conclusion "first-visit steps are unpredictable" should be revised to
"**the wrong structure was used**" -- the right structure is not the code graph but the
agent's own observation provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.code_channel import load_graph                  # noqa: E402
from pilot.channels.parsed_command import parse_step                # noqa: E402
from pilot.harness import rollout as R                              # noqa: E402
from pilot.probes import task as T                                  # noqa: E402
from pilot.analyze.v3g_regime import (                              # noqa: E402
    p_hist, p_graph, rank_ge, exp_hit, exp_recip, eff_cand, ece, Acc, KS,
)
from pilot.analyze.v3i_graph_probe import (                         # noqa: E402
    multi_source_dist, seed_adjacency,
)

SEED = 20260814
# Path pattern: relative path with an extension, or grep's `path:line:` prefix
PATH_RE = re.compile(
    r"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\.(?:py|pyx|pyi|rst|txt|cfg|toml|ini|md|json|ya?ml|c|h|cpp|hpp|js|html))"
)


def load_episodes_with_output(runs_root: Path) -> list[dict]:
    """Isomorphic to v3_nextstep_dual.load_episodes, but **keeps each step's output text**."""
    insts = {i.instance_id: i for i in T.load_instances()}
    eps = []
    for f in sorted(runs_root.glob("shard*/rollouts/*.jsonl")):
        iid = f.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None:
            continue
        g = load_graph(iid)
        if not g.ok:
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
            seq.append({"tool": p.kind, "obj": p.obj,
                        "node": g.resolve(p.path or p.obj),
                        "out": s.output_text or ""})
        eps.append({"id": iid, "project": inst.project, "seq": seq,
                    "graph": g, "gold": inst.gold})
    return eps


def mentioned_nodes(g, text: str, cap: int = 4000) -> list[str]:
    """Extract, from a chunk of tool output, the paths resolvable to graph nodes (deduplicated, order preserved)."""
    if not text:
        return []
    out, seen = [], set()
    for m in PATH_RE.finditer(text[:200_000]):
        raw = m.group(1)
        n = g.resolve(raw)
        if n and not n.startswith("dir:") and n not in seen:
            seen.add(n)
            out.append(n)
            if len(out) >= cap:
                break
    return out


def p_prov(mentions: dict[str, list[int]], t: int, mode: str,
           gamma: float = 0.8, exclude: set[str] | None = None) -> dict[str, float]:
    """Provenance prior. mentions[node] = list of step indices where it was mentioned.

    mode: uniform / recency (decay by the most recent mention) / freq (by mention count)
    """
    w: dict[str, float] = {}
    for node, occ in mentions.items():
        if exclude and node in exclude:
            continue
        if mode == "uniform":
            w[node] = 1.0
        elif mode == "recency":
            w[node] = gamma ** max(0, t - max(occ))
        elif mode == "freq":
            w[node] = float(len(occ))
        else:
            raise ValueError(mode)
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def boot_ci(vals: list[float], b: int = 2000, seed: int = SEED) -> dict:
    if not vals:
        return {"note": "no samples"}
    rng = random.Random(seed)
    pts = []
    for _ in range(b):
        pts.append(sum(vals[rng.randrange(len(vals))]
                       for _ in range(len(vals))) / len(vals))
    pts.sort()
    return {"mean": round(statistics.mean(vals), 4),
            "ci95": [round(pts[int(0.025 * b)], 4), round(pts[int(0.975 * b)], 4)],
            "n": len(vals)}


def paired_ci(a: list[float], b: list[float], bnum: int = 2000,
              seed: int = SEED) -> dict:
    if len(a) != len(b) or not a:
        return {"note": f"point sets do not match {len(a)} vs {len(b)}"}
    rng = random.Random(seed)
    pts = []
    for _ in range(bnum):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        pts.append(sum(a[i] - b[i] for i in idx) / len(idx))
    pts.sort()
    lo, hi = pts[int(0.025 * bnum)], pts[int(0.975 * bnum)]
    return {"diff": round(statistics.mean(a) - statistics.mean(b), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3j_trace_graph.json")
    args = ap.parse_args()

    eps = load_episodes_with_output(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    print(f"loaded {len(eps)} episodes (graphs built, output text included)")

    PRED = ["prov_uniform", "prov_recency", "prov_freq",
            "prov_lastout", "prov_lastout_ranked",
            "hist_freq", "marginal_repo", "graph_C1",
            "routed_prov", "routed_graph"]
    # Pointwise expected hits, bucketed by regime; the F bucket carries the key comparison
    hits = {p: {r: {k: [] for k in KS} for r in ("all", "R", "F")} for p in PRED}
    cov = {p: {r: [] for r in ("all", "R", "F")} for p in PRED}
    effc = {p: [] for p in PRED}

    g0_mentioned_before, g0_first_visit = [], 0
    g0_by_regime = {"R": [], "F": []}
    n_R = n_F = 0
    # G3 structural quantities
    g3_rows = []

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [x for j, x in enumerate(group) if j != hi]
            objfreq = Counter(s["node"] for x in train for s in x["seq"] if s["node"])
            g, seq = held["graph"], held["seq"]
            nodes = [s["node"] for s in seq]
            # Pre-extract the nodes mentioned in each step's output
            outs = [mentioned_nodes(g, s["out"]) for s in seq]

            mentions: dict[str, list[int]] = defaultdict(list)
            visits: list[str] = []
            seen: set[str] = set()
            prev_rev = False
            # regime transition rate (same method as V3g, estimated from train)
            rc: dict[bool, Counter] = {True: Counter(), False: Counter()}
            for x in train:
                sq, sn = x["seq"], set()
                pr = False
                for t2 in range(len(sq) - 1):
                    if sq[t2]["node"]:
                        sn.add(sq[t2]["node"])
                    if not sq[t2 + 1]["node"]:
                        continue
                    isr = sq[t2 + 1]["node"] in sn
                    rc[pr][isr] += 1
                    pr = isr

            def pi_hat(pr: bool) -> float:
                row = rc[pr]
                tot = sum(row.values())
                return (row[True] + 0.5) / (tot + 1.0) if tot else 0.5

            for t in range(len(seq) - 1):
                if nodes[t]:
                    visits.append(nodes[t])
                    seen.add(nodes[t])
                # Step t's output is **already visible** when predicting step t+1
                for nd in outs[t]:
                    mentions[nd].append(t)
                truth = nodes[t + 1]
                if not truth:
                    continue
                is_rev = truth in seen
                reg = "R" if is_rev else "F"
                if is_rev:
                    n_R += 1
                else:
                    n_F += 1
                    g0_first_visit += 1
                    g0_mentioned_before.append(1.0 if truth in mentions else 0.0)
                g0_by_regime[reg].append(1.0 if truth in mentions else 0.0)

                pi = pi_hat(prev_rev)
                fseeds = {x for x in seen if not x.startswith("dir:") and x in g.files}
                # Code graph C1 multi-source proximity (V3i's strongest form), excluding already visited
                if fseeds:
                    dd = multi_source_dist(g, fseeds)
                    aa = seed_adjacency(g, fseeds)
                    mx = max(dd.values()) or 1
                    c1 = {x: (mx - dd[x] + aa.get(x, 0) / 100.0)
                          for x in dd if x not in seen}
                    z1 = sum(c1.values())
                    c1 = {k: v / z1 for k, v in c1.items()} if z1 else {}
                else:
                    c1 = {}
                prov_uns = p_prov(mentions, t, "freq", exclude=seen)
                ph = p_hist(visits, nodes[t], "freq")
                # A sharper mechanism test: use only the **most recent output that had mentions**.
                # If it already reaches prov_recency's level, then the mechanism is
                # "read the files listed in that last output", not "a weighting of the whole observation history".
                last: list[str] = []
                for tt in range(t, -1, -1):
                    if outs[tt]:
                        last = outs[tt]
                        break
                lo_u = _norm({x: 1.0 for x in last})
                lo_r = _norm({x: 1.0 / (i + 1) for i, x in enumerate(last)})
                dists = {
                    "prov_uniform": p_prov(mentions, t, "uniform"),
                    "prov_recency": p_prov(mentions, t, "recency"),
                    "prov_freq":    p_prov(mentions, t, "freq"),
                    "prov_lastout": lo_u,
                    "prov_lastout_ranked": lo_r,
                    "hist_freq":    ph,
                    "marginal_repo": _norm({k: float(v) for k, v in objfreq.items()}),
                    "graph_C1":     c1,
                    "routed_prov":  _mix(ph, prov_uns, pi),
                    "routed_graph": _mix(ph, c1, pi),
                }
                for nm, d in dists.items():
                    rs = rank_ge(d, truth)
                    for bucket in ("all", reg):
                        cov[nm][bucket].append(1.0 if rs else 0.0)
                        for k in KS:
                            hits[nm][bucket][k].append(
                                exp_hit(rs[0], rs[1], k) if rs else 0.0)
                    if rs:
                        effc[nm].append(eff_cand(d))
                prev_rev = is_rev

            # ---- G3 provenance graph structural quantities ----
            n_edges = sum(1 for t in range(len(seq))
                          for nd in outs[t] if nd in nodes[t + 1:t + 2])
            g3_rows.append({
                "id": held["id"], "n_steps": len(seq),
                "n_mentioned_total": len({n for o in outs for n in o}),
                "n_visited": len({n for n in nodes if n}),
                "prov_immediate_edges": n_edges,
                "mention_to_visit_rate": (
                    len({n for o in outs for n in o} & {n for n in nodes if n})
                    / max(1, len({n for o in outs for n in o}))),
            })

    res: dict = {
        "n_episodes": len(eps),
        "regime_split": {"n_revisit": n_R, "n_first_visit": n_F},
        "G0_provenance_coverage": {
            "first_visit_target_was_mentioned_before": boot_ci(g0_mentioned_before),
            "by_regime": {r: boot_ci(v) for r, v in g0_by_regime.items()},
            "note": "fraction of first-visit objects that were \"already mentioned in some earlier step's output\" "
                    "-- the upper bound of the provenance hypothesis",
        },
        "G1_predictors": {
            nm: {
                "all": {"coverage": boot_ci(cov[nm]["all"])["mean"],
                        "topk": {f"top{k}": round(statistics.mean(hits[nm]["all"][k]), 4)
                                 for k in KS}},
                "R": {"coverage": boot_ci(cov[nm]["R"])["mean"],
                      "topk": {f"top{k}": round(statistics.mean(hits[nm]["R"][k]), 4)
                               for k in KS}},
                "F": {"coverage": boot_ci(cov[nm]["F"])["mean"],
                      "topk": {f"top{k}": boot_ci(hits[nm]["F"][k]) for k in KS}},
                "eff_candidates": (round(statistics.median(effc[nm]), 2)
                                   if effc[nm] else None),
            } for nm in PRED},
        "G1_contrasts_on_first_visit": {
            "prov_freq_vs_marginal_top10": paired_ci(hits["prov_freq"]["F"][10],
                                                     hits["marginal_repo"]["F"][10]),
            "prov_freq_vs_marginal_top3": paired_ci(hits["prov_freq"]["F"][3],
                                                    hits["marginal_repo"]["F"][3]),
            "prov_freq_vs_graphC1_top10": paired_ci(hits["prov_freq"]["F"][10],
                                                    hits["graph_C1"]["F"][10]),
            "prov_freq_vs_graphC1_top3": paired_ci(hits["prov_freq"]["F"][3],
                                                   hits["graph_C1"]["F"][3]),
            "prov_recency_vs_prov_freq_top3": paired_ci(hits["prov_recency"]["F"][3],
                                                        hits["prov_freq"]["F"][3]),
            # Mechanism discrimination: whole observation history vs only the most recent output
            "prov_recency_vs_lastout_top3": paired_ci(hits["prov_recency"]["F"][3],
                                                      hits["prov_lastout"]["F"][3]),
            "prov_recency_vs_lastout_top10": paired_ci(hits["prov_recency"]["F"][10],
                                                       hits["prov_lastout"]["F"][10]),
            "lastout_ranked_vs_lastout_top3": paired_ci(
                hits["prov_lastout_ranked"]["F"][3], hits["prov_lastout"]["F"][3]),
        },
        "G2_routing_contrasts_all": {
            "routed_prov_vs_routed_graph_top10": paired_ci(
                hits["routed_prov"]["all"][10], hits["routed_graph"]["all"][10]),
            "routed_prov_vs_routed_graph_top3": paired_ci(
                hits["routed_prov"]["all"][3], hits["routed_graph"]["all"][3]),
            "routed_prov_vs_hist_top10": paired_ci(
                hits["routed_prov"]["all"][10], hits["hist_freq"]["all"][10]),
        },
        "G3_trace_graph_structure": {
            "per_episode_mean": {
                k: round(statistics.mean([r[k] for r in g3_rows]), 4)
                for k in ("n_steps", "n_mentioned_total", "n_visited",
                          "mention_to_visit_rate")},
            "note": "n_mentioned_total far exceeding n_visited means the agent sees far more than it goes to; "
                    "mention_to_visit_rate is the fraction of \"mentioned objects that end up visited\"",
        },
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ------------------------------------------------------------- print
    g0 = res["G0_provenance_coverage"]
    print(f"\n=== G0 provenance coverage (upper bound of the hypothesis) ===")
    b = g0["first_visit_target_was_mentioned_before"]
    print(f"  first-visit object \"already mentioned in an output\": {b['mean']:.4f} CI{b['ci95']} (n={b['n']})")
    for r in ("R", "F"):
        v = g0["by_regime"][r]
        print(f"    regime {r}: {v['mean']:.4f} CI{v['ci95']} (n={v['n']})")

    print(f"\n=== G1 predictors: all points (denominator={n_R + n_F}) ===")
    print(f"{'predictor':<16}{'coverage':>8}{'top1':>8}{'top3':>8}{'top10':>8}{'eff_cand':>10}")
    for nm in PRED:
        d = res["G1_predictors"][nm]["all"]
        print(f"{nm:<16}{d['coverage']:>8.4f}{d['topk']['top1']:>8.4f}"
              f"{d['topk']['top3']:>8.4f}{d['topk']['top10']:>8.4f}"
              f"{str(res['G1_predictors'][nm]['eff_candidates']):>10}")

    print(f"\n=== G1 first-visit steps F (n={n_F}) -- the decisive battleground ===")
    print(f"{'predictor':<16}{'coverage':>8}{'top3':>18}{'top10':>18}")
    for nm in PRED:
        d = res["G1_predictors"][nm]["F"]
        t3, t10 = d["topk"]["top3"], d["topk"]["top10"]
        if "note" in t3:
            continue
        print(f"{nm:<16}{d['coverage']:>8.4f}"
              f"{t3['mean']:>10.4f} CI{t3['ci95']}"
              f"{t10['mean']:>8.4f} CI{t10['ci95']}")

    print(f"\n=== Paired contrasts on first-visit steps ===")
    for k, v in res["G1_contrasts_on_first_visit"].items():
        if "note" in v:
            print(f"  {k:<34} {v['note']}")
            continue
        print(f"  {k:<34} {v['diff']:+.4f}  CI[{v['ci95'][0]:+.4f}, "
              f"{v['ci95'][1]:+.4f}]  {'significant' if v['significant'] else 'not significant'}")

    print(f"\n=== G2 routing: provenance vs code graph ===")
    for k, v in res["G2_routing_contrasts_all"].items():
        if "note" in v:
            continue
        print(f"  {k:<38} {v['diff']:+.4f}  CI[{v['ci95'][0]:+.4f}, "
              f"{v['ci95'][1]:+.4f}]  {'significant' if v['significant'] else 'not significant'}")

    print(f"\n=== G3 provenance graph scale ===")
    for k, v in res["G3_trace_graph_structure"]["per_episode_mean"].items():
        print(f"  {k:<24} {v}")

    print(f"\n→ {args.out}")
    return 0


def _norm(w: dict[str, float]) -> dict[str, float]:
    z = sum(w.values())
    return {k: v / z for k, v in w.items()} if z else {}


def _mix(a: dict[str, float], b: dict[str, float], pi: float) -> dict[str, float]:
    w: dict[str, float] = {}
    for k, v in a.items():
        w[k] = w.get(k, 0.0) + pi * v
    for k, v in b.items():
        w[k] = w.get(k, 0.0) + (1 - pi) * v
    return _norm(w)


if __name__ == "__main__":
    raise SystemExit(main())
