"""V3i: multi-angle evidence on the code graph. Zero API.

Why multiple angles are needed
------------------------------
V3g already showed the code graph is useless on next-object (first-visit top-10 0.077,
significantly worse than repository term frequency 0.163).
But a single failure is not enough to assert "the code graph has no value", because there
are at least five alternative explanations:

  (a) measurement: the agent's objects never resolved to graph nodes at all → the graph never had a chance
  (b) usage: only "one-hop neighbors of the current node" was used, not "the whole visited set" for multi-source proximity
  (c) granularity: node level fails, community level might work
  (d) target: the graph does not predict "the next step", but might predict "which files are gold"
  (e) description vs prediction: the graph may characterize the trace (adjacent steps closer than random), just not enough to rank

This script checks each in turn, and every item comes with a null hypothesis. Only if all
these angles fail is "the code graph is ineffective for characterizing the movement of agent
attention" an assertable conclusion; if one succeeds, that one is the graph's correct use.

Angles and criteria
-------------------
A measurement validity  A1 object→graph-node resolution rate   A2 is gold in the graph, how far from visited
B descriptive assoc.    B1 adjacent-step graph distance vs permutation null (**separates self-repeats at distance 0**)
                        B2 adjacent-step same-community rate vs permutation null
C stronger predictors   C1 multi-source proximity (whole visited set as seeds, ranked over the whole repository)
                        C2 k-hop sweep (coverage-precision tradeoff at 1/2/3 hops)
                        C3 pure degree-centrality baseline (no trace information) -- if on par
                           with C1, then "the graph-trace relation" adds no incremental gain
                           and the graph is merely recommending popular files
D change of target      D1 can graph proximity predict gold files (a possible positive use)
                        D2 is the visited set more clustered than a random set of the same
                           size (a descriptive claim)

Key pitfall
-----------
W_L1=0.62 comes from "62% of the 548 adjacent step pairs are within one hop". If that
statistic also counts **two consecutive steps on the same file (distance 0)** as "within one
hop", the weight is a self-repetition artifact
(V3h already showed immediately adjacent object repeats are 1.83x random). B1 explicitly
separates d=0 from d=1.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.code_channel import load_graph          # noqa: E402
from pilot.analyze.v3_nextstep_dual import load_episodes    # noqa: E402
from pilot.probes import task as T                          # noqa: E402

B_PERM = 1000
SEED = 20260814
KS = (1, 3, 5, 10, 20, 50)


# ---------------------------------------------------------------- helpers

def multi_source_dist(g, seeds: set[str], max_hops: int = 4) -> dict[str, int]:
    """Multi-source BFS from seeds; returns the shortest hop count per reachable node."""
    dist: dict[str, int] = {s: 0 for s in seeds if s in g.edges or s in g.files}
    dq = deque(dist)
    while dq:
        x = dq.popleft()
        d = dist[x]
        if d >= max_hops:
            continue
        for y in g.edges.get(x, ()):
            if y not in dist:
                dist[y] = d + 1
                dq.append(y)
    return dist


def seed_adjacency(g, seeds: set[str]) -> Counter:
    """How many seeds each node is directly adjacent to (tie-break signal for multi-source proximity)."""
    c: Counter = Counter()
    for s in seeds:
        for y in g.edges.get(s, ()):
            c[y] += 1
    return c


def exp_hit(g_: int, e_: int, k: int) -> float:
    """Closed-form expectation of P(rank ≤ k), with random tie-breaking."""
    return min(1.0, max(0.0, (k - g_) / e_))


def rank_stats(scores: dict[str, float], truth: str) -> tuple[int, int] | None:
    if truth not in scores:
        return None
    p = scores[truth]
    return (sum(1 for v in scores.values() if v > p),
            sum(1 for v in scores.values() if v == p))


def _best_rank(cand: dict[str, float], targets: set[str], hits: dict,
               rankfrac: list[float], n_pool: int) -> None:
    """Convert the best-ranked member of targets into top-k hits and a rank fraction, accumulating into hits."""
    best = None
    for gp in targets:
        rs = rank_stats(cand, gp)
        if rs:
            r = rs[0] + (rs[1] + 1) / 2          # expected rank (random tie-breaking)
            best = r if best is None else min(best, r)
    if best is None:
        return
    for k in KS:
        hits[k].append(1.0 if best <= k else 0.0)
    rankfrac.append(best / max(1, n_pool))


def boot_ci(vals: list[float], b: int = 2000, seed: int = SEED) -> dict:
    if not vals:
        return {"note": "no samples"}
    rng = random.Random(seed)
    pts = []
    for _ in range(b):
        s = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        pts.append(statistics.mean(s))
    pts.sort()
    return {"mean": round(statistics.mean(vals), 4),
            "ci95": [round(pts[int(0.025 * b)], 4), round(pts[int(0.975 * b)], 4)],
            "n": len(vals)}


# ---------------------------------------------------------------- main flow

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3i_graph_probe.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    eps = load_episodes(Path(args.runs_root))
    print(f"loaded {len(eps)} episodes (graphs built)")

    # ---- A1 resolution rate ----
    a1 = {"n_steps": 0, "n_with_obj": 0, "resolved": 0, "dir_node": 0,
          "file_node": 0, "unresolved": 0}
    # ---- B ----
    b1_obs_d0, b1_obs_d1, b1_obs_le3 = [], [], []
    b1_null_d0, b1_null_d1, b1_null_le3 = [], [], []
    b1_obs_d1_excl0, b1_null_d1_excl0 = [], []
    b2_obs, b2_null = [], []
    # ---- C/D accumulators ----
    C1 = {k: [] for k in KS}
    C3 = {k: [] for k in KS}
    C1_cov, C3_cov = [], []
    C2 = {h: {"cov": [], "size": []} for h in (1, 2, 3)}
    D1 = {k: [] for k in KS}
    D1_uns = {k: [] for k in KS}
    D1_pre = {k: [] for k in KS}
    D1_rankfrac, D1_gold_in_graph, D1_gold_dist = [], [], []
    D1_uns_rankfrac, D1_pre_rankfrac, D1_uns_n = [], [], []
    # E paired on the same point set: four predictors compete on **exactly the same** first-visit points
    E_names = ("graph_1hop_current", "C1_multisource", "C3_degree", "marginal_repo")
    E = {nm: {k: [] for k in KS} for nm in E_names}
    E_cov = {nm: [] for nm in E_names}
    D2_obs, D2_null = [], []
    n_first_visit = 0

    rng = random.Random(SEED)

    for e in eps:
        g, seq, iid = e["graph"], e["seq"], e["id"]
        inst = insts.get(iid)
        nodes = [s["node"] for s in seq]
        # A1
        for s in seq:
            a1["n_steps"] += 1
            if s["obj"]:
                a1["n_with_obj"] += 1
            if s["node"]:
                a1["resolved"] += 1
                if s["node"].startswith("dir:"):
                    a1["dir_node"] += 1
                else:
                    a1["file_node"] += 1
            elif s["obj"]:
                a1["unresolved"] += 1

        # keep only the real file-node sequence usable for graph analysis
        fseq = [n for n in nodes if n and not n.startswith("dir:") and n in g.files]
        if len(fseq) >= 3:
            # ---- B1 adjacent-step graph distance (separating d=0) ----
            def dist_profile(s: list[str]) -> tuple[float, float, float, float]:
                d0 = d1 = le3 = tot = 0
                d1x = tot_x = 0                     # one-hop rate after excluding d=0
                for a, b in zip(s, s[1:]):
                    dd = g.graph_distance(a, b, max_hops=3)
                    tot += 1
                    if dd == 0:
                        d0 += 1
                    else:
                        tot_x += 1
                        if dd == 1:
                            d1x += 1
                    if dd == 1:
                        d1 += 1
                    if dd is not None and dd <= 3:
                        le3 += 1
                return (d0 / tot, d1 / tot, le3 / tot,
                        (d1x / tot_x if tot_x else float("nan")))

            o = dist_profile(fseq)
            b1_obs_d0.append(o[0]); b1_obs_d1.append(o[1]); b1_obs_le3.append(o[2])
            if not math.isnan(o[3]):
                b1_obs_d1_excl0.append(o[3])
            work = list(fseq)
            n0 = n1 = n3 = 0.0
            n1x, cnt_x = 0.0, 0
            NP = 60          # permutations per episode (graph distance is expensive)
            for _ in range(NP):
                rng.shuffle(work)
                v = dist_profile(work)
                n0 += v[0]; n1 += v[1]; n3 += v[2]
                if not math.isnan(v[3]):
                    n1x += v[3]; cnt_x += 1
            b1_null_d0.append(n0 / NP); b1_null_d1.append(n1 / NP)
            b1_null_le3.append(n3 / NP)
            if cnt_x:
                b1_null_d1_excl0.append(n1x / cnt_x)

            # ---- B2 same-community rate (excluding the same node) ----
            def comm_rate(s: list[str]) -> float | None:
                hit = tot = 0
                for a, b in zip(s, s[1:]):
                    if a == b:
                        continue
                    ca, cb = g.community.get(a), g.community.get(b)
                    if ca is None or cb is None:
                        continue
                    tot += 1
                    if ca == cb:
                        hit += 1
                return hit / tot if tot else None

            co = comm_rate(fseq)
            if co is not None:
                b2_obs.append(co)
                acc, cnt = 0.0, 0
                for _ in range(200):
                    rng.shuffle(work)
                    v = comm_rate(work)
                    if v is not None:
                        acc += v; cnt += 1
                if cnt:
                    b2_null.append(acc / cnt)

        # ---- C1/C3 stronger predictors (evaluated on first-visit steps only) ----
        seen: set[str] = set()
        for t in range(len(seq) - 1):
            if nodes[t]:
                seen.add(nodes[t])
            truth = nodes[t + 1]
            if not truth or truth in seen:
                continue                      # first-visit steps only
            n_first_visit += 1
            fseeds = {x for x in seen if not x.startswith("dir:") and x in g.files}
            if not fseeds:
                continue
            dist = multi_source_dist(g, fseeds)
            adj = seed_adjacency(g, fseeds)
            # C1 multi-source proximity: shortest hop count first, then "how many seeds it is adjacent to"
            cand = {x: (-dist[x] * 1000 + adj.get(x, 0))
                    for x in dist if x not in seen}
            C1_cov.append(1.0 if truth in cand else 0.0)
            rs = rank_stats(cand, truth)
            for k in KS:
                C1[k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)
            # C3 pure degree centrality (no trace)
            deg = {x: float(len(g.edges.get(x, ()))) for x in g.files if x not in seen}
            C3_cov.append(1.0 if truth in deg else 0.0)
            rs3 = rank_stats(deg, truth)
            for k in KS:
                C3[k].append(exp_hit(rs3[0], rs3[1], k) if rs3 else 0.0)
            # C2 k-hop coverage vs size
            for h in (1, 2, 3):
                reach = {x for x, d in dist.items() if 0 < d <= h and x not in seen}
                C2[h]["cov"].append(1.0 if truth in reach else 0.0)
                C2[h]["size"].append(len(reach))

        # ---- D1 can graph proximity predict gold ----
        # ⚠️ Leakage control: the agent success rate is 94%, so gold is usually **already
        # visited** (distance 0). Leaving visited files in the candidate set makes "hitting
        # gold" a mere restatement of the agent's own history.
        # Hence three tiers are reported side by side:
        #   full   : ranking over the whole repository (visited included) -- leaky, control only
        #   unseen : visited excluded -- really asks "can the graph point at gold not yet seen"
        #   prefix : only the first 30% of the trace as seeds, and that prefix's visited excluded -- strictest
        if inst and inst.gold:
            gold_in = {p for p in inst.gold if p in g.files}
            D1_gold_in_graph.append(len(gold_in) / len(inst.gold))
            fseeds = {x for x in seen if not x.startswith("dir:") and x in g.files}
            if gold_in and fseeds:
                dist = multi_source_dist(g, fseeds)
                adj = seed_adjacency(g, fseeds)
                cand_full = {x: (-dist.get(x, 99) * 1000 + adj.get(x, 0))
                             for x in g.files}
                _best_rank(cand_full, gold_in, D1, D1_rankfrac, len(g.files))
                # unseen: exclude visited from both candidates and gold
                gold_unseen = gold_in - fseeds
                cand_uns = {x: v for x, v in cand_full.items() if x not in fseeds}
                if gold_unseen:
                    _best_rank(cand_uns, gold_unseen, D1_uns, D1_uns_rankfrac,
                               len(cand_uns))
                    D1_uns_n.append(1.0)
                else:
                    D1_uns_n.append(0.0)
                for gp in gold_in:
                    d = dist.get(gp)
                    D1_gold_dist.append(d if d is not None else 99)
                # prefix: use only the visited set from the first 30% of steps
                cut = max(1, int(0.3 * len(seq)))
                pre = {n for n in nodes[:cut]
                       if n and not n.startswith("dir:") and n in g.files}
                gold_pre = gold_in - pre
                if pre and gold_pre:
                    dp = multi_source_dist(g, pre)
                    ap = seed_adjacency(g, pre)
                    cp = {x: (-dp.get(x, 99) * 1000 + ap.get(x, 0))
                          for x in g.files if x not in pre}
                    _best_rank(cp, gold_pre, D1_pre, D1_pre_rankfrac, len(cp))

        # ---- D2 is the visited set more clustered than a random set ----
        fs = {x for x in seen if not x.startswith("dir:") and x in g.files}
        if 3 <= len(fs) <= 60 and len(g.files) > len(fs) * 3:
            def clustering(S: set[str]) -> float:
                """Internal edge density of a set: internal edges / possible pairs."""
                S = set(S)
                inner = sum(1 for x in S for y in g.edges.get(x, ()) if y in S) / 2
                m = len(S) * (len(S) - 1) / 2
                return inner / m if m else 0.0
            D2_obs.append(clustering(fs))
            pool = list(g.files)
            acc = 0.0
            for _ in range(200):
                acc += clustering(set(rng.sample(pool, len(fs))))
            D2_null.append(acc / 200)

    # ------------------------------------------------ E paired head-to-head
    # V3g's marginal is fit with within-repo LOO, so we use the same LOO and
    # `len(group)>=6` filter here, guaranteeing E's point set is **pointwise identical** to
    # V3g's F regime; otherwise C1(n=381) and marginal(n=341) are not comparable.
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [x for j, x in enumerate(group) if j != hi]
            objfreq = Counter(s["node"] for x in train for s in x["seq"] if s["node"])
            g, seq = held["graph"], held["seq"]
            nodes = [s["node"] for s in seq]
            seen: set[str] = set()
            for t in range(len(seq) - 1):
                if nodes[t]:
                    seen.add(nodes[t])
                truth = nodes[t + 1]
                if not truth or truth in seen:
                    continue
                cur = nodes[t]
                fseeds = {x for x in seen if not x.startswith("dir:") and x in g.files}
                # 1) V3g usage: one hop from the current node + same community, visited excluded
                s1: dict[str, float] = {}
                if cur and not cur.startswith("dir:"):
                    L1 = g.neighbors(cur) - seen
                    L2 = (g.same_community(cur) - g.neighbors(cur)) - seen
                    for x in L1:
                        s1[x] = 2.0
                    for x in L2:
                        s1[x] = 1.0
                # 2) multi-source proximity
                s2: dict[str, float] = {}
                if fseeds:
                    dd = multi_source_dist(g, fseeds)
                    aa = seed_adjacency(g, fseeds)
                    s2 = {x: -dd[x] * 1000 + aa.get(x, 0)
                          for x in dd if x not in seen}
                # 3) degree centrality
                s3 = {x: float(len(g.edges.get(x, ()))) for x in g.files
                      if x not in seen}
                # 4) repository term frequency (LOO)
                s4 = {x: float(c) for x, c in objfreq.items() if x not in seen}
                for nm, sc in (("graph_1hop_current", s1), ("C1_multisource", s2),
                               ("C3_degree", s3), ("marginal_repo", s4)):
                    rs = rank_stats(sc, truth)
                    E_cov[nm].append(1.0 if rs else 0.0)
                    for k in KS:
                        E[nm][k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)

    def paired(a: str, b: str, k: int, bnum: int = 2000) -> dict:
        va, vb = E[a][k], E[b][k]
        if len(va) != len(vb) or not va:
            return {"note": "point sets do not match"}
        r = random.Random(SEED)
        pts = []
        for _ in range(bnum):
            idx = [r.randrange(len(va)) for _ in range(len(va))]
            pts.append(sum(va[i] - vb[i] for i in idx) / len(idx))
        pts.sort()
        lo, hi = pts[int(0.025 * bnum)], pts[int(0.975 * bnum)]
        return {"diff": round(statistics.mean(va) - statistics.mean(vb), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "significant": bool(lo > 0 or hi < 0)}

    # ---------------------------------------------------------- summary
    res: dict = {
        "n_episodes": len(eps),
        "A1_resolution": {
            **a1,
            "resolve_rate_of_steps_with_obj": round(
                a1["resolved"] / max(1, a1["n_with_obj"]), 4),
            "file_node_share_of_resolved": round(
                a1["file_node"] / max(1, a1["resolved"]), 4),
            "dir_node_share_of_resolved": round(
                a1["dir_node"] / max(1, a1["resolved"]), 4),
        },
        "B1_adjacent_step_distance": {
            "d0_selfrepeat": {"obs": boot_ci(b1_obs_d0), "null": boot_ci(b1_null_d0)},
            "d1_onehop_incl_d0_denominator": {
                "obs": boot_ci(b1_obs_d1), "null": boot_ci(b1_null_d1)},
            "d1_onehop_EXCLUDING_selfrepeat": {
                "obs": boot_ci(b1_obs_d1_excl0), "null": boot_ci(b1_null_d1_excl0)},
            "within_3hops": {"obs": boot_ci(b1_obs_le3), "null": boot_ci(b1_null_le3)},
        },
        "B2_same_community": {"obs": boot_ci(b2_obs), "null": boot_ci(b2_null)},
        "C1_multisource_proximity": {
            "coverage": boot_ci(C1_cov),
            "topk": {f"top{k}": boot_ci(C1[k]) for k in KS},
            "n_first_visit_points": len(C1_cov),
        },
        "C3_degree_centrality_baseline": {
            "coverage": boot_ci(C3_cov),
            "topk": {f"top{k}": boot_ci(C3[k]) for k in KS},
        },
        "C2_khop_tradeoff": {
            f"{h}hop": {"coverage": boot_ci(C2[h]["cov"]),
                        "median_candidate_set": (statistics.median(C2[h]["size"])
                                                 if C2[h]["size"] else None)}
            for h in (1, 2, 3)},
        "D1_gold_prediction": {
            "gold_in_graph_share": boot_ci(D1_gold_in_graph),
            "leakage_warning": "the full tier includes visited files and gold is usually "
                               "already visited ⇒ leaky, control only; draw conclusions "
                               "from the unseen and prefix tiers",
            "full_LEAKY": {
                "topk_any_gold": {f"top{k}": boot_ci(D1[k]) for k in KS},
                "best_rank_frac": boot_ci(D1_rankfrac)},
            "unseen": {
                "topk_any_gold": {f"top{k}": boot_ci(D1_uns[k]) for k in KS},
                "best_rank_frac": boot_ci(D1_uns_rankfrac),
                "share_episodes_with_unvisited_gold": boot_ci(D1_uns_n)},
            "prefix30pct": {
                "topk_any_gold": {f"top{k}": boot_ci(D1_pre[k]) for k in KS},
                "best_rank_frac": boot_ci(D1_pre_rankfrac)},
            "gold_hop_distance_from_visited": (
                dict(sorted(Counter(D1_gold_dist).items())) if D1_gold_dist else None),
        },
        "E_paired_headtohead_first_visit": {
            "n_points": len(E_cov["C1_multisource"]),
            "note": "pointwise identical to V3g's F regime (within-repo LOO, group>=6)",
            "coverage": {nm: boot_ci(E_cov[nm])["mean"] for nm in E_names},
            "topk": {nm: {f"top{k}": round(statistics.mean(E[nm][k]), 4)
                          for k in KS} for nm in E_names},
            "contrasts": {
                "C1_vs_graph1hop_top10": paired("C1_multisource",
                                                "graph_1hop_current", 10),
                "C1_vs_marginal_top10": paired("C1_multisource",
                                               "marginal_repo", 10),
                "C1_vs_marginal_top3": paired("C1_multisource",
                                              "marginal_repo", 3),
                "C1_vs_C3degree_top10": paired("C1_multisource", "C3_degree", 10),
                "C1_vs_marginal_top50": paired("C1_multisource",
                                               "marginal_repo", 50),
            },
        },
        "D2_visited_set_clustering": {
            "obs": boot_ci(D2_obs), "null_random_sets": boot_ci(D2_null),
            "lift": (round(statistics.mean(D2_obs) / statistics.mean(D2_null), 3)
                     if D2_obs and D2_null and statistics.mean(D2_null) > 0 else None),
        },
        "n_first_visit_total": n_first_visit,
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ---------------------------------------------------------- print
    a = res["A1_resolution"]
    print(f"\n=== A1 measurement validity: object→graph-node resolution ===")
    print(f"  steps with an object {a['n_with_obj']}/{a['n_steps']}, "
          f"resolution success rate {a['resolve_rate_of_steps_with_obj']:.1%}")
    print(f"  among resolved: real file nodes {a['file_node_share_of_resolved']:.1%}, "
          f"directory nodes {a['dir_node_share_of_resolved']:.1%}")

    print(f"\n=== B1 adjacent-step graph distance vs permutation null ===")
    for key, label in (("d0_selfrepeat", "distance 0 (same file)"),
                       ("d1_onehop_incl_d0_denominator", "distance 1 (all pairs as denominator)"),
                       ("d1_onehop_EXCLUDING_selfrepeat", "distance 1 (**excluding distance 0**)"),
                       ("within_3hops", "within 3 hops")):
        d = res["B1_adjacent_step_distance"][key]
        if "note" in d["obs"]:
            continue
        o, n = d["obs"], d["null"]
        lift = o["mean"] / n["mean"] if n["mean"] else float("nan")
        print(f"  {label:<38} observed {o['mean']:.4f} CI{o['ci95']}   "
              f"null {n['mean']:.4f} CI{n['ci95']}   lift {lift:.2f}")

    d = res["B2_same_community"]
    if "note" not in d["obs"]:
        lift = d["obs"]["mean"] / d["null"]["mean"] if d["null"]["mean"] else float("nan")
        print(f"\n=== B2 adjacent-step same-community rate (same node excluded) ===")
        print(f"  observed {d['obs']['mean']:.4f} CI{d['obs']['ci95']}   "
              f"null {d['null']['mean']:.4f} CI{d['null']['ci95']}   lift {lift:.2f}")

    c1, c3 = res["C1_multisource_proximity"], res["C3_degree_centrality_baseline"]
    print(f"\n=== C1/C3 stronger predictors (first-visit steps only, n={c1['n_first_visit_points']}) ===")
    print(f"{'':<32}" + "".join(f"{'top'+str(k):>9}" for k in KS))
    print(f"{'C1 multi-source (uses trace)':<32}" +
          "".join(f"{c1['topk']['top'+str(k)]['mean']:>9.4f}" for k in KS))
    print(f"{'C3 degree centrality (no trace)':<32}" +
          "".join(f"{c3['topk']['top'+str(k)]['mean']:>9.4f}" for k in KS))
    print(f"  coverage: C1 {c1['coverage']['mean']:.4f}  C3 {c3['coverage']['mean']:.4f}")

    print(f"\n=== C2 k-hop coverage vs candidate-set size tradeoff ===")
    for h in (1, 2, 3):
        d = res["C2_khop_tradeoff"][f"{h}hop"]
        print(f"  {h} hops: coverage {d['coverage']['mean']:.4f} "
              f"CI{d['coverage']['ci95']}   median candidate-set size {d['median_candidate_set']}")

    ee = res["E_paired_headtohead_first_visit"]
    print(f"\n=== E paired head-to-head on the same point set (first-visit steps, n={ee['n_points']}, matches V3g's F) ===")
    print(f"{'predictor':<22}{'coverage':>8}" + "".join(f"{'top'+str(k):>9}" for k in KS))
    for nm in E_names:
        print(f"{nm:<22}{ee['coverage'][nm]:>8.4f}" +
              "".join(f"{ee['topk'][nm]['top'+str(k)]:>9.4f}" for k in KS))
    print("  paired contrasts (significant only if the diff CI excludes 0):")
    for kk, vv in ee["contrasts"].items():
        if "note" in vv:
            print(f"    {kk:<28} {vv['note']}")
            continue
        print(f"    {kk:<28} {vv['diff']:+.4f}  "
              f"CI[{vv['ci95'][0]:+.4f}, {vv['ci95'][1]:+.4f}]  "
              f"{'significant' if vv['significant'] else 'n.s.'}")

    d1 = res["D1_gold_prediction"]
    print(f"\n=== D1 change of target: can graph proximity predict gold files ===")
    print(f"  share of gold inside the graph {d1['gold_in_graph_share']['mean']:.4f} "
          f"CI{d1['gold_in_graph_share']['ci95']}")
    print(f"  ⚠️ {d1['leakage_warning']}")
    print(f"  hop-distance distribution of gold from the visited set {d1['gold_hop_distance_from_visited']}"
          f"  ← distance 0 means \"already visited\"")
    for tag, label in (("full_LEAKY", "full (visited included, leaky)"),
                       ("unseen", "unseen (visited excluded)"),
                       ("prefix30pct", "prefix30% (first 30% of trace only)")):
        blk = d1[tag]
        tk = blk["topk_any_gold"]
        if "note" in tk[f"top{KS[0]}"]:
            print(f"  {label:<34} no samples")
            continue
        print(f"  {label:<34}" +
              "  ".join(f"top{k} {tk['top'+str(k)]['mean']:.3f}" for k in KS))
        rf = blk["best_rank_frac"]
        if "note" not in rf:
            print(f"  {'':<34}best gold rank fraction {rf['mean']:.4f} "
                  f"CI{rf['ci95']} (random=0.5, n={rf['n']})")

    d2 = res["D2_visited_set_clustering"]
    print(f"\n=== D2 graph clustering of the visited set ===")
    if "note" not in d2["obs"]:
        print(f"  observed internal edge density {d2['obs']['mean']:.5f} CI{d2['obs']['ci95']}")
        print(f"  random sets of the same size {d2['null_random_sets']['mean']:.5f} "
              f"CI{d2['null_random_sets']['ci95']}   lift {d2['lift']}")

    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
