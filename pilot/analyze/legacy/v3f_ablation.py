"""V3f: ablation of the three-layer structural prior -- rules out the "it is all historical prior" alternative explanation. Zero API.

Why this must come first
------------------------
V3b produced a positive result (object distribution top-3=0.513 above baseline, ECE=0.0202),
but candidate scoring has three layers:
    L1 one-hop neighbors (W=0.62) / L2 same community (W=0.30) / L3 visited history (W=0.08 + self-repeat 0.04)
If only-L3 (historical prior alone) reaches the same top-3, then **the code graph contributes
nothing incremental**, and V3b's positive conclusion must be restated as "the self-repeat prior
works" rather than "code structure works".

This is the alternative explanation most in need of ruling out, with higher priority than the other V3 sub-experiments.

Ablation arms
-------------
  full     L1+L2+L3          baseline
  no_L1    drop one-hop neighbors    are dependency-graph edges useful
  no_L2    drop same community       is Leiden clustering useful
  no_L3    drop historical prior     does self-repeat dominate
  only_L3  history only              **key arm**: does graph structure contribute anything incremental
  only_L1  one-hop neighbors only    the standalone power of graph edges
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.code_channel import W_L1, W_L2, W_L3               # noqa: E402
from pilot.analyze.v3_nextstep_dual import load_episodes               # noqa: E402

KS = (1, 2, 3, 5, 10)
ARMS = ("full", "no_L1", "no_L2", "no_L3", "only_L3", "only_L1")


def candidates(g, cur_node, seen, arm: str, topk_seen: int = 30) -> dict[str, float]:
    """Build candidates and weights according to the ablation arm."""
    use_l1 = arm in ("full", "no_L2", "no_L3", "only_L1")
    use_l2 = arm in ("full", "no_L1", "no_L3")
    use_l3 = arm in ("full", "no_L1", "no_L2", "only_L3")
    c: dict[str, float] = {}
    if cur_node and not cur_node.startswith("dir:"):
        L1 = g.neighbors(cur_node)
        L2 = g.same_community(cur_node) - L1
        if use_l1:
            for x in L1:
                c[x] = c.get(x, 0.0) + W_L1 / max(1, len(L1))
        if use_l2:
            for x in L2:
                c[x] = c.get(x, 0.0) + W_L2 / max(1, len(L2))
    if use_l3:
        hist = list(seen)[-topk_seen:]
        for x in hist:
            c[x] = c.get(x, 0.0) + W_L3 / max(1, len(hist))
        if cur_node:
            c[cur_node] = c.get(cur_node, 0.0) + W_L3 / 2
    return c


def ece(pairs: list[tuple[float, int]], n_bins: int = 10) -> float:
    if not pairs:
        return 0.0
    bins: list[list] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        bins[min(n_bins - 1, int(p * n_bins))].append((p, y))
    e, n = 0.0, len(pairs)
    for b in bins:
        if b:
            e += len(b) / n * abs(statistics.mean(x for x, _ in b)
                                  - statistics.mean(y for _, y in b))
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3f_ablation.json")
    args = ap.parse_args()

    eps = load_episodes(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    print(f"loaded {len(eps)} episodes")

    stat = {a: {"hits": {k: 0 for k in KS}, "tot": 0, "cov_num": 0, "cov_den": 0,
                "ranks": [], "calib": [], "eff": [], "nll": 0.0} for a in ARMS}

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            g, seq = held["graph"], held["seq"]
            seen: set[str] = set()
            for t in range(len(seq) - 1):
                cur, nxt = seq[t], seq[t + 1]
                if cur["node"]:
                    seen.add(cur["node"])
                if not nxt["node"]:
                    continue
                for arm in ARMS:
                    s = stat[arm]
                    cands = candidates(g, cur["node"], seen, arm)
                    s["cov_den"] += 1
                    if not cands or nxt["node"] not in cands:
                        continue
                    s["cov_num"] += 1
                    z = sum(cands.values())
                    p = {k: v / z for k, v in cands.items()}
                    ranked = sorted(p, key=lambda k: -p[k])
                    pos = ranked.index(nxt["node"]) + 1
                    s["tot"] += 1
                    s["ranks"].append(pos)
                    for k in KS:
                        if pos <= k:
                            s["hits"][k] += 1
                    s["nll"] += -math.log(max(p[nxt["node"]], 1e-12))
                    s["eff"].append(math.exp(-sum(v * math.log(v)
                                                  for v in p.values() if v > 0)))
                    s["calib"].append((p[nxt["node"]], 1))
                    for cc in ranked[:20]:
                        if cc != nxt["node"]:
                            s["calib"].append((p[cc], 0))

    res = {"arms": {}}
    for a in ARMS:
        s = stat[a]
        if not s["tot"]:
            res["arms"][a] = {"note": "no valid points"}
            continue
        res["arms"][a] = {
            "n": s["tot"],
            "coverage": round(s["cov_num"] / s["cov_den"], 4) if s["cov_den"] else None,
            "topk": {f"top{k}": round(s["hits"][k] / s["tot"], 4) for k in KS},
            "MRR": round(statistics.mean(1 / r for r in s["ranks"]), 4),
            "median_rank": statistics.median(s["ranks"]),
            "ECE": round(ece(s["calib"]), 4),
            "eff_candidates": round(statistics.median(s["eff"]), 2),
            "perplexity": round(math.exp(s["nll"] / s["tot"]), 2),
        }

    full = res["arms"].get("full", {})
    only3 = res["arms"].get("only_L3", {})
    if full.get("topk") and only3.get("topk"):
        d3 = full["topk"]["top3"] - only3["topk"]["top3"]
        res["key_contrast"] = {
            "full_top3": full["topk"]["top3"],
            "only_L3_top3": only3["topk"]["top3"],
            "graph_increment": round(d3, 4),
            "verdict": ("code graph contributes incrementally: full is significantly above only_L3"
                        if d3 > 0.03 else
                        "⚠️ code graph contributes nothing incremental: the positive result actually comes from the historical prior, "
                        "the V3b conclusion must be restated"),
        }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== V3f three-layer prior ablation ===")
    print(f"{'arm':<10}{'n':>6}{'cov':>8}{'top1':>8}{'top3':>8}{'top10':>8}"
          f"{'MRR':>8}{'ECE':>8}{'eff_cand':>9}")
    for a in ARMS:
        r = res["arms"][a]
        if "note" in r:
            print(f"{a:<10} {r['note']}")
            continue
        print(f"{a:<10}{r['n']:>6}{r['coverage']:>8}{r['topk']['top1']:>8}"
              f"{r['topk']['top3']:>8}{r['topk']['top10']:>8}"
              f"{r['MRR']:>8}{r['ECE']:>8}{r['eff_candidates']:>9}")
    if "key_contrast" in res:
        kc = res["key_contrast"]
        print(f"\nkey contrast full({kc['full_top3']}) - only_L3({kc['only_L3_top3']}) "
              f"= {kc['graph_increment']:+.4f}")
        print(f"verdict: {kc['verdict']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
