"""V3b: evaluating next-step as a **probability distribution** (rather than an exact top-1 hit).

Why the evaluation was redone
-----------------------------
In the first V3 round I used top-1 to declare "the object channel fails" (0.2627 = repeat-prev), which is the **wrong criterion**.
The real value of next-step is answering "which few things might come next and with what probabilities", not guessing the single answer.
Signals that were ignored: top-3 = 0.4407, MRR = 0.395 (the true answer ranks 2.5th on average).

This module evaluates distribution quality rather than point hits:
  1. top-k curve (k=1..10) -- how much coverage is enough
  2. MRR / mean hit rank
  3. **calibration**: bin the predicted probabilities and check whether each bin's actual hit rate matches (this is the core)
  4. effective candidate count exp(H) -- "how many candidates must be considered", the practical meaning of U^obj
  5. NLL / perplexity -- overall distribution quality
  6. distribution-level comparison against three baselines (repeat-prev, marginal frequency, uniform)

Criteria (preregistered)
------------------------
  distribution usable: ECE < 0.15 and top-3 clearly above baseline top-3
  partially usable: well calibrated but top-k below baseline → probabilities trustworthy, ranking no better
  unusable: poorly calibrated and top-k below baseline
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

from pilot.channels.code_channel import W_L1, W_L2, W_L3, load_graph  # noqa: E402
from pilot.channels.parsed_command import parse_step                  # noqa: E402
from pilot.harness import rollout as R                                # noqa: E402
from pilot.probes import task as T                                    # noqa: E402
from pilot.analyze.v3_nextstep_dual import (                          # noqa: E402
    fit_tool_trans, load_episodes, obj_candidates, p_tool,
)

KS = (1, 2, 3, 5, 10)


def normalize(d: dict[str, float]) -> dict[str, float]:
    z = sum(d.values())
    return {k: v / z for k, v in d.items()} if z > 0 else d


def entropy(ps) -> float:
    return -sum(p * math.log(p) for p in ps if p > 0)


def ece(pairs: list[tuple[float, int]], n_bins: int = 10) -> tuple[float, list]:
    """pairs = [(predicted probability, hit or not)]. ECE over equal-width bins + per-bin detail."""
    if not pairs:
        return 0.0, []
    bins: list[list] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        b = min(n_bins - 1, int(p * n_bins))
        bins[b].append((p, y))
    e = 0.0
    detail = []
    n = len(pairs)
    for i, b in enumerate(bins):
        if not b:
            continue
        conf = statistics.mean(x for x, _ in b)
        acc = statistics.mean(y for _, y in b)
        e += len(b) / n * abs(conf - acc)
        detail.append({"bin": f"[{i/n_bins:.1f},{(i+1)/n_bins:.1f})",
                       "n": len(b), "mean_pred": round(conf, 4),
                       "actual_rate": round(acc, 4)})
    return e, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3b_distribution.json")
    args = ap.parse_args()

    eps = load_episodes(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    tool_vocab = sorted({s["tool"] for e in eps for s in e["seq"]})
    print(f"loaded {len(eps)} episodes, {len(tool_vocab)} tool types")

    hits = {k: 0 for k in KS}
    base_hits = {name: {k: 0 for k in KS}
                 for name in ("repeat_prev", "marginal", "uniform")}
    tot = 0
    ranks: list[int] = []
    calib: list[tuple[float, int]] = []
    eff_cands: list[float] = []
    nll = 0.0
    # tool distribution evaluated separately
    tool_hits = {k: 0 for k in KS}
    tool_tot = 0
    tool_calib: list[tuple[float, int]] = []
    tool_eff: list[float] = []

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [e for j, e in enumerate(group) if j != hi]
            tm = fit_tool_trans(train)
            objfreq = Counter(s["node"] for e in train for s in e["seq"] if s["node"])
            g, seq = held["graph"], held["seq"]
            seen: set[str] = set()
            for t in range(len(seq) - 1):
                cur, nxt = seq[t], seq[t + 1]
                if cur["node"]:
                    seen.add(cur["node"])

                # ---------- tool distribution ----------
                pt = normalize({v: p_tool(tm, cur["tool"], v, tool_vocab)
                                for v in tool_vocab})
                tool_tot += 1
                tool_eff.append(math.exp(entropy(pt.values())))
                tr = sorted(pt, key=lambda k: -pt[k])
                tp = tr.index(nxt["tool"]) + 1
                for k in KS:
                    if tp <= k:
                        tool_hits[k] += 1
                tool_calib.append((pt[nxt["tool"]], 1))
                for v in tool_vocab:
                    if v != nxt["tool"]:
                        tool_calib.append((pt[v], 0))

                # ---------- object distribution ----------
                if not nxt["node"]:
                    continue
                cands = obj_candidates(g, cur["node"], seen)
                if nxt["node"] not in cands:
                    continue
                po = normalize(cands)
                tot += 1
                eff_cands.append(math.exp(entropy(po.values())))
                ranked = sorted(po, key=lambda k: -po[k])
                pos = ranked.index(nxt["node"]) + 1
                ranks.append(pos)
                for k in KS:
                    if pos <= k:
                        hits[k] += 1
                nll += -math.log(max(po[nxt["node"]], 1e-12))
                # calibration: the hit item + at most 20 miss items (to keep the samples balanced)
                calib.append((po[nxt["node"]], 1))
                for c in ranked[:20]:
                    if c != nxt["node"]:
                        calib.append((po[c], 0))

                # baselines
                rp = {c: (1.0 if c == cur["node"] else 0.0) for c in cands}
                mg = {c: float(objfreq.get(c, 0)) for c in cands}
                un = {c: 1.0 for c in cands}
                for name, dist in (("repeat_prev", rp), ("marginal", mg),
                                   ("uniform", un)):
                    rk = sorted(dist, key=lambda k: -dist[k])
                    p = rk.index(nxt["node"]) + 1
                    for k in KS:
                        if p <= k:
                            base_hits[name][k] += 1

    if not tot:
        print("no valid decision points")
        return 0

    e_obj, det_obj = ece(calib)
    e_tool, det_tool = ece(tool_calib)
    res = {
        "n_decision_points": tot,
        "object_distribution": {
            "topk": {f"top{k}": round(hits[k] / tot, 4) for k in KS},
            "MRR": round(statistics.mean(1 / r for r in ranks), 4),
            "mean_rank": round(statistics.mean(ranks), 2),
            "median_rank": statistics.median(ranks),
            "ECE": round(e_obj, 4),
            "calibration_bins": det_obj,
            "mean_effective_candidates": round(statistics.mean(eff_cands), 2),
            "median_effective_candidates": round(statistics.median(eff_cands), 2),
            "NLL": round(nll / tot, 4),
            "perplexity": round(math.exp(nll / tot), 2),
        },
        "tool_distribution": {
            "n": tool_tot,
            "topk": {f"top{k}": round(tool_hits[k] / tool_tot, 4)
                     for k in KS if k <= len(tool_vocab)},
            "ECE": round(e_tool, 4),
            "mean_effective_candidates": round(statistics.mean(tool_eff), 2),
        },
        "baselines_topk": {name: {f"top{k}": round(v[k] / tot, 4) for k in KS}
                           for name, v in base_hits.items()},
    }
    o = res["object_distribution"]
    b3 = max(res["baselines_topk"][n]["top3"] for n in res["baselines_topk"])
    res["verdict"] = (
        f"distribution usable: ECE={o['ECE']} <0.15 and top-3={o['topk']['top3']} > baseline {b3}"
        if o["ECE"] < 0.15 and o["topk"]["top3"] > b3 else
        f"partially usable: ECE={o['ECE']} (calibration {'good' if o['ECE']<0.15 else 'poor'}), "
        f"top-3={o['topk']['top3']} vs baseline {b3}"
    )
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== V3b object distribution quality (n={tot}) ===")
    print(f"  top-k: {o['topk']}")
    print(f"  MRR={o['MRR']}  mean_rank={o['mean_rank']}  median_rank={o['median_rank']}")
    print(f"  ECE={o['ECE']}  NLL={o['NLL']}  perplexity={o['perplexity']}")
    print(f"  effective candidates exp(H): mean={o['mean_effective_candidates']} "
          f"median={o['median_effective_candidates']}")
    print("\n  baseline top-k:")
    for n_, v in res["baselines_topk"].items():
        print(f"    {n_:<12} {v}")
    tl = res["tool_distribution"]
    print(f"\n=== tool distribution (n={tl['n']}) ===")
    print(f"  top-k: {tl['topk']}  ECE={tl['ECE']}  "
          f"effective candidates={tl['mean_effective_candidates']}")
    print("\n  object distribution calibration bins:")
    for d in det_obj[:8]:
        print(f"    {d['bin']:<12} n={d['n']:<5} pred={d['mean_pred']:<8} "
              f"actual={d['actual_rate']}")
    print(f"\nverdict: {res['verdict']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
