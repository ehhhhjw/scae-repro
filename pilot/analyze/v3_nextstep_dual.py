"""V3: next-step prediction redone -- dual-channel decomposition (tool × object). Zero API.

Why the first round of next-step failed
---------------------------------------
All three settings lost to repeat-prev (0.354/0.307/0.220 vs 0.371/0.294/0.236),
candidate coverage only 48%. Root cause: the candidate set was a **set of string paths**,
paths do not exist across repos, and "why one step follows another" was not modeled.

Changes in this round
---------------------
Dual-channel decomposition P(u_{t+1}) = P(tool_{t+1}|·) · P(obj_{t+1}|obj_t, G):
  tool component    the official 6-class conditional transition model of V1' (verified top-1 56.6%)
  object component  candidate set replaced by the **structural neighborhood of the code graph**:
                      L1 one-hop neighbors / L2 same community / L3 the rest
                    scored with structural priors (W_L1=0.62 etc., fixed from E3 measurements
                    of graph distance between adjacent steps)

Evaluation
----------
Within-repo leave-one-episode-out (the correct setting for next-step: the actual use is
predicting the next step inside the current repository, no cross-repository transfer needed).
Metrics top-1/top-3/MRR + candidate coverage.
Baselines: repeat-prev (which once beat us), marginal frequency, random.
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


def load_episodes(runs_root: Path, graph_only: bool = True) -> list[dict]:
    """Load (tool, obj) sequences. With graph_only=True keep only repositories that have a graph."""
    insts = {i.instance_id: i for i in T.load_instances()}
    eps = []
    for f in sorted(runs_root.glob("shard*/rollouts/*.jsonl")):
        iid = f.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None:
            continue
        g = load_graph(iid)
        if graph_only and not g.ok:
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
            node = g.resolve(p.path or p.obj) if g.ok else None
            seq.append({"tool": p.kind, "obj": p.obj, "node": node})
        eps.append({"id": iid, "project": inst.project, "seq": seq, "graph": g})
    return eps


def obj_candidates(g, cur_node: str | None, seen: set[str],
                   topk_seen: int = 30) -> dict[str, float]:
    """Object candidate set + structural prior weights.

    L1 one-hop neighbors / L2 same community / L3 already visited (history prior)
    L3 uses "already visited" rather than "whole repository", because whole-repository
    candidates dilute top-1 to the point of meaninglessness.
    """
    cands: dict[str, float] = {}
    if cur_node and not cur_node.startswith("dir:"):
        L1 = g.neighbors(cur_node)
        L2 = g.same_community(cur_node) - L1
        for x in L1:
            cands[x] = cands.get(x, 0.0) + W_L1 / max(1, len(L1))
        for x in L2:
            cands[x] = cands.get(x, 0.0) + W_L2 / max(1, len(L2))
    hist = list(seen)[-topk_seen:]
    for x in hist:
        cands[x] = cands.get(x, 0.0) + W_L3 / max(1, len(hist))
    if cur_node:
        cands[cur_node] = cands.get(cur_node, 0.0) + W_L3 / 2   # self-repeat prior
    return cands


def fit_tool_trans(eps: list[dict], alpha: float = 0.5) -> dict:
    tr: dict[str, Counter] = defaultdict(Counter)
    for e in eps:
        seq = e["seq"]
        for a, b in zip(seq, seq[1:]):
            tr[a["tool"]][b["tool"]] += 1
    return {"trans": tr, "alpha": alpha}


def p_tool(m: dict, cur: str, nxt: str, vocab: list[str]) -> float:
    a = m["alpha"]
    row = m["trans"].get(cur)
    if not row or sum(row.values()) == 0:
        return 1.0 / len(vocab)
    return (row.get(nxt, 0) + a) / (sum(row.values()) + a * len(vocab))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3_nextstep_dual.json")
    args = ap.parse_args()

    eps = load_episodes(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    tool_vocab = sorted({s["tool"] for e in eps for s in e["seq"]})
    print(f"loaded {len(eps)} episodes (repositories with a graph), "
          f"{sum(len(e['seq'])-1 for e in eps)} step transitions, {len(tool_vocab)} tool classes")

    tot = h1 = h3 = 0
    rr = 0.0
    cov_num = cov_den = 0
    base = {"repeat_prev": 0, "marginal": 0}
    tool_h1 = obj_h1 = tool_den = 0

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [e for j, e in enumerate(group) if j != hi]
            tm = fit_tool_trans(train)
            objfreq = Counter(s["node"] for e in train for s in e["seq"] if s["node"])
            g = held["graph"]
            seq = held["seq"]
            seen: set[str] = set()
            for t in range(len(seq) - 1):
                cur, nxt = seq[t], seq[t + 1]
                if cur["node"]:
                    seen.add(cur["node"])
                # --- tool component ---
                pt = {v: p_tool(tm, cur["tool"], v, tool_vocab) for v in tool_vocab}
                tool_den += 1
                if max(pt, key=pt.get) == nxt["tool"]:
                    tool_h1 += 1
                # --- object component ---
                if not nxt["node"]:
                    continue
                cands = obj_candidates(g, cur["node"], seen)
                cov_den += 1
                if nxt["node"] not in cands:
                    continue
                cov_num += 1
                # joint score: P(tool) * P(obj)
                joint = {}
                for node, w in cands.items():
                    joint[node] = w * pt.get(nxt["tool"], 1e-6)
                ranked = sorted(joint, key=lambda k: -joint[k])
                pos = ranked.index(nxt["node"]) + 1
                tot += 1
                if pos == 1:
                    h1 += 1
                    obj_h1 += 1
                if pos <= 3:
                    h3 += 1
                rr += 1.0 / pos
                if cur["node"] == nxt["node"]:
                    base["repeat_prev"] += 1
                if objfreq and max(cands, key=lambda c: objfreq.get(c, 0)) == nxt["node"]:
                    base["marginal"] += 1

    if not tot:
        print("no valid decision points")
        return 0
    res = {
        "setting": "within-repo LOO, dual-channel (tool x object-graph)",
        "n_episodes": len(eps), "n_decision_points": tot,
        "candidate_coverage": round(cov_num / cov_den, 4) if cov_den else None,
        "top1": round(h1 / tot, 4), "top3": round(h3 / tot, 4),
        "MRR": round(rr / tot, 4),
        "baselines_top1": {k: round(v / tot, 4) for k, v in base.items()},
        "tool_component_top1": round(tool_h1 / max(1, tool_den), 4),
        "tool_component_n": tool_den,
        "prior_result": {"note": "three settings of the first round", "coverage": [0.4815, 0.6074, 0.9774],
                         "top1": [0.354, 0.307, 0.220],
                         "repeat_prev": [0.371, 0.294, 0.236]},
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n=== V3 dual-channel next-step ({res['setting']}) ===")
    print(f"  decision points {tot}   candidate coverage {res['candidate_coverage']}")
    print(f"  top-1 {res['top1']}   top-3 {res['top3']}   MRR {res['MRR']}")
    print(f"  baseline top-1: {res['baselines_top1']}")
    print(f"  tool component alone top-1: {res['tool_component_top1']}")
    print(f"\n  vs first round: coverage 48.2/60.7/97.7% → {res['candidate_coverage']:.1%}")
    print(f"                  top-1 0.354/0.307/0.220 → {res['top1']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
