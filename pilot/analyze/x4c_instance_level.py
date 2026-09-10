"""X4c: the **level-matched** re-check of X4, written to disk. Zero API.

Why a separate script
---------------------
X4b showed that the structure of U^exec is at the **instance level** (residualized ICC 0.640), not the
step level (no step position, no tool effect). Using **step-level** H_tool against it is therefore level
mismatched, so an instance-level test must be added.
That test was previously computed ad hoc and **never written to disk**, so the paper's rho=−0.219 / −0.425
could not be traced -- exactly the situation pitfall #12 is meant to avoid. This script pins it down.

Estimand
--------
Spearman after instance-level aggregation:
    mean/max H_tool  ~  mean/max U^exec_t
Control quantities (used to show the dependent variable does respond): instance recall, v0, step count ~ mean U^exec.
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

from pilot.channels.parsed_command import parse_step        # noqa: E402
from pilot.harness import rollout as R                      # noqa: E402
from pilot.analyze.v3k_trace_causal import _spearman_boot   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4prime", default="pilot/runs/v4prime/attribution.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x4c_instance_level.json")
    args = ap.parse_args()

    # global tool-transition model → H_tool(current tool)
    trans: dict[str, Counter] = defaultdict(Counter)
    step_tool: dict[str, list[str]] = {}
    for p in sorted(Path(args.runs_root).glob("shard*/rollouts/*.jsonl")):
        iid = p.stem.split("__k")[0]
        try:
            st = R.steps(R.load(p))
        except (OSError, ValueError):
            continue
        tools = [parse_step(s).kind for s in st]
        step_tool[iid] = tools
        for a, b in zip(tools, tools[1:]):
            trans[a][b] += 1
    vocab = sorted({t for ts in step_tool.values() for t in ts})

    def h_tool(cur: str, alpha: float = 0.5) -> float:
        row = trans.get(cur) or Counter()
        tot = sum(row.values()) + alpha * len(vocab)
        ps = [(row.get(v, 0) + alpha) / tot for v in vocab]
        return -sum(q * math.log(q) for q in ps if q > 0)

    recs = [json.loads(l) for l in open(args.v4prime) if l.strip()]
    rows = []
    for r in recs:
        tools = step_tool.get(r["instance_id"]) or []
        us = [statistics.variance([float(y) for y in s["ys_t"]])
              for s in (r.get("causal_segments") or [])
              if s.get("ys_t") and len(s["ys_t"]) > 1]
        if not us or not tools:
            continue
        rows.append({
            "instance_id": r["instance_id"],
            "u_mean": statistics.mean(us), "u_max": max(us),
            "h_mean": statistics.mean(h_tool(t) for t in tools),
            "h_max": max(h_tool(t) for t in tools),
            "n_steps": float(len(tools)),
            "v0": r.get("causal_v0"), "recall": r.get("recall"),
        })

    res: dict = {
        "n_instances": len(rows),
        "note": "X4b shows the structure of U^exec is at the instance level, so this is the level-matched test",
        "LEVEL_MATCHED": {}, "CONTROLS": {},
    }
    for hx in ("h_mean", "h_max"):
        for ux in ("u_mean", "u_max"):
            res["LEVEL_MATCHED"][f"{hx}_vs_{ux}"] = _spearman_boot(
                [x[hx] for x in rows], [x[ux] for x in rows])
    for k in ("n_steps", "v0", "recall"):
        sub = [x for x in rows if x[k] is not None]
        res["CONTROLS"][f"{k}_vs_u_mean"] = _spearman_boot(
            [float(x[k]) for x in sub], [x["u_mean"] for x in sub])
    res["per_instance"] = rows

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"instance-level samples n={len(rows)}")
    print("\n=== level matched: instance-level H_tool ~ instance-level U^exec ===")
    for k, v in res["LEVEL_MATCHED"].items():
        print(f"  {k:<18} rho={v['rho']:+.4f} CI{v['ci95']}  "
              f"{'significant' if v['significant'] else 'not significant'}")
    print("\n=== controls (showing the dependent variable responds) ===")
    for k, v in res["CONTROLS"].items():
        print(f"  {k:<18} rho={v['rho']:+.4f} CI{v['ci95']}  n={v['n']}  "
              f"{'significant' if v['significant'] else 'not significant'}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
