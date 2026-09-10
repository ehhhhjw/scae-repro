"""R1+R2: three-state accounting + manipulation check. Zero API, only parses existing rollouts.

Pain point
----------
The status quo uses a single boolean `degenerate` (Δ all 0), conflating two semantically distinct cases:
  (a) the intervention **cannot be applied / cannot be estimated** → should leave the denominator
  (b) the intervention **is applied successfully but has no causal effect** → should enter the
      denominator and be recorded as zero effect
which makes `NNondegen=54/70` uninterpretable.

RFEval's contrast precondition delta + contrast coverage points at the correct accounting:
delta=0 leaves the denominator and coverage is reported alongside; kappa=0 enters the denominator
as a failure. This module implements the equivalent.

The first E1 case empirically supports this distinction: django-12155 gives recall=0.5 with zero
variance across all m=4 runs of each of the four arms, while the injected observation was verified
to be real file content (4884 characters). That is, delta!=0 but kappa=0 -- the failure is not
"never saw the right file" but the stop/report policy deciding not to report it.

Three states
------------
  INESTIMABLE  no valid intervention can be applied / MC1 not passed → leaves the denominator, counted in coverage
  NO_EFFECT    intervention applicable and MC1 passed, but Δ_t = 0 → enters the denominator, zero effect
  EFFECT       intervention applicable and Δ_t != 0 → enters the denominator
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                      # noqa: E402
from pilot.probes import task as T                          # noqa: E402

_DEFS = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]{3,})")
_IDENTS = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{5,})\b")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval -- the correct interval for small-sample proportions (R4)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 4), round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def trackable_tokens(step) -> set[str]:
    """Tokens in this step's observation usable for MC1 tracking.

    Prefer def/class definition names; if there are none (e.g. sed only captured the import
    block), fall back to long identifiers (>=6 characters), which are still discriminative.
    """
    out = step.output_text or ""
    defs = set(_DEFS.findall(out))
    if defs:
        return defs
    idents = Counter(_IDENTS.findall(out))
    return {w for w, c in idents.most_common(12) if c <= 3}


def mc1(steps, t: int) -> dict:
    """MC1: whether the content of step t's observation is read/referenced by later steps. t is 1-based."""
    if t < 1 or t > len(steps):
        return {"ok": None, "why": "t out of range"}
    toks = trackable_tokens(steps[t - 1])
    if not toks:
        return {"ok": None, "why": "no trackable token in this step's observation"}
    later = " ".join((s.args_raw or "") + " " + (s.output_text or "")
                     for s in steps[t:])
    hit = sum(1 for x in toks if x in later)
    return {"ok": hit > 0, "n_tokens": len(toks), "n_hit": hit, "why": ""}


def classify(rec: dict, steps) -> list[dict]:
    out = []
    for s in rec.get("causal_segments", []):
        t, d = s.get("t"), s.get("delta")
        if t is None or d is None:
            out.append({"t": t, "state": "INESTIMABLE", "why": "missing t/delta"})
            continue
        c = mc1(steps, t)
        if c["ok"] is None:
            out.append({"t": t, "state": "INESTIMABLE", "delta": d,
                        "why": "MC1 undecidable: " + c["why"]})
        elif not c["ok"]:
            out.append({"t": t, "state": "INESTIMABLE", "delta": d,
                        "n_tokens": c["n_tokens"],
                        "why": "MC1 not passed (early-assimilation failure)"})
        else:
            out.append({"t": t,
                        "state": "NO_EFFECT" if abs(d) < 1e-9 else "EFFECT",
                        "delta": d, "n_tokens": c["n_tokens"],
                        "n_hit": c["n_hit"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="pilot/runs/full_attr/attribution.jsonl")
    ap.add_argument("--locate", default="pilot/runs/full_locate/results.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/estimability.json")
    args = ap.parse_args()

    loc = {}
    for line in Path(args.locate).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            loc[r["instance_id"]] = r
    attr = [json.loads(l) for l in Path(args.attr).read_text().splitlines() if l.strip()]
    attr = [r for r in attr if not r.get("error")]

    states = Counter()
    by_outcome = {1: Counter(), 0: Counter()}
    early_fail = undecidable = 0
    rows = []
    for r in attr:
        lrec = loc.get(r["instance_id"])
        if not lrec or not lrec.get("rollout"):
            continue
        cands = list(Path(args.runs_root).glob(f"shard*/{lrec['rollout']}"))
        if not cands:
            continue
        try:
            steps = R.steps(R.load(cands[0]))
        except (OSError, ValueError):
            continue
        segs = classify(r, steps)
        y = r.get("Y")
        for s in segs:
            states[s["state"]] += 1
            if y in by_outcome:
                by_outcome[y][s["state"]] += 1
            why = s.get("why", "")
            if "early-assimilation" in why:
                early_fail += 1
            elif "undecidable" in why:
                undecidable += 1
        rows.append({"instance_id": r["instance_id"], "Y": y, "segments": segs})

    tot = sum(states.values())
    est = states["NO_EFFECT"] + states["EFFECT"]
    p, lo, hi = wilson(est, tot)
    pe, ple, phe = wilson(states["EFFECT"], est) if est else (0, 0, 0)

    res = {
        "n_episodes": len(rows), "n_segments": tot,
        "state_counts": dict(states),
        "estimability_coverage": {"point": p, "wilson_ci95": [lo, hi],
                                  "n_estimable": est, "n_total": tot},
        "effect_rate_among_estimable": {"point": pe, "wilson_ci95": [ple, phe],
                                        "n_effect": states["EFFECT"],
                                        "n_estimable": est},
        "inestimable_breakdown": {"early_assimilation_failure": early_fail,
                                  "mc1_undecidable": undecidable},
        "by_outcome": {("succeeded" if k == 1 else "failed"): dict(v)
                       for k, v in by_outcome.items()},
        "note": ("RFEval-style accounting: INESTIMABLE leaves the denominator and coverage is reported; "
                 "NO_EFFECT enters the denominator as zero effect. The old degenerate field conflated both."),
        "episodes": rows,
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"=== three-state accounting ({len(rows)} episodes, {tot} segments) ===")
    for k in ("EFFECT", "NO_EFFECT", "INESTIMABLE"):
        v = states[k]
        print(f"  {k:<12} {v:>4}  {v/tot:.1%}" if tot else f"  {k}: 0")
    print(f"\nestimable coverage {p:.1%}  Wilson CI {[lo, hi]}  ({est}/{tot})")
    print(f"effect rate among estimable segments {pe:.1%}  Wilson CI {[ple, phe]}")
    print(f"INESTIMABLE breakdown: early-assimilation {early_fail}, undecidable {undecidable}")
    print("\nstratified by outcome:")
    for k, v in res["by_outcome"].items():
        print(f"  {k:<10} {dict(v)}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
