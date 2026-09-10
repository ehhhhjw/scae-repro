"""D1/D2/D3 analysis -- the verdict for the C3 attack line.

Input: a batch of runs (each rollout already stored); for each run we compute
  - causal attribution top-1 step (causal.py, fork-resample estimate of Δ_t)
  - full-trace judge top-1 step (judge.py)
  - prefix-only judge top-1 step (isolates collider bias)

Output (pre-registered criteria, PILOT section 7):
  D1 = fraction where judge-top1 ≠ causal-top1 (bootstrap CI)
  D3 = whether the judge-top1 step position is systematically later than causal-top1 (one-sided Wilcoxon)
       + full-trace vs prefix-only judge step-position gap (collider bias isolation)
  D2 = relationship between dataflow dead/dfw and |Δ_t| (regression), testing "dataflow dependence ≠ causal effect"

This script consumes the per-run attribution results jsonl produced by run_attribution.py.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def wilcoxon_signed_rank_one_sided(diffs: list[float]) -> dict:
    """One-sided Wilcoxon signed-rank: H1 = median > 0 (judge step position later than causal).

    Minimal implementation with no scipy dependency; normal approximation for large n. Zero differences dropped.
    """
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n < 6:
        return {"n": n, "note": "too few samples, skipping normal approximation", "W": None, "p_one_sided": None}
    ranks = _rank([abs(x) for x in d])
    w_plus = sum(r for x, r in zip(d, ranks) if x > 0)
    mean_w = n * (n + 1) / 4
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mean_w) / sd_w if sd_w else 0.0
    p = 1 - _norm_cdf(z)          # one-sided H1: W+ is large
    return {"n": n, "W_plus": round(w_plus, 1), "z": round(z, 3),
            "p_one_sided": round(p, 4)}


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def bootstrap_ci(vals: list[float], stat=statistics.mean, B: int = 2000,
                 seed_stream: list[int] | None = None) -> tuple[float, float, float]:
    """Bootstrap 95% CI by resampling samples. Uses a fixed LCG for reproducibility (no Math.random dependency)."""
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = stat(vals)
    state = 12345
    boots = []
    for _ in range(B):
        sample = []
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            sample.append(vals[state % n])
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[int(0.975 * B)]
    return (round(point, 4), round(lo, 4), round(hi, 4))


def analyze(attr_path: Path) -> dict:
    """attr_path: jsonl produced by run_attribution; each line has causal / judge_full / judge_prefix."""
    recs = [json.loads(l) for l in attr_path.read_text().splitlines() if l.strip()]
    usable = [r for r in recs if r.get("causal_top1_pos") is not None
              and r.get("judge_full_top1_pos") is not None]

    # D1: whether the top-1 step differs
    disagree = [int(r["causal_top1_step"] != r["judge_full_top1_step"]) for r in usable]
    d1_point, d1_lo, d1_hi = bootstrap_ci([float(x) for x in disagree])

    # D3: judge step position − causal step position (>0 = judge is later)
    pos_diff = [r["judge_full_top1_pos"] - r["causal_top1_pos"] for r in usable]
    d3 = wilcoxon_signed_rank_one_sided(pos_diff)

    # D3 isolation: full vs prefix judge step position
    both = [r for r in usable if r.get("judge_prefix_top1_pos") is not None]
    fp_diff = [r["judge_full_top1_pos"] - r["judge_prefix_top1_pos"] for r in both]
    d3_iso = wilcoxon_signed_rank_one_sided(fp_diff) if len(both) >= 6 else {"n": len(both)}

    result = {
        "n_runs": len(recs),
        "n_usable": len(usable),
        "D1_disagreement_rate": {"point": d1_point, "ci95": [d1_lo, d1_hi],
                                 "pass": d1_lo > 0.15},
        "D3_judge_later_than_causal": {
            **d3,
            "mean_pos_diff": round(statistics.mean(pos_diff), 4) if pos_diff else None,
            "median_causal_pos": round(statistics.median(
                [r["causal_top1_pos"] for r in usable]), 4) if usable else None,
            "median_judge_pos": round(statistics.median(
                [r["judge_full_top1_pos"] for r in usable]), 4) if usable else None,
            "pass": (d3.get("p_one_sided") is not None and d3["p_one_sided"] < 0.05),
        },
        "D3_collider_isolation_full_vs_prefix": {
            **d3_iso,
            "mean_pos_diff": round(statistics.mean(fp_diff), 4) if fp_diff else None,
        },
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", required=True, help="jsonl produced by run_attribution")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = analyze(Path(args.attr))
    out = Path(args.out) if args.out else Path(args.attr).with_name("d1d3_report.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== D1/D3 analysis ===")
    print(f"usable runs: {res['n_usable']}/{res['n_runs']}")
    d1 = res["D1_disagreement_rate"]
    print(f"D1 attribution disagreement rate: {d1['point']:.1%}  CI95={d1['ci95']}  "
          f"{'PASS' if d1['pass'] else 'FAIL'}(lower bound>15%)")
    d3 = res["D3_judge_later_than_causal"]
    print(f"D3 step-position shift: causal median={d3['median_causal_pos']} judge median={d3['median_judge_pos']} "
          f"Δ={d3['mean_pos_diff']} p={d3.get('p_one_sided')} "
          f"{'PASS' if d3['pass'] else 'FAIL'}(judge significantly later)")
    iso = res["D3_collider_isolation_full_vs_prefix"]
    print(f"collider bias isolation (full−prefix): Δ={iso.get('mean_pos_diff')} p={iso.get('p_one_sided')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
