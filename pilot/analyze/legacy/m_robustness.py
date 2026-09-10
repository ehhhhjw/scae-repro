"""R6: does the choice of m need defending -- robustness of the Δ_t ranking between m=3 and m=8.

Motivation (borrowing RFEval's practice of defending hyperparameters)
---------------------------------------------------------------------
The main experiment uses m=3 as a cost compromise, but the paper only claims
**ranking-type conclusions** (which step is most critical, whether the judge and causal
attribution point at the same step). If the Δ_t rankings from m=3 and m=8 are highly
correlated, then a small m suffices to support ranking-type conclusions, and the "cost
compromise" becomes "a defensible choice".

Method
------
On the same batch of cases, compare m=3 (full_attr) against m=8 (m8) on:
  1. Spearman rank correlation of the Δ_t sequences (computed per case, then report the distribution)
  2. whether the top-1 step agrees (this is the quantity the paper actually relies on)
  3. whether |Δ| shrinks as m grows (the expected signature of reduced Monte Carlo noise)

Criterion
---------
  high top-1 agreement AND median Spearman > 0.5 -> m=3 suffices to support ranking conclusions
  otherwise -> m must be raised, or the conclusions downgraded to exploratory
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def load(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        out[r["instance_id"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", default="pilot/runs/full_attr/attribution.jsonl")
    ap.add_argument("--m8-glob", default="pilot/runs/m8/shard*/attribution.jsonl")
    ap.add_argument("--out", default="pilot/runs/m_robustness.json")
    args = ap.parse_args()

    a3 = load(Path(args.m3))
    a8: dict[str, dict] = {}
    for p in Path().glob(args.m8_glob):
        a8.update(load(p))
    common = sorted(set(a3) & set(a8))
    print(f"m=3: {len(a3)} cases, m=8: {len(a8)} cases, overlap {len(common)}")
    if not common:
        print("no overlapping cases, cannot run R6")
        return 0

    rows = []
    rhos, top1_same, absr = [], [], []
    for iid in common:
        s3 = {s["t"]: s["delta"] for s in a3[iid].get("causal_segments", [])
              if s.get("delta") is not None}
        s8 = {s["t"]: s["delta"] for s in a8[iid].get("causal_segments", [])
              if s.get("delta") is not None}
        ts = sorted(set(s3) & set(s8))
        rho = spearman([s3[t] for t in ts], [s8[t] for t in ts]) if len(ts) >= 3 else None
        t1_3, t1_8 = a3[iid].get("causal_top1_step"), a8[iid].get("causal_top1_step")
        same = (t1_3 == t1_8) if (t1_3 and t1_8) else None
        m3abs = statistics.mean([abs(s3[t]) for t in ts]) if ts else None
        m8abs = statistics.mean([abs(s8[t]) for t in ts]) if ts else None
        rows.append({"instance_id": iid, "Y": a3[iid].get("Y"),
                     "n_shared_points": len(ts), "spearman": rho,
                     "top1_m3": t1_3, "top1_m8": t1_8, "top1_same": same,
                     "mean_abs_delta_m3": (round(m3abs, 4) if m3abs is not None else None),
                     "mean_abs_delta_m8": (round(m8abs, 4) if m8abs is not None else None)})
        if rho is not None:
            rhos.append(rho)
        if same is not None:
            top1_same.append(1.0 if same else 0.0)
        if m3abs is not None and m8abs is not None:
            absr.append((m3abs, m8abs))

    res = {
        "n_common": len(common),
        "n_with_spearman": len(rhos),
        "spearman_median": (round(statistics.median(rhos), 4) if rhos else None),
        "spearman_mean": (round(statistics.mean(rhos), 4) if rhos else None),
        "spearman_all": [round(r, 3) for r in rhos],
        "top1_agreement": (round(statistics.mean(top1_same), 4) if top1_same else None),
        "n_top1_compared": len(top1_same),
        "mean_abs_delta": {
            "m3": (round(statistics.mean([a for a, _ in absr]), 4) if absr else None),
            "m8": (round(statistics.mean([b for _, b in absr]), 4) if absr else None),
        },
        "rows": rows,
    }
    med = res["spearman_median"]
    agr = res["top1_agreement"]
    res["verdict"] = (
        "insufficient data" if med is None or agr is None else
        f"top-1 agreement {agr:.0%}, median Spearman {med:.2f} → "
        + ("m=3 suffices to support ranking-type conclusions; the choice is defensible"
           if agr >= 0.5 and med >= 0.5 else
           "the m=3 ranking is not robust; ranking-type conclusions must be downgraded to exploratory or m raised")
    )
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== R6 m robustness (overlap {len(common)} cases) ===")
    print(f"  median Spearman {res['spearman_median']}  mean {res['spearman_mean']}"
          f"  (n={len(rhos)})")
    print(f"  top-1 agreement {res['top1_agreement']}  (n={len(top1_same)})")
    print(f"  mean |Δ| m3={res['mean_abs_delta']['m3']}  m8={res['mean_abs_delta']['m8']}")
    print("\nper case:")
    for r in rows:
        print(f"  {r['instance_id'][:28]:<28} Y={r['Y']} pts={r['n_shared_points']} "
              f"rho={r['spearman']} top1: {r['top1_m3']}->{r['top1_m8']} "
              f"{'same' if r['top1_same'] else 'diff' if r['top1_same'] is not None else '-'}")
    print(f"\nverdict: {res['verdict']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
