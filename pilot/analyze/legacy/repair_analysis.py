"""E1 repair experiment analysis -- primary criterion is Δmean recall, not binary succ.

Metric design correction
------------------------
The first 4 E1 cases exposed: succ (requires recall=1.0) is too strict on multi-file tasks.
django-13344's mean recall rising from 0.5(none) to 0.667(top1) is a substantive improvement,
but succ is 0 throughout and the signal is discarded entirely. This replays the earlier
outcome-definition mistake (strict set equality vs recall), so the primary criterion must be
a continuous quantity.

Three contrasts
---------------
  primary criterion     mean(top1_missed) − mean(none)
  position specificity  mean(top1_missed) − mean(rand_missed)
  content specificity   mean(top1_missed) − mean(top1_irrelev)
Plus two sham contrasts (rand/irrelev vs none), to judge "whether any injection works".

How to read this
----------------
If the three intervention arms are close to each other and all above none → the gain comes from
"injecting any content", attribution specificity does not hold.
If irrelev is markedly below the other arms → the content control arm is doing work, injections
are not equivalent (sphinx-7590 is such a case).

Statistical method (RFEval R4)
------------------------------
Paired permutation test (sign flipping, B=2e4), more reliable than bootstrap at small n and free
of distributional assumptions; the difference additionally gets a bootstrap CI for readability.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ARMS = ["none", "top1_missed", "rand_missed", "top1_irrelev"]


def paired_perm_test(a: list[float], b: list[float], B: int = 20000,
                     seed: int = 23) -> dict:
    """Paired permutation test: randomly flip the sign of each paired difference, test mean(a−b) != 0."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 3:
        return {"n": n, "note": "insufficient samples"}
    obs = statistics.mean(d)
    rng = np.random.default_rng(seed)
    arr = np.asarray(d, dtype=float)
    cnt = sum(1 for _ in range(B)
              if abs(float((arr * rng.choice([-1.0, 1.0], size=n)).mean()))
              >= abs(obs) - 1e-12)
    return {"n": n, "mean_diff": round(obs, 4),
            "p_two_sided": round((cnt + 1) / (B + 1), 4)}


def boot_ci(vals: list[float], B: int = 10000, seed: int = 29) -> list[float]:
    if not vals:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    bs = sorted(float(arr[rng.integers(0, len(arr), len(arr))].mean())
                for _ in range(B))
    return [round(bs[int(0.025 * B)], 4), round(bs[int(0.975 * B)], 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", default="pilot/runs/repair2/repair.jsonl")
    ap.add_argument("--out", default="pilot/runs/repair_analysis.json")
    args = ap.parse_args()

    p = Path(args.repair)
    if not p.exists():
        print("no repair results yet")
        return 0
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if not r.get("error")
            and all(f"mean_{a}" in r for a in ARMS)]
    if len(recs) < 3:
        print(f"only {len(recs)} complete four-arm records, waiting for more data")
        return 0

    means = {a: [r[f"mean_{a}"] for r in recs] for a in ARMS}
    succs = {a: [r.get(f"succ_{a}", 0.0) for r in recs] for a in ARMS}

    contrasts = {
        "main_top1_vs_none": ("top1_missed", "none"),
        "position_top1_vs_rand": ("top1_missed", "rand_missed"),
        "content_top1_vs_irrelev": ("top1_missed", "top1_irrelev"),
        "sham_rand_vs_none": ("rand_missed", "none"),
        "sham_irrelev_vs_none": ("top1_irrelev", "none"),
    }
    res = {
        "n_cases": len(recs),
        "arm_mean_recall": {a: round(statistics.mean(means[a]), 4) for a in ARMS},
        "arm_mean_recall_ci95": {a: boot_ci(means[a]) for a in ARMS},
        "arm_succ_rate": {a: round(statistics.mean(succs[a]), 4) for a in ARMS},
        "contrasts": {},
    }
    for name, (x, y) in contrasts.items():
        res["contrasts"][name] = {
            **paired_perm_test(means[x], means[y]),
            "diff_ci95": boot_ci([u - v for u, v in zip(means[x], means[y])]),
        }

    iv = {a: statistics.mean(means[a]) for a in
          ("top1_missed", "rand_missed", "top1_irrelev")}
    spread = round(max(iv.values()) - min(iv.values()), 4)
    base = statistics.mean(means["none"])
    res["intervention_arms_spread"] = spread
    res["fragile_failure_cases"] = sum(1 for v in means["none"] if v >= 0.999)
    if spread < 0.02 and iv["top1_missed"] > base + 0.02:
        diag = ("three intervention arms differ by <0.02 and all exceed none → the gain comes from "
                "\"injecting any content\", attribution specificity is not supported")
    elif iv["top1_irrelev"] < min(iv["top1_missed"], iv["rand_missed"]) - 0.05:
        diag = ("irrelevant-content arm markedly lower → content control is effective, injections "
                "are not equivalent; position specificity can be tested further")
    else:
        diag = "intervention arms differ; more samples needed to determine specificity"
    res["diagnosis"] = diag
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"=== E1 repair experiment (n={len(recs)}) ===")
    print(f"{'arm':<16}{'mean recall':>13}{'CI95':>20}{'succ':>8}")
    for a in ARMS:
        print(f"{a:<16}{res['arm_mean_recall'][a]:>13}"
              f"{str(res['arm_mean_recall_ci95'][a]):>20}"
              f"{res['arm_succ_rate'][a]:>8}")
    print("\ncontrasts (paired permutation test):")
    for name, c in res["contrasts"].items():
        if "note" in c:
            print(f"  {name:<26} {c['note']}")
            continue
        sig = " *" if c["p_two_sided"] < 0.05 else ""
        print(f"  {name:<26} Δ={c['mean_diff']:>8} "
              f"CI={str(c['diff_ci95']):<18} p={c['p_two_sided']}{sig}")
    print(f"\nintervention arm range {spread}")
    print(f"none arm all-success (fragile failure) {res['fragile_failure_cases']}/{len(recs)}")
    print(f"diagnosis: {diag}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
