"""X7 (zero API): fragile success ↔ L1 predictive entropy. Supplements RESEARCH_PLAN_v5 §G8.

Proposition
-----------
Fragile-success instances (0 < causal_v0 < 1, i.e. replay is unstable) have
significantly higher mean predictive entropy H_tool and H_prov over their traces
than robust-success instances (v0 = 1).

Estimands
---------
1. Fragile vs robust success, paired contrast on mean H_tool (instance level)
2. Fragile success vs total failure (v0 = 0), mean H_tool contrast
3. H_tool mean trend across the three groups (fail / fragile / robust)
4. Spearman correlation causal_v0 ~ mean H_tool (continuous version)
5. Precision and recall of "high entropy → fragile/fail"

Design
------
- Fragile: 0 < causal_v0 < 1 (replay variance > 0)
- Robust success: causal_v0 = 1.0 (every replay succeeds)
- Failure: causal_v0 = 0.0 or close to it (replay usually fails)
- Data source: the 70 multi-file instances in full_attr

RESEARCH_PLAN_v5 criteria
-------------------------
| Result | Path forward |
|---|---|
| Fragile H_tool significantly above robust, trend monotonic | Practical: H_tool works as an early "needs replay" signal |
| Trend present but not significant | Exploratory: fold into X4's joint H_t ~ U^exec analysis |
| No trend | Report as is; note L1 quantities are decoupled from execution instability |

Statistical discipline
----------------------
- instance-level bootstrap B=2000
- three-group comparison BH-FDR q=0.10 (3 contrasts)
- Spearman + bootstrap CI
- Pitfall #22: first confirm H_tool has detectable structure along some dimension
  (position / tool stratification) before concluding "unrelated to v0"
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.channels.parsed_command import parse_step   # noqa: E402
from pilot.harness import rollout as R                  # noqa: E402
from pilot.probes import task as T                      # noqa: E402

SEED = 20260814


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


def paired_ci(a: list[float], b_: list[float], bnum: int = 2000) -> dict:
    if len(a) != len(b_) or not a:
        return {"note": f"point sets do not match {len(a)} vs {len(b_)}"}
    rng = random.Random(SEED)
    pts = []
    for _ in range(bnum):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        pts.append(sum(a[i] - b_[i] for i in idx) / len(idx))
    pts.sort()
    lo, hi = pts[int(0.025 * bnum)], pts[int(0.975 * bnum)]
    return {"diff": round(statistics.mean(a) - statistics.mean(b_), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}


def _rankdata(v: list[float]) -> list[float]:
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def _pearson(a: list[float], b_: list[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(a) / n, sum(b_) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b_))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b_) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else 0.0


def spearman_boot(xs: list[float], ys: list[float], b: int = 2000) -> dict:
    if len(xs) < 5:
        return {"note": "insufficient samples"}
    rho = _pearson(_rankdata(xs), _rankdata(ys))
    rng = random.Random(SEED)
    pts = []
    for _ in range(b):
        idx = [rng.randrange(len(xs)) for _ in range(len(xs))]
        sx = [xs[i] for i in idx]
        sy = [ys[i] for i in idx]
        pts.append(_pearson(_rankdata(sx), _rankdata(sy)))
    pts.sort()
    lo, hi = pts[int(0.025 * b)], pts[int(0.975 * b)]
    return {"rho": round(rho, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n": len(xs), "significant": bool(lo > 0 or hi < 0)}


def bh_fdr(pvals: list[float], q: float = 0.10) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns rejection mask."""
    n = len(pvals)
    if n == 0:
        return []
    ranked = sorted(range(n), key=lambda i: pvals[i])
    reject = [False] * n
    last = -1
    for rank, idx in enumerate(ranked):
        if pvals[idx] <= (rank + 1) * q / n:
            last = rank
    for rank, idx in enumerate(ranked):
        if rank <= last:
            reject[idx] = True
    return reject


def normal_ci_diff(n1: int, m1: float, s1: float,
                   n2: int, m2: float, s2: float) -> dict:
    """Normal-approximation CI for a two-sample difference (for large samples)."""
    se = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 0.5
    diff = m1 - m2
    return {"diff": round(diff, 4),
            "ci95": [round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4)],
            "significant": bool(abs(diff) > 1.96 * se)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr-dir", default="pilot/runs/full_attr")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--tool-model", default="pilot/runs/tool_model_official.json")
    ap.add_argument("--out", default="pilot/runs/x7_fragile_l1.json")
    args = ap.parse_args()

    # ---- 1. load tool-channel entropy_rows ----
    tm = json.loads(Path(args.tool_model).read_text())
    h_tool_idx: dict[tuple[str, int], float] = {
        (row["instance_id"], row["t"]): row["H_tool"]
        for row in tm.get("entropy_rows", [])
    }
    print(f"tool-channel entropy_rows: {len(h_tool_idx)} steps")

    # ---- 2. load attribution data (causal_v0) ----
    recs: list[dict] = []
    for f in sorted(Path(args.attr_dir).glob("shard*/attribution.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    print(f"attribution records: {len(recs)}")
    v0_map: dict[str, float] = {}
    for r in recs:
        v0 = r.get("causal_v0")
        if v0 is not None:
            v0_map[r["instance_id"]] = float(v0)

    # ---- 3. compute mean H_tool per instance from rollouts ----
    insts = {i.instance_id: i for i in T.load_instances()}
    runs_root = Path(args.runs_root)
    h_per_inst: dict[str, float] = {}
    n_steps_per_inst: dict[str, int] = {}

    for rfile in sorted(runs_root.glob("shard*/rollouts/*.jsonl")):
        iid = rfile.stem.split("__k")[0]
        if iid not in v0_map:
            continue   # only analyze instances that have v0 (multi-file subset)
        try:
            rl = R.load(rfile)
            steps = R.steps(rl)
        except (OSError, ValueError):
            continue
        if not steps:
            continue
        hs = [h_tool_idx[(iid, s.idx)] for s in steps
              if (iid, s.idx) in h_tool_idx]
        if not hs:
            continue
        h_per_inst[iid] = statistics.mean(hs)
        n_steps_per_inst[iid] = len(hs)

    print(f"instances with both H_tool and v0: {len(h_per_inst)}")

    # ---- 4. three-group split ----
    EPS = 1e-6
    fail_ids = [i for i in h_per_inst if v0_map.get(i, 1) < EPS]
    fragile_ids = [i for i in h_per_inst if EPS < v0_map.get(i, 1) < 1 - EPS]
    robust_ids = [i for i in h_per_inst if v0_map.get(i, 1) > 1 - EPS]

    fail_h = [h_per_inst[i] for i in fail_ids]
    fragile_h = [h_per_inst[i] for i in fragile_ids]
    robust_h = [h_per_inst[i] for i in robust_ids]

    print(f"fail: {len(fail_ids)}  fragile: {len(fragile_ids)}  robust: {len(robust_ids)}")

    # group means
    groups = {
        "fail": boot_ci(fail_h),
        "fragile": boot_ci(fragile_h),
        "robust": boot_ci(robust_h),
    }

    # paired contrasts (BH-FDR correction over 3 contrasts)
    # unpaired bootstrap difference
    def nonpaired_ci(a: list[float], b_: list[float], bnum: int = 2000) -> dict:
        if not a or not b_:
            return {"note": "no samples"}
        rng = random.Random(SEED)
        pts = []
        for _ in range(bnum):
            ma = sum(a[rng.randrange(len(a))] for _ in range(len(a))) / len(a)
            mb = sum(b_[rng.randrange(len(b_))] for _ in range(len(b_))) / len(b_)
            pts.append(ma - mb)
        pts.sort()
        lo, hi = pts[int(0.025 * bnum)], pts[int(0.975 * bnum)]
        return {"diff": round(statistics.mean(a) - statistics.mean(b_), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "significant": bool(lo > 0 or hi < 0)}

    c_fragile_vs_robust = nonpaired_ci(fragile_h, robust_h)
    c_fail_vs_robust = nonpaired_ci(fail_h, robust_h)
    c_fragile_vs_fail = nonpaired_ci(fragile_h, fail_h)

    contrasts = {
        "fragile_vs_robust": c_fragile_vs_robust,
        "fail_vs_robust": c_fail_vs_robust,
        "fragile_vs_fail": c_fragile_vs_fail,
    }

    # ---- 5. continuous correlation: mean H_tool ~ causal_v0 ----
    iids_sorted = sorted(h_per_inst.keys())
    xs = [h_per_inst[i] for i in iids_sorted]
    ys = [v0_map[i] for i in iids_sorted]
    corr_h_v0 = spearman_boot(xs, ys)

    # ---- 6. precision/recall of a "high entropy threshold" (positive class = "needs replay" = fragile+fail) ----
    med_h = statistics.median(list(h_per_inst.values()))
    pos_ids = set(fail_ids) | set(fragile_ids)
    neg_ids = set(robust_ids)
    # prediction: H_tool > median → needs replay
    tp = sum(1 for i in pos_ids if h_per_inst[i] >= med_h)
    fp = sum(1 for i in neg_ids if h_per_inst[i] >= med_h)
    fn = sum(1 for i in pos_ids if h_per_inst[i] < med_h)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    # AUROC (simple two-class)
    all_sorted = sorted(h_per_inst.keys(), key=lambda i: -h_per_inst[i])
    tpr_pts, fpr_pts = [0.0], [0.0]
    n_pos, n_neg = len(pos_ids), len(neg_ids)
    tp_c = fp_c = 0
    for iid in all_sorted:
        if iid in pos_ids:
            tp_c += 1
        else:
            fp_c += 1
        tpr_pts.append(tp_c / n_pos if n_pos else 0)
        fpr_pts.append(fp_c / n_neg if n_neg else 0)
    auroc = sum((fpr_pts[i + 1] - fpr_pts[i]) * (tpr_pts[i + 1] + tpr_pts[i]) / 2
                for i in range(len(fpr_pts) - 1))

    predict_fragile = {
        "threshold_H_median": round(med_h, 4),
        "n_pos": len(pos_ids), "n_neg": len(neg_ids),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "note": "positive class = fragile+fail (needs replay), negative class = robust success",
    }

    # ---- 7. power check: does H_tool itself have detectable structure (before claiming "no relation") ----
    # X1 already showed H_tool has a position effect (low early, high mid/late);
    # here we run one more check against n_steps
    ns = [n_steps_per_inst[i] for i in iids_sorted if i in n_steps_per_inst]
    corr_h_nsteps = spearman_boot(
        [h_per_inst[i] for i in iids_sorted if i in n_steps_per_inst], ns)

    res = {
        "n_instances_with_h_and_v0": len(h_per_inst),
        "group_sizes": {
            "fail": len(fail_ids),
            "fragile": len(fragile_ids),
            "robust": len(robust_ids),
        },
        "group_mean_H_tool": groups,
        "contrasts": contrasts,
        "continuous_spearman_H_vs_v0": corr_h_v0,
        "predictability_of_fragile": predict_fragile,
        "power_check_H_vs_nsteps": corr_h_nsteps,
        "per_instance": [
            {"id": i, "v0": v0_map[i], "mean_H_tool": round(h_per_inst[i], 4),
             "n_steps": n_steps_per_inst.get(i, 0),
             "group": "fail" if i in fail_ids else ("fragile" if i in fragile_ids else "robust")}
            for i in sorted(h_per_inst.keys())
        ],
    }

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ---- print summary ----
    print(f"\n=== X7: fragile success ↔ L1 predictive entropy ===")
    print(f"\nmean H_tool by group:")
    for g, v in groups.items():
        if "note" not in v:
            print(f"  {g:10s}: {v['mean']:.4f} CI{v['ci95']}  (n={v['n']})")
    print(f"\ncontrasts (bootstrap difference):")
    for name, v in contrasts.items():
        if "note" not in v:
            print(f"  {name}: {v['diff']:+.4f} CI{v['ci95']}"
                  f"  {'significant' if v['significant'] else 'not significant'}")
    c = corr_h_v0
    if "note" not in c:
        print(f"\nSpearman H_tool ~ v0: rho={c['rho']:+.4f} CI{c['ci95']}"
              f"  {'significant' if c['significant'] else 'not significant'}")
    pf = predict_fragile
    print(f"\npredictive performance of \"high H_tool → needs replay\":")
    print(f"  threshold={pf['threshold_H_median']:.4f}  prec={pf['precision']:.3f}  "
          f"rec={pf['recall']:.3f}  f1={pf['f1']:.3f}  AUROC={pf['auroc']:.3f}")
    pw = res["power_check_H_vs_nsteps"]
    if "note" not in pw:
        print(f"\npower check: H_tool ~ n_steps  rho={pw['rho']:+.4f} CI{pw['ci95']}"
              f"  {'significant' if pw['significant'] else 'not significant'}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
