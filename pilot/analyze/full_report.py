"""Full-scale experiment analysis -- produces all table data the paper needs in one pass.

Input: full localization results (500 instances) + full attribution results (multi-file tasks)
Output: analysis.json + console tables, one per table of the paper's Experiment section.

Contains:
  T1 localization baseline (three definitions × stratified by gold count × stratified by repo)
  T2 leave-one-repository-out evaluation of the v_0 predictor (AUROC/ECE/Brier/AURC)
  T3 UQ three-way decomposition (real data)
  T4 attribution contrasts D1/D3 (with Wilcoxon, p-value reported when n suffices)
  T5 entropy structure statistics (positivity locality)
  T6 degeneration rate and failure-mode statistics
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.probes import task as T                                    # noqa: E402
from pilot.uq import features as F                                    # noqa: E402
from pilot.uq import predictor as P                                   # noqa: E402
from pilot.analyze.attribution_compare import (                       # noqa: E402
    wilcoxon_signed_rank_one_sided, bootstrap_ci,
)


# --------------------------------------------------------------------------- T1
def t1_baseline(recs: list[dict]) -> dict:
    ok = [r for r in recs if r.get("ok")]
    insts = {i.instance_id: i for i in T.load_instances()}
    ny = nyx = nyf = 0
    by_gold: dict[int, list[int]] = defaultdict(list)
    by_proj: dict[str, list[int]] = defaultdict(list)
    steps: list[int] = []
    walls: list[float] = []
    compaction = 0
    fails: list[dict] = []
    for r in ok:
        inst = insts.get(r["instance_id"])
        if inst is None:
            continue
        pred = frozenset(r["answer"]) if r.get("answer") else None
        oc = T.outcome(pred, inst.gold)
        ny += oc["Y"]; nyx += oc["Y_exact"]; nyf += oc["Y_filtered"]
        by_gold[len(inst.gold)].append(oc["Y"])
        by_proj[r.get("project", "?")].append(oc["Y"])
        steps.append(r.get("n_steps", 0))
        walls.append(r.get("wall_s", 0.0))
        compaction += 1 if r.get("compaction", 0) > 0 else 0
        if oc["Y"] == 0:
            fails.append({"id": r["instance_id"], "n_gold": len(inst.gold),
                          "recall": round(oc["recall"], 3),
                          "n_steps": r.get("n_steps"),
                          "missed": T.missed_files(pred, inst.gold)})
    n = len(ok)
    return {
        "n_total": len(recs), "n_ok": n, "n_error": len(recs) - n,
        "primary_recall_rate": round(ny / n, 4) if n else None,
        "exact_match_rate": round(nyx / n, 4) if n else None,
        "filtered_match_rate": round(nyf / n, 4) if n else None,
        "mean_steps": round(statistics.mean(steps), 2) if steps else None,
        "median_steps": statistics.median(steps) if steps else None,
        "mean_wall_s": round(statistics.mean(walls), 1) if walls else None,
        "n_with_compaction": compaction,
        "by_n_gold": {str(k): {"n": len(v), "rate": round(sum(v) / len(v), 4)}
                      for k, v in sorted(by_gold.items())},
        "by_project": {k: {"n": len(v), "rate": round(sum(v) / len(v), 4)}
                       for k, v in sorted(by_proj.items(), key=lambda kv: -len(kv[1]))},
        "n_failures": len(fails),
        "failures": fails[:40],
    }


# --------------------------------------------------------------------------- T2
def t2_v0_loro(recs: list[dict]) -> dict:
    """leave-one-repository-out evaluation of v_0 = E[Y | q, G]."""
    insts = {i.instance_id: i for i in T.load_instances()}
    rows = []
    for r in recs:
        if not r.get("ok"):
            continue
        inst = insts.get(r["instance_id"])
        if inst is None:
            continue
        pred = frozenset(r["answer"]) if r.get("answer") else None
        rows.append({"instance_id": r["instance_id"], "project": inst.project,
                     "Y": T.outcome(pred, inst.gold)["Y"],
                     "problem_statement": inst.problem_statement,
                     "repo_path": str(inst.repo)})
    if len(rows) < 20:
        return {"note": f"insufficient samples({len(rows)}), skipping LORO"}
    X, y, groups, ids = F.build_matrix(rows)
    projects = sorted(set(groups))
    preds = np.zeros(len(y)); epis = np.zeros(len(y))
    for p in projects:
        te = np.array([g == p for g in groups])
        tr = ~te
        if tr.sum() < 10 or te.sum() == 0:
            continue
        if len(set(y[tr].tolist())) < 2:      # training fold single-class, degenerate to base rate
            preds[te] = y[tr].mean(); epis[te] = 0.0
            continue
        m = P.BayesianLogisticLaplace(l2=2.0).fit(X[tr], y[tr])
        preds[te] = m.predict_proba(X[te])
        epis[te] = m.epistemic_var(X[te])
    mask = preds > 0
    # Validity condition: LORO needs enough repo groups, else train/test distribution shift dominates.
    # With fewer than 5 repos AUROC is uninterpretable (measured 0.359 < 0.5 at 2 repos).
    valid = len(projects) >= 5 and int(mask.sum()) >= 50
    return {
        "n": int(len(y)), "n_projects": len(projects),
        "loro_interpretable": valid,
        "validity_note": ("interpretable" if valid else
                          f"uninterpretable: repo groups={len(projects)}(need ≥5), n={int(mask.sum())}(need ≥50)"),
        "base_rate": round(float(y.mean()), 4),
        "AUROC": round(P.auroc(preds[mask], y[mask]), 4) if mask.sum() > 5 else None,
        "ECE": round(P.ece(preds[mask], y[mask]), 4) if mask.sum() > 5 else None,
        "Brier": round(P.brier(preds[mask], y[mask]), 4) if mask.sum() > 5 else None,
        "NLL": round(P.nll(preds[mask], y[mask]), 4) if mask.sum() > 5 else None,
        "AURC": round(P.risk_coverage(preds[mask], y[mask])[2], 4) if mask.sum() > 5 else None,
        "mean_epistemic": round(float(epis[mask].mean()), 4) if mask.sum() else None,
    }


# --------------------------------------------------------------------------- T3
def t3_uq_decomposition(recs: list[dict]) -> dict:
    """Three-way decomposition on real data. U_policy needs K>1 to be estimable; marked inestimable when K=1."""
    by_inst: dict[str, list[int]] = defaultdict(list)
    insts = {i.instance_id: i for i in T.load_instances()}
    for r in recs:
        if not r.get("ok"):
            continue
        inst = insts.get(r["instance_id"])
        if inst is None:
            continue
        pred = frozenset(r["answer"]) if r.get("answer") else None
        by_inst[r["instance_id"]].append(T.outcome(pred, inst.gold)["Y"])
    ks = [len(v) for v in by_inst.values()]
    multi = {k: v for k, v in by_inst.items() if len(v) > 1}
    u_policy = None
    if multi:
        u_policy = round(float(np.mean([np.var(v, ddof=0) for v in multi.values()])), 4)
    return {
        "n_instances": len(by_inst),
        "K_per_instance": dict(Counter(ks)),
        "U_policy": u_policy,
        "U_policy_note": ("estimable (instances with K>1 exist)" if u_policy is not None
                          else "inestimable: all instances have K=1, repeated rollouts needed"),
    }


# --------------------------------------------------------------------------- T4
def t4_attribution(attr_path: Path) -> dict:
    if not attr_path.exists():
        return {"note": "no attribution results yet"}
    recs = [json.loads(l) for l in attr_path.read_text().splitlines() if l.strip()]
    usable = [r for r in recs
              if r.get("causal_top1_pos") is not None
              and r.get("judge_full_top1_pos") is not None]
    if not usable:
        return {"n_runs": len(recs), "note": "no usable attribution records"}
    disagree = [float(r["causal_top1_step"] != r["judge_full_top1_step"]) for r in usable]
    d1, d1lo, d1hi = bootstrap_ci(disagree)
    posdiff = [r["judge_full_top1_pos"] - r["causal_top1_pos"] for r in usable]
    d3 = wilcoxon_signed_rank_one_sided(posdiff)
    both = [r for r in usable if r.get("judge_prefix_top1_pos") is not None]
    fp = [r["judge_full_top1_pos"] - r["judge_prefix_top1_pos"] for r in both]
    iso = wilcoxon_signed_rank_one_sided(fp) if len(fp) >= 6 else {"n": len(fp)}
    degen = [r for r in recs if r.get("degenerate")]
    return {
        "n_runs": len(recs), "n_usable": len(usable),
        "D1_disagreement": {"point": d1, "ci95": [d1lo, d1hi]},
        "D3_displacement": {
            "mean": round(statistics.mean(posdiff), 4),
            "median_causal_pos": round(statistics.median(
                [r["causal_top1_pos"] for r in usable]), 4),
            "median_judge_pos": round(statistics.median(
                [r["judge_full_top1_pos"] for r in usable]), 4),
            **d3,
        },
        "collider_isolation": {
            "n": len(fp),
            "mean": round(statistics.mean(fp), 4) if fp else None, **iso},
        "degenerate_rate": round(len(degen) / len(recs), 4) if recs else None,
        "n_degenerate": len(degen),
    }


# --------------------------------------------------------------------------- T5
def t5_entropy(scan_dirs: list[Path]) -> dict:
    from pilot.analyze.entropy_scan import spearman
    pos: list[float] = []; H: list[float] = []; nhigh = 0; ntot = 0
    per_task = []
    for rp in scan_dirs:
        for f in rp.glob("*/v5_report.json"):
            d = json.loads(f.read_text())
            ns = d.get("rollout_n_steps", 0)
            seq = []
            for s in d.get("per_step", []):
                if s.get("n_samples", 0) < 2:
                    continue
                p = s["fork_step"] / ns if ns else 0
                pos.append(p); H.append(s["entropy_nats"]); ntot += 1
                if s["positivity_ok"]:
                    nhigh += 1
                seq.append({"pos": round(p, 3), "H": s["entropy_nats"],
                            "high": s["positivity_ok"]})
            per_task.append({"instance": d["instance"], "seq": seq})
    if not ntot:
        return {"note": "no entropy scan data"}
    return {
        "n_forkpoints": ntot,
        "high_entropy_fraction": round(nhigh / ntot, 4),
        "spearman_pos_vs_entropy": round(spearman(pos, H), 4),
        "per_task": per_task,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locate", default="pilot/runs/full_locate/results.jsonl")
    ap.add_argument("--attr", default="pilot/runs/full_locate/attribution.jsonl")
    ap.add_argument("--scan", nargs="*", default=["pilot/runs/v5_scan", "pilot/runs/v5"])
    ap.add_argument("--out", default="pilot/runs/analysis.json")
    args = ap.parse_args()

    lp = Path(args.locate)
    if not lp.exists():
        # fallback: merge shards
        shards = sorted(Path(lp).parent.glob("shard*/results.jsonl"))
        recs = []
        for s in shards:
            recs += [json.loads(l) for l in s.read_text().splitlines() if l.strip()]
    else:
        recs = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()]

    out = {
        "T1_baseline": t1_baseline(recs),
        "T2_v0_loro": t2_v0_loro(recs),
        "T3_uq": t3_uq_decomposition(recs),
        "T4_attribution": t4_attribution(Path(args.attr)),
        "T5_entropy": t5_entropy([Path(s) for s in args.scan]),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    b = out["T1_baseline"]
    print(f"\n=== T1 localization baseline (n={b['n_ok']}/{b['n_total']}, err={b['n_error']}) ===")
    print(f"  recall definition {b['primary_recall_rate']}  strict {b['exact_match_rate']}  "
          f"filtered {b['filtered_match_rate']}")
    print(f"  steps mean={b['mean_steps']} median={b['median_steps']}  "
          f"wall clock {b['mean_wall_s']}s  compacted runs={b['n_with_compaction']}")
    print("  by gold count:", {k: f"{v['rate']}({v['n']})" for k, v in b["by_n_gold"].items()})
    print(f"  failures {b['n_failures']}")
    print(f"\n=== T2 v0 LORO ===\n  {out['T2_v0_loro']}")
    print(f"\n=== T3 UQ ===\n  {out['T3_uq']}")
    print(f"\n=== T4 attribution ===\n  {json.dumps(out['T4_attribution'], ensure_ascii=False)[:600]}")
    t5 = out["T5_entropy"]
    print(f"\n=== T5 entropy structure ===\n  fork points={t5.get('n_forkpoints')} "
          f"high-entropy fraction={t5.get('high_entropy_fraction')} "
          f"Spearman={t5.get('spearman_pos_vs_entropy')}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
