"""X2: can the LLM judge's choice be reproduced by a single **zero-LLM structural rule**. Zero API.

Motivation
----------
V3k found that 90.0% of judge top-1 picks land on steps "with a provenance relation to gold" (causal top-1: only 28.6%).
If a trivial structural rule can reproduce the judge's choice, then **observational process attribution carries no
information beyond structure** -- a strong disproof of that class of methods, at zero cost.

Selection-space mismatch that must be fixed first
-------------------------------------------------
V3k used **the same** baseline for the judge and for causal top-1 (36.2%, the share of gold provenance steps among
estimation points). But the judge reads the whole trace and picks from **all steps**; causal top-1 only picks among
~5 **equally spaced estimation points**. The two selection spaces differ, so the baselines must be computed separately:

    judge baseline  = gold provenance steps / all steps
    causal baseline = gold provenance steps / estimation points

This script reports both spaces side by side, plus a fair comparison **restricted to the common space**.

Candidate rules (all zero-LLM)
------------------------------
  R1 first_gold      first step that brings gold into the observation history
  R2 visit_gold      first step that directly operates on a gold file
  R3 last_gold       last step with a provenance relation to gold (the judge is known to skew late, A12)
  R4 last_step       final step (pure positional baseline: does "skewing late" alone explain the judge?)
  R5 max_gold        step mentioning gold the most
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                          # noqa: E402
from pilot.probes import task as T                              # noqa: E402
from pilot.analyze.v3k_trace_causal import gold_relation, boot_ci, paired_ci  # noqa: E402

SEED = 20260814
RULES = ("R1_first_gold", "R2_visit_gold", "R3_last_gold",
         "R4_last_step", "R5_max_gold")


def rule_pick(rel: dict[int, dict], all_steps: list[int]) -> dict[str, int | None]:
    """Top-1 pick of each rule (step index)."""
    gold_steps = [t for t in all_steps if rel[t]["any_gold"]]
    firsts = [t for t in all_steps if rel[t]["first_gold"]]
    visits = [t for t in all_steps if rel[t]["visit_gold"]]
    return {
        "R1_first_gold": firsts[0] if firsts else None,
        "R2_visit_gold": visits[0] if visits else None,
        "R3_last_gold": gold_steps[-1] if gold_steps else None,
        "R4_last_step": all_steps[-1] if all_steps else None,
        "R5_max_gold": (max(all_steps, key=lambda t: rel[t]["n_first"])
                        if all_steps else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr-dir", default="pilot/runs/full_attr")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x2_judge_mechanism.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    roll = {p.stem.split("__k")[0]: p
            for p in Path(args.runs_root).glob("shard*/rollouts/*.jsonl")}
    recs = []
    for f in sorted(Path(args.attr_dir).glob("shard*/attribution.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
    print(f"loaded {len(recs)} attribution records")

    # Selection-space diagnostics
    judge_in_estpts = []
    share_allsteps, share_estpts = [], []
    judge_hit_all, judge_hit_est = [], []
    causal_hit_est = []
    # Rule agreement rate
    agree = {r: [] for r in RULES}
    agree_chance = {r: [] for r in RULES}
    # Fair comparison (restricted to instances whose judge pick is inside the estimation points)
    fair_judge, fair_causal = [], []
    n_used = 0
    judge_pos, causal_pos = [], []
    residual_when_rule_misses: Counter = Counter()
    # Within-class position: judge's relative position **among** gold provenance steps (1=latest), closed-form null 0.5
    within_gold_pos, within_gold_null = [], []
    n_gold_steps = []

    for rec in recs:
        iid = rec["instance_id"]
        inst = insts.get(iid)
        segs = rec.get("causal_segments") or []
        if inst is None or not inst.gold or iid not in roll or not segs:
            continue
        try:
            steps = R.steps(R.load(roll[iid]))
        except (OSError, ValueError):
            continue
        rel = gold_relation(steps, set(inst.gold))
        all_steps = sorted(rel)
        est_pts = sorted({s["t"] for s in segs if s.get("t") in rel})
        if len(all_steps) < 3 or len(est_pts) < 2:
            continue
        n_used += 1

        jt = rec.get("judge_full_top1_step")
        ct = rec.get("causal_top1_step")
        s_all = sum(1 for t in all_steps if rel[t]["any_gold"]) / len(all_steps)
        s_est = sum(1 for t in est_pts if rel[t]["any_gold"]) / len(est_pts)
        share_allsteps.append(s_all)
        share_estpts.append(s_est)

        if jt in rel:
            judge_in_estpts.append(1.0 if jt in est_pts else 0.0)
            judge_hit_all.append(1.0 if rel[jt]["any_gold"] else 0.0)
            judge_pos.append(jt / len(all_steps))
        if ct in rel:
            causal_hit_est.append(1.0 if rel[ct]["any_gold"] else 0.0)
            causal_pos.append(ct / len(all_steps))
        if jt in est_pts and ct in rel:
            fair_judge.append(1.0 if rel[jt]["any_gold"] else 0.0)
            fair_causal.append(1.0 if rel[ct]["any_gold"] else 0.0)

        # Within-class position: given the judge picked a gold provenance step, its rank among those steps (from the end)
        gsteps = [t for t in all_steps if rel[t]["any_gold"]]
        n_gold_steps.append(len(gsteps))
        if jt in rel and rel.get(jt, {}).get("any_gold") and len(gsteps) >= 2:
            # Normalized "lateness": 1.0=last gold step, 0.0=first
            r_idx = gsteps.index(jt)
            within_gold_pos.append(r_idx / (len(gsteps) - 1))
            within_gold_null.append(0.5)          # uniform expectation, closed form

        # Rule vs judge agreement; chance = 1/n_steps (a random rule hitting the same step)
        picks = rule_pick(rel, all_steps)
        for r in RULES:
            p = picks[r]
            if p is None or jt not in rel:
                continue
            agree[r].append(1.0 if p == jt else 0.0)
            agree_chance[r].append(1.0 / len(all_steps))
        # Residual: what kind of step the judge picks when R3 misses
        if jt in rel and picks["R3_last_gold"] is not None and picks["R3_last_gold"] != jt:
            residual_when_rule_misses[
                "gold" if rel[jt]["any_gold"] else "non_gold"] += 1

    res = {
        "n_instances": n_used,
        "SELECTION_SPACE": {
            "note": "the judge picks from all steps, causal top-1 only from equally spaced estimation points "
                    "→ baselines must be computed separately",
            "judge_pick_inside_estimation_points": boot_ci(judge_in_estpts),
            "gold_provenance_share_of_all_steps": boot_ci(share_allsteps),
            "gold_provenance_share_of_estimation_points": boot_ci(share_estpts),
        },
        "judge_vs_own_baseline": {
            "observed": boot_ci(judge_hit_all),
            "correct_baseline_all_steps": boot_ci(share_allsteps),
            "diff": paired_ci(judge_hit_all, share_allsteps[:len(judge_hit_all)]),
        },
        "causal_vs_own_baseline": {
            "observed": boot_ci(causal_hit_est),
            "correct_baseline_estimation_points": boot_ci(share_estpts),
            "diff": paired_ci(causal_hit_est, share_estpts[:len(causal_hit_est)]),
        },
        "fair_comparison_same_space": {
            "note": "only instances whose judge pick falls exactly on an estimation point",
            "n": len(fair_judge),
            "judge": boot_ci(fair_judge), "causal": boot_ci(fair_causal),
            "diff": paired_ci(fair_judge, fair_causal),
        },
        "position": {"judge_mean_pos": boot_ci(judge_pos),
                     "causal_mean_pos": boot_ci(causal_pos)},
        "RULE_AGREEMENT_WITH_JUDGE": {
            r: {"agreement": boot_ci(agree[r]),
                "closed_form_chance": boot_ci(agree_chance[r]),
                "diff": paired_ci(agree[r], agree_chance[r])}
            for r in RULES},
        "residual_when_R3_misses": dict(residual_when_rule_misses),
        "WITHIN_CLASS_position": {
            "note": "given the judge picked a gold provenance step, its normalized lateness among those steps "
                    "(1=last gold step); closed-form null 0.5",
            "observed": boot_ci(within_gold_pos),
            "uniform_null": 0.5,
            "diff": paired_ci(within_gold_pos, within_gold_null),
            "mean_n_gold_steps_per_instance": boot_ci(
                [float(x) for x in n_gold_steps]),
            "uniform_within_class_agreement": boot_ci(
                [1.0 / x for x in n_gold_steps if x > 0]),
        },
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    ss = res["SELECTION_SPACE"]
    print(f"\nusable instances: {n_used}")
    print(f"\n=== Selection-space diagnostics (fixing V3k's baseline mismatch) ===")
    print(f"  fraction of judge picks landing inside estimation points "
          f"{ss['judge_pick_inside_estimation_points']['mean']:.4f} "
          f"CI{ss['judge_pick_inside_estimation_points']['ci95']}")
    print(f"  gold provenance steps as share of **all steps** {ss['gold_provenance_share_of_all_steps']['mean']:.4f} "
          f"CI{ss['gold_provenance_share_of_all_steps']['ci95']}")
    print(f"  gold provenance steps as share of **estimation points** {ss['gold_provenance_share_of_estimation_points']['mean']:.4f} "
          f"CI{ss['gold_provenance_share_of_estimation_points']['ci95']}")

    for key, label in (("judge_vs_own_baseline", "LLM judge (baseline=all steps)"),
                       ("causal_vs_own_baseline", "causal top-1 (baseline=estimation points)")):
        b = res[key]
        d = b["diff"]
        obs = b["observed"]
        base = b.get("correct_baseline_all_steps") or b.get(
            "correct_baseline_estimation_points")
        if "note" in obs:
            continue
        print(f"\n=== {label} landing on gold provenance steps ===")
        print(f"  observed {obs['mean']:.4f} CI{obs['ci95']}   "
              f"correct baseline {base['mean']:.4f} CI{base['ci95']}")
        if "note" not in d:
            print(f"  diff {d['diff']:+.4f} CI[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                  f"  {'significant' if d['significant'] else 'not significant'}")

    fc = res["fair_comparison_same_space"]
    if "note" not in fc["judge"]:
        print(f"\n=== Fair same-space comparison (n={fc['n']}) ===")
        print(f"  judge {fc['judge']['mean']:.4f}   causal {fc['causal']['mean']:.4f}")
        d = fc["diff"]
        if "note" not in d:
            print(f"  diff {d['diff']:+.4f} CI[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                  f"  {'significant' if d['significant'] else 'not significant'}")

    print(f"\n=== Structural rules reproducing the judge's top-1 (exact same-step hit) ===")
    print(f"{'rule':<18}{'agreement':>9}{'chance':>12}   diff CI")
    for r in RULES:
        b = res["RULE_AGREEMENT_WITH_JUDGE"][r]
        if "note" in b["agreement"]:
            print(f"{r:<18} no samples")
            continue
        d = b["diff"]
        ds = (f"{d['diff']:+.4f} CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
              f"{'significant' if d['significant'] else 'not significant'}") if "note" not in d else "-"
        print(f"{r:<18}{b['agreement']['mean']:>9.4f}"
              f"{b['closed_form_chance']['mean']:>12.4f}   {ds}")

    p = res["position"]
    print(f"\n  judge mean relative position {p['judge_mean_pos']['mean']:.4f}   "
          f"causal {p['causal_mean_pos']['mean']:.4f}")
    print(f"  judge's pick category when R3 misses {res['residual_when_R3_misses']}")
    wc = res["WITHIN_CLASS_position"]
    if "note" not in wc["observed"]:
        d = wc["diff"]
        print(f"\n=== Within-class position: which gold provenance step the judge picks ===")
        print(f"  normalized lateness {wc['observed']['mean']:.4f} CI{wc['observed']['ci95']}"
              f"   uniform null 0.5")
        if "note" not in d:
            print(f"  diff {d['diff']:+.4f} CI[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                  f"  {'significant' if d['significant'] else 'not significant'}")
        print(f"  mean {wc['mean_n_gold_steps_per_instance']['mean']:.2f} "
              f"gold provenance steps per instance → within-class random hit rate "
              f"{wc['uniform_within_class_agreement']['mean']:.4f}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
