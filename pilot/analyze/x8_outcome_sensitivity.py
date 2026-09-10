"""X8 first half: pick the **most sensitive outcome definition** for the intervention experiment. Zero API.

Why this must come first (it is a prerequisite for X3)
------------------------------------------------------
The primary definition Y=(gold⊆pred) has a baseline rate of 94% -- the ceiling effect compresses
the effect size of any intervention: if 94% of instances already succeed, masking one provenance
edge can at most flip 6% of instances "from success to failure", while changes within the
remaining 94% are flattened out by binarization. **So before spending 11.7h on X3, we must first
confirm the definition can carry an effect.**

Selection criterion (key point: "baseline rate near 50%" is not what matters)
-----------------------------------------------------------------------------
An intervention experiment detects a **difference in means**, whose detectability depends on

    effect size / standard error ∝ (available headroom) / sd

so three quantities must be read together:
  1. **baseline mean** -- how far it is from the ceiling/floor (headroom = min(mean, 1-mean))
  2. **standard deviation** -- larger sd is more easily swamped by noise, but also indicates
     discriminative power
  3. **effective dynamic range** -- how much this definition can **in principle** move when the
     gold provenance is removed. Approximated by a "counterfactual upper bound": the value of
     the definition after removing all gold files from pred
     → this gives the value under "masking works perfectly", and its gap to the baseline is the
     **maximum detectable effect**

Candidate definitions
---------------------
  Y            gold ⊆ pred (primary definition, binary)
  Y_exact      strict set equality (binary)
  Y_filtered   equality or containment after filtering test/docs (binary)
  recall       continuous [0,1]
  precision    continuous (penalizes over-reporting)
  f1           continuous
  n_steps      efficiency (step count, continuous, unbounded above)

Criteria (pre-registered)
-------------------------
| Result | Path forward |
|---|---|
| some definition has "max detectable effect/sd" ≥50% above the primary one | **X3 switches to that definition as primary**, recall becomes secondary |
| all definitions comparable | stay with recall (continuous, already better than binary Y), no change needed |
| max detectable effect <0.1 for every definition | the definition is not the bottleneck → the task itself is too easy, need a harder subset |
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                  # noqa: E402
from pilot.probes import task as T                      # noqa: E402
from pilot.analyze.v3j_trace_graph import boot_ci       # noqa: E402
from pilot.analyze.v3k_trace_causal import gold_relation  # noqa: E402

METRICS = ("Y", "Y_exact", "Y_filtered", "recall", "precision", "f1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x8_outcome_sensitivity.json")
    ap.add_argument("--multifile-only", action="store_true",
                    help="only multi-file instances (the actual population for X3)")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    rows = []
    for p in sorted(Path(args.runs_root).glob("shard*/rollouts/*.jsonl")):
        iid = p.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None or not inst.gold:
            continue
        if args.multifile_only and len(inst.gold) < 2:
            continue
        try:
            items = R.load(p)
        except (OSError, ValueError):
            continue
        ans = T.answer_from_rollout(items)
        oc = T.outcome(ans, inst.gold)
        if not oc.get("parsed"):
            continue
        # Counterfactual: drop **only** the gold files first exposed at t* (the first_gold step).
        # The first version removed all gold, which zeroed out every definition and degenerated
        # "max effect" into the baseline mean, turning the criterion into mean/sd (which only
        # selects definitions with high mean and low variance) rather than sensitivity.
        st = R.steps(items)
        rel = gold_relation(st, set(inst.gold))
        # find the first_gold step and take the gold set it first exposes
        lost: set[str] = set()
        gold_seen: set[str] = set()
        for s in st:
            txt = s.output_text or ""
            ment = {gp for gp in inst.gold
                    if gp in txt or gp.split("/")[-1] in txt}
            first = ment - gold_seen
            gold_seen |= ment
            if first:
                lost = first            # the first step that has a first_gold
                break
        if not lost:
            continue
        cf_pred = frozenset((ans or frozenset()) - lost)
        cf = T.outcome(cf_pred, inst.gold)
        rows.append({
            "instance_id": iid, "n_gold": len(inst.gold),
            "n_pred": len(ans) if ans else 0,
            "n_lost_if_masked": len(lost),
            "n_steps": len(st),
            **{m: oc[m] for m in METRICS},
            **{f"cf_{m}": cf[m] for m in METRICS},
        })

    if not rows:
        print("no samples")
        return 1
    print(f"loaded {len(rows)} instances"
          f" ({'multi-file only' if args.multifile_only else 'all'})")

    res: dict = {"n_instances": len(rows),
                 "multifile_only": bool(args.multifile_only),
                 "metrics": {}}
    n = len(rows)
    for m in METRICS:
        vals = [float(r[m]) for r in rows]
        cfs = [float(r[f"cf_{m}"]) for r in rows]
        # **paired difference**: how much the definition drops per instance when the gold
        # exposed at t* is dropped
        diffs = [a - b for a, b in zip(vals, cfs)]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if n > 1 else 0.0
        d_mean = statistics.mean(diffs)
        d_sd = statistics.stdev(diffs) if n > 1 else 0.0
        # detectability of a paired design ∝ mean(Δ)/sd(Δ)·sqrt(n) (paired t statistic)
        tstat = (d_mean / d_sd * (n ** 0.5)) if d_sd > 1e-9 else None
        res["metrics"][m] = {
            "mean": round(mean, 4), "sd": round(sd, 4),
            "headroom_to_ceiling_or_floor": round(min(mean, 1.0 - mean), 4),
            "counterfactual_mean": round(statistics.mean(cfs), 4),
            "paired_effect_mean": round(d_mean, 4),
            "paired_effect_sd": round(d_sd, 4),
            "paired_t_stat": (round(tstat, 2) if tstat is not None else None),
            "effect_over_sd": (round(d_mean / d_sd, 3) if d_sd > 1e-9 else None),
            "frac_instances_moved": round(
                sum(1 for d in diffs if abs(d) > 1e-9) / n, 4),
            "ci_mean": boot_ci(vals)["ci95"],
        }
    # step count (unbounded above, handled separately)
    ns = [float(r["n_steps"]) for r in rows]
    res["n_steps"] = {"mean": round(statistics.mean(ns), 2),
                      "sd": round(statistics.stdev(ns), 2) if len(ns) > 1 else 0.0,
                      "cv": (round(statistics.stdev(ns) / statistics.mean(ns), 3)
                             if len(ns) > 1 and statistics.mean(ns) > 0 else None),
                      "note": "efficiency definition: no ceiling, but only indirectly related to gold discoverability"}

    # verdict
    base = res["metrics"]["recall"]
    best = max((m for m in METRICS
                if res["metrics"][m]["effect_over_sd"] is not None),
               key=lambda m: res["metrics"][m]["effect_over_sd"])
    b_eos = res["metrics"][best]["effect_over_sd"]
    r_eos = base["effect_over_sd"]
    gain = (b_eos / r_eos - 1.0) if (r_eos and r_eos > 0) else None
    res["VERDICT"] = {
        "best_metric": best, "best_effect_over_sd": b_eos,
        "recall_effect_over_sd": r_eos,
        "relative_gain_over_recall": (round(gain, 3) if gain is not None else None),
        "decision": (
            f"switch to {best} as the primary X3 definition (effect/sd is {gain:.0%} above recall)"
            if gain is not None and gain >= 0.5 else
            "keep recall as the primary X3 definition (no substantive gain from the others)"),
        "all_effects_tiny": bool(all(
            res["metrics"][m]["paired_effect_mean"] < 0.1 for m in METRICS)),
    }

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== intervention sensitivity of each outcome definition ===")
    print(f"  counterfactual = drop only the gold files first exposed at t* (first_gold step), i.e. the real consequence of masking")
    print(f"{'metric':<12}{'base':>8}{'counterf':>9}{'pairedEff':>10}"
          f"{'effSd':>9}{'eff/sd':>9}{'pairT':>8}{'fracMoved':>9}")
    for m in METRICS:
        d = res["metrics"][m]
        eos, ts = d["effect_over_sd"], d["paired_t_stat"]
        print(f"{m:<12}{d['mean']:>8.4f}{d['counterfactual_mean']:>9.4f}"
              f"{d['paired_effect_mean']:>10.4f}{d['paired_effect_sd']:>9.4f}"
              f"{(f'{eos:.3f}' if eos is not None else '-'):>9}"
              f"{(f'{ts:.1f}' if ts is not None else '-'):>8}"
              f"{d['frac_instances_moved']:>9.4f}")
    n = res["n_steps"]
    print(f"\n  step count: mean {n['mean']} sd {n['sd']} cv {n['cv']}  ({n['note']})")
    v = res["VERDICT"]
    print(f"\n=== Verdict ===")
    print(f"  most sensitive definition **{v['best_metric']}** (effect/sd = {v['best_effect_over_sd']})"
          f"   recall = {v['recall_effect_over_sd']}")
    print(f"  {v['decision']}")
    if v["all_effects_tiny"]:
        print(f"  ⚠️ max effect <0.1 for every definition → the definition is not the bottleneck, need a harder subset")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
