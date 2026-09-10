"""X1b: completing X1 -- the **measured** evaluation of the joint distribution and **a test of
the conditional independence assumption**. Zero API.

What X1 was missing
-------------------
X1 reported "joint effective candidates = exp(H_tool) × exp(H_prov) = 2.757 × 3.46 = 9.54".
But that multiplication relies on the conditional independence assumption
**P(tool,obj) = P(tool)·P(obj)**, while X1 itself noted "independence not yet tested".
If they are not independent, the number 9.54 is wrong (under dependence the joint entropy <
the sum of the marginal entropies, so 9.54 is an **upper bound**, not an estimate).

What plan X1 asked for is the **measured top-k / ECE / exp(H)** of the joint distribution, not
a product of marginals. This script fills that in:

  J1 measured joint: candidates = Cartesian product (tool, obj), P = P(tool)·P(obj_prov),
     common denominator, closed-form expectation for ties, giving top-k / ECE / eff. candidates
  J2 independence test: I(tool_{t+1}; objcat_{t+1}) vs **a within-episode permutation null**
     objcat uses the coarse classes {in the most recent output's paths / in the visited set /
     other}, because the object space is too large to estimate mutual information directly
  J3 if not independent: fit a tool-conditioned object model (one provenance weighting per tool
     class), compare it to the factored model by NLL and joint top-k, quantifying the
     misspecification cost
  J4 joint ECE stratification: find "when to trust" by (tool type × provenance coverage × step position)

Criteria (pre-registered)
-------------------------
| J2 result | consequence |
|---|---|
| MI significantly > null | factorization misspecified → 9.54 may only be reported as an upper bound; use the J1 measured values and the J3 conditional model |
| not significant | independence acceptable → 9.54 usable as an estimate, but the J1 measured values still govern |
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.analyze.v3g_regime import rank_ge, exp_hit, eff_cand, ece, KS  # noqa: E402
from pilot.analyze.v3_nextstep_dual import fit_tool_trans, p_tool        # noqa: E402
from pilot.analyze.v3j_trace_graph import (                              # noqa: E402
    load_episodes_with_output, mentioned_nodes, boot_ci, paired_ci,
)

SEED = 20260814
B_PERM = 2000
OBJCATS = ("in_last_output", "in_visited", "other")


def objcat(truth: str, last_out: list[str], seen: set[str]) -> str:
    if truth in last_out:
        return "in_last_output"
    if truth in seen:
        return "in_visited"
    return "other"


def mutual_info(pairs: list[tuple[str, str]]) -> float:
    """I(X;Y) in nats."""
    if not pairs:
        return 0.0
    n = len(pairs)
    pj, px, py = Counter(pairs), Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    return sum((c / n) * math.log((c / n) / ((px[a] / n) * (py[b] / n)))
               for (a, b), c in pj.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x1b_joint_independence.json")
    args = ap.parse_args()

    eps = load_episodes_with_output(Path(args.runs_root))
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_repo[e["project"]].append(e)
    tool_vocab = sorted({s["tool"] for e in eps for s in e["seq"]})
    print(f"loaded {len(eps)} episodes, {len(tool_vocab)} tool classes")

    # J1 measured joint
    joint_hits = {k: [] for k in KS}
    joint_cov, joint_eff, joint_cal = [], [], []
    # NLL of factored vs conditional
    nll_fact, nll_cond = [], []
    factored_hits = {k: [] for k in KS}       # object component only (control)
    # J2 independence
    mi_pairs_per_ep: list[list[tuple[str, str]]] = []
    # J4 stratification
    strat: dict[str, list[tuple[float, int]]] = defaultdict(list)
    strat_top3: dict[str, list[float]] = defaultdict(list)

    for proj, group in by_repo.items():
        if len(group) < 6:
            continue
        for hi, held in enumerate(group):
            train = [x for j, x in enumerate(group) if j != hi]
            tm = fit_tool_trans(train)
            # conditional model: one set of provenance position weights per "current tool" class (estimated from train)
            cond_w: dict[str, Counter] = defaultdict(Counter)
            for x in train:
                sq = x["seq"]
                o_prev: list[str] = []
                for t2 in range(len(sq) - 1):
                    mo = mentioned_nodes(x["graph"], sq[t2]["out"])
                    if mo:
                        o_prev = mo
                    tr = sq[t2 + 1]["node"]
                    if tr and o_prev:
                        # position bucket of the truth inside the most recent output (0,1,2,3+)
                        if tr in o_prev:
                            b = min(3, o_prev.index(tr))
                            cond_w[sq[t2 + 1]["tool"]][b] += 1
            g, seq = held["graph"], held["seq"]
            nodes = [s["node"] for s in seq]
            outs = [mentioned_nodes(g, s["out"]) for s in seq]
            seen: set[str] = set()
            last_out: list[str] = []
            ep_pairs: list[tuple[str, str]] = []
            for t in range(len(seq) - 1):
                if nodes[t]:
                    seen.add(nodes[t])
                if outs[t]:
                    last_out = outs[t]
                truth = nodes[t + 1]
                if not truth:
                    continue
                nxt_tool = seq[t + 1]["tool"]
                ep_pairs.append((nxt_tool, objcat(truth, last_out, seen)))

                # tool component
                pt = {v: p_tool(tm, seq[t]["tool"], v, tool_vocab) for v in tool_vocab}
                zt = sum(pt.values())
                pt = {k: v / zt for k, v in pt.items()}
                # object component (prov_lastout_ranked: the most recent output weighted by position)
                po_raw = {x: 1.0 / (i + 1) for i, x in enumerate(last_out)}
                zo = sum(po_raw.values())
                po = {k: v / zo for k, v in po_raw.items()} if zo else {}

                # --- J1 joint ---
                if po:
                    jd = {(a, b): pa * pb
                          for a, pa in pt.items() for b, pb in po.items()}
                    rs = rank_ge(jd, (nxt_tool, truth))
                    joint_cov.append(1.0 if rs else 0.0)
                    for k in KS:
                        joint_hits[k].append(exp_hit(rs[0], rs[1], k) if rs else 0.0)
                    if rs:
                        joint_eff.append(eff_cand(jd))
                        pt_true = jd[(nxt_tool, truth)]
                        joint_cal.append((pt_true, 1))
                        for v in sorted((v for kk, v in jd.items()
                                         if kk != (nxt_tool, truth)),
                                        reverse=True)[:20]:
                            joint_cal.append((v, 0))
                    # object component alone (control)
                    rso = rank_ge(po, truth)
                    for k in KS:
                        factored_hits[k].append(
                            exp_hit(rso[0], rso[1], k) if rso else 0.0)
                    # --- J3 NLL: factored vs conditional ---
                    if truth in po:
                        nll_fact.append(-math.log(max(po[truth], 1e-12)))
                        row = cond_w.get(nxt_tool)
                        if row and sum(row.values()) >= 5:
                            tot = sum(row.values()) + 4 * 0.5
                            w2 = {x: (row.get(min(3, i), 0) + 0.5) / tot
                                  for i, x in enumerate(last_out)}
                            z2 = sum(w2.values())
                            pc = {k: v / z2 for k, v in w2.items()} if z2 else {}
                            nll_cond.append(-math.log(max(pc.get(truth, 1e-12), 1e-12)))
                        else:
                            nll_cond.append(-math.log(max(po[truth], 1e-12)))
                    # --- J4 stratification ---
                    covered = truth in po
                    pos = "early" if t < len(seq) / 3 else (
                        "mid" if t < 2 * len(seq) / 3 else "late")
                    # ⚠️ Stratified ECE must collect positives and negatives together, otherwise
                    # what it computes is just 1−mean(p_true), not calibration error (the first
                    # version of this script made exactly this mistake).
                    negs = sorted((v for kk, v in jd.items()
                                   if kk != (nxt_tool, truth)), reverse=True)[:20]
                    # ⚠️ prov_cov conditions on the **truth** (truth in po), which is unknown at
                    # prediction time, so it is only "conditional precision given coverage" and
                    # **cannot be used as an actionable rule**. Only the three keys below are
                    # observable at prediction time and can form a real "when to trust" rule.
                    nlast = len(last_out)
                    nb = ("nlast=0" if nlast == 0 else "nlast=1-3" if nlast <= 3
                          else "nlast=4-10" if nlast <= 10 else "nlast=11+")
                    ec = eff_cand(jd)
                    eb = ("eff<=5" if ec <= 5 else "eff5-15" if ec <= 15
                          else "eff>15")
                    for key in (f"tool={seq[t]['tool']}", f"pos={pos}",
                                f"OBS_{nb}", f"OBS_{eb}",
                                f"POSTHOC_prov_cov={covered}"):
                        if rs:
                            strat[key].append((jd[(nxt_tool, truth)], 1))
                            for v in negs:
                                strat[key].append((v, 0))
                        strat_top3[key].append(
                            exp_hit(rs[0], rs[1], 3) if rs else 0.0)
            if ep_pairs:
                mi_pairs_per_ep.append(ep_pairs)

    # ---- J2 independence: permute tool labels within each episode ----
    rng = random.Random(SEED)
    obs_mi, null_mi, per_ep_p = [], [], []
    for pairs in mi_pairs_per_ep:
        if len(pairs) < 4:
            continue
        o = mutual_info(pairs)
        tools = [a for a, _ in pairs]
        cats = [b for _, b in pairs]
        nulls = []
        for _ in range(200):
            rng.shuffle(tools)
            nulls.append(mutual_info(list(zip(tools, cats))))
        obs_mi.append(o)
        null_mi.append(statistics.mean(nulls))
        per_ep_p.append((sum(1 for v in nulls if v >= o) + 1) / (len(nulls) + 1))
    # globally pooled mutual information (more power)
    allp = [p for ps in mi_pairs_per_ep for p in ps]
    g_obs = mutual_info(allp)
    tools_all = [a for a, _ in allp]
    cats_all = [b for _, b in allp]
    g_nulls = []
    for _ in range(B_PERM):
        rng.shuffle(tools_all)
        g_nulls.append(mutual_info(list(zip(tools_all, cats_all))))
    g_nulls.sort()
    g_p = (sum(1 for v in g_nulls if v >= g_obs) + 1) / (len(g_nulls) + 1)

    n_pts = len(joint_cov)
    res = {
        "n_episodes": len(eps), "n_decision_points": n_pts,
        "J1_measured_joint": {
            "coverage": boot_ci(joint_cov),
            "topk_uncond": {f"top{k}": boot_ci(joint_hits[k]) for k in KS},
            "eff_candidates_median": (round(statistics.median(joint_eff), 2)
                                      if joint_eff else None),
            "ECE_cond": round(ece(joint_cal), 4),
            "object_component_only_topk": {
                f"top{k}": round(statistics.mean(factored_hits[k]), 4) for k in KS},
        },
        "J2_independence_test": {
            "note": "objcat={in_last_output,in_visited,other}; null = permute tool labels within episode",
            "pooled_MI_observed_nats": round(g_obs, 5),
            "pooled_MI_null_mean": round(statistics.mean(g_nulls), 5),
            "pooled_MI_null_p975": round(g_nulls[int(0.975 * len(g_nulls))], 5),
            "pooled_perm_p": g_p,
            "per_episode_MI_obs": boot_ci(obs_mi),
            "per_episode_MI_null": boot_ci(null_mi),
            "per_episode_frac_p_below_05": (
                round(sum(1 for p in per_ep_p if p < 0.05) / len(per_ep_p), 4)
                if per_ep_p else None),
            "verdict": None,
        },
        "J3_factored_vs_conditional_NLL": {
            "nll_factored": boot_ci(nll_fact),
            "nll_tool_conditional": boot_ci(nll_cond),
            "diff_factored_minus_conditional": paired_ci(nll_fact, nll_cond),
            "note": "diff >0 and significant ⇒ conditional is better ⇒ factorization misspecified",
        },
        "J4_stratified_top3_and_ECE": {
            k: {"top3": boot_ci(v)["mean"], "n": len(v),
                "ECE": round(ece(strat.get(k, [])), 4),
                "n_calib_pairs": len(strat.get(k, []))}
            for k, v in sorted(strat_top3.items())},
        "independence_upper_bound_note": (
            "X1's exp(H_tool)×exp(H_prov)=9.54 is an **upper bound** under dependence: "
            "mutual information I>0 ⇒ joint entropy = H_tool+H_prov−I < sum of the marginals"),
    }
    mi_excess = g_obs - statistics.mean(g_nulls)
    res["J2_independence_test"]["verdict"] = (
        f"independence **does not hold** (pooled mutual information exceeds the null by {mi_excess:.5f} nat, "
        f"perm p={g_p:.4g}) ⇒ 9.54 may only be used as an upper bound"
        if g_p < 0.05 else
        f"independence acceptable (perm p={g_p:.4g}) ⇒ 9.54 usable as an estimate, but the J1 measured values still govern")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    j1 = res["J1_measured_joint"]
    print(f"\n=== J1 measured joint distribution (n={n_pts}, common denominator) ===")
    print(f"  coverage {j1['coverage']['mean']:.4f} CI{j1['coverage']['ci95']}")
    for k in KS:
        b = j1["topk_uncond"][f"top{k}"]
        print(f"  joint top-{k:<2} {b['mean']:.4f} CI{b['ci95']}")
    print(f"  joint effective candidates (measured median) **{j1['eff_candidates_median']}**"
          f"   ECE {j1['ECE_cond']}")
    print(f"  object component alone top-k: " +
          "  ".join(f"{k}={v}" for k, v in j1["object_component_only_topk"].items()))

    j2 = res["J2_independence_test"]
    print(f"\n=== J2 conditional independence test ===")
    print(f"  pooled mutual information: observed {j2['pooled_MI_observed_nats']} nat   "
          f"null mean {j2['pooled_MI_null_mean']}   "
          f"null 97.5% quantile {j2['pooled_MI_null_p975']}")
    print(f"  permutation p = {j2['pooled_perm_p']:.4g}   "
          f"per-episode significant rate {j2['per_episode_frac_p_below_05']}")
    print(f"  verdict: {j2['verdict']}")

    j3 = res["J3_factored_vs_conditional_NLL"]
    if "note" not in j3["nll_factored"]:
        d = j3["diff_factored_minus_conditional"]
        print(f"\n=== J3 factored vs tool-conditional (NLL, lower is better) ===")
        print(f"  factored {j3['nll_factored']['mean']:.4f}   "
              f"tool-conditional {j3['nll_tool_conditional']['mean']:.4f}")
        if "note" not in d:
            print(f"  diff(factored−conditional) {d['diff']:+.4f} "
                  f"CI[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
                  f"{'significant (conditional better)' if d['significant'] and d['diff'] > 0 else 'not significant' if not d['significant'] else 'significant (factored better)'}")

    print(f"\n=== J4 stratified top-3 / ECE (finding 'when to trust') ===")
    print(f"  Note: tool=X refers to the tool of the **current step** (how predictable the next step is after doing X);")
    print(f"      OBS_* are strata **observable** at prediction time and can serve as actionable rules;")
    print(f"      POSTHOC_* condition on the truth, **cannot serve as rules**, only as conditional-precision reference.")
    print(f"{'stratum':<26}{'n':>7}{'top3':>9}{'ECE':>9}{'calib':>8}")
    for k, v in res["J4_stratified_top3_and_ECE"].items():
        print(f"{k:<26}{v['n']:>7}{v['top3']:>9.4f}{v['ECE']:>9.4f}"
              f"{v['n_calib_pairs']:>8}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
