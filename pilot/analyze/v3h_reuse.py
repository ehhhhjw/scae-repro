"""V3h: a rigorous test of the hypothesis "the agent tends to reuse tools/files it has already used". Zero API.

Why a null-hypothesis control is needed
---------------------------------------
The naive reading is to look at the reuse rate (P(next ∈ something seen before)). But that quantity
is a **permutation invariant**: the number of "first occurrences" among sequence positions 2..n is
always #distinct−1, hence

    reuse rate = (n − #distinct) / (n − 1)

is fully determined by "how many steps, how many distinct objects", and is unchanged by any shuffle
of the sequence. So a 33.0% revisit rate **is not evidence of any "tendency"** -- it is merely the
arithmetic consequence of "only 14 distinct objects touched over 24 steps". This script verifies
that explicitly as a zero-variance check (O0), to avoid mistaking arithmetic for a finding.

What is actually testable is the **structure** of reuse, i.e. whether, given the composition, the temporal arrangement is more "clustered" than random:

  T1/O1  adjacent repeat   P(x_{t+1} = x_t)                                   permutation test
  T2/O2  sequential dep    mutual information I(x_t; x_{t+1})                 permutation test
  O3     burstiness        gap between adjacent occurrences of same object     permutation test
  O4     frequency pref    revisit target's historical visit count vs uniform  closed-form expectation
  O5     recency pref      revisit target's recency rank vs uniform            closed-form expectation

Null hypothesis: **within-episode permutation** (preserves the episode's composition and length,
shuffles only the order). That precisely isolates "tendency" (temporal structure) from "small
vocabulary" (composition) as competing explanations.
O4/O5 use closed-form expectations rather than sampling (pitfall #15).
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

from pilot.analyze.v3_nextstep_dual import load_episodes    # noqa: E402

B = 2000
SEED = 20260814


# ---------------------------------------------------------------- statistics

def self_repeat(seq: list) -> tuple[float, int]:
    """Adjacent repeat rate P(x_{t+1} = x_t)."""
    n = sum(1 for a, b in zip(seq, seq[1:]))
    if not n:
        return float("nan"), 0
    return sum(1 for a, b in zip(seq, seq[1:]) if a == b) / n, n


def reuse_rate(seq: list) -> tuple[float, int]:
    """Reuse rate P(x_{t+1} ∈ {x_1..x_t}). **Permutation invariant**, used only for the O0 check."""
    seen, hit, n = {seq[0]}, 0, 0
    for x in seq[1:]:
        n += 1
        if x in seen:
            hit += 1
        seen.add(x)
    return (hit / n if n else float("nan")), n


def mutual_info(seq: list) -> float:
    """I(x_t; x_{t+1}), in nat. Measures sequential dependence beyond the marginal distribution."""
    pairs = list(zip(seq, seq[1:]))
    if not pairs:
        return 0.0
    n = len(pairs)
    pj = Counter(pairs)
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    mi = 0.0
    for (a, b), c in pj.items():
        mi += (c / n) * math.log((c / n) / ((pa[a] / n) * (pb[b] / n)))
    return mi


def mean_gap(seq: list) -> float | None:
    """Mean gap between two adjacent occurrences of the same object. Smaller = more "bursty/clustered"."""
    pos: dict = defaultdict(list)
    for i, x in enumerate(seq):
        pos[x].append(i)
    gaps = [b - a for v in pos.values() for a, b in zip(v, v[1:])]
    return statistics.mean(gaps) if gaps else None


def perm_test(seq: list, stat_fn, n_perm: int = B, seed: int = SEED,
              alternative: str = "greater") -> dict:
    """Within-episode permutation test. Returns the observed value, null-distribution mean, and p-value."""
    obs = stat_fn(seq)
    if obs is None or (isinstance(obs, float) and math.isnan(obs)):
        return {"obs": None}
    rng = random.Random(seed)
    work = list(seq)
    null = []
    for _ in range(n_perm):
        rng.shuffle(work)
        v = stat_fn(work)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            null.append(v)
    if not null:
        return {"obs": obs}
    if alternative == "greater":
        ge = sum(1 for v in null if v >= obs)
    else:
        ge = sum(1 for v in null if v <= obs)
    return {"obs": obs, "null_mean": statistics.mean(null),
            "null_sd": (statistics.stdev(null) if len(null) > 1 else 0.0),
            "p": (ge + 1) / (len(null) + 1)}


# ---------------------------------------------------------------- closed-form preference

def freq_recency_bias(seq: list) -> dict:
    """Frequency/recency preference on revisit events, with a **closed-form expectation** as null.

    For every revisit event (x_{t+1} ∈ seen_t):
      frequency: the target's historical visit count c*  vs  E[c] of a uniformly drawn seen object
      recency:   the target's recency rank r* (1 = most recently visited) vs uniform expectation (m+1)/2
    """
    obs_c, exp_c, obs_r, exp_r, n = [], [], [], [], 0
    counts: Counter = Counter()
    order: list = []            # sorted by most recent visit (front = most recent)
    for i, x in enumerate(seq):
        if i > 0 and counts and x in counts:
            n += 1
            m = len(counts)
            obs_c.append(counts[x])
            exp_c.append(sum(counts.values()) / m)          # expected count of a uniformly drawn seen object
            obs_r.append(order.index(x) + 1)
            exp_r.append((m + 1) / 2)                        # uniform rank expectation
        counts[x] += 1
        if x in order:
            order.remove(x)
        order.insert(0, x)
    if not n:
        return {"n": 0}
    return {
        "n_revisit_events": n,
        "freq_obs": round(statistics.mean(obs_c), 4),
        "freq_null": round(statistics.mean(exp_c), 4),
        "freq_lift": round(statistics.mean(obs_c) / statistics.mean(exp_c), 4),
        "recency_obs_rank": round(statistics.mean(obs_r), 4),
        "recency_null_rank": round(statistics.mean(exp_r), 4),
        "recency_lift": round(statistics.mean(exp_r) / statistics.mean(obs_r), 4),
    }


# ---------------------------------------------------------------- summary

def agg(per_ep: list[dict], key: str) -> dict:
    """Cross-episode summary: episode-level bootstrap over each episode's statistic."""
    vals = [e[key] for e in per_ep if e.get(key) is not None]
    if not vals:
        return {"note": "no samples"}
    rng = random.Random(SEED)
    pts = []
    for _ in range(B):
        s = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        pts.append(statistics.mean(s))
    pts.sort()
    return {"mean": round(statistics.mean(vals), 4),
            "ci95": [round(pts[int(0.025 * B)], 4), round(pts[int(0.975 * B)], 4)],
            "n_episodes": len(vals)}


def fisher_combine(ps: list[float]) -> dict:
    """Combine the per-episode permutation p-values (Fisher's method)."""
    ps = [p for p in ps if p is not None and 0 < p <= 1]
    if not ps:
        return {"note": "no p-values"}
    x2 = -2 * sum(math.log(p) for p in ps)
    df = 2 * len(ps)
    # chi-square survival function (closed form for even df): P = e^{-t}·Σ_{i<k} t^i/i!
    # Computing it directly overflows (k reaches the hundreds), so use log-sum-exp in **log space**.
    k, t = df // 2, x2 / 2
    terms = [i * math.log(t) - math.lgamma(i + 1) for i in range(k)] if t > 0 else [0.0]
    mx = max(terms)
    log_p = -t + mx + math.log(sum(math.exp(v - mx) for v in terms))
    return {"n_episodes": len(ps), "chi2": round(x2, 2), "df": df,
            "log10_p_combined": round(log_p / math.log(10), 2),
            "p_combined": (math.exp(log_p) if log_p > -700 else 0.0),
            "median_p": round(statistics.median(ps), 4),
            "frac_p_below_0.05": round(sum(1 for p in ps if p < 0.05) / len(ps), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/v3h_reuse.json")
    args = ap.parse_args()

    eps = load_episodes(Path(args.runs_root), graph_only=False)
    print(f"loaded {len(eps)} episodes (not restricted to graph-built)")

    rows = []
    for e in eps:
        tools = [s["tool"] for s in e["seq"]]
        objs = [s["obj"] for s in e["seq"] if s["obj"]]
        if len(tools) < 4:
            continue
        r: dict = {"id": e["id"], "n_steps": len(tools),
                   "n_tools_distinct": len(set(tools)),
                   "n_objs": len(objs), "n_objs_distinct": len(set(objs))}
        # --- tools ---
        t1 = perm_test(tools, lambda s: self_repeat(s)[0])
        t2 = perm_test(tools, mutual_info)
        r["tool_selfrep"] = t1.get("obs")
        r["tool_selfrep_null"] = t1.get("null_mean")
        r["tool_selfrep_p"] = t1.get("p")
        r["tool_mi"] = t2.get("obs")
        r["tool_mi_null"] = t2.get("null_mean")
        r["tool_mi_p"] = t2.get("p")
        r["tool_reuse"] = reuse_rate(tools)[0]
        r["tool_reuse_perm"] = perm_test(tools, lambda s: reuse_rate(s)[0]).get("null_sd")
        # --- objects ---
        if len(objs) >= 4:
            o1 = perm_test(objs, lambda s: self_repeat(s)[0])
            o2 = perm_test(objs, mutual_info)
            o3 = perm_test(objs, mean_gap, alternative="less")
            r["obj_selfrep"] = o1.get("obs")
            r["obj_selfrep_null"] = o1.get("null_mean")
            r["obj_selfrep_p"] = o1.get("p")
            r["obj_mi"] = o2.get("obs")
            r["obj_mi_null"] = o2.get("null_mean")
            r["obj_mi_p"] = o2.get("p")
            r["obj_gap"] = o3.get("obs")
            r["obj_gap_null"] = o3.get("null_mean")
            r["obj_gap_p"] = o3.get("p")
            r["obj_reuse"] = reuse_rate(objs)[0]
            r["obj_reuse_perm"] = perm_test(objs, lambda s: reuse_rate(s)[0]).get("null_sd")
            r.update({f"bias_{k}": v for k, v in freq_recency_bias(objs).items()})
        rows.append(r)

    res: dict = {
        "n_episodes": len(rows),
        "note": "null hypothesis = within-episode permutation (preserves composition and length, shuffles order), B=2000",
        "O0_reuse_is_permutation_invariant": {
            "tool_reuse_null_sd_max": max((r.get("tool_reuse_perm") or 0)
                                          for r in rows),
            "obj_reuse_null_sd_max": max((r.get("obj_reuse_perm") or 0)
                                         for r in rows),
            "tool_reuse_mean": agg(rows, "tool_reuse"),
            "obj_reuse_mean": agg(rows, "obj_reuse"),
            "verdict": None,      # filled in below
        },
        "T1_tool_self_repeat": {"obs": agg(rows, "tool_selfrep"),
                                "null": agg(rows, "tool_selfrep_null"),
                                "perm": fisher_combine([r.get("tool_selfrep_p")
                                                        for r in rows])},
        "T2_tool_mutual_info": {"obs": agg(rows, "tool_mi"),
                                "null": agg(rows, "tool_mi_null"),
                                "perm": fisher_combine([r.get("tool_mi_p")
                                                        for r in rows])},
        "O1_obj_self_repeat": {"obs": agg(rows, "obj_selfrep"),
                               "null": agg(rows, "obj_selfrep_null"),
                               "perm": fisher_combine([r.get("obj_selfrep_p")
                                                       for r in rows])},
        "O2_obj_mutual_info": {"obs": agg(rows, "obj_mi"),
                               "null": agg(rows, "obj_mi_null"),
                               "perm": fisher_combine([r.get("obj_mi_p")
                                                       for r in rows])},
        "O3_obj_burstiness_gap": {"obs": agg(rows, "obj_gap"),
                                  "null": agg(rows, "obj_gap_null"),
                                  "perm": fisher_combine([r.get("obj_gap_p")
                                                          for r in rows])},
        "O4_frequency_bias": {"obs": agg(rows, "bias_freq_obs"),
                              "null": agg(rows, "bias_freq_null"),
                              "lift": agg(rows, "bias_freq_lift")},
        "O5_recency_bias": {"obs_rank": agg(rows, "bias_recency_obs_rank"),
                            "null_rank": agg(rows, "bias_recency_null_rank"),
                            "lift": agg(rows, "bias_recency_lift")},
        "per_episode": rows,
    }
    sd_max = max(res["O0_reuse_is_permutation_invariant"]["tool_reuse_null_sd_max"],
                 res["O0_reuse_is_permutation_invariant"]["obj_reuse_null_sd_max"])
    res["O0_reuse_is_permutation_invariant"]["verdict"] = (
        f"max sd of the permutation null distribution {sd_max:.2e} ⇒ the reuse rate is indeed a permutation invariant, "
        f"and cannot serve as evidence of a \"tendency\"" if sd_max < 1e-9 else
        f"⚠️ sd {sd_max:.2e} is non-zero, contradicting theory, check the implementation")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ---------------------------------------------------------- print
    o0 = res["O0_reuse_is_permutation_invariant"]
    print(f"\n=== O0 the reuse rate is a permutation invariant (rule out the arithmetic artifact first) ===")
    print(f"  tool reuse rate {o0['tool_reuse_mean']['mean']}  "
          f"CI{o0['tool_reuse_mean']['ci95']}")
    print(f"  object reuse rate {o0['obj_reuse_mean']['mean']}  "
          f"CI{o0['obj_reuse_mean']['ci95']}")
    print(f"  {o0['verdict']}")

    print(f"\n=== permutation tests of temporal structure (these are the real \"tendencies\") ===")
    print(f"{'statistic':<22}{'obs':>9}{'null':>9}{'ratio':>8}   combined p")
    for key, label, inv in (
        ("T1_tool_self_repeat", "tool adjacent repeat", False),
        ("T2_tool_mutual_info", "tool MI (nat)", False),
        ("O1_obj_self_repeat", "obj adjacent repeat", False),
        ("O2_obj_mutual_info", "obj MI (nat)", False),
        ("O3_obj_burstiness_gap", "obj mean gap ↓", True),
    ):
        d = res[key]
        o, nl = d["obs"].get("mean"), d["null"].get("mean")
        if o is None or nl is None:
            continue
        ratio = (nl / o if inv and o else (o / nl if nl else float("nan")))
        lp = d["perm"].get("log10_p_combined")
        ps = (f"1e{lp:.0f}" if lp is not None and lp < -3
              else ("%.3g" % d["perm"]["p_combined"]) if lp is not None else "-")
        frac = d["perm"].get("frac_p_below_0.05")
        print(f"{label:<22}{o:>9.4f}{nl:>9.4f}{ratio:>8.2f}   {ps}"
              f"   per-ep signif. rate {frac}   (n={d['obs']['n_episodes']} ep)")
        print(f"{'':<22}obs CI{d['obs']['ci95']}  null CI{d['null']['ci95']}")

    print(f"\n=== selection preference among revisit targets (closed-form expectation as null) ===")
    o4, o5 = res["O4_frequency_bias"], res["O5_recency_bias"]
    print(f"  frequency preference: target's historical visit count {o4['obs']['mean']} vs uniform expectation "
          f"{o4['null']['mean']}   lift {o4['lift']['mean']} CI{o4['lift']['ci95']}")
    print(f"  recency preference: target's recency rank {o5['obs_rank']['mean']} vs uniform expectation "
          f"{o5['null_rank']['mean']}   lift {o5['lift']['mean']} CI{o5['lift']['ci95']}")

    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
