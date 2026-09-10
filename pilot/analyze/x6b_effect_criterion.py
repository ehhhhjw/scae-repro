"""X6b: is X6's 5x difference a real signal, or **manufactured by the binary criterion**? Zero API.

Origin
------
X6 reported: EFFECT rate 0.013 at low-H_tool steps vs 0.066 at high-H_tool steps (5x),
which is the **opposite** direction from X4's null result (H_tool ~ U^exec, rho=−0.088 n.s.).
But X6's EFFECT is a binary criterion (pointwise p<0.05) with only 7 events in the whole
sample, so that 5x could easily be an artifact of thresholding.

The question this script asks
-----------------------------
On the same points and the same H_tool, does the relationship survive when the response
variable is switched from the **binary EFFECT** to a **continuous quantity**?
Three response variables:

  binary   1[p < 0.05]        (X6's criterion)
  absdelta |Δ_t|              (the effect size itself)
  neglogp  −log10(p)          (evidence strength, discards no information)

Criterion (preregistered)
-------------------------
| Result | Conclusion |
|---|---|
| binary related, both continuous ones unrelated | the 5x is a **thresholding artifact**; X6b closes this doubt and X4's null result stands unchallenged |
| the continuous quantities are related too | X4's null result needs revisiting (its aggregation may have masked a step-level relationship) |
| the binary relationship itself fails bootstrap | simpler explanation: n=7 noise |

How the pointwise p values are computed
---------------------------------------
Each estimation segment has ys_t (recall over m replays) and ys_prev. Δ≠0 is tested with a
two-sample permutation test (pool the 2m values and reshuffle; statistic = absolute mean
difference), consistent with X5.
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

from pilot.channels.parsed_command import parse_step        # noqa: E402
from pilot.harness import rollout as R                      # noqa: E402
from pilot.analyze.v3k_trace_causal import _spearman_boot   # noqa: E402
from pilot.analyze.v3j_trace_graph import boot_ci, paired_ci  # noqa: E402

SEED = 20260817
B_PERM = 2000


def perm_p(a: list[float], b: list[float], n_perm: int = B_PERM,
           seed: int = SEED) -> float:
    """Two-sided p of a two-sample permutation test (statistic = |mean difference|)."""
    obs = abs(statistics.mean(a) - statistics.mean(b))
    pool = a + b
    na = len(a)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(statistics.mean(pool[:na]) - statistics.mean(pool[na:])) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4prime", default="pilot/runs/v4prime/attribution.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x6b_effect_criterion.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    # H_tool covariate (global tool-transition model)
    trans: dict[str, Counter] = defaultdict(Counter)
    step_tool: dict[str, list[str]] = {}
    for p in sorted(Path(args.runs_root).glob("shard*/rollouts/*.jsonl")):
        iid = p.stem.split("__k")[0]
        try:
            st = R.steps(R.load(p))
        except (OSError, ValueError):
            continue
        tools = [parse_step(s).kind for s in st]
        step_tool[iid] = tools
        for x, y in zip(tools, tools[1:]):
            trans[x][y] += 1
    vocab = sorted({t for ts in step_tool.values() for t in ts})

    def h_tool(cur: str, alpha: float = 0.5) -> float:
        row = trans.get(cur) or Counter()
        tot = sum(row.values()) + alpha * len(vocab)
        ps = [(row.get(v, 0) + alpha) / tot for v in vocab]
        return -sum(q * math.log(q) for q in ps if q > 0)

    recs = [json.loads(l) for l in open(args.v4prime) if l.strip()]
    rows = []
    for r in recs:
        tools = step_tool.get(r["instance_id"]) or []
        for s in r.get("causal_segments") or []:
            yt, yp = s.get("ys_t"), s.get("ys_prev")
            if not yt or not yp or len(yt) < 2 or len(yp) < 2:
                continue
            t = s["t"]
            cur = tools[t - 1] if 1 <= t <= len(tools) else None
            if cur is None:
                continue
            a = [float(x) for x in yt]
            b = [float(x) for x in yp]
            pv = perm_p(a, b)
            rows.append({
                "iid": r["instance_id"], "t": t,
                "h_tool": h_tool(cur),
                "absdelta": abs(statistics.mean(a) - statistics.mean(b)),
                "p": pv,
                "neglogp": -math.log10(max(pv, 1e-4)),
                "binary": 1.0 if pv < args.alpha else 0.0,
            })
    n = len(rows)
    n_eff = int(sum(x["binary"] for x in rows))
    print(f"pointwise samples {n}, EFFECT (p<{args.alpha}) = {n_eff} ({n_eff/n:.1%})")

    # stratify at the H_tool median (reproducing X6's procedure)
    hs = sorted(x["h_tool"] for x in rows)
    med = hs[len(hs) // 2]
    lo = [x for x in rows if x["h_tool"] <= med]
    hi = [x for x in rows if x["h_tool"] > med]

    res: dict = {
        "n_points": n, "n_effect": n_eff,
        "effect_rate": round(n_eff / n, 4), "alpha": args.alpha,
        "h_tool_median": round(med, 4),
        "STRATIFIED": {
            "low_H_n": len(lo), "high_H_n": len(hi),
            "low_H_effect_rate": round(statistics.mean(x["binary"] for x in lo), 4),
            "high_H_effect_rate": round(statistics.mean(x["binary"] for x in hi), 4),
        },
        "CONTINUOUS_VS_BINARY": {},
        "SENSITIVITY_TO_ALPHA": {},
    }
    # bootstrap the binary difference (this is the reliability of X6's 5x)
    rng = random.Random(SEED)
    d = []
    for _ in range(4000):
        sl = [lo[rng.randrange(len(lo))]["binary"] for _ in range(len(lo))]
        sh = [hi[rng.randrange(len(hi))]["binary"] for _ in range(len(hi))]
        d.append(statistics.mean(sh) - statistics.mean(sl))
    d.sort()
    res["STRATIFIED"]["binary_diff_high_minus_low"] = {
        "diff": round(statistics.mean(x["binary"] for x in hi)
                      - statistics.mean(x["binary"] for x in lo), 4),
        "ci95": [round(d[100], 4), round(d[3899], 4)],
        "significant": bool(d[100] > 0 or d[3899] < 0),
    }

    # relationship between H_tool and each of the three response variables
    for resp in ("binary", "absdelta", "neglogp"):
        res["CONTINUOUS_VS_BINARY"][f"h_tool_vs_{resp}"] = _spearman_boot(
            [x["h_tool"] for x in rows], [x[resp] for x in rows])

    # sensitivity of the binary criterion to the threshold
    for a in (0.01, 0.05, 0.10, 0.20):
        b_lo = statistics.mean(1.0 if x["p"] < a else 0.0 for x in lo)
        b_hi = statistics.mean(1.0 if x["p"] < a else 0.0 for x in hi)
        res["SENSITIVITY_TO_ALPHA"][f"alpha={a}"] = {
            "n_effect": int(sum(1 for x in rows if x["p"] < a)),
            "low_H_rate": round(b_lo, 4), "high_H_rate": round(b_hi, 4),
            "ratio": (round(b_hi / b_lo, 2) if b_lo > 0 else None),
        }

    cb = res["CONTINUOUS_VS_BINARY"]
    bin_sig = cb["h_tool_vs_binary"]["significant"]
    cont_sig = (cb["h_tool_vs_absdelta"]["significant"]
                or cb["h_tool_vs_neglogp"]["significant"])
    res["VERDICT"] = (
        "binary related but the continuous quantities are not ⇒ X6's 5x is a **thresholding artifact**; X4's null result stands unchallenged"
        if bin_sig and not cont_sig else
        "the continuous quantities are related too ⇒ X4's null result needs revisiting"
        if cont_sig else
        "**neither** binary nor continuous is related ⇒ X6's 5x is noise from too small an n; "
        "this doubt is closed and poses no challenge to X4")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    st = res["STRATIFIED"]
    print(f"\n=== X6's stratification reproduced (split at the H_tool median {med:.4f}) ===")
    print(f"  low H_tool EFFECT rate {st['low_H_effect_rate']} (n={st['low_H_n']})")
    print(f"  high H_tool EFFECT rate {st['high_H_effect_rate']} (n={st['high_H_n']})")
    b = st["binary_diff_high_minus_low"]
    print(f"  difference {b['diff']:+.4f} CI{b['ci95']}  "
          f"{'significant' if b['significant'] else '**not significant**'}")

    print(f"\n=== after switching to continuous response variables (same {n} points, same H_tool) ===")
    for k, v in cb.items():
        print(f"  {k:<22} rho={v['rho']:+.4f} CI{v['ci95']}  "
              f"{'significant' if v['significant'] else 'not significant'}")

    print(f"\n=== sensitivity of the binary criterion to the threshold ===")
    print(f"{'alpha':<10}{'n_effect':>10}{'lowH':>9}{'highH':>9}{'ratio':>8}")
    for k, v in res["SENSITIVITY_TO_ALPHA"].items():
        print(f"{k:<10}{v['n_effect']:>10}{v['low_H_rate']:>9}{v['high_H_rate']:>9}"
              f"{str(v['ratio']):>8}")

    print(f"\n=== verdict ===\n  {res['VERDICT']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
