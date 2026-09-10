"""X4b: does U^exec have **non-mechanical** structure -- decides whether X4's null result is publishable. Zero API.

Question
--------
X4 concluded "structural entropy does not predict execution uncertainty" (rho=-0.088, CI contains 0),
and claimed power was established with two probes:
    |Δ| ~ U^exec  rho=+0.534 significant
    v_t ~ U^exec  rho=-0.432 significant
But both probes are **mechanically related** to U^exec and cannot serve as evidence of power:
  · v_t ~ U^exec: variance is necessarily small when the mean approaches 0/1 (the mean-variance
    constraint) -- arithmetic, not structure.
  · |Δ| ~ U^exec: |Δ|=|v_t-v_{t-1}| is derived from the difference of two 8-replay means, so
    large variance ⇒ noisy Δ estimate ⇒ |Δ| inflated by noise; equally mechanical.

Also: U^exec has **median 0, with variance exactly zero at 50.5% of the points** (recall at m=8 takes
only a few discrete values), which by itself shows the quantity is very coarse at m=8.

So we must switch to probes **with no mechanical link to replay noise**:
  V1 instance-level ICC  does U^exec have instance-layer structure (null hypothesis scrambles instance grouping)
  V2 step position       U^exec ~ normalized step position (Spearman)
  V3 tool type           does U^exec differ when grouped by the current step's tool (permute tool labels)
  V4 residualization     first bin by v_t to absorb the mechanical mean-variance relation, then check
                         whether the **residual** U^exec is still related to H_tool / step position / tool

Criteria (pre-registered)
-------------------------
| V1-V3 result | Path forward |
|---|---|
| at least one significant | U^exec has real structure ⇒ X4's null result is a **substantive conclusion** (structural entropy decoupled from execution uncertainty), publishable |
| none significant | U^exec has no detectable structure at m=8 ⇒ X4's null result **only reflects insufficient power**, not publishable;
             need larger m or a different definition of U^exec (e.g. bootstrap variance of continuous recall) |
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
from pilot.probes import task as T                          # noqa: E402
from pilot.analyze.v3k_trace_causal import _spearman_boot    # noqa: E402
from pilot.analyze.v3j_trace_graph import boot_ci            # noqa: E402

SEED = 20260814
B_PERM = 2000


def var(xs: list[float]) -> float:
    return statistics.variance(xs) if len(xs) > 1 else 0.0


def icc(groups: dict[str, list[float]]) -> float:
    """Share of total variance that is between-group (simplified ICC)."""
    allv = [x for v in groups.values() for x in v]
    if len(allv) < 3:
        return float("nan")
    gm = statistics.mean(allv)
    n = len(allv)
    between = sum(len(v) * (statistics.mean(v) - gm) ** 2 for v in groups.values())
    total = sum((x - gm) ** 2 for x in allv)
    return between / total if total > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4prime", default="pilot/runs/v4prime/attribution.jsonl")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x4b_uexec_validity.json")
    args = ap.parse_args()

    # ---- global tool transition model (used for the H_tool covariate) ----
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
        for a, b in zip(tools, tools[1:]):
            trans[a][b] += 1
    vocab = sorted({t for ts in step_tool.values() for t in ts})

    def h_tool(cur: str, alpha: float = 0.5) -> float:
        row = trans.get(cur) or Counter()
        tot = sum(row.values()) + alpha * len(vocab)
        ps = [(row.get(v, 0) + alpha) / tot for v in vocab]
        return -sum(q * math.log(q) for q in ps if q > 0)

    # ---- per estimation point ----
    recs = [json.loads(l) for l in open(args.v4prime) if l.strip()]
    rows = []
    for r in recs:
        iid = r["instance_id"]
        tools = step_tool.get(iid) or []
        nst = r.get("n_steps") or max(
            [s["t"] for s in (r.get("causal_segments") or [])] or [1])
        for s in r.get("causal_segments") or []:
            ys = s.get("ys_t")
            if not ys or len(ys) < 2:
                continue
            t = s["t"]
            cur = tools[t - 1] if 1 <= t <= len(tools) else None
            rows.append({
                "iid": iid, "t": t, "pos": t / max(1, nst),
                "u_exec": var([float(y) for y in ys]),
                "v_t": float(s.get("v_t") or 0.0),
                "tool": cur,
                "h_tool": (h_tool(cur) if cur else None),
                "n_gold": r.get("n_gold"),
            })
    print(f"per-point samples {len(rows)} (from {len(recs)} instances)")
    u = [x["u_exec"] for x in rows]
    print(f"U^exec: mean={statistics.mean(u):.6f} median={statistics.median(u):.6f} "
          f"nonzero rate={sum(1 for x in u if x > 1e-12)/len(u):.4f}")

    rng = random.Random(SEED)
    res: dict = {"n_points": len(rows), "n_instances": len(recs),
                 "u_exec": {"mean": round(statistics.mean(u), 6),
                            "median": round(statistics.median(u), 6),
                            "nonzero_rate": round(
                                sum(1 for x in u if x > 1e-12) / len(u), 4)}}

    # ---- V1 instance-level ICC ----
    by_inst: dict[str, list[float]] = defaultdict(list)
    for x in rows:
        by_inst[x["iid"]].append(x["u_exec"])
    obs_icc = icc(by_inst)
    sizes = [len(v) for v in by_inst.values()]
    nulls = []
    for _ in range(B_PERM):
        pool = u[:]
        rng.shuffle(pool)
        g, i = {}, 0
        for k, sz in zip(by_inst, sizes):
            g[k] = pool[i:i + sz]
            i += sz
        nulls.append(icc(g))
    nulls = sorted(v for v in nulls if not math.isnan(v))
    p_icc = (sum(1 for v in nulls if v >= obs_icc) + 1) / (len(nulls) + 1)
    res["V1_instance_ICC"] = {
        "observed": round(obs_icc, 4),
        "null_mean": round(statistics.mean(nulls), 4),
        "null_p975": round(nulls[int(0.975 * len(nulls))], 4),
        "perm_p": p_icc, "significant": bool(p_icc < 0.05),
        "n_instances": len(by_inst),
    }

    # ---- V2 step position ----
    res["V2_position"] = _spearman_boot([x["pos"] for x in rows], u)

    # ---- V3 tool type (permute tool labels, statistic = between-group variance ratio) ----
    tr = [x for x in rows if x["tool"]]
    by_tool: dict[str, list[float]] = defaultdict(list)
    for x in tr:
        by_tool[x["tool"]].append(x["u_exec"])
    obs_t = icc(by_tool)
    ut = [x["u_exec"] for x in tr]
    tsizes = [len(v) for v in by_tool.values()]
    tn = []
    for _ in range(B_PERM):
        pool = ut[:]
        rng.shuffle(pool)
        g, i = {}, 0
        for k, sz in zip(by_tool, tsizes):
            g[k] = pool[i:i + sz]
            i += sz
        tn.append(icc(g))
    tn = sorted(v for v in tn if not math.isnan(v))
    p_t = (sum(1 for v in tn if v >= obs_t) + 1) / (len(tn) + 1)
    res["V3_tool_type"] = {
        "observed_between_group_ratio": round(obs_t, 4),
        "null_mean": round(statistics.mean(tn), 4),
        "perm_p": p_t, "significant": bool(p_t < 0.05),
        "group_means": {k: round(statistics.mean(v), 5)
                        for k, v in sorted(by_tool.items())},
        "group_n": {k: len(v) for k, v in sorted(by_tool.items())},
    }

    # ---- V4 residualization: bin by v_t first to absorb the mechanical mean-variance relation ----
    NB = 5
    srt = sorted(rows, key=lambda x: x["v_t"])
    bin_of: dict[int, list[float]] = defaultdict(list)
    for i, x in enumerate(srt):
        x["_bin"] = min(NB - 1, i * NB // len(srt))
        bin_of[x["_bin"]].append(x["u_exec"])
    bmean = {b: statistics.mean(v) for b, v in bin_of.items()}
    for x in rows:
        x["resid"] = x["u_exec"] - bmean[x["_bin"]]
    # **Decisive test**: instance-level ICC after residualization.
    # V1's raw ICC may still be mechanical -- v_t depends strongly on the instance (some
    # instances sit at v≈1 always ⇒ variance must be 0), so between-group structure may be
    # nothing but between-group differences in v_t. Only if the ICC remains significant
    # after absorbing v_t can we say U^exec has genuine, non-mechanical instance-layer structure.
    by_inst_r: dict[str, list[float]] = defaultdict(list)
    for x in rows:
        by_inst_r[x["iid"]].append(x["resid"])
    obs_ricc = icc(by_inst_r)
    rvals = [x["resid"] for x in rows]
    rsizes = [len(v) for v in by_inst_r.values()]
    rn = []
    for _ in range(B_PERM):
        pool = rvals[:]
        rng.shuffle(pool)
        g, i = {}, 0
        for k, sz in zip(by_inst_r, rsizes):
            g[k] = pool[i:i + sz]
            i += sz
        rn.append(icc(g))
    rn = sorted(v for v in rn if not math.isnan(v))
    p_ricc = (sum(1 for v in rn if v >= obs_ricc) + 1) / (len(rn) + 1)

    hr = [(x["h_tool"], x["resid"]) for x in rows if x["h_tool"] is not None]
    res["V4_residualized"] = {
        "note": "relations of the residual U^exec after five v_t bins absorb the mechanical mean-variance relation",
        "bin_means_of_u_exec_by_v_t": {str(k): round(v, 5)
                                       for k, v in sorted(bmean.items())},
        "resid_vs_h_tool": (_spearman_boot([a for a, _ in hr], [b for _, b in hr])
                            if len(hr) > 20 else {"note": "insufficient samples"}),
        "resid_vs_position": _spearman_boot([x["pos"] for x in rows],
                                            [x["resid"] for x in rows]),
        "RESIDUALIZED_instance_ICC": {
            "observed": round(obs_ricc, 4),
            "null_mean": round(statistics.mean(rn), 4),
            "null_p975": round(rn[int(0.975 * len(rn))], 4),
            "perm_p": p_ricc, "significant": bool(p_ricc < 0.05),
            "note": "still significant after absorbing v_t ⇒ U^exec has non-mechanical instance-layer structure",
        },
    }

    sig = [k for k in ("V2_position", "V3_tool_type") if res[k].get("significant")]
    if res["V4_residualized"]["RESIDUALIZED_instance_ICC"]["significant"]:
        sig.append("RESIDUALIZED_instance_ICC")
    res["VERDICT"] = {
        "non_mechanical_probes_significant": sig,
        "decision": (
            f"U^exec has non-mechanical structure ({', '.join(sig)} significant) ⇒ X4's null result is a **substantive conclusion**, publishable"
            if sig else
            "U^exec has no detectable non-mechanical structure at m=8 ⇒ X4's null result **only reflects insufficient power**, "
            "and cannot serve as evidence for \"structure decoupled from execution uncertainty\"; need larger m or a revised U^exec definition"),
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== V1 instance-level ICC (does U^exec have instance-layer structure) ===")
    v = res["V1_instance_ICC"]
    print(f"  observed {v['observed']}  null mean {v['null_mean']}  "
          f"null 97.5% {v['null_p975']}  perm p={v['perm_p']:.4g}  "
          f"{'significant' if v['significant'] else 'not significant'}  (n_inst={v['n_instances']})")
    print(f"\n=== V2 U^exec ~ normalized step position ===")
    v = res["V2_position"]
    print(f"  rho={v['rho']:+.4f} CI{v['ci95']}  n={v['n']}  "
          f"{'significant' if v['significant'] else 'not significant'}")
    print(f"\n=== V3 U^exec grouped by the current step's tool ===")
    v = res["V3_tool_type"]
    print(f"  between-group variance ratio {v['observed_between_group_ratio']}  null {v['null_mean']}  "
          f"perm p={v['perm_p']:.4g}  {'significant' if v['significant'] else 'not significant'}")
    for k in v["group_means"]:
        print(f"    {k:<14} mean={v['group_means'][k]:<10} n={v['group_n'][k]}")
    print(f"\n=== V4 residualized (after absorbing the mechanical v_t effect) ===")
    v4 = res["V4_residualized"]
    print(f"  mean U^exec by v_t quintile: {v4['bin_means_of_u_exec_by_v_t']}")
    for key in ("resid_vs_h_tool", "resid_vs_position"):
        d = v4[key]
        if "note" in d:
            print(f"  {key:<20} {d['note']}")
            continue
        print(f"  {key:<20} rho={d['rho']:+.4f} CI{d['ci95']}  "
              f"{'significant' if d['significant'] else 'not significant'}")
    ri = res["V4_residualized"]["RESIDUALIZED_instance_ICC"]
    print(f"  **residualized instance ICC**: observed {ri['observed']}  "
          f"null {ri['null_mean']} (97.5%={ri['null_p975']})  "
          f"perm p={ri['perm_p']:.4g}  {'significant' if ri['significant'] else 'not significant'}")
    print(f"\n=== Verdict ===\n  {res['VERDICT']['decision']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
