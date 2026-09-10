"""Process uncertainty quantification + outcome-stratified attribution -- the right home for the UQ line.

Design correction log
---------------------
Two earlier attempts both tied UQ to "predicting the binary outcome Y across episodes":
  (1) v_0 (pre-execution features) → LORO AUROC ≈ 0.5
  (2) anytime prefix features → all bootstrap CIs contain 0.5
The fundamental obstacle: with a 94% success rate there are only 30 failed samples, so no discriminative boundary can be learned.

Correct home: uncertainty comes from the **replay variance of a single execution**, not from cross-episode outcome labels.
v̂_t is estimated from m fork replays; v̂_t=1 means success is certain from that prefix (zero uncertainty),
v̂_t≈0.5 means half the replays succeed (maximum uncertainty). Every episode has this quantity, whatever the outcome.

Three reportable quantities
---------------------------
  U1 process entropy trace  H(v̂_t): how uncertainty evolves along execution ("at which step it converges")
  U2 fragile success rate   Y=1 and ∃t: v̂_t < 1−ε: succeeded but replay-unstable -- a blind spot of outcome-only evaluation
  U3 convergence point t*   step position after which v̂_t last crosses the threshold and never dips back

Attribution stratified by outcome (successful cases are attributed too)
-----------------------------------------------------------------------
Failed case: argmin_t Δ̂_t -- the step that hurt the outcome most
Successful case: argmax_t Δ̂_t -- the most decisive positive step
Both come from the same Theorem 1 decomposition; failed samples are not a prerequisite.
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


def binary_entropy(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def reconstruct_traj(rec: dict) -> list[tuple[int, float]] | None:
    """Reconstruct the v̂_t trace from causal_v0 plus cumulative deltas.

    causal_segments only persists delta (a field gap in the implementation), but
    delta_i = v̂_{t_i} − v̂_{t_{i-1}}, so v̂ can be recovered by accumulating from v0.
    """
    v = rec.get("causal_v0")
    if v is None:
        return None
    out = [(0, float(v))]
    for s in sorted(rec.get("causal_segments", []), key=lambda x: x["t"]):
        d = s.get("delta")
        if d is None:
            continue
        v = v + float(d)
        out.append((int(s["t"]), round(min(max(v, 0.0), 1.0), 6)))
    return out


def bootstrap_ci(vals: list[float], stat=statistics.mean, B: int = 4000,
                 seed: int = 11) -> tuple[float, float, float]:
    if not vals:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    boots = [stat(arr[rng.integers(0, len(arr), len(arr))].tolist()) for _ in range(B)]
    boots.sort()
    return (round(stat(arr.tolist()), 4),
            round(boots[int(0.025 * B)], 4),
            round(boots[int(0.975 * B)], 4))


def analyze(recs: list[dict], eps: float = 1e-3, theta: float = 0.9) -> dict:
    ok = [r for r in recs if not r.get("error") and reconstruct_traj(r)]
    succ = [r for r in ok if r.get("Y") == 1]
    fail = [r for r in ok if r.get("Y") == 0]

    # ---------------- U1 process entropy trace ----------------
    ent_bins: dict[float, list[float]] = defaultdict(list)
    for r in ok:
        n = max(1, r.get("n_steps", 1))
        for t, v in reconstruct_traj(r):
            b = round(min(1.0, t / n) * 4) / 4
            ent_bins[b].append(binary_entropy(v))
    u1 = []
    for b in sorted(ent_bins):
        pt, lo, hi = bootstrap_ci(ent_bins[b])
        u1.append({"pos": b, "n": len(ent_bins[b]), "mean_H": pt, "ci95": [lo, hi]})

    # ---------------- U2 fragile success ----------------
    frag_flags = []
    frag_min_v = []
    for r in succ:
        vs = [v for _, v in reconstruct_traj(r)]
        is_frag = any(v < 1.0 - eps for v in vs)
        frag_flags.append(1.0 if is_frag else 0.0)
        if is_frag:
            frag_min_v.append(min(vs))
    fr, flo, fhi = bootstrap_ci(frag_flags)

    # ---------------- U3 convergence point ----------------
    conv = []
    for r in ok:
        tr = reconstruct_traj(r)
        n = max(1, r.get("n_steps", 1))
        last_below = None
        for t, v in tr:
            if v < theta:
                last_below = t
        # the first point after the last dip below the threshold is the convergence point; 0 if it never dips
        if last_below is None:
            conv.append(0.0)
        else:
            after = [t for t, _ in tr if t > last_below]
            conv.append((after[0] if after else n) / n)
    cpt, clo, chi = bootstrap_ci(conv)

    # ---------------- attribution stratified by outcome (successful cases too) ----------------
    def top_step(r: dict, mode: str):
        segs = [s for s in r.get("causal_segments", []) if s.get("delta") is not None]
        if not segs:
            return None
        s = (min(segs, key=lambda x: x["delta"]) if mode == "min"
             else max(segs, key=lambda x: x["delta"]))
        return s

    strat = {}
    for name, group, mode in (("failed", fail, "min"), ("succeeded", succ, "max")):
        tops = [top_step(r, mode) for r in group]
        tops = [s for s in tops if s]
        deltas = [s["delta"] for s in tops]
        poss = [s["pos"] for s in tops]
        nz = [d for d in deltas if abs(d) > 1e-9]
        dpt, dlo, dhi = bootstrap_ci(deltas) if deltas else (0, 0, 0)
        ppt, plo, phi = bootstrap_ci(poss) if poss else (0, 0, 0)
        strat[name] = {
            "n": len(group),
            "n_with_segments": len(tops),
            "n_nonzero_top": len(nz),
            "nonzero_frac": round(len(nz) / len(tops), 4) if tops else None,
            "mean_top_delta": dpt, "delta_ci95": [dlo, dhi],
            "mean_top_pos": ppt, "pos_ci95": [plo, phi],
            "mode": "argmin (most harmful step)" if mode == "min"
                    else "argmax (most decisive positive step)",
        }

    # distribution of v̂_t values (exposes the granularity of m)
    allv = [v for r in ok for _, v in reconstruct_traj(r)]
    return {
        "n_episodes": len(ok),
        "n_succeeded": len(succ),
        "n_failed": len(fail),
        "U1_entropy_trajectory": u1,
        "U2_fragile_success": {
            "rate": fr, "ci95": [flo, fhi],
            "n_fragile": int(sum(frag_flags)), "n_success": len(succ),
            "mean_min_v_among_fragile": (round(statistics.mean(frag_min_v), 4)
                                        if frag_min_v else None),
            "definition": f"Y=1 and exists t with v̂_t < 1-{eps}",
        },
        "U3_convergence_point": {
            "mean_normalized": cpt, "ci95": [clo, chi], "theta": theta,
            "definition": "normalized position of first point after the last dip below theta",
        },
        "attribution_by_outcome": strat,
        "v_hat_value_counts": dict(sorted(Counter(round(v, 3) for v in allv).items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default="pilot/runs/full_attr/attribution.jsonl")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--theta", type=float, default=0.9)
    ap.add_argument("--out", default="pilot/runs/process_uncertainty.json")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.attr).read_text().splitlines() if l.strip()]
    res = analyze(recs, eps=args.eps, theta=args.theta)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"=== process uncertainty (n={res['n_episodes']}: "
          f"succeeded {res['n_succeeded']}, failed {res['n_failed']})===")
    print("\nU1 entropy trace H(v̂_t):")
    for e in res["U1_entropy_trajectory"]:
        print(f"  pos={e['pos']:.2f}  n={e['n']:>3}  H={e['mean_H']:.4f}  CI={e['ci95']}")
    u2 = res["U2_fragile_success"]
    print(f"\nU2 fragile success rate: {u2['rate']:.1%}  CI={u2['ci95']}  "
          f"({u2['n_fragile']}/{u2['n_success']})")
    print(f"   lowest v̂ among fragile: {u2['mean_min_v_among_fragile']}")
    u3 = res["U3_convergence_point"]
    print(f"\nU3 convergence point (normalized): {u3['mean_normalized']}  CI={u3['ci95']}  θ={u3['theta']}")
    print("\nAttribution stratified by outcome (successful cases too):")
    for k, v in res["attribution_by_outcome"].items():
        print(f"  {k:<10} n={v['n']:>2}  nonzero_top_frac={v['nonzero_frac']}  "
              f"mean|top Δ|={v['mean_top_delta']} CI={v['delta_ci95']}  "
              f"pos={v['mean_top_pos']} CI={v['pos_ci95']}")
        print(f"             ({v['mode']})")
    print(f"\nv̂_t distinct value levels {len(res['v_hat_value_counts'])} → granularity limited by m")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
