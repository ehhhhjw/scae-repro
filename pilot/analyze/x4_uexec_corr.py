"""X4 (zero API, post-V4′): U^exec (true replay variance) ↔ effective structural quantity H_tool + H_prov.

Proposition
-----------
L1's predictive entropy can proxy replay variance.

Estimand
--------
corr(H_tool_t + H_prov_t, U^exec_t), Spearman + bootstrap CI;
partial correlations: H_tool alone, H_prov alone;
**negative control**: also report the invalidated u_obj (the Bernoulli approximation
4·v̂·(1−v̂), expected non-significant).

U^exec definition for the V4′ data
----------------------------------
segments[i]["ys_t"]   → recall values over m replays (length m)
U^exec_t = Var(ys_t)   true variance, no Bernoulli assumption needed

Control group (U^exec_legacy from the invalidated V4):
U^exec_legacy_t = 4 * v_hat_t * (1 - v_hat_t)

Three-tier criterion (RESEARCH_PLAN_v5 X4)
------------------------------------------
| ρ ≥ 0.4, significant | cheap quantity can proxy expensive replay → main claim of the paper                |
| 0.2 ≤ ρ < 0.4        | auxiliary signal, must be combined with replay                                     |
| ρ not significant    | first check whether U^exec itself has detectable structure (else insufficient power) |
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


def spearman_boot(xs: list[float], ys: list[float], b: int = 4000) -> dict:
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


def partial_spearman(x: list[float], y: list[float], z: list[float]) -> dict:
    """Approximation of partial Spearman corr(x, y | z) (via residual ranking)."""
    def ols_residual(dep: list[float], ind: list[float]) -> list[float]:
        n = len(dep)
        if n < 3:
            return dep[:]
        rx, ry = _rankdata(ind), _rankdata(dep)
        mx, my = sum(rx) / n, sum(ry) / n
        b_ = sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / max(
            1e-9, sum((a - mx) ** 2 for a in rx))
        a_ = my - b_ * mx
        return [ry[i] - (a_ + b_ * rx[i]) for i in range(n)]
    ex = ols_residual(x, z)
    ey = ols_residual(y, z)
    rho = _pearson(ex, ey)
    return {"partial_rho": round(rho, 4), "n": len(x)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4prime-dir", default="pilot/runs/v4prime",
                    help="V4′ shard directory (contains shard0..5/attribution.jsonl)")
    ap.add_argument("--tool-model", default="pilot/runs/tool_model_official.json")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate")
    ap.add_argument("--out", default="pilot/runs/x4_uexec_corr.json")
    args = ap.parse_args()

    # ---- 1. Load tool entropy ----
    tm = json.loads(Path(args.tool_model).read_text())
    h_tool_idx: dict[tuple[str, int], float] = {
        (row["instance_id"], row["t"]): row["H_tool"]
        for row in tm.get("entropy_rows", [])
    }
    H_PROV_EFF = 3.46   # global value from v3j prov_lastout_ranked
    H_PROV = math.log(H_PROV_EFF)   # ≈ 1.241
    print(f"H_tool steps: {len(h_tool_idx)},  H_prov global: {H_PROV:.4f}")

    # ---- 2. Load V4′ attribution ----
    recs: list[dict] = []
    v4p = Path(args.v4prime_dir)
    for f in sorted(v4p.glob("shard*/attribution.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    print(f"V4′ attribution records: {len(recs)}")

    if not recs:
        print("⚠️  No V4′ attribution data yet. Wait for V4′ to finish, then rerun this script.")
        return 1

    # ---- 3. Build the (H_struct, U^exec, U^exec_legacy) triple per segment ----
    rows: list[dict] = []
    for r in recs:
        iid = r["instance_id"]
        segs = r.get("causal_segments") or []
        for seg in segs:
            t = seg.get("t")
            ys_t = seg.get("ys_t") or []
            v_t = seg.get("v_t")
            if not ys_t or v_t is None or t is None:
                continue
            # true U^exec = Var(ys)
            if len(ys_t) < 2:
                continue
            u_exec = statistics.variance(ys_t)  # true replay variance
            # legacy Bernoulli approximation (V4's wrong method, kept as negative control)
            u_exec_legacy = 4.0 * v_t * (1.0 - v_t)
            # H_tool (per step)
            h_t = h_tool_idx.get((iid, t))
            # H_struct = H_tool + H_prov (global H_prov used as the per-step proxy)
            h_struct = (h_t + H_PROV) if h_t is not None else None
            rows.append({
                "iid": iid, "t": t,
                "u_exec": u_exec, "u_exec_legacy": u_exec_legacy,
                "h_tool": h_t, "h_struct": h_struct,
                "v_t": v_t,
                "delta": seg.get("delta", 0.0),
            })

    n_rows = len(rows)
    n_with_h = sum(1 for r in rows if r["h_struct"] is not None)
    print(f"per-segment records: {n_rows}, of which with H_tool: {n_with_h}")

    # ---- 4. Main correlation: H_struct ~ U^exec ----
    pairs_struct = [(r["h_struct"], r["u_exec"]) for r in rows if r["h_struct"] is not None]
    pairs_htool = [(r["h_tool"], r["u_exec"]) for r in rows if r["h_tool"] is not None]
    pairs_legacy = [(r["h_struct"], r["u_exec_legacy"]) for r in rows if r["h_struct"] is not None]

    corr_struct = spearman_boot([x for x, _ in pairs_struct],
                                [y for _, y in pairs_struct])
    corr_htool = spearman_boot([x for x, _ in pairs_htool],
                               [y for _, y in pairs_htool])
    corr_legacy = spearman_boot([x for x, _ in pairs_legacy],
                                [y for _, y in pairs_legacy])

    # ---- 5. U^exec's own structure (power check, pitfall #22) ----
    all_u_exec = [r["u_exec"] for r in rows]
    u_exec_stats = {
        "n": len(all_u_exec),
        "mean": round(statistics.mean(all_u_exec), 6),
        "median": round(statistics.median(all_u_exec), 6),
        "sd": round(statistics.stdev(all_u_exec), 6),
        "p90": round(sorted(all_u_exec)[int(0.9 * len(all_u_exec))], 6),
        "nonzero_rate": round(sum(1 for x in all_u_exec if x > 1e-9) / len(all_u_exec), 4),
    }

    # U^exec ~ delta (the effect should correlate with the variance)
    pairs_d = [(abs(r["delta"]), r["u_exec"]) for r in rows]
    corr_delta = spearman_boot([x for x, _ in pairs_d], [y for _, y in pairs_d])

    # U^exec ~ v_t (mean-variance relation)
    pairs_vt = [(r["v_t"], r["u_exec"]) for r in rows]
    corr_vt = spearman_boot([x for x, _ in pairs_vt], [y for _, y in pairs_vt])

    # ---- 6. Partial correlation (H_tool vs U^exec after controlling for v_t) ----
    common = [(r["h_tool"], r["u_exec"], r["v_t"])
              for r in rows if r["h_tool"] is not None]
    if len(common) >= 10:
        partial = partial_spearman(
            [x for x, _, _ in common],
            [y for _, y, _ in common],
            [z for _, _, z in common])
    else:
        partial = {"note": "insufficient samples"}

    # ---- 7. Stratification (three layers by delta magnitude) ----
    rows_sorted_d = sorted(rows, key=lambda r: abs(r["delta"]))
    n3 = len(rows_sorted_d) // 3
    layer_corrs = {}
    for name, subset in (("delta_low_33pct", rows_sorted_d[:n3]),
                         ("delta_mid_33pct", rows_sorted_d[n3:2 * n3]),
                         ("delta_high_33pct", rows_sorted_d[2 * n3:])):
        p = [(r["h_struct"], r["u_exec"]) for r in subset if r["h_struct"] is not None]
        layer_corrs[name] = spearman_boot([x for x, _ in p], [y for _, y in p]) if len(p) >= 10 else {"note": "insufficient samples"}

    res = {
        "n_rows": n_rows,
        "n_rows_with_h": n_with_h,
        "U_exec_stats": u_exec_stats,
        "MAIN_CORRELATIONS": {
            "H_struct_vs_U_exec": corr_struct,
            "H_tool_only_vs_U_exec": corr_htool,
            "H_struct_vs_U_exec_LEGACY": corr_legacy,
            "note_legacy": "V4's Bernoulli approximation (invalidated, kept as negative control)",
        },
        "POWER_CHECK_U_exec": {
            "abs_delta_vs_U_exec": corr_delta,
            "v_t_vs_U_exec": corr_vt,
            "note": "if U^exec has no structure w.r.t. position or effect, the negative result only reflects insufficient power",
        },
        "partial_spearman_H_tool_vs_Uexec_given_vt": partial,
        "stratified_by_delta_magnitude": layer_corrs,
    }

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"\n=== X4: U^exec ↔ structural entropy correlation ===")
    print(f"U^exec mean={u_exec_stats['mean']:.6f}, nonzero rate={u_exec_stats['nonzero_rate']:.3f}")
    print()
    mc = res["MAIN_CORRELATIONS"]
    for k, v in mc.items():
        if k == "note_legacy" or "note" in v:
            continue
        sig = "SIGNIFICANT" if v.get("significant") else "n.s."
        print(f"  {k}: rho={v['rho']:+.4f} CI{v['ci95']} [{sig}] n={v.get('n','?')}")
    print()
    pw = res["POWER_CHECK_U_exec"]
    for k, v in pw.items():
        if k == "note" or "note" in v:
            continue
        sig = "SIGNIFICANT" if v.get("significant") else "n.s."
        print(f"  power {k}: rho={v['rho']:+.4f} CI{v['ci95']} [{sig}]")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
