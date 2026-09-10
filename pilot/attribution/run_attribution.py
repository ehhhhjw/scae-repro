"""Attribution experiment orchestration -- the entry point for C3.

For the runs produced by the runner (each with a rollout copy), run all of the following
on a selected subset:
  1) causal attribution (causal.py): fork-resample to estimate Δ_t → top-1 root-cause step
  2) full-trace judge attribution (judge.py)
  3) prefix-only judge attribution (isolates collider bias; the prefix is cut before the
     causal top-1 to avoid leakage)

Produces a per-run jsonl, handed to analyze/attribution_compare.py to derive D1/D3.

Which runs get attributed
-------------------------
Failed runs first (Y=0, where "which step hurt" is well defined). If failures are too few,
also attribute successful runs (asking "which step mattered most"), turning C3 into
"are the critical steps consistent". Both record outcome_Y so the analysis can stratify.

Cost control
------------
- causal attribution uses a window (by default W equidistant points along the trace, avoiding
  tail bias -- a methodological requirement of D3)
- m defaults to 4; the judge is called twice per run (full + prefix)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                                   # noqa: E402
from pilot.harness.appserver import AppServer                            # noqa: E402
from pilot.harness.runner import write_codex_config                     # noqa: E402
from pilot.probes import task as T                                       # noqa: E402
from pilot.attribution import causal as C                                # noqa: E402
from pilot.attribution import judge as J                                 # noqa: E402


def equidistant_window_points(n_steps: int, w: int) -> list[int]:
    """Take w equidistant points over [1, n] (including near both ends), so tail bias does not contaminate D3."""
    if n_steps <= w:
        return list(range(1, n_steps + 1))
    return sorted({max(1, round(1 + i * (n_steps - 1) / (w - 1))) for i in range(w)})


def causal_top1_over_points(
    srv, base_items, inst, workdir, model, provider, m, points, turn_timeout,
    target_file: str | None = None,
) -> dict:
    """Estimate v_t at the given discrete points, approximate Δ by differencing adjacent points, and return the top-1 (most negative) step.

    target_file non-empty → per-file attribution: estimate v_t^(f), answering "which step
    decided whether f was found" (DESIGN_v2 §4.1.1; a missed file is the strongest negative signal).

    Note: differencing over discrete points is a "segment effect" approximation; the full
    experiment uses a continuous window, while the pilot uses equidistant points to control cost.
    """
    st = R.steps(base_items)
    vals = {}
    pts = sorted(set([0] + points))
    for t in pts:
        vals[t] = C.estimate_v(srv, base_items, t, inst, workdir, model, provider,
                               m, turn_timeout, target_file=target_file)
    seg = []
    for i in range(1, len(pts)):
        t0, t1 = pts[i - 1], pts[i]
        d = vals[t1].v_hat - vals[t0].v_hat
        # attribute the segment effect to the segment's end point t1 (the last step executed in it)
        a, b = R.action_bucket(st[t1 - 1], inst.repo)
        # persist the m raw recalls of each estimation point (fixes the data gap found by the
        # V4 diagnostic): storing only delta would force U^exec to be back-derived from v̂ as a
        # Bernoulli variance, but recall is a continuous value in [0,1], so the back-derivation
        # distorts it -- in practice the derived value is locked to v̂'s coarse granularity while
        # the true variance is fine-grained.
        seg.append({"t": t1, "seg_from": t0, "delta": round(d, 4),
                    "action": f"{a}#{b}", "pos": round(t1 / len(st), 4),
                    "v_t": round(vals[t1].v_hat, 4),
                    "v_prev": round(vals[t0].v_hat, 4),
                    "ys_t": [round(y, 4) for y in vals[t1].ys],
                    "ys_prev": [round(y, 4) for y in vals[t0].ys]})
    if not seg:
        return {"top1_step": None}
    top1 = min(seg, key=lambda s: s["delta"])       # the segment that hurt the outcome most
    return {"top1_step": top1["t"], "top1_pos": top1["pos"],
            "top1_delta": top1["delta"], "segments": seg,
            "v0": round(vals[0].v_hat, 4),
            "ys_v0": [round(y, 4) for y in vals[0].ys],
            "n_steps": len(st)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, help="runner output directory (containing results.jsonl and rollouts/)")
    ap.add_argument("--model", default="gpt-5.5-0424-global")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--m", type=int, default=4, help="number of fork-resamples")
    ap.add_argument("--window-points", type=int, default=6, help="number of equidistant estimation points")
    ap.add_argument("--only-failed", action="store_true", help="attribute only runs with Y=0")
    ap.add_argument("--per-file", action="store_true",
                    help="per-file recall attribution: estimate v_t^(f) for the missed gold file (DESIGN §4.1.1)")
    ap.add_argument("--multifile-only", action="store_true",
                    help="run only multi-file tasks (gold>=2) -- single-file tasks measured 100% signal-free")
    ap.add_argument("--outcome", choices=["binary", "per_file", "recall"], default="per_file",
                    help="outcome variable: binary=task-level Y; per_file=hit on a single gold file; "
                         "recall=continuous recall (fixes the Δ≡0 degeneracy, recommended)")
    ap.add_argument("--prefix-frac", type=float, default=0.6,
                    help="truncation fraction for the prefix-only judge (a fixed value, to avoid a constructive bias from depending on the causal result)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--turn-timeout", type=float, default=600.0)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    results = [json.loads(l) for l in (runs_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    results = [r for r in results if r.get("ok") and r.get("rollout")]
    if args.only_failed:
        results = [r for r in results if r.get("Y") == 0]
    if args.multifile_only:
        results = [r for r in results if r.get("n_gold", 1) >= 2]
    if args.limit > 0:
        results = results[: args.limit]

    out_path = Path(args.out) if args.out else runs_dir / "attribution.jsonl"
    workroot = runs_dir / "attr_work"
    codex_home = runs_dir / "attr_codex_home"
    write_codex_config(codex_home, args.model)

    insts = {i.instance_id: i for i in T.load_instances()}
    srv = AppServer(codex_bin=args.codex_bin, codex_home=codex_home,
                    log_path=runs_dir / "attr_appserver.log", experimental_api=True)
    n = 0
    t0 = time.time()
    try:
        with open(out_path, "a", encoding="utf-8") as fh:
            for r in results:
                iid = r["instance_id"]
                inst = insts[iid]
                rollout_file = runs_dir / r["rollout"]
                if not rollout_file.exists():
                    continue
                base_items = R.load(rollout_file)
                st = R.steps(base_items)
                if len(st) < 3:
                    continue
                # per-file attribution target: prefer the missed gold file (strongest negative
                # signal), else the first gold of a multi-file task; single-file tasks fall back
                # to the task-level binary Y.
                pred = frozenset(r["answer"]) if r.get("answer") else None
                missed = T.missed_files(pred, inst.gold)
                target_file = None
                if args.outcome == "recall":
                    # continuous outcome variable, fixes the Δ≡0 degeneracy (50% of per-file cases measured as degenerate)
                    target_file = "__recall__"
                elif args.per_file:
                    if missed:
                        target_file = missed[0]
                    elif len(inst.gold) >= 2:
                        target_file = sorted(inst.gold)[0]

                rec = {"instance_id": iid, "project": r.get("project"),
                       "Y": r["Y"], "n_steps": len(st),
                       "n_gold": len(inst.gold), "recall": r.get("recall"),
                       "missed_files": missed, "target_file": target_file,
                       "attr_mode": "per_file" if target_file else "task_binary"}
                try:
                    pts = equidistant_window_points(len(st), args.window_points)
                    wd = workroot / iid
                    caus = causal_top1_over_points(
                        srv, base_items, inst, wd, args.model, args.model_provider,
                        args.m, pts, args.turn_timeout, target_file=target_file)
                    rec["causal_top1_step"] = caus.get("top1_step")
                    rec["causal_top1_pos"] = caus.get("top1_pos")
                    rec["causal_top1_delta"] = caus.get("top1_delta")
                    rec["causal_v0"] = caus.get("v0")
                    # degeneracy check: all Δ are 0 ⇒ attribution cannot distinguish any step (must be reported honestly, not hidden)
                    segs = caus.get("segments", [])
                    rec["causal_segments"] = segs
                    rec["degenerate"] = bool(segs) and all(
                        abs(s.get("delta", 0.0)) < 1e-9 for s in segs)

                    jf = J.judge_attribution(srv, base_items, inst, r["Y"],
                                             args.model, args.model_provider,
                                             mode="full", turn_timeout=args.turn_timeout)
                    if jf and jf.get("parsed"):
                        rec["judge_full_top1_step"] = jf["top1_step"]
                        rec["judge_full_top1_pos"] = jf["top1_pos"]

                    # prefix-only: truncate at a **fixed fraction** (independent of the causal result).
                    # ⚠️ We used to cut "just before the causal top1", which artificially lowered the
                    # step positions available to the prefix judge, manufacturing the artifact that
                    # "full is later than prefix" -- a constructive bias, now fixed (PREREGISTRATION §3).
                    upto = max(2, int(round(len(st) * args.prefix_frac)))
                    jp = J.judge_attribution(srv, base_items, inst, r["Y"],
                                             args.model, args.model_provider,
                                             mode="prefix", prefix_upto=upto,
                                             turn_timeout=args.turn_timeout)
                    if jp and jp.get("parsed"):
                        rec["judge_prefix_top1_step"] = jp["top1_step"]
                        rec["judge_prefix_top1_pos"] = jp["top1_pos"]
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    rec["trace"] = traceback.format_exc()[-500:]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                print(f"[{n}/{len(results)}] {iid} Y={r['Y']} "
                      f"causal_top1={rec.get('causal_top1_step')} "
                      f"judge_top1={rec.get('judge_full_top1_step')}", flush=True)
    finally:
        srv.close()
    print(f"\ncompleted {n} attributions in {(time.time()-t0)/60:.1f}min → {out_path}")
    print(f"next: python -m pilot.analyze.attribution_compare --attr {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
