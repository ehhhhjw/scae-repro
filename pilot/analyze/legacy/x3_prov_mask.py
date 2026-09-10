"""X3 (API, highest priority) provenance-mask intervention: making A3 causal.

Claim
-----
The provenance edge by which the agent moves to "the path it just saw in the output" is
**causal**: masking it lowers recall.

Estimand (path-specific effect)
-------------------------------
PSE(t*) = E[Y|τ_{1:t*}] - E[Y|τ_{1:t*}, do(gold paths in output_{t*} are masked)]

Three-arm design (paired within instance)
-----------------------------------------
  C control : output unchanged (the standard estimate of v_t*)
  M mask    : drop the output lines that mention gold (do(observation))
  P placebo : drop the same number of non-gold lines (internal validity check)

Fork point t* = the first_gold step (the step that first lets gold enter the agent's observation history)
  - 71 candidate instances, median first_gold_t = 2 (step index, not a fraction)
  - m=8 forks per arm × ~2.5min = 20min/instance; 6 in parallel → ~8.3h total

Three-way criterion
-------------------
(i)  M ≪ C and P ≈ C → the provenance edge is causally effective; main result of the paper
(ii) M ≈ C and P ≈ C → information redundancy; run dose-response (mask k=1..K gold lines)
(iii)P drops too     → operator disturbance; fix mask_output_lines and rerun

Usage
-----
  # generate the intervention plan (zero API)
  python3 -m pilot.analyze.x3_prov_mask --mode plan --out pilot/runs/x3_plan.json

  # run the intervention (API, ~8.3h)
  python3 -m pilot.analyze.x3_prov_mask --mode run \
      --plan pilot/runs/x3_plan.json \
      --out pilot/runs/x3_pse.json

  # analyze the results (zero API)
  python3 -m pilot.analyze.x3_prov_mask --mode analyze \
      --out pilot/runs/x3_pse.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                          # noqa: E402
from pilot.harness.appserver import AppServer                   # noqa: E402
from pilot.harness.runner import write_codex_config             # noqa: E402
from pilot.probes import task as T                              # noqa: E402
from pilot.attribution.causal import estimate_v                 # noqa: E402
from pilot.analyze.v3k_trace_causal import gold_relation, boot_ci, paired_ci  # noqa: E402
from pilot.analyze.v3j_trace_graph import PATH_RE                # noqa: E402

SEED = 20260814
M_FORKS = 8          # forks per arm
N_BOOT = 4000        # bootstrap resamples


def _candidate_instances(
    runs_root: str,
    runs_root2: str | None = None,
) -> list[dict]:
    """Find multi-file instances that have a first_gold step, plus their first_gold_t.

    Returns a list: [{"instance_id", "rollout_path", "first_gold_t", "n_gold", "gold", "n_steps"}]
    """
    insts = {i.instance_id: i for i in T.load_instances()}
    # collect rollout paths (full_locate first, then v4prime)
    roll_map: dict[str, Path] = {}
    for root in ([runs_root] + ([runs_root2] if runs_root2 else [])):
        for p in Path(root).glob("shard*/rollouts/*.jsonl"):
            iid = p.stem.split("__k")[0]
            if iid not in roll_map:
                roll_map[iid] = p

    candidates = []
    for iid, rollout_path in roll_map.items():
        inst = insts.get(iid)
        if inst is None or len(inst.gold) < 2:
            continue
        try:
            base_items = R.load(rollout_path)
        except Exception:
            continue
        st = R.steps(base_items)
        if len(st) < 3:
            continue
        rel = gold_relation(st, inst.gold)
        first_gold_t = None
        for tidx in sorted(rel.keys()):
            if rel[tidx].get("first_gold"):
                first_gold_t = tidx
                break
        if first_gold_t is None:
            continue
        candidates.append({
            "instance_id": iid,
            "rollout_path": str(rollout_path),
            "first_gold_t": first_gold_t,
            "n_gold": len(inst.gold),
            "gold": sorted(inst.gold),
            "n_steps": len(st),
        })
    return sorted(candidates, key=lambda x: x["instance_id"])


def plan_mode(args: argparse.Namespace) -> int:
    """Generate the intervention plan (zero API)."""
    print("computing the first_gold candidate list...")
    cands = _candidate_instances(args.runs_root, args.runs_root2)
    print(f"candidate instances: {len(cands)}")

    # stats
    fg_ts = [c["first_gold_t"] for c in cands]
    fg_ratios = [c["first_gold_t"] / c["n_steps"] for c in cands]
    within_5 = sum(1 for t in fg_ts if t <= 5)

    summary = {
        "n_candidates": len(cands),
        "first_gold_t": {
            "median": statistics.median(fg_ts),
            "p25": fg_ts[len(fg_ts) // 4],
            "p75": fg_ts[3 * len(fg_ts) // 4],
            "within_5_steps": within_5,
            "within_5_rate": round(within_5 / max(1, len(cands)), 3),
        },
        "first_gold_ratio": {
            "median": round(statistics.median(fg_ratios), 3),
            "mean": round(statistics.mean(fg_ratios), 3),
        },
        "estimated_budget": {
            "forks": len(cands) * 3 * M_FORKS,  # 3 arms * m
            "parallel_workers": 6,
            "min_per_fork": 2.5,
            "estimated_hours": len(cands) * 3 * M_FORKS * 2.5 / 60 / 6,
        },
        "candidates": cands,
    }

    Path(args.out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n=== X3 intervention plan ===")
    print(f"candidate instances: {len(cands)}")
    print(f"first_gold_t median: {statistics.median(fg_ts):.0f}")
    print(f"first_gold_t ≤5: {within_5}/{len(cands)} = {within_5/max(1,len(cands))*100:.0f}%")
    print(f"estimated budget: {len(cands)*3*M_FORKS} forks, "
          f"≈{len(cands)*3*M_FORKS*2.5/60/6:.1f}h (6 in parallel)")
    print(f"\n→ {args.out}")
    return 0


def _estimate_arm(
    srv: AppServer,
    base_items: list[dict],
    inst: T.Instance,
    workdir: Path,
    model: str,
    provider: str,
    t_star: int,
    arm: str,             # "control" / "mask" / "placebo"
    gold_paths: list[str],
    m: int,
    turn_timeout: float,
) -> dict:
    """Given the prefix through t_star (inclusive), modify the output per arm and estimate v_{t*}.

    control : original prefix (truncate_including_step(t_star))
    mask    : mask the gold lines
    placebo : mask the same number of non-gold lines (line count = gold line count, sampled at random from non-gold lines)
    """
    prefix = R.truncate_including_step(base_items, t_star)
    R.assert_no_dangling_calls(prefix)
    diag = {}

    if arm == "mask":
        try:
            prefix, diag = R.mask_output_lines(base_items, t_star, gold_paths, mode="drop")
        except ValueError as e:
            return {"error": str(e), "arm": arm}
    elif arm == "placebo":
        # first count the gold lines
        n_gold_lines = R.count_output_lines_matching(base_items, t_star, gold_paths)
        if n_gold_lines == 0:
            return {"error": "no gold lines to match for placebo", "arm": arm}
        st = R.steps(base_items)
        if not (1 <= t_star <= len(st)) or st[t_star - 1].out_line is None:
            return {"error": "placebo: no output at t_star", "arm": arm}
        txt = R.payload(base_items[st[t_star - 1].out_line]).get("output", "") or ""
        all_lines = txt.split("\n")
        non_gold = [ln for ln in all_lines
                    if not any(gp in ln for gp in gold_paths) and ln.strip()]
        # Fix 1: the placebo should remove non-gold lines that **also contain paths**, otherwise
        # what it drops are uninformative header lines like `Chunk ID:` / `Output:`, carrying far
        # less information than the mask arm and thus not a fair control ("same amount of
        # perturbation, different information" is the whole point of a placebo).
        pathy = [ln for ln in non_gold if PATH_RE.search(ln)]
        pool = pathy if len(pathy) >= n_gold_lines else (
            pathy + [ln for ln in non_gold if ln not in pathy])
        # Fix 2: `hash()` is randomized by PYTHONHASHSEED and unstable across processes → not reproducible.
        # Use a string seed instead (random.Random accepts str, deterministic), seeded per instance.
        rng = random.Random(f"{SEED}:{inst.instance_id}")
        rng.shuffle(pool)
        placebo_targets = pool[:n_gold_lines]
        if not placebo_targets:
            return {"error": "placebo: no non-gold lines available", "arm": arm}
        try:
            prefix, diag = R.mask_output_lines(
                base_items, t_star, placebo_targets, mode="placebo"
            )
        except ValueError as e:
            return {"error": str(e), "arm": arm}
        # Fix 3: record the balancing explicitly. Do not silently pass when there are too few
        # lines -- the main analysis uses only balanced samples, and unbalanced ones are
        # reported separately (they can still feed the dose-response analysis).
        diag["n_gold_lines_target"] = n_gold_lines
        diag["n_pathy_available"] = len(pathy)
        diag["balanced"] = bool(diag.get("n_lines_dropped") == n_gold_lines)
        diag["placebo_all_pathy"] = bool(len(pathy) >= n_gold_lines)

    # write the prefix
    workdir.mkdir(parents=True, exist_ok=True)
    fork_file = workdir / f"prefix_t{t_star}_{arm}.jsonl"
    R.dump(prefix, fork_file)

    # m forks
    ys: list[float] = []
    errs: list[str] = []
    for i in range(m):
        try:
            T.reset_repo(inst)
            from pilot.harness.appserver import thread_rollout_path
            fk = srv.thread_fork(
                path=fork_file, cwd=inst.repo, model=model,
                model_provider=provider,
                sandbox="read-only", approval_policy="never",
                base_instructions=T.BASE_INSTRUCTIONS,
            )
            tid = fk["thread"]["id"]
            tr = srv.turn_run(tid, T.CONTINUE_PROMPT, timeout=turn_timeout)
            # ⚠️ Key guardrail: when a turn fails there is no answer in the rollout,
            # `answer_from_rollout` returns None, and `T.outcome(None, gold)["recall"] == 0.0`
            # -- so **any API/environment failure silently turns into "recall=0" and is treated
            # as a valid observation**. If failures intermittently hit the mask arm more often,
            # that **manufactures a spurious positive PSE**.
            # So we must first decide whether the turn really succeeded; on failure record NaN
            # (excluded) rather than 0.
            err_msgs = []
            for nt in (tr.get("notifications") or []):
                if nt.get("method") == "error":
                    pr_ = nt.get("params") or {}
                    # willRetry=true is a transient "reconnecting" notification, not a terminal failure.
                    # Only turn/completed.status=="failed" is terminal (observed in practice:
                    # it can still succeed after Reconnecting 1..5).
                    if pr_.get("willRetry"):
                        continue
                    err_msgs.append(str((pr_.get("error") or {}).get("message"))[:120])
                elif nt.get("method") == "turn/completed":
                    tn = (nt.get("params") or {}).get("turn") or {}
                    if tn.get("status") == "failed":
                        err_msgs.append("turn failed: " +
                                        str((tn.get("error") or {}).get("message"))[:120])
            if err_msgs:
                ys.append(float("nan")); errs.append("TURN_FAILED " + err_msgs[0]); continue
            pth = thread_rollout_path(fk)
            if not (pth and Path(pth).exists()):
                ys.append(float("nan")); errs.append("NO_ROLLOUT_FILE"); continue
            items = R.load(pth)
            # not a single assistant message in the continuation ⇒ invalid (a real agent always
            # says something), rather than "answered but recall=0"
            has_assistant = any(
                it.get("type") == "response_item"
                and (it.get("payload") or {}).get("role") == "assistant"
                for it in items)
            ans = T.answer_from_rollout(items)
            if ans is None and not has_assistant:
                ys.append(float("nan")); errs.append("NO_ASSISTANT_MESSAGE"); continue
            ys.append(T.outcome(ans, inst.gold)["recall"])
        except Exception as exc:
            # not silent: record the exception type to distinguish "environment failure" from "a genuine recall=0"
            ys.append(float("nan"))
            errs.append(f"{type(exc).__name__}: {str(exc)[:120]}")

    ys_clean = [y for y in ys if not math.isnan(y)]
    return {
        "arm": arm,
        "t_star": t_star,
        "ys": ys,
        "v_hat": sum(ys_clean) / len(ys_clean) if ys_clean else float("nan"),
        "n_valid": len(ys_clean),
        "n_errors": len(errs),
        "errors": errs[:5],
        "diag": diag,
    }


def run_mode(args: argparse.Namespace) -> int:
    """Run the three-arm intervention (API, ~8.3h)."""
    plan = json.loads(Path(args.plan).read_text())
    candidates = plan["candidates"]
    if args.shard_count > 0 and args.shard_index >= 0:
        # round-robin the shards by project to keep repository diversity and load balance across shards (same as attr)
        by_proj = {}
        for c in candidates:
            by_proj.setdefault(c["instance_id"].split("__")[0], []).append(c)
        ordered = []
        pools = list(by_proj.values())
        while any(pools):
            for pl in pools:
                if pl:
                    ordered.append(pl.pop(0))
        candidates = [c for i, c in enumerate(ordered)
                      if i % args.shard_count == args.shard_index]
        print(f"[shard {args.shard_index}/{args.shard_count}] assigned {len(candidates)} instances")
    if args.limit > 0:
        candidates = candidates[:args.limit]
    print(f"will process {len(candidates)} instances (limit={args.limit or 'none'})")

    out_path = Path(args.out)
    workroot = Path(args.workdir)
    workroot.mkdir(parents=True, exist_ok=True)
    codex_home = workroot / "codex_home"
    write_codex_config(codex_home, args.model)

    # pre-check: a missing API key makes every fork fail, and failures used to be silently recorded as recall=0
    import os
    if not os.environ.get("IDEALAB_API_KEY"):
        print("❌ Environment variable IDEALAB_API_KEY is not set. Every X3 fork will fail, "
              "and failures used to be silently recorded as recall=0 (disguised as an "
              "intervention effect). Export it first, then rerun.")
        return 2

    insts = {i.instance_id: i for i in T.load_instances()}
    srv = AppServer(
        codex_bin=args.codex_bin, codex_home=codex_home,
        log_path=workroot / "x3_appserver.log", experimental_api=True,
    )
    results = []
    n = 0
    t0 = time.time()
    try:
        for c in candidates:
            iid = c["instance_id"]
            inst = insts.get(iid)
            if inst is None:
                continue
            rollout_path = Path(c["rollout_path"])
            if not rollout_path.exists():
                print(f"[SKIP] {iid}: rollout not found at {rollout_path}")
                continue
            try:
                base_items = R.load(rollout_path)
            except Exception as exc:
                print(f"[SKIP] {iid}: load error {exc}")
                continue

            t_star = c["first_gold_t"]
            gold_paths = c["gold"]
            wd = workroot / iid

            rec = {
                "instance_id": iid,
                "first_gold_t": t_star,
                "n_gold": c["n_gold"],
                "gold": gold_paths,
            }
            arms = {}
            for arm in ("control", "mask", "placebo"):
                print(f"  [{n+1}/{len(candidates)}] {iid} arm={arm} t*={t_star}...",
                      flush=True)
                try:
                    res = _estimate_arm(
                        srv, base_items, inst, wd,
                        args.model, args.model_provider,
                        t_star, arm, gold_paths,
                        args.m, args.turn_timeout,
                    )
                except Exception as exc:
                    res = {"arm": arm, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                           "trace": traceback.format_exc()[-300:]}
                arms[arm] = res
            rec["arms"] = arms
            # fail-fast: if any arm of the first instance is entirely invalid, the
            # environment/config is broken, so abort immediately. Otherwise 12h later we get
            # all-zero data, and it **looks exactly like branch (ii) "information redundancy"**
            # -- a misconfiguration disguised as a scientific conclusion.
            if n == 0:
                bad = [a for a, r in arms.items() if not r.get("n_valid")]
                if bad:
                    print(f"❌ FAIL-FAST: arms {bad} of the first instance replayed entirely invalid.")
                    for a in bad:
                        for e in (arms[a].get("errors") or [])[:3]:
                            print(f"     {a}: {e}")
                    print("   Aborted, no budget wasted. Fix it and rerun.")
                    return 3

            # compute PSE and the placebo effect
            c_v = arms.get("control", {}).get("v_hat", float("nan"))
            m_v = arms.get("mask", {}).get("v_hat", float("nan"))
            p_v = arms.get("placebo", {}).get("v_hat", float("nan"))
            rec["PSE"] = round(c_v - m_v, 4) if not math.isnan(c_v - m_v) else None
            rec["placebo_effect"] = round(c_v - p_v, 4) if not math.isnan(c_v - p_v) else None
            rec["control_v"] = round(c_v, 4) if not math.isnan(c_v) else None
            rec["mask_v"] = round(m_v, 4) if not math.isnan(m_v) else None
            rec["placebo_v"] = round(p_v, 4) if not math.isnan(p_v) else None

            results.append(rec)
            n += 1
            print(f"  → PSE={rec['PSE']} placebo_eff={rec['placebo_effect']} "
                  f"(control={rec['control_v']}, mask={rec['mask_v']}, placebo={rec['placebo_v']})",
                  flush=True)

            # append-write (recoverable from an interruption partway through)
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finally:
        srv.close()

    elapsed = (time.time() - t0) / 60
    print(f"\ncompleted {n} X3 interventions in {elapsed:.1f}min → {out_path}")
    return 0


def analyze_mode(args: argparse.Namespace) -> int:
    """Analyze the X3 results (zero API)."""
    out_path = Path(args.out)
    if not out_path.exists():
        print(f"⚠️ result file does not exist: {out_path}")
        return 1

    recs = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    if not recs:
        print("⚠️ no records")
        return 1

    print(f"analyzing {len(recs)} X3 results")

    # PSE and the placebo effect
    pse_vals = [r["PSE"] for r in recs if r.get("PSE") is not None]
    plac_vals = [r["placebo_effect"] for r in recs if r.get("placebo_effect") is not None]
    control_vs = [r["control_v"] for r in recs if r.get("control_v") is not None]
    mask_vs = [r["mask_v"] for r in recs if r.get("mask_v") is not None]

    pse_ci = boot_ci(pse_vals) if pse_vals else {"note": "no samples"}
    plac_ci = boot_ci(plac_vals) if plac_vals else {"note": "no samples"}

    # paired comparison (within instance)
    pse_paired = paired_ci(
        [r["control_v"] for r in recs if r.get("control_v") is not None and r.get("mask_v") is not None],
        [r["mask_v"] for r in recs if r.get("control_v") is not None and r.get("mask_v") is not None],
    )
    plac_paired = paired_ci(
        [r["control_v"] for r in recs if r.get("control_v") is not None and r.get("placebo_v") is not None],
        [r["placebo_v"] for r in recs if r.get("control_v") is not None and r.get("placebo_v") is not None],
    )

    # three-way criterion
    pse_sig = pse_paired.get("significant", False)
    plac_sig = plac_paired.get("significant", False)
    pse_dir = (pse_paired.get("diff", 0.0) or 0.0) > 0   # control > mask

    if pse_sig and not plac_sig and pse_dir:
        verdict = "(i) STRONG_POSITIVE: the provenance edge is causally effective, M≪C and P≈C"
    elif not pse_sig and not plac_sig:
        verdict = "(ii) REDUNDANCY: information redundancy M≈C, a dose-response experiment is advised"
    elif plac_sig:
        verdict = "(iii) OPERATOR_DISTURBANCE: P drops too, fix the mask operator and rerun"
    else:
        verdict = "AMBIGUOUS: only some conditions met, check the detailed numbers"

    # dose-response stratification (by n_gold)
    by_ngold: dict[int, list[float]] = {}
    for r in recs:
        ng = r.get("n_gold", 0)
        if r.get("PSE") is not None:
            by_ngold.setdefault(ng, []).append(r["PSE"])
    dose_response = {str(ng): boot_ci(vals) for ng, vals in sorted(by_ngold.items())}

    analysis = {
        "n_instances": len(recs),
        "PSE": {
            "mean_PSE": pse_ci,
            "paired_control_minus_mask": pse_paired,
        },
        "PLACEBO": {
            "mean_placebo_effect": plac_ci,
            "paired_control_minus_placebo": plac_paired,
        },
        "VERDICT": verdict,
        "DOSE_RESPONSE_BY_N_GOLD": dose_response,
        "error_rate": round(sum(1 for r in recs
                               if any("error" in (r.get("arms") or {}).get(a, {})
                                      for a in ("control", "mask", "placebo")))
                            / max(1, len(recs)), 3),
    }

    # write the analysis file (append the suffix _analysis.json)
    ana_path = out_path.with_suffix("") .parent / (out_path.stem + "_analysis.json")
    ana_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n=== X3: provenance-mask intervention analysis ===")
    print(f"n={len(recs)}")
    if "note" not in pse_paired:
        pse_d = pse_paired.get("diff", "?")
        pse_c = pse_paired.get("ci95", "?")
        sig_str = "SIGNIFICANT" if pse_sig else "n.s."
        print(f"  PSE (control-mask): {pse_d:+.4f} CI{pse_c} [{sig_str}]")
    if "note" not in plac_paired:
        plac_d = plac_paired.get("diff", "?")
        plac_c = plac_paired.get("ci95", "?")
        sig_str = "SIGNIFICANT" if plac_sig else "n.s."
        print(f"  placebo (control-placebo): {plac_d:+.4f} CI{plac_c} [{sig_str}]")
    print(f"\ncriterion: {verdict}")
    print(f"\n→ {ana_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plan", "run", "analyze"], default="plan")
    ap.add_argument("--runs-root", default="pilot/runs/full_locate",
                    help="first rollout source (full localization results)")
    ap.add_argument("--runs-root2", default=None,
                    help="second rollout source (optional, extra coverage)")
    ap.add_argument("--plan", default="pilot/runs/x3_plan.json",
                    help="intervention plan path (produced by plan mode, read by run mode)")
    ap.add_argument("--out", default="pilot/runs/x3_pse.json",
                    help="output path for run mode / input path for analyze mode")
    ap.add_argument("--workdir", default="pilot/runs/x3_work")
    ap.add_argument("--model", default="gpt-5.5-0424-global")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--m", type=int, default=M_FORKS)
    ap.add_argument("--turn-timeout", type=float, default=900.0)
    ap.add_argument("--shard-index", type=int, default=-1,
                    help="shard index (0-based). 1680 forks serially is ≈70h, so sharding in parallel is required.")
    ap.add_argument("--shard-count", type=int, default=0,
                    help="total number of shards. Each shard uses its own --workdir → its own CODEX_HOME+AppServer")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N instances (for debugging, 0=all)")
    args = ap.parse_args()

    if args.mode == "plan":
        return plan_mode(args)
    elif args.mode == "run":
        return run_mode(args)
    elif args.mode == "analyze":
        return analyze_mode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
