#!/usr/bin/env python3
"""S0 Smoke: before investing in any formal experiment, settle five unknowns that decide whether the method is feasible.

    [V1] Can the experimental gate be opened? Is thread.path obtainable? Does fork by path work?
    [V2] Can the tool surface be narrowed? (measure which tools the agent actually used, and the violation rate)
    [V3] What is the minimal set of item types that must be kept to fork a "hand-built/truncated rollout"?
    [V4] After a fork, does advancing with CONTINUE_PROMPT "carry on" or "restart the task"?
    [V5] Forking the same prefix repeatedly, is the entropy of the action distribution > 0?   ← the highest-risk one

[V5] If the entropy ≈ 0, positivity fails outright, the policy baseline term of Theorem 1 is identically 0,
and the whole estimation strategy must switch to soft interventions (turn/steer or instruction perturbation)
as the alternative action source. So V5 must be answered first.

Usage
-----
    python pilot/smoke/smoke.py --instance astropy__astropy-12907 \\
        --model <model> --out pilot/runs/smoke --v5-samples 8

Output
------
    <out>/<instance>/report.json        structured conclusions (PASS/FAIL per item)
    <out>/<instance>/base.rollout.jsonl copy of the raw rollout
    <out>/<instance>/forks/*.jsonl      truncated files used by each ablation
    <out>/<instance>/appserver.log      full JSON-RPC send/receive log
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                      # noqa: E402
from pilot.harness.appserver import AppServer, AppServerError, thread_rollout_path  # noqa: E402
from pilot.probes import task as T                          # noqa: E402

# Allowed tool names (V2's criterion). Anything not matching is recorded as a violation.
ALLOWED_TOOL_HINTS = ("read", "search", "grep", "glob", "list", "find", "view", "answer")
# Explicitly counted as violations (they mutate repository state or go beyond read-only localization)
FORBIDDEN_TOOL_HINTS = ("apply_patch", "write", "edit", "update_plan_apply", "patch")


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c)


def normalized_entropy(counts: Counter) -> float:
    k = len(counts)
    return entropy(counts) / math.log(k) if k > 1 else 0.0


# ---------------------------------------------------------------------------
# V3's ablation plan: drop item types progressively to find the minimal set fork needs
# ---------------------------------------------------------------------------
ABLATIONS: list[tuple[str, set[str] | None]] = [
    ("keep_all", None),
    ("drop_event_msg", {"event_msg"}),
    ("drop_world_state", {"world_state"}),
    ("drop_event_and_world", {"event_msg", "world_state"}),
    ("drop_turn_context", {"turn_context"}),
    ("only_meta_and_response", {"event_msg", "world_state", "turn_context",
                                "inter_agent_communication",
                                "inter_agent_communication_metadata"}),
]


def apply_ablation(items: list[dict], drop: set[str] | None) -> list[dict]:
    if not drop:
        return items
    return [it for it in items if R.line_type(it) not in drop]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--model", default=None, help="leave empty to use the codex default model")
    ap.add_argument("--model-provider", default=None, help="e.g. idealab")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--out", default="pilot/runs/smoke")
    ap.add_argument("--v5-samples", type=int, default=8)
    ap.add_argument("--fork-step", type=int, default=None,
                    help="truncate before which step; defaults to the trace midpoint")
    ap.add_argument("--turn-timeout", type=float, default=1200.0)
    ap.add_argument("--skip-v5", action="store_true")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    if args.instance not in insts:
        print(f"unknown instance: {args.instance}", file=sys.stderr)
        return 2
    inst = insts[args.instance]

    outdir = Path(args.out) / inst.instance_id
    (outdir / "forks").mkdir(parents=True, exist_ok=True)
    codex_home = outdir / "codex_home"

    report: dict = {
        "instance": inst.instance_id,
        "project": inst.project,
        "gold": sorted(inst.gold),
        "n_gold": len(inst.gold),
        "model": args.model,
        "checks": {},
    }

    # ---------------------------------------------------------------- preflight checks
    # The dataset worktree has the gold patch applied; before running we must reset back to a clean base_commit (buggy state)
    repo_state = T.reset_repo(inst)
    report["repo"] = repo_state
    if not repo_state["ok"]:
        report["checks"]["PRE_repo_clean"] = {"pass": False, "detail": repo_state["reason"]}
        _save(report, outdir)
        print(f"FAIL preflight check (still not clean after reset): {repo_state['reason']}", file=sys.stderr)
        return 1
    report["checks"]["PRE_repo_clean"] = {"pass": True, "detail": "reset to base_commit succeeded"}
    report["repo_n_py"] = T.n_py_files(inst)

    t0 = time.time()
    with AppServer(
        codex_bin=args.codex_bin,
        codex_home=codex_home,
        log_path=outdir / "appserver.log",
        experimental_api=True,
    ) as srv:
        report["server_info"] = srv.server_info

        # ------------------------------------------------------ V1a: start + path
        started = srv.thread_start(
            cwd=inst.repo,
            model=args.model, model_provider=args.model_provider,
            sandbox="read-only",
            approval_policy="never",
            base_instructions=T.BASE_INSTRUCTIONS,
        )
        base_thread = started.get("thread", {})
        base_tid = base_thread.get("id")
        base_path = thread_rollout_path(started)
        if base_path is None:
            base_path = _scan_rollout(codex_home, base_tid)
        report["checks"]["V1a_thread_path_available"] = {
            "pass": base_path is not None,
            "detail": base_path or "Thread.path missing and the sessions/ scan found nothing",
        }
        if base_path is None:
            _save(report, outdir)
            return 1

        # ------------------------------------------------- run the full task (V2 data source)
        run = srv.turn_run(base_tid, T.build_prompt(inst), timeout=args.turn_timeout)
        report["base_run"] = {
            "n_items": len(run["items"]),
            "interrupted": run["interrupted"],
            "wall_s": round(time.time() - t0, 1),
        }
        answer = T.parse_answer(T.final_message_text(run["items"]))
        report["base_run"]["outcome"] = T.outcome(answer, inst.gold)
        report["base_run"]["answer"] = sorted(answer) if answer else None

        shutil.copy(base_path, outdir / "base.rollout.jsonl")
        items = R.load(base_path)
        st = R.steps(items)
        report["base_run"]["n_steps"] = len(st)
        report["base_run"]["compaction_lines"] = R.compaction_lines(items)
        report["base_run"]["line_types"] = dict(Counter(R.line_type(i) for i in items))

        # --------------------------------------------------------------- V2
        tools = Counter(s.tool for s in st)
        violations = {
            t: c for t, c in tools.items()
            if any(f in t.lower() for f in FORBIDDEN_TOOL_HINTS)
        }
        unknown = {
            t: c for t, c in tools.items()
            if not any(a in t.lower() for a in ALLOWED_TOOL_HINTS) and t not in violations
        }
        report["checks"]["V2_tool_surface"] = {
            "pass": not violations,
            "tools": dict(tools),
            "forbidden_used": violations,
            "unclassified": unknown,
            "action_buckets": dict(Counter(
                f"{a}#{b}" for a, b in (R.action_bucket(s, inst.repo) for s in st)
            )),
        }
        # Server-initiated requests should be empty: non-empty means approvalPolicy/sandbox did not take effect
        report["checks"]["V2b_no_server_requests"] = {
            "pass": not srv.server_requests,
            "n": len(srv.server_requests),
            "methods": [m.get("method") for m in srv.server_requests[:5]],
        }

        if len(st) < 3:
            report["checks"]["V3_min_item_set"] = {
                "pass": False, "detail": f"too few steps ({len(st)}), cannot truncate a prefix"}
            _save(report, outdir)
            return 1

        k = args.fork_step or max(2, len(st) // 2)
        report["fork_step"] = k

        # --------------------------------------------------------------- V3
        prefix = R.truncate_before_step(items, k)
        R.assert_no_dangling_calls(prefix)
        v3: dict = {"variants": {}}
        good_variant: str | None = None
        good_file: Path | None = None
        for name, drop in ABLATIONS:
            trimmed = apply_ablation(prefix, drop)
            f = R.dump(trimmed, outdir / "forks" / f"{name}.jsonl")
            try:
                forked = srv.thread_fork(
                    path=f, cwd=inst.repo, model=args.model, model_provider=args.model_provider,
                    sandbox="read-only", approval_policy="never",
                    base_instructions=T.BASE_INSTRUCTIONS, exclude_turns=True,
                )
                ok = bool(forked.get("thread", {}).get("id"))
                v3["variants"][name] = {"pass": ok, "n_lines": len(trimmed)}
                if ok and good_variant is None:
                    good_variant, good_file = name, f
            except AppServerError as exc:
                v3["variants"][name] = {"pass": False, "n_lines": len(trimmed),
                                        "error": str(exc)[:400]}
        v3["pass"] = good_variant is not None
        v3["first_working_variant"] = good_variant
        # minimal set = the most aggressive ablation that passes
        v3["minimal_working_variant"] = next(
            (n for n, _ in reversed(ABLATIONS) if v3["variants"].get(n, {}).get("pass")), None
        )
        report["checks"]["V3_min_item_set"] = v3
        if good_file is None:
            _save(report, outdir)
            return 1

        # --------------------------------------------------------------- V4
        f4 = srv.thread_fork(
            path=good_file, cwd=inst.repo, model=args.model, model_provider=args.model_provider,
            sandbox="read-only", approval_policy="never",
            base_instructions=T.BASE_INSTRUCTIONS,
        )
        tid4 = f4["thread"]["id"]
        run4 = srv.turn_run(tid4, T.CONTINUE_PROMPT, timeout=args.turn_timeout)
        p4 = thread_rollout_path(f4) or _scan_rollout(codex_home, tid4)
        cont_items = R.load(p4) if p4 else []
        cont_steps = R.steps(cont_items)
        # Criterion: total steps after continuation > prefix steps (it really moved forward),
        # and the action buckets of the first k-1 steps match the original trace (no task restart)
        orig_buckets = [R.action_bucket(s, inst.repo) for s in st[: k - 1]]
        cont_buckets = [R.action_bucket(s, inst.repo) for s in cont_steps[: k - 1]]
        ans4 = T.parse_answer(T.final_message_text(run4["items"]))
        report["checks"]["V4_continue_semantics"] = {
            "pass": len(cont_steps) >= k and orig_buckets == cont_buckets,
            "n_steps_prefix": k - 1,
            "n_steps_after_continue": len(cont_steps),
            "prefix_preserved": orig_buckets == cont_buckets,
            "produced_answer": ans4 is not None,
            "outcome": T.outcome(ans4, inst.gold),
        }

        # --------------------------------------------------------------- V5
        if args.skip_v5:
            report["checks"]["V5_action_entropy"] = {"pass": None, "detail": "skipped"}
        else:
            first_actions: list[str] = []
            errors: list[str] = []
            for i in range(args.v5_samples):
                try:
                    fk = srv.thread_fork(
                        path=good_file, cwd=inst.repo, model=args.model, model_provider=args.model_provider,
                        sandbox="read-only", approval_policy="never",
                        base_instructions=T.BASE_INSTRUCTIONS,
                    )
                    tid = fk["thread"]["id"]
                    seen: list[dict] = []

                    def _stop(p: dict, _seen: list[dict] = seen) -> bool:
                        """Interrupt on the first tool call -- we only need to sample A_k, not run to the end."""
                        it = p.get("item", p)
                        kind = str(it.get("itemType") or it.get("type") or "")
                        if "ommand" in kind or "oolCall" in kind or "tool_call" in kind:
                            _seen.append(it)
                            return True
                        return False

                    srv.turn_run(tid, T.CONTINUE_PROMPT, timeout=args.turn_timeout,
                                 on_item=_stop)
                    pth = thread_rollout_path(fk) or _scan_rollout(codex_home, tid)
                    sample_steps = R.steps(R.load(pth)) if pth else []
                    if len(sample_steps) >= k:
                        a, b = R.action_bucket(sample_steps[k - 1], inst.repo)
                        first_actions.append(f"{a}#{b}")
                    else:
                        errors.append(f"sample{i}: continuation produced no step {k}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"sample{i}: {type(exc).__name__}: {str(exc)[:200]}")

            counts = Counter(first_actions)
            H = entropy(counts)
            report["checks"]["V5_action_entropy"] = {
                # Criterion: at least 2 distinct actions. Zero entropy → positivity fails → the method must change
                "pass": len(counts) >= 2,
                "n_samples": len(first_actions),
                "n_distinct": len(counts),
                "entropy_nats": round(H, 4),
                "normalized_entropy": round(normalized_entropy(counts), 4),
                "distribution": dict(counts),
                "errors": errors,
                "note": ("zero entropy means the policy is near-deterministic on this prefix: positivity fails "
                         "outright, the policy baseline term of Δ_t is identically 0, and soft interventions "
                         "must be used as the alternative action source"),
            }

    report["wall_s_total"] = round(time.time() - t0, 1)
    report["verdict"] = _verdict(report)
    _save(report, outdir)
    _print(report)
    return 0 if report["verdict"]["go"] else 1


def _verdict(report: dict) -> dict:
    checks = report["checks"]
    hard = ["PRE_repo_clean", "V1a_thread_path_available", "V3_min_item_set",
            "V4_continue_semantics"]
    failed_hard = [k for k in hard if checks.get(k, {}).get("pass") is False]
    v5 = checks.get("V5_action_entropy", {}).get("pass")
    return {
        "go": not failed_hard,
        "failed_hard_checks": failed_hard,
        "v5_entropy_ok": v5,
        "next": (
            "blocked: first fix " + ", ".join(failed_hard) if failed_hard
            else 'V5 entropy is 0 → switch to the soft-intervention route; the estimand must be rewritten as "the effect relative to a perturbed policy"'
            if v5 is False
            else "all passed → proceed to S1 (75 tasks × K=4)"
        ),
    }


def _scan_rollout(codex_home: Path, thread_id: str | None) -> str | None:
    """Fallback when Thread.path is unavailable: search under CODEX_HOME/sessions by thread id."""
    if not thread_id:
        return None
    root = Path(codex_home) / "sessions"
    if not root.is_dir():
        return None
    cands = [p for p in root.rglob("*.jsonl") if thread_id in p.name]
    if not cands:
        cands = sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)[-1:]
    return str(cands[-1]) if cands else None


def _save(report: dict, outdir: Path) -> None:
    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _print(report: dict) -> None:
    print(f"\n=== S0 Smoke: {report['instance']} ===")
    for name, c in report["checks"].items():
        flag = {True: "PASS", False: "FAIL", None: "SKIP"}[c.get("pass")]
        print(f"  [{flag}] {name}")
    v5 = report["checks"].get("V5_action_entropy", {})
    if v5.get("n_samples"):
        print(f"\n  V5 action distribution (n={v5['n_samples']}, distinct={v5['n_distinct']}, "
              f"H={v5['entropy_nats']} nats):")
        for a, c in sorted(v5["distribution"].items(), key=lambda kv: -kv[1]):
            print(f"      {c:>3}x  {a}")
    v3 = report["checks"].get("V3_min_item_set", {})
    if v3.get("minimal_working_variant"):
        print(f"\n  V3 minimal forkable item set: {v3['minimal_working_variant']}")
    print(f"\n  verdict: {report['verdict']['next']}\n")


if __name__ == "__main__":
    raise SystemExit(main())
