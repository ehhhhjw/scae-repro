"""Batch file-localization runner -- the core of S1.

Runs file localization over all 500 processed SWE-bench instances:
  reset repo → thread/start → turn(problem_statement) → parse <answer> from the rollout → decide Y

Each task can be run K times (different randomness), for estimating U_policy and avg@K.

Design
------
- **per-task reset**: the dataset workspace has the gold patch applied, so every task is forced back to base_commit before running.
- **answer parsed from the rollout** (answer_from_rollout), not relying on the notification field structure.
- **resumable**: completed (instance, k) pairs are recorded in results.jsonl and skipped automatically on restart.
- **per-task CODEX_HOME**: isolates rollouts to avoid cross-contamination; includes the idealab config.toml.
- **concurrency**: one AppServer subprocess per worker. Serial by default (concurrency amplifies API pressure; stay stable in the pilot phase).
- **persistence**: results.jsonl holds one (instance, k) result per line plus the rollout copy path, for reuse by later attribution.

Usage
-----
    python -m pilot.harness.runner \\
        --model gpt-5.5-0424-global --model-provider idealab \\
        --k 1 --limit 5 --out pilot/runs/s1
    # full run: --limit 0 (or omit it)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                                   # noqa: E402
from pilot.harness.appserver import AppServer, thread_rollout_path       # noqa: E402
from pilot.probes import task as T                                       # noqa: E402

IDEALAB_CONFIG = """\
model = "{model}"
model_provider = "idealab"

[model_providers.idealab]
name = "Idealab AI"
base_url = "https://idealab.alibaba-inc.com/api/openai/v1"
env_key = "IDEALAB_API_KEY"
wire_api = "{wire_api}"
"""


def write_codex_config(codex_home: Path, model: str,
                       wire_api: str = "responses") -> None:
    """Write the codex config.

    wire_api defaults to "responses" (used by the gpt-5.5 family), but on that endpoint the qwen family
    only supports "chat": with responses the stream disconnects mid-flight, showing up as
    `stream disconnected before completion`, and the turn ends with status=failed.
    Measured: /chat/completions returns 200 for qwen3.6-plus, /responses is unusable.
    """
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "config.toml").write_text(
        IDEALAB_CONFIG.format(model=model, wire_api=wire_api), encoding="utf-8")


def done_keys(results_path: Path) -> set[tuple[str, int]]:
    """Completed (instance_id, k) pairs, for resuming."""
    done: set[tuple[str, int]] = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((rec["instance_id"], rec["k"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_one(
    srv: AppServer,
    inst: T.Instance,
    k: int,
    outdir: Path,
    model: str | None,
    model_provider: str | None,
    turn_timeout: float,
) -> dict:
    """Run a single (instance, k). Returns a structured result record."""
    t0 = time.time()
    reset = T.reset_repo(inst)
    if not reset["ok"]:
        return {
            "instance_id": inst.instance_id, "k": k, "ok": False,
            "error": f"reset failed: {reset['reason']}", "wall_s": round(time.time() - t0, 1),
        }

    started = srv.thread_start(
        cwd=inst.repo, model=model, model_provider=model_provider,
        sandbox="read-only", approval_policy="never",
        base_instructions=T.BASE_INSTRUCTIONS, timeout=120,
    )
    tid = started["thread"]["id"]
    rpath = thread_rollout_path(started)

    run = srv.turn_run(tid, T.build_prompt(inst), timeout=turn_timeout)

    # prefer parsing from the rollout file (reliable), fall back to the notification text
    items: list[dict] = []
    if rpath and Path(rpath).exists():
        items = R.load(rpath)
        ans = T.answer_from_rollout(items)
    else:
        ans = T.parse_answer(T.final_message_text(run["items"]))

    steps = R.steps(items) if items else []
    oc = T.outcome(ans, inst.gold)

    # save a rollout copy for reuse by later attribution
    saved = ""
    if rpath and Path(rpath).exists():
        dst = outdir / "rollouts" / f"{inst.instance_id}__k{k}.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rpath, dst)
        saved = str(dst.relative_to(outdir))

    tools = {}
    for s in steps:
        tools[s.tool] = tools.get(s.tool, 0) + 1

    return {
        "instance_id": inst.instance_id,
        "project": inst.project,
        "k": k,
        "ok": True,
        "Y": oc["Y"],
        "f1": round(oc["f1"], 4),
        "precision": round(oc["precision"], 4),
        "recall": round(oc["recall"], 4),
        "parsed": oc["parsed"],
        "n_gold": len(inst.gold),
        "gold": sorted(inst.gold),
        "answer": sorted(ans) if ans else None,
        "n_steps": len(steps),
        "tools": tools,
        "compaction": len(R.compaction_lines(items)) if items else 0,
        "interrupted": run["interrupted"],
        "rollout": saved,
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5-0424-global")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--k", type=int, default=1, help="repetitions per task")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N (0=all 500)")
    ap.add_argument("--instances", nargs="*", default=None, help="explicit list of instance_ids")
    ap.add_argument("--out", default="pilot/runs/s1")
    ap.add_argument("--turn-timeout", type=float, default=900.0)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    if not os.environ.get("IDEALAB_API_KEY"):
        print("warning: IDEALAB_API_KEY is not set", file=sys.stderr)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"
    codex_home = outdir / "codex_home"
    write_codex_config(codex_home, args.model)

    all_insts = T.load_instances()
    if args.instances:
        want = set(args.instances)
        insts = [i for i in all_insts if i.instance_id in want]
    else:
        insts = all_insts
        if args.limit > 0:
            insts = insts[: args.limit]

    done = done_keys(results_path)
    todo = [(i, k) for i in insts for k in range(args.k) if (i.instance_id, k) not in done]
    print(f"instances {len(insts)} × K={args.k} = {len(insts) * args.k} runs; done {len(done)}; todo {len(todo)}")

    srv = AppServer(
        codex_bin=args.codex_bin, codex_home=codex_home,
        log_path=outdir / "appserver.log", experimental_api=True,
    )
    n_done = 0
    n_ok = 0
    n_y = 0
    t_start = time.time()
    try:
        with open(results_path, "a", encoding="utf-8") as fh:
            for inst, k in todo:
                try:
                    rec = run_one(srv, inst, k, outdir, args.model,
                                  args.model_provider, args.turn_timeout)
                except Exception as exc:  # noqa: BLE001
                    rec = {
                        "instance_id": inst.instance_id, "k": k, "ok": False,
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "trace": traceback.format_exc()[-800:],
                    }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                if rec.get("ok"):
                    n_ok += 1
                    n_y += rec.get("Y", 0)
                if n_done % args.progress_every == 0 or n_done == len(todo):
                    rate = n_y / n_ok if n_ok else 0
                    eta = (time.time() - t_start) / n_done * (len(todo) - n_done)
                    print(f"[{n_done}/{len(todo)}] ok={n_ok} localization_rate={rate:.2%} "
                          f"ETA={eta/60:.1f}min", flush=True)
    finally:
        srv.close()

    # summary
    _summarize(results_path, outdir)
    return 0


def _summarize(results_path: Path, outdir: Path) -> None:
    from collections import defaultdict
    recs = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    ok = [r for r in recs if r.get("ok")]
    by_proj: dict[str, list] = defaultdict(list)
    for r in ok:
        by_proj[r.get("project", "?")].append(r)

    summary = {
        "n_total": len(recs),
        "n_ok": len(ok),
        "n_error": len(recs) - len(ok),
        "solve_rate": round(sum(r["Y"] for r in ok) / len(ok), 4) if ok else None,
        "parse_rate": round(sum(r["parsed"] for r in ok) / len(ok), 4) if ok else None,
        "mean_steps": round(sum(r["n_steps"] for r in ok) / len(ok), 2) if ok else None,
        "mean_wall_s": round(sum(r["wall_s"] for r in ok) / len(ok), 1) if ok else None,
        "n_with_compaction": sum(1 for r in ok if r.get("compaction", 0) > 0),
        "by_project": {
            p: {"n": len(rs), "solve_rate": round(sum(r["Y"] for r in rs) / len(rs), 4)}
            for p, rs in sorted(by_proj.items())
        },
        "by_n_gold": {},
    }
    from collections import Counter
    ng = defaultdict(list)
    for r in ok:
        ng[r["n_gold"]].append(r["Y"])
    summary["by_n_gold"] = {
        str(k): {"n": len(v), "solve_rate": round(sum(v) / len(v), 4)}
        for k, v in sorted(ng.items())
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== summary ===")
    print(f"done {summary['n_ok']}/{summary['n_total']}  errors {summary['n_error']}")
    print(f"localization rate (Y=1): {summary['solve_rate']}")
    print(f"answer parse rate: {summary['parse_rate']}")
    print(f"mean steps: {summary['mean_steps']}  mean wall clock: {summary['mean_wall_s']}s")
    print(f"runs with compaction events: {summary['n_with_compaction']}")
    print("solve rate by repo:")
    for p, s in summary["by_project"].items():
        print(f"  {p:<14} n={s['n']:<4} solve={s['solve_rate']}")


if __name__ == "__main__":
    raise SystemExit(main())
