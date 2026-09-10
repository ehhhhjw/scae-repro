#!/usr/bin/env python3
"""V5 standalone probe: action entropy of repeated forks from the same prefix.

This is the single point of failure of the whole method. Given an existing rollout, truncate it
before step k, then repeatedly fork + continue, sampling only "the first action A_k produced by
the continuation", and measure the entropy of its distribution.

Criterion
---------
  H > 0 (≥2 distinct actions) → the policy is stochastic on that prefix → fork-resample can estimate v_t
                              → the estimation path for C2 attribution / C4 prediction holds
  H ≈ 0 (only 1 action)       → positivity fails → the policy baseline term of Δ_t is identically 0
                              → must switch to soft intervention (turn/steer), estimand rewritten as "effect relative to a perturbed policy"

Why do it separately
--------------------
smoke.py's V5 requires running the full task first. This one eats an existing rollout directly, can
scan many (repository, k) pairs quickly, and reuses one AppServer subprocess to cut handshake cost.

Usage
-----
    python -m pilot.smoke.probe_v5 \\
        --rollout pilot/runs/smoke/astropy__astropy-12907/base.rollout.jsonl \\
        --instance astropy__astropy-12907 \\
        --model gpt-5.5-0424-global --model-provider idealab \\
        --fork-steps 2 4 6 --samples 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                                   # noqa: E402
from pilot.harness.appserver import AppServer, thread_rollout_path       # noqa: E402
from pilot.probes import task as T                                       # noqa: E402


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c) if n else 0.0


def norm_entropy(counts: Counter) -> float:
    k = len(counts)
    return entropy(counts) / math.log(k) if k > 1 else 0.0


def first_action_after(items_after: list[dict], k: int, repo: Path) -> str | None:
    """Action bucket of step k in the continuation trace (steps 1..k-1 are the pinned prefix)."""
    st = R.steps(items_after)
    if len(st) < k:
        return None
    a, b = R.action_bucket(st[k - 1], repo)
    return f"{a}#{b}"


def probe_one_step(
    srv: AppServer, base_items: list[dict], k: int, inst: T.Instance,
    outdir: Path, model: str | None, provider: str | None,
    samples: int, turn_timeout: float,
) -> dict:
    prefix = R.truncate_before_step(base_items, k)
    R.assert_no_dangling_calls(prefix)
    fork_file = outdir / f"prefix_k{k}.jsonl"
    R.dump(prefix, fork_file)

    orig_first = None
    st0 = R.steps(base_items)
    if len(st0) >= k:
        a, b = R.action_bucket(st0[k - 1], inst.repo)
        orig_first = f"{a}#{b}"

    actions: list[str] = []
    errors: list[str] = []
    for i in range(samples):
        T.reset_repo(inst)                       # back to a clean state before each continuation
        try:
            fk = srv.thread_fork(
                path=fork_file, cwd=inst.repo, model=model, model_provider=provider,
                sandbox="read-only", approval_policy="never",
                base_instructions=T.BASE_INSTRUCTIONS,
            )
            tid = fk["thread"]["id"]

            captured: dict = {"hit": False}

            def _stop(p: dict, _c=captured) -> bool:
                it = p.get("item", p)
                kind = str(it.get("itemType") or it.get("type") or "")
                if "ommand" in kind or "oolCall" in kind or "tool_call" in kind or "unction" in kind:
                    _c["hit"] = True
                    return True
                return False

            srv.turn_run(tid, T.CONTINUE_PROMPT, timeout=turn_timeout, on_item=_stop)
            pth = thread_rollout_path(fk)
            fa = first_action_after(R.load(pth), k, inst.repo) if pth and Path(pth).exists() else None
            if fa:
                actions.append(fa)
            else:
                errors.append(f"s{i}: did not sample the step-{k} action")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"s{i}: {type(exc).__name__}: {str(exc)[:150]}")

    counts = Counter(actions)
    return {
        "fork_step": k,
        "orig_first_action": orig_first,
        "n_samples": len(actions),
        "n_distinct": len(counts),
        "entropy_nats": round(entropy(counts), 4),
        "normalized_entropy": round(norm_entropy(counts), 4),
        "distribution": dict(counts),
        "positivity_ok": len(counts) >= 2,
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--model", default="gpt-5.5-0424-global")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--fork-steps", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default="pilot/runs/v5")
    ap.add_argument("--turn-timeout", type=float, default=600.0)
    args = ap.parse_args()

    inst = {i.instance_id: i for i in T.load_instances()}[args.instance]
    base_items = R.load(args.rollout)
    n_steps = len(R.steps(base_items))
    steps_to_probe = [k for k in args.fork_steps if 2 <= k <= n_steps]

    outdir = Path(args.out) / inst.instance_id
    outdir.mkdir(parents=True, exist_ok=True)

    from pilot.harness.runner import write_codex_config
    codex_home = outdir / "codex_home"
    write_codex_config(codex_home, args.model)

    srv = AppServer(codex_bin=args.codex_bin, codex_home=codex_home,
                    log_path=outdir / "appserver.log", experimental_api=True)
    report = {"instance": inst.instance_id, "rollout_n_steps": n_steps,
              "samples_per_step": args.samples, "per_step": []}
    t0 = time.time()
    try:
        for k in steps_to_probe:
            res = probe_one_step(srv, base_items, k, inst, outdir,
                                 args.model, args.model_provider,
                                 args.samples, args.turn_timeout)
            report["per_step"].append(res)
            print(f"[k={k}] distinct={res['n_distinct']}/{res['n_samples']} "
                  f"H={res['entropy_nats']} nats  positivity_ok={res['positivity_ok']}",
                  flush=True)
            for a, c in sorted(res["distribution"].items(), key=lambda kv: -kv[1]):
                print(f"      {c:>2}x  {a}")
    finally:
        srv.close()

    report["wall_s"] = round(time.time() - t0, 1)
    any_ok = any(s["positivity_ok"] for s in report["per_step"])
    report["verdict"] = {
        "any_step_has_positivity": any_ok,
        "conclusion": (
            "at least one fork point has action stochasticity → fork-resample can estimate v_t, main line holds"
            if any_ok else
            "action entropy is 0 at every fork point → positivity fails → switch to the soft-intervention route, estimand must be rewritten"
        ),
    }
    (outdir / "v5_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVerdict: {report['verdict']['conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
