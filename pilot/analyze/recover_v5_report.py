#!/usr/bin/env python3
"""Recover the crashed V5 entropy probe report from leftover on-disk data (read-only, zero API cost).

Background
----------
`pilot/smoke/probe_v5.py` writes `v5_report.json` only at the very end of main(), so a crash midway
throws away every fork sample already paid for. But the samples themselves are persisted: each
`thread_fork` + `turn_run` drops one complete continuation rollout at
`<outdir>/codex_home/sessions/YYYY/MM/DD/rollout-*.jsonl`. This script regroups those leftover rollouts
back onto their fork points and reuses `pilot.harness.rollout`'s `steps()` / `action_bucket()` to
recompute the distributions and entropies, producing a `v5_report.json` isomorphic to the one a
successful run would have written.

Grouping criterion (structural, no timestamps)
----------------------------------------------
At fork point k, `probe_v5` builds the prefix with `truncate_before_step(base, k)`, and that prefix
contains exactly k-1 steps. The forked continuation rollout carries that prefix verbatim and then
appends step k. Hence:

    m := the **length of the longest common step prefix** between this rollout and the base rollout
         (compared call_id by call_id)
    k  = m + 1

call_id is a random string generated model-side, so a continuation can never reproduce the base's
call_id; m is therefore well defined. Plus two hard checks:
  1. every `prefix_kN.jsonl` must equal `truncate_before_step(base, N)` line by line
     -- confirming we found the right base rollout;
  2. the response_item sequence for the first k-1 steps of each rollout must equal the response_item
     sequence of `prefix_k{k}.jsonl` (ignoring differences caused by codex re-serializing
     `reasoning.content`).

If any rollout cannot be grouped onto a declared fork point, or fails a check, the script exits with a
failure and writes **no output at all**. Better to fail recovery than to emit guessed numbers.

Timestamps are used only to order samples within a single fork point (probe_v5's sampling loop is
strictly serial); they play no part in grouping.

Usage
-----
    python3 -m pilot.analyze.recover_v5_report \\
        --run-dir pilot/runs/v5_scan/django__django-10097
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R      # noqa: E402
from pilot.probes import task as T          # noqa: E402


class Unrecoverable(RuntimeError):
    """Data insufficient for reliable grouping. Callers must report the failure honestly, never fall back to guessing."""


# ---------------------------------------------------------------------------
# Statistics defined verbatim as in probe_v5 (do not "improve" here, or old and new reports become incomparable)
# ---------------------------------------------------------------------------

def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c) if n else 0.0


def norm_entropy(counts: Counter) -> float:
    k = len(counts)
    return entropy(counts) / math.log(k) if k > 1 else 0.0


def first_action_after(items_after: list[dict], k: int, repo: Path) -> str | None:
    st = R.steps(items_after)
    if len(st) < k:
        return None
    a, b = R.action_bucket(st[k - 1], repo)
    return f"{a}#{b}"


# ---------------------------------------------------------------------------
# Structural comparison
# ---------------------------------------------------------------------------

def _canon_response_items(items: list[dict]) -> list[dict]:
    """Take the response_item payloads, stripping the noise fields introduced by fork re-serialization.

    The only observed difference: the `reasoning` payload in post-fork rollouts carries an extra `content`
    key (absent from the original rollout). All other fields are identical verbatim.
    """
    out = []
    for it in items:
        if R.line_type(it) != "response_item":
            continue
        p = R.payload(it)
        if p.get("type") == "reasoning":
            p = {k: v for k, v in p.items() if k != "content"}
        out.append(p)
    return out


def _common_step_prefix(fork_ids: list[str | None], base_ids: list[str | None]) -> int:
    m = 0
    while m < len(fork_ids) and m < len(base_ids):
        a, b = fork_ids[m], base_ids[m]
        if a is None or b is None or a != b:
            break
        m += 1
    return m


_TS_RE = re.compile(r"rollout-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-")


def _sort_key(p: Path) -> tuple:
    m = _TS_RE.search(p.name)
    return (m.group(1) if m else "", p.name)


# ---------------------------------------------------------------------------
# base rollout localization (verified discovery, not guesswork)
# ---------------------------------------------------------------------------

def _prefixes_match(base_items: list[dict], prefix_files: dict[int, Path]) -> bool:
    for k, pf in prefix_files.items():
        try:
            trunc = R.truncate_before_step(base_items, k)
        except (IndexError, ValueError):
            return False
        if json.dumps(trunc, sort_keys=True) != json.dumps(R.load(pf), sort_keys=True):
            return False
    return True


def discover_base(repo_root: Path, instance: str, prefix_files: dict[int, Path],
                  explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise Unrecoverable(f"the file --base points to does not exist: {p}")
        if not _prefixes_match(R.load(p), prefix_files):
            raise Unrecoverable(f"--base {p} disagrees with prefix_k*.jsonl; it is not the base rollout this probe used")
        return p

    cands = sorted(set(repo_root.glob(f"pilot/runs/**/{instance}__k*.jsonl"))
                   | set(repo_root.glob(f"pilot/runs/**/{instance}/base.rollout.jsonl")))
    hits = [c for c in cands if _prefixes_match(R.load(c), prefix_files)]
    if not hits:
        raise Unrecoverable(
            f"no base rollout under pilot/runs agrees with {instance}'s prefix_k*.jsonl"
            f" (tried {len(cands)} candidates). Specify one explicitly with --base.")
    if len({json.dumps(R.load(h), sort_keys=True) for h in hits}) > 1:
        raise Unrecoverable(f"found multiple base rollout candidates with differing content: {[str(h) for h in hits]}")
    return hits[0]


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def recover(run_dir: Path, repo_root: Path, instance: str | None,
            base_path: str | None, samples_expected: int | None) -> dict:
    instance = instance or run_dir.name

    prefix_files = {}
    for pf in run_dir.glob("prefix_k*.jsonl"):
        m = re.fullmatch(r"prefix_k(\d+)\.jsonl", pf.name)
        if m:
            prefix_files[int(m.group(1))] = pf
    if not prefix_files:
        raise Unrecoverable(f"no prefix_k*.jsonl under {run_dir}; cannot tell which fork points were probed")
    fork_steps = sorted(prefix_files)

    base = discover_base(repo_root, instance, prefix_files, base_path)
    base_items = R.load(base)
    base_steps = R.steps(base_items)
    base_ids = [s.call_id for s in base_steps]
    if any(i is None for i in base_ids):
        raise Unrecoverable("the base rollout contains steps without a call_id; structural grouping is impossible")
    if len(set(base_ids)) != len(base_ids):
        raise Unrecoverable("the base rollout has duplicate call_ids; the longest common prefix is no longer well defined")

    repo = T.PROJECTS_ROOT / instance

    sessions_dir = run_dir / "codex_home" / "sessions"
    if not sessions_dir.is_dir():
        raise Unrecoverable(f"{sessions_dir} missing; the forked continuation rollouts are not on disk, unrecoverable")
    session_files = sorted(sessions_dir.rglob("rollout-*.jsonl"), key=_sort_key)
    if not session_files:
        raise Unrecoverable(f"no rollout-*.jsonl under {sessions_dir}, unrecoverable")

    prefix_canon = {k: _canon_response_items(R.load(p)) for k, p in prefix_files.items()}

    groups: dict[int, list[Path]] = {k: [] for k in fork_steps}
    for sp in session_files:
        items = R.load(sp)
        st = R.steps(items)
        m = _common_step_prefix([s.call_id for s in st], base_ids)
        k = m + 1
        if k not in groups:
            raise Unrecoverable(
                f"{sp} has a common step prefix of length {m} with base → k={k}, which is not among the "
                f"declared fork points {fork_steps}. Cannot group reliably; refusing to guess.")
        # Hard check: the response_item sequence for the first k-1 steps must equal prefix_k{k}
        want = prefix_canon[k]
        got = _canon_response_items(items)[: len(want)]
        if json.dumps(got, sort_keys=True) != json.dumps(want, sort_keys=True):
            raise Unrecoverable(
                f"{sp} was grouped to k={k}, but its prefix response_item sequence disagrees with "
                f"prefix_k{k}.jsonl; refusing to trust it.")
        groups[k].append(sp)

    per_step = []
    for k in fork_steps:
        orig_first = None
        if len(base_steps) >= k:
            a, b = R.action_bucket(base_steps[k - 1], repo)
            orig_first = f"{a}#{b}"

        actions: list[str] = []
        errors: list[str] = []
        for i, sp in enumerate(groups[k]):
            fa = first_action_after(R.load(sp), k, repo)
            if fa:
                actions.append(fa)
            else:
                errors.append(f"s{i}: step {k} action not collected")
        n_found = len(groups[k])
        if samples_expected and n_found < samples_expected:
            errors.append(
                f"[recovered] only {n_found}/{samples_expected} samples on disk"
                f" (the probe crashed at this fork point)")

        counts = Counter(actions)
        per_step.append({
            "fork_step": k,
            "orig_first_action": orig_first,
            "n_samples": len(actions),
            "n_distinct": len(counts),
            "entropy_nats": round(entropy(counts), 4),
            "normalized_entropy": round(norm_entropy(counts), 4),
            "distribution": dict(counts),
            "positivity_ok": len(counts) >= 2,
            "errors": errors,
        })

    # wall_s cannot be recovered from a crashed process: use the leftover rollouts' timestamp span as a lower bound.
    stamps = [_sort_key(p)[0] for p in session_files if _sort_key(p)[0]]
    wall_s = None
    if len(stamps) >= 2:
        import datetime as _dt
        fmt = "%Y-%m-%dT%H-%M-%S"
        t0 = _dt.datetime.strptime(min(stamps), fmt)
        t1 = _dt.datetime.strptime(max(stamps), fmt)
        wall_s = round((t1 - t0).total_seconds(), 1)

    any_ok = any(s["positivity_ok"] for s in per_step)
    return {
        "instance": instance,
        "rollout_n_steps": len(base_steps),
        "samples_per_step": samples_expected or max(
            (len(groups[k]) for k in fork_steps), default=0),
        "per_step": per_step,
        "wall_s": wall_s,
        "verdict": {
            "any_step_has_positivity": any_ok,
            "conclusion": (
                "at least one fork point shows action randomness → fork-resample can estimate v_t, main line holds"
                if any_ok else
                "action entropy is 0 at every fork point → positivity fails → switch to the soft-intervention route, the estimator must be rewritten"
            ),
        },
        "recovery": {
            "recovered_from_disk": True,
            "reason": "probe_v5 crashed without writing v5_report.json; recomputed from the leftover fork rollouts",
            "base_rollout": str(base),
            "n_session_rollouts": len(session_files),
            "grouping": {str(k): [p.name for p in groups[k]] for k in fork_steps},
            "wall_s_is_timestamp_span_lower_bound": True,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="probe output directory, e.g. pilot/runs/v5_scan/<instance>")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--instance", default=None, help="defaults to the run-dir directory name")
    ap.add_argument("--base", default=None, help="base rollout path; if omitted, search under pilot/runs with verification")
    ap.add_argument("--samples", type=int, default=None,
                    help="the original probe's --samples; used to flag fork points with missing samples")
    ap.add_argument("--out", default=None, help="defaults to <run-dir>/v5_report.json")
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing v5_report.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "v5_report.json"
    if out.exists() and not args.force:
        print(f"[refused] {out} already exists. This script does not overwrite existing reports (pass --force to allow).", file=sys.stderr)
        return 2

    try:
        report = recover(run_dir, Path(args.repo_root), args.instance, args.base, args.samples)
    except Unrecoverable as exc:
        print(f"[unrecoverable] {exc}", file=sys.stderr)
        return 1

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"recovered → {out}")
    print(f"base rollout: {report['recovery']['base_rollout']}  "
          f"n_steps={report['rollout_n_steps']}  "
          f"leftover rollouts: {report['recovery']['n_session_rollouts']}")
    for s in report["per_step"]:
        print(f"[k={s['fork_step']}] distinct={s['n_distinct']}/{s['n_samples']} "
              f"H={s['entropy_nats']} nats  positivity_ok={s['positivity_ok']}")
        for a, c in sorted(s["distribution"].items(), key=lambda kv: -kv[1]):
            print(f"      {c:>2}x  {a}")
        for e in s["errors"]:
            print(f"      ! {e}")
    print(f"\nverdict: {report['verdict']['conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
