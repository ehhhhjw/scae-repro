"""A3 replay consistency test -- zero API cost, provides empirical support for the paper's assumption A3.

The paper's Method §Identification claims A3 (environment replayability) holds by construction: tool outputs
are totally ordered, truncation positions fixed, locale/TZ/terminal width/hash seed all pinned. This is a
**testable assertion**, and should not be left as a mere declaration.

Method: take read-only actions that were actually executed in existing rollouts, re-run them R times in the
same repository (reset to base_commit), and check whether the outputs are byte-identical.

report:
  - consistency_rate: fraction of actions that are byte-identical
  - the inconsistent actions and the source of their divergence (used to localize residual nondeterminism)

This maps directly onto the "A3 replay consistency rate" that the paper should report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                      # noqa: E402
from pilot.harness import action_exec as AE                 # noqa: E402
from pilot.probes import task as T                          # noqa: E402


def extract_shell_cmd(step) -> str | None:
    """Pull the shell command out of a step's args (codex's exec_command tool)."""
    a = step.args
    cmd = a.get("cmd") or a.get("command")
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    return cmd if isinstance(cmd, str) and cmd.strip() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True,
                    help="rollout jsonl paths (glob-expandable)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-actions", type=int, default=120)
    ap.add_argument("--out", default="pilot/runs/replay_check.json")
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    checked = 0
    consistent = 0
    skipped_nonreadonly = 0
    mismatches: list[dict] = []
    per_repo: Counter = Counter()

    for rp in args.rollouts:
        p = Path(rp)
        if not p.exists():
            continue
        # rollout filenames look like <instance_id>__k0.jsonl
        iid = p.stem.split("__k")[0]
        inst = insts.get(iid)
        if inst is None or not inst.repo.is_dir():
            continue
        state = T.reset_repo(inst)
        if not state.get("ok"):
            continue
        steps = R.steps(R.load(p))
        for s in steps:
            if checked >= args.max_actions:
                break
            cmd = extract_shell_cmd(s)
            if not cmd:
                continue
            if not AE.is_readonly(cmd):
                skipped_nonreadonly += 1
                continue
            digests = set()
            for _ in range(args.repeats):
                out = AE.exec_readonly_action(inst.repo, cmd)
                digests.add(hashlib.sha256(out.encode("utf-8", "ignore")).hexdigest())
            checked += 1
            per_repo[inst.project] += 1
            if len(digests) == 1:
                consistent += 1
            else:
                mismatches.append({"instance": iid, "step": s.idx,
                                   "cmd": cmd[:200], "n_distinct": len(digests)})

    rate = consistent / checked if checked else None
    report = {
        "n_actions_checked": checked,
        "repeats_each": args.repeats,
        "n_consistent": consistent,
        "consistency_rate": round(rate, 4) if rate is not None else None,
        "n_skipped_non_readonly": skipped_nonreadonly,
        "by_project": dict(per_repo),
        "n_mismatch": len(mismatches),
        "mismatches": mismatches[:20],
        "interpretation": (
            "A3 holds within the tested scope (100% byte-identical)"
            if rate == 1.0 else
            f"A3 has residual nondeterminism: {len(mismatches)} actions produce unstable output, must be localized and pinned"
            if rate is not None else "no testable actions"),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("=== A3 replay consistency test ===")
    print(f"  actions checked {checked} (each repeated {args.repeats} times)")
    print(f"  byte-identical {consistent} → consistency rate {report['consistency_rate']}")
    print(f"  skipped non-read-only {skipped_nonreadonly}")
    print(f"  inconsistent {len(mismatches)}")
    for m in mismatches[:5]:
        print(f"    {m['instance']} step{m['step']}: {m['cmd'][:80]} ({m['n_distinct']} distinct outputs)")
    print(f"  conclusion: {report['interpretation']}")
    print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
