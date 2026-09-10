#!/usr/bin/env python3
"""Prepare the shard directories for the attribution stage.

Background
----------
`pilot.harness.parallel --cmd attr` does not pick instances itself: it requires that
`shard*/` already exist under `--out`, each shard holding a `results.jsonl` (the
localization results that shard is responsible for) and `rollouts/` (readable rollout
JSONL). In the original experiment this step was done by hand; this script pins it down
so that stage 2 is reproducible.

What it does
------------
1. Read the localization-stage results (merged results.jsonl or shard*/results.jsonl)
2. Deduplicate by instance_id, keeping only records that are ok and have a locatable rollout
3. Shard greedily by repo (reusing pilot.harness.parallel.shard_by_repo, which guarantees
   a given repository directory is used by at most one worker at a time)
4. Write results.jsonl per shard and symlink (default) or copy the rollouts into it

Usage
-----
# main attribution sample (the 70 episodes of paper D.10, m=3, for the RQ5 judge analysis)
python3 setup/stage_attr_shards.py --locate pilot/runs/full_locate \
    --out pilot/runs/full_attr --workers 6

# m=8 subsample (the 190 estimation points / 38 episodes of paper §5.6, for the RQ4 step-level power analysis)
python3 setup/stage_attr_shards.py --locate pilot/runs/full_locate \
    --out pilot/runs/v4prime --workers 6 --limit 38

Note
----
Instance filtering (keeping only multi-file instances) is done by run_attribution's
`--multifile-only`; this script does not pre-filter, to avoid creating a second source
of truth alongside the paper's definition.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pilot.harness.parallel import shard_by_repo  # noqa: E402
from pilot.probes import task as T  # noqa: E402


def load_locate(locate_dir: Path) -> dict[str, dict]:
    """Read localization results, deduplicating by instance_id (ok records win)."""
    files = [locate_dir / "results.jsonl"] + sorted(locate_dir.glob("shard*/results.jsonl"))
    uniq: dict[str, dict] = {}
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = r.get("instance_id")
            if not iid:
                continue
            # ok records win; among equally-ok records keep the one with more steps
            old = uniq.get(iid)
            if (
                old is None
                or (r.get("ok") and not old.get("ok"))
                or (r.get("ok") == old.get("ok") and (r.get("n_steps") or 0) > (old.get("n_steps") or 0))
            ):
                uniq[iid] = r
    return uniq


def find_rollouts(locate_dir: Path) -> dict[str, Path]:
    """Collect instance_id -> rollout path (taking the most complete k0)."""
    out: dict[str, Path] = {}
    for p in sorted(locate_dir.glob("shard*/rollouts/*.jsonl")) + sorted(
        locate_dir.glob("rollouts/*.jsonl")
    ):
        iid = p.name.split("__k")[0]
        # with several copies of the same name keep the larger (more complete) one
        if iid not in out or p.stat().st_size > out[iid].stat().st_size:
            out[iid] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare attribution shard directories")
    ap.add_argument("--locate", type=Path, default=REPO_ROOT / "pilot" / "runs" / "full_locate")
    ap.add_argument("--out", type=Path, required=True, help="attribution output root, e.g. pilot/runs/full_attr")
    ap.add_argument("--workers", type=int, default=6, help="number of shards (= number of parallel workers)")
    ap.add_argument("--limit", type=int, default=0, help="take only the first N multi-file instances (for the m=8 subsample)")
    ap.add_argument("--copy", action="store_true", help="copy rollouts instead of symlinking (symlink is the default, to save disk)")
    ap.add_argument("--force", action="store_true", help="overwrite existing shard directories")
    args = ap.parse_args()

    if not args.locate.exists():
        sys.exit(f"localization results directory {args.locate} not found; run scripts/01_locate.sh first")

    uniq = load_locate(args.locate)
    rollouts = find_rollouts(args.locate)
    if not uniq:
        sys.exit(f"no localization results read under {args.locate}.")
    if not rollouts:
        sys.exit(f"no rollouts/*.jsonl found under {args.locate} -- attribution needs rollouts to replay prefixes.")

    # keep only records that are ok and have a rollout
    usable = {i: r for i, r in uniq.items() if r.get("ok") and i in rollouts}
    dropped_notok = sum(1 for r in uniq.values() if not r.get("ok"))
    dropped_noroll = sum(1 for i, r in uniq.items() if r.get("ok") and i not in rollouts)

    # multi-file count (for cross-checking the paper's definition; the actual filtering is done by --multifile-only)
    try:
        gold = {i.instance_id: i.gold for i in T.load_instances()}
        multi = [i for i in usable if len(gold.get(i, ())) > 1]
    except Exception:  # noqa: BLE001
        multi = []

    ids = sorted(usable)
    if args.limit > 0:
        # take multi-file instances first, so enough episodes survive --multifile-only
        prefer = sorted(multi) if multi else ids
        ids = prefer[: args.limit]

    print(f"localization  {len(uniq)} unique instances")
    print(f"  usable      {len(usable)} (dropped: not ok {dropped_notok}, no rollout {dropped_noroll})")
    if multi:
        print(f"  multi-file  {len(multi)}  ← attribution actually runs only these (--multifile-only)")
    print(f"sharding now  {len(ids)} instances → {args.workers} shards")

    args.out.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.out.glob("shard*"))
    if existing and not args.force:
        sys.exit(f"{args.out} already holds {len(existing)} shard directories. Pass --force to overwrite.")
    for d in existing:
        shutil.rmtree(d)

    shards = shard_by_repo(ids, args.workers)
    total_links = 0
    for i, sids in enumerate(shards):
        sdir = args.out / f"shard{i}"
        (sdir / "rollouts").mkdir(parents=True, exist_ok=True)
        with open(sdir / "results.jsonl", "w", encoding="utf-8") as fh:
            for iid in sids:
                fh.write(json.dumps(usable[iid], ensure_ascii=False) + "\n")
        # rollouts: attribution must find a rollout per instance; linking all of them is simplest and safest
        for iid in sids:
            src = rollouts[iid].resolve()
            dst = sdir / "rollouts" / rollouts[iid].name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.copy:
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
            total_links += 1
        print(f"  shard{i}: {len(sids)} instances → {sdir}")

    mode = "copy" if args.copy else "symlink"
    print(f"\ndone: {len(shards)} shards, {total_links} rollouts ({mode})")
    print(f"next: bash scripts/02_attribution.sh  (or directly python3 -m pilot.harness.parallel --cmd attr --out {args.out} ...)")


if __name__ == "__main__":
    main()
