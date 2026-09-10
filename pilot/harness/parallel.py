"""Parallel execution orchestration -- a required component for the full-scale experiment.

Measured serial speed: one fork continuation takes ~200s (29-step task). Full attribution
over 71 tasks would take ~71h serially. Parallelized (W=6) this compresses to ~12h.

Isolation essentials (each worker must be fully independent, otherwise they contaminate each other):
  - separate CODEX_HOME (each with its own config.toml) -> rollouts do not mix
  - separate AppServer subprocess -> JSON-RPC streams do not cross
  - **repository isolation**: multiple workers resetting the same repo concurrently destroy each other.
    Hence we shard by instance_id, guaranteeing a given repo directory is used by exactly one worker at a time.
  - separate result files, merged at the end (avoids concurrent writes to the same jsonl)

Usage: called as a library by run_full.py; can also be run directly
    python -m pilot.harness.parallel --cmd locate --workers 6 --out pilot/runs/full_locate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.probes import task as T  # noqa: E402


def shard_by_repo(instance_ids: list[str], n_shards: int) -> list[list[str]]:
    """Shard by repository: the repo directory of a given instance belongs to exactly one shard.

    In the dataset each instance_id has its own separate repo directory (projects/<instance_id>),
    so there is in fact no cross-instance directory contention; but grouping by project still helps:
    repos within the same project have similar sizes, which balances the load across shards.
    """
    by_proj: dict[str, list[str]] = {}
    for iid in instance_ids:
        by_proj.setdefault(iid.split("__")[0], []).append(iid)
    # Greedy bin packing: groups in descending size, into the currently smallest shard
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    for proj in sorted(by_proj, key=lambda p: -len(by_proj[p])):
        for iid in by_proj[proj]:
            shards.sort(key=len)
            shards[0].append(iid)
    return [s for s in shards if s]


def launch_locate_shards(
    instance_ids: list[str], out_root: Path, workers: int,
    model: str, provider: str, k: int, turn_timeout: float,
) -> list[subprocess.Popen]:
    """Launch localization runners in parallel, one process per shard."""
    out_root.mkdir(parents=True, exist_ok=True)
    shards = shard_by_repo(instance_ids, workers)
    procs = []
    for i, ids in enumerate(shards):
        sdir = out_root / f"shard{i}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "ids.txt").write_text("\n".join(ids), encoding="utf-8")
        log = open(sdir / "run.log", "w")
        p = subprocess.Popen(
            [sys.executable, "-m", "pilot.harness.runner",
             "--model", model, "--model-provider", provider,
             "--k", str(k), "--out", str(sdir),
             "--turn-timeout", str(turn_timeout),
             "--progress-every", "1",
             "--instances", *ids],
            stdout=log, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[2]),
            env={**os.environ},
        )
        procs.append(p)
        print(f"  shard{i}: {len(ids)} instances → PID {p.pid}", flush=True)
    return procs


def launch_attr_shards(
    runs_dirs: list[Path], out_root: Path, workers: int,
    model: str, provider: str, m: int, window_points: int,
    outcome: str, turn_timeout: float,
) -> list[subprocess.Popen]:
    """Launch attribution in parallel, one process per shard (inputs are results.jsonl already split by shard)."""
    procs = []
    for i, rd in enumerate(runs_dirs):
        log = open(rd / "attr.log", "w")
        p = subprocess.Popen(
            [sys.executable, "-m", "pilot.attribution.run_attribution",
             "--runs-dir", str(rd),
             "--model", model, "--model-provider", provider,
             "--m", str(m), "--window-points", str(window_points),
             "--outcome", outcome, "--multifile-only",
             "--prefix-frac", "0.6",
             "--turn-timeout", str(turn_timeout)],
            stdout=log, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[2]),
            env={**os.environ},
        )
        procs.append(p)
        print(f"  attr shard{i}: {rd} → PID {p.pid}", flush=True)
    return procs


def wait_and_report(procs: list[subprocess.Popen], label: str,
                    poll: float = 60.0) -> None:
    t0 = time.time()
    while True:
        alive = [p for p in procs if p.poll() is None]
        if not alive:
            break
        print(f"[{label}] {len(procs)-len(alive)}/{len(procs)} shards done "
              f"({(time.time()-t0)/60:.0f}min)", flush=True)
        time.sleep(poll)
    codes = [p.returncode for p in procs]
    print(f"[{label}] all done, exit codes {codes}, elapsed {(time.time()-t0)/60:.1f}min",
          flush=True)


def merge_jsonl(shard_dirs: list[Path], fname: str, out: Path) -> int:
    """Merge the jsonl of each shard."""
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for d in shard_dirs:
            f = d / fname
            if not f.exists():
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    fh.write(line + "\n")
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", choices=["locate", "attr"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-5.5-0424-global")
    ap.add_argument("--model-provider", default="idealab")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--window-points", type=int, default=5)
    ap.add_argument("--outcome", default="recall")
    ap.add_argument("--turn-timeout", type=float, default=900.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    if args.cmd == "locate":
        insts = T.load_instances()
        ids = [i.instance_id for i in insts]
        if args.limit > 0:
            ids = ids[: args.limit]
        print(f"full-scale localization: {len(ids)} instances, {args.workers} in parallel")
        procs = launch_locate_shards(ids, out_root, args.workers,
                                     args.model, args.model_provider,
                                     args.k, args.turn_timeout)
        wait_and_report(procs, "locate")
        shard_dirs = sorted(out_root.glob("shard*"))
        n = merge_jsonl(shard_dirs, "results.jsonl", out_root / "results.jsonl")
        print(f"merged {n} results → {out_root/'results.jsonl'}")
    else:
        shard_dirs = sorted(out_root.glob("shard*"))
        print(f"full-scale attribution: {len(shard_dirs)} shards")
        procs = launch_attr_shards(shard_dirs, out_root, args.workers,
                                   args.model, args.model_provider, args.m,
                                   args.window_points, args.outcome,
                                   args.turn_timeout)
        wait_and_report(procs, "attr")
        n = merge_jsonl(shard_dirs, "attribution.jsonl",
                        out_root / "attribution.jsonl")
        print(f"merged {n} attributions → {out_root/'attribution.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
