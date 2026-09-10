#!/usr/bin/env python3
"""Build SCAE's dataset interface from SWE-bench-Verified and materialize repo snapshots.

The reproduction package ships no data. This script covers step 1: turning SWE-bench-Verified
into the two things `pilot/probes/task.py` expects:

  1. instance_summary.csv    fields instance_id, base_commit, problem_statement, patch_file
                             patch_file = files the gold patch touches (newline-separated) = paper's F*
  2. projects/<instance_id>/ one git repo snapshot per instance, HEAD parked at base_commit

Usage
-----
# A) You already have SWE-bench-Verified as JSON (list of records with instance_id/base_commit/
#    problem_statement/patch fields)
python3 setup/prepare_dataset.py --from-json /path/to/instance_questions.json

# B) Pull from HuggingFace (needs `pip install datasets`)
python3 setup/prepare_dataset.py --from-hf

# C) Write the CSV only, no repo clones (about 900KB, done in seconds)
python3 setup/prepare_dataset.py --from-hf --skip-clone

# D) Clone/verify repo snapshots (about 20-40GB of disk, hours of runtime)
python3 setup/prepare_dataset.py --from-json data.json --clone --workers 4

# E) Only check that existing snapshots sit at base_commit (self-check, changes nothing)
python3 setup/prepare_dataset.py --verify-only

Why the gold set is recomputed from the patch
---------------------------------------------
The paper's result definition is recall over F* (Appendix C.6), and F* is defined as "the set of
files modified by the reference patch". Parsing it straight from the patch's unified diff headers
keeps F* word-for-word identical to the paper's definition, rather than relying on any field that
a downstream dataset happens to ship.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "dataset" / "instance_summary.csv"
DEFAULT_PROJECTS = REPO_ROOT / "dataset" / "projects"

HF_DATASET = "princeton-nlp/SWE-bench_Verified"

# --- gold patch file-path parsing (matches SWE-bench's diff format) -------------------
# "diff --git a/<path> b/<path>"; paths with special characters are wrapped in quotes
DIFF_GIT_RE = re.compile(
    r'^diff --git\s+(?:"a/(?P<qa>.+?)"|a/(?P<a>\S+))\s+(?:"b/(?P<qb>.+?)"|b/(?P<b>.+?))\s*$'
)
OLD_FILE_RE = re.compile(r'^--- (?:"?a/)?(?P<path>.+?)"?\s*$')
NEW_FILE_RE = re.compile(r'^\+\+\+ (?:"?b/)?(?P<path>.+?)"?\s*$')


def extract_diff_paths(patch: str) -> list[str]:
    """Extract modified file paths from a unified diff, keeping order of appearance and deduping.

    Prefer the b/ side (new path); on file deletion the b side is /dev/null, so fall back to a/.
    """
    if not patch:
        return []

    paths: list[str] = []
    seen: set[str] = set()

    def add(p: str | None) -> None:
        if p and p != "/dev/null" and p not in seen:
            seen.add(p)
            paths.append(p)

    lines = patch.splitlines()
    matched_git_header = False
    for line in lines:
        m = DIFF_GIT_RE.match(line)
        if m:
            matched_git_header = True
            new_path = m.group("qb") or m.group("b")
            old_path = m.group("qa") or m.group("a")
            add(new_path if new_path and new_path != "/dev/null" else old_path)
    if matched_git_header:
        return paths

    # fall back to the ---/+++ lines when there is no "diff --git" header
    pending_old = None
    for line in lines:
        m = NEW_FILE_RE.match(line)
        if m and not line.startswith("+++ +"):
            p = m.group("path").split("\t")[0]
            add(p if p != "/dev/null" else pending_old)
            pending_old = None
            continue
        m = OLD_FILE_RE.match(line)
        if m and not line.startswith("--- -"):
            pending_old = m.group("path").split("\t")[0]
    return paths


# --- Data sources ---------------------------------------------------------------------
def load_from_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [data] if isinstance(data, dict) else data


def load_from_hf(split: str = "test") -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "The `datasets` package is missing. Run `pip install datasets`, "
            "or switch to --from-json pointing at a local SWE-bench-Verified JSON."
        )
    print(f"Pulling {HF_DATASET} [{split}] from HuggingFace ...")
    ds = load_dataset(HF_DATASET, split=split)
    return [dict(r) for r in ds]


def write_csv(records: list[dict], out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, no_path = [], 0
    for item in records:
        paths = extract_diff_paths(item.get("patch", "") or "")
        if not paths:
            no_path += 1
        rows.append(
            {
                "instance_id": (item.get("instance_id") or "").strip(),
                "base_commit": (item.get("base_commit") or "").strip(),
                "problem_statement": item.get("problem_statement") or "",
                "patch_file": "\n".join(paths),
            }
        )
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["instance_id", "base_commit", "problem_statement", "patch_file"],
            quoting=csv.QUOTE_ALL,
        )
        w.writeheader()
        w.writerows(rows)
    multi = sum(1 for r in rows if len(r["patch_file"].split("\n")) > 1)
    return {"n": len(rows), "no_gold_path": no_path, "multi_file": multi, "csv": str(out)}


# --- Repo snapshots -------------------------------------------------------------------
def repo_url(instance_id: str) -> str:
    """astropy__astropy-12907 -> https://github.com/astropy/astropy"""
    stem = instance_id.rsplit("-", 1)[0]
    if "__" not in stem:
        raise ValueError(f"cannot derive the repo name from instance_id: {instance_id}")
    org, name = stem.split("__", 1)
    return f"https://github.com/{org}/{name}"


def _git(cwd: Path | None, *args: str, timeout: float = 1800) -> subprocess.CompletedProcess:
    cmd = ["git"] + (["-C", str(cwd)] if cwd else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def ensure_repo(instance_id: str, base_commit: str, projects: Path) -> tuple[str, str]:
    """Ensure projects/<instance_id> exists with HEAD == base_commit. Returns (status, detail)."""
    dest = projects / instance_id
    try:
        if not (dest / ".git").exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = repo_url(instance_id)
            p = _git(None, "clone", "--quiet", url, str(dest))
            if p.returncode != 0:
                return "clone_failed", p.stderr.strip()[:200]
        # fetch the requested commit (needed when a shallow clone or the default branch lacks it)
        if _git(dest, "cat-file", "-e", f"{base_commit}^{{commit}}").returncode != 0:
            _git(dest, "fetch", "--quiet", "origin", base_commit)
        for args in (
            ("checkout", "-f", base_commit),
            ("reset", "--hard", base_commit),
            ("clean", "-fdq"),
        ):
            p = _git(dest, *args)
            if p.returncode != 0:
                return "reset_failed", " ".join(args) + ": " + p.stderr.strip()[:160]
        return verify_repo(instance_id, base_commit, projects)
    except subprocess.TimeoutExpired:
        return "timeout", "git operation timed out"
    except Exception as e:  # noqa: BLE001
        return "error", f"{type(e).__name__}: {e}"[:200]


def verify_repo(instance_id: str, base_commit: str, projects: Path) -> tuple[str, str]:
    """Read-only self-check: is HEAD == base_commit, is the working tree clean."""
    dest = projects / instance_id
    if not (dest / ".git").exists():
        return "missing", "repo snapshot does not exist"
    head = _git(dest, "rev-parse", "HEAD").stdout.strip()
    if head != base_commit:
        return "wrong_commit", f"HEAD={head[:12]} != base={base_commit[:12]}"
    dirty = _git(dest, "status", "--porcelain").stdout.strip()
    if dirty:
        n = len(dirty.splitlines())
        return "dirty", f"working tree has {n} uncommitted change(s) (gold patch leakage risk)"
    return "ok", ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the SCAE dataset interface (instance_summary.csv + repo snapshots)"
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-json", type=Path, help="local SWE-bench-Verified JSON")
    src.add_argument("--from-hf", action="store_true", help=f"pull {HF_DATASET} from HuggingFace")
    ap.add_argument("--hf-split", default="test")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"output CSV (default {DEFAULT_CSV})")
    ap.add_argument("--projects", type=Path, default=DEFAULT_PROJECTS)
    ap.add_argument("--clone", action="store_true", help="clone/fix up repo snapshots (tens of GB, hours)")
    ap.add_argument("--skip-clone", action="store_true", help="write the CSV only (default behavior)")
    ap.add_argument("--verify-only", action="store_true", help="only self-check existing snapshots, change nothing")
    ap.add_argument("--workers", type=int, default=4, help="number of parallel git jobs")
    ap.add_argument("--limit", type=int, help="process only the first N instances (for debugging)")
    args = ap.parse_args()

    # --- self-check only ---
    if args.verify_only:
        if not args.csv.exists():
            sys.exit(f"{args.csv} not found; run --from-json/--from-hf once to generate the CSV.")
        with open(args.csv, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if args.limit:
            rows = rows[: args.limit]
        stats: dict[str, int] = {}
        problems: list[str] = []
        for r in rows:
            st, msg = verify_repo(r["instance_id"].strip(), r["base_commit"].strip(), args.projects)
            stats[st] = stats.get(st, 0) + 1
            if st != "ok":
                problems.append(f"  {r['instance_id']}: {st} -- {msg}")
        print(f"\nSelf-check of {len(rows)} instances (projects={args.projects}):")
        for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
            print(f"  {k:14s} {v}")
        if problems:
            print(f"\nFirst 20 problem instances:")
            for line in problems[:20]:
                print(line)
            print(
                "\n⚠️  Non-ok instances make the agent read already-fixed code (leakage) or make replays incomparable."
                "\n   Run --clone to fix them, or see docs/PITFALLS.md."
            )
        sys.exit(0 if not problems else 1)

    if not args.from_json and not args.from_hf:
        ap.error("a data source is required: --from-json <path> or --from-hf (or --verify-only to self-check)")

    records = load_from_json(args.from_json) if args.from_json else load_from_hf(args.hf_split)
    if args.limit:
        records = records[: args.limit]

    info = write_csv(records, args.csv)
    print("\n=== instance_summary.csv ===")
    print(f"  instances total       {info['n']}")
    print(f"  multi-file instances  {info['multi_file']}")
    print(f"  gold not parsed       {info['no_gold_path']}")
    print(f"  output                {info['csv']}")
    if info["no_gold_path"]:
        print("  ⚠️  Some instances yielded no gold path; their F* is empty and they must be dropped from the corpus.")

    # --- repo snapshots ---
    if not args.clone or args.skip_clone:
        print(
            f"\nRepo snapshots not cloned (skipped by default). The localization experiments need them:\n"
            f"  python3 setup/prepare_dataset.py --from-{'json ' + str(args.from_json) if args.from_json else 'hf'} --clone\n"
            f"or point SCAE_PROJECTS at an existing snapshot directory."
        )
        return

    with open(args.csv, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    print(f"\nCloning/fixing up {len(rows)} repos into {args.projects} (workers={args.workers}) ...")
    args.projects.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                ensure_repo, r["instance_id"].strip(), r["base_commit"].strip(), args.projects
            ): r["instance_id"].strip()
            for r in rows
        }
        for i, fut in enumerate(cf.as_completed(futs), 1):
            iid = futs[fut]
            st, msg = fut.result()
            stats[st] = stats.get(st, 0) + 1
            if st != "ok":
                print(f"  [{i}/{len(rows)}] {iid}: {st} -- {msg}")
            elif i % 25 == 0:
                print(f"  [{i}/{len(rows)}] ok={stats.get('ok', 0)}")
    print("\n=== repo snapshots ===")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v}")
    print(
        "\nNote: the localization experiments force-reset to base_commit again before every task"
        " (pilot/probes/task.py::reset_repo); that is the last gate against answer leakage."
    )


if __name__ == "__main__":
    main()
