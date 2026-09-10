#!/usr/bin/env python3
"""Environment self-check: confirm the prerequisites are in place before running any experiment.

Reports two profiles:
  [offline analysis]  needs only python3 + numpy. Enough to run scripts/04_analysis_offline.sh
  [data collection]   additionally needs codex CLI + API key + dataset snapshot. Required to rerun the 12.7-hour agent execution

Usage
-----
python3 setup/check_env.py
python3 setup/check_env.py --collect   # also check the data-collection profile prerequisites
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "
_results: list[tuple[str, str, str]] = []


def record(level: str, item: str, detail: str = "") -> None:
    _results.append((level, item, detail))
    print(f"[{level}] {item}" + (f" -- {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 10):
        record(OK, "Python", f"{s} (requires >= 3.10)")
    else:
        record(BAD, "Python", f"{s}, but the code uses 3.10+ syntax (X | Y type annotations)")


def check_numpy() -> None:
    if importlib.util.find_spec("numpy") is None:
        record(BAD, "numpy", "not installed -- some analysis scripts need it. pip install numpy")
        return
    import numpy as np

    record(OK, "numpy", np.__version__)


def check_package_imports() -> None:
    """Confirm the pilot package can be imported (reproduction package integrity)."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import pilot.probes.task as T  # noqa: F401
        import pilot.harness.rollout  # noqa: F401
        import pilot.attribution.causal  # noqa: F401
        import pilot.uq.predictor  # noqa: F401

        record(OK, "pilot package import", "harness / attribution / uq / probes all importable")
    except Exception as e:  # noqa: BLE001
        record(BAD, "pilot package import", f"{type(e).__name__}: {e}")


def check_dataset() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from pilot.probes import task as T

    csv_path, projects = T.CSV_PATH, T.PROJECTS_ROOT
    src_csv = "SCAE_CSV" if os.environ.get("SCAE_CSV") else "default"
    src_prj = "SCAE_PROJECTS" if os.environ.get("SCAE_PROJECTS") else "default"

    if not csv_path.exists():
        record(
            BAD,
            f"instance_summary.csv ({src_csv})",
            f"cannot find {csv_path}. Run setup/prepare_dataset.py to generate it, or export SCAE_CSV=...",
        )
        return
    try:
        insts = T.load_instances()
    except Exception as e:  # noqa: BLE001
        record(BAD, "instance_summary.csv parse", f"{type(e).__name__}: {e}")
        return

    n_gold0 = sum(1 for i in insts if not i.gold)
    multi = sum(1 for i in insts if len(i.gold) > 1)
    detail = f"{len(insts)} instances, multi-file {multi}, no gold {n_gold0} ({csv_path})"
    record(OK if n_gold0 == 0 else WARN, f"instance_summary.csv ({src_csv})", detail)

    if not projects.exists():
        record(
            WARN,
            f"repository snapshots ({src_prj})",
            f"cannot find {projects} -- ignorable if you only run offline analysis; required for the localization experiment",
        )
        return
    present = sum(1 for i in insts if (projects / i.instance_id / ".git").exists())
    lvl = OK if present == len(insts) else WARN
    record(lvl, f"repository snapshots ({src_prj})", f"{present}/{len(insts)} instances have a git snapshot ({projects})")
    if present:
        # spot-check base_commit; answer leakage is the biggest pitfall in this project
        sample = [i for i in insts if (projects / i.instance_id / ".git").exists()][:20]
        bad = []
        for i in sample:
            r = subprocess.run(
                ["git", "-C", str(projects / i.instance_id), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            if r.stdout.strip() != i.base_commit:
                bad.append(i.instance_id)
        if bad:
            record(
                WARN,
                "base_commit spot-check",
                f"{len(bad)}/{len(sample)} snapshots have HEAD != base_commit (an automatic reset "
                f"runs before the task, but confirm no gold patch is layered on)",
            )
        else:
            record(OK, "base_commit spot-check", f"spot-checked {len(sample)}, HEAD equals base_commit for all")


def check_codex() -> None:
    exe = shutil.which(os.environ.get("SCAE_CODEX_BIN", "codex"))
    if not exe:
        record(BAD, "codex CLI", "not in PATH -- required for data collection (the paper uses codex 0.145.0)")
        return
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
        ver = (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else "?"
        record(OK, "codex CLI", f"{ver} ({exe})")
    except Exception as e:  # noqa: BLE001
        record(WARN, "codex CLI", f"present but version could not be read: {e}")


def check_api_key() -> None:
    key = os.environ.get("IDEALAB_API_KEY")
    if key:
        record(OK, "IDEALAB_API_KEY", f"set ({len(key)} chars)")
    else:
        record(BAD, "IDEALAB_API_KEY", "not set -- required for data collection and the judge experiment")


def check_git() -> None:
    exe = shutil.which("git")
    if not exe:
        record(BAD, "git", "not in PATH -- required for repository reset")
        return
    r = subprocess.run(["git", "--version"], capture_output=True, text=True)
    record(OK, "git", r.stdout.strip())


def check_runs_dir() -> None:
    runs = REPO_ROOT / "pilot" / "runs"
    if not runs.exists():
        record(WARN, "pilot/runs/", "does not exist -- created automatically during collection (the reproduction package ships no recorded data)")
        return
    n = len(list(runs.glob("*")))
    record(OK if n else WARN, "pilot/runs/", f"{n} entries" if n else "empty directory (not yet collected)")


def main() -> None:
    ap = argparse.ArgumentParser(description="SCAE reproduction environment self-check")
    ap.add_argument("--collect", action="store_true", help="also check the data-collection profile prerequisites")
    args = ap.parse_args()

    print("=" * 66)
    print("SCAE reproduction environment self-check")
    print("=" * 66)
    print("\n--- offline analysis profile (needs only python + numpy) ---")
    check_python()
    check_numpy()
    check_package_imports()
    check_runs_dir()

    print("\n--- dataset ---")
    check_dataset()

    if args.collect:
        print("\n--- data collection profile (rerun agent execution) ---")
        check_git()
        check_codex()
        check_api_key()

    n_bad = sum(1 for lvl, _, _ in _results if lvl == BAD)
    n_warn = sum(1 for lvl, _, _ in _results if lvl == WARN)
    print("\n" + "=" * 66)
    print(f"result: {len(_results)} checks, FAIL {n_bad}, warn {n_warn}")
    if n_bad:
        print("There are FAIL items; handle them per the hints above first. See REPRODUCE.md.")
    elif n_warn:
        print("No FAIL. Whether the warn items need handling depends on which stage you plan to run.")
    else:
        print("All passed.")
    print("=" * 66)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
