#!/usr/bin/env python3
"""环境自检：在跑任何实验之前确认前置条件齐备。

分两档报告：
  [离线分析]  只需 python3 + numpy。够用就能跑 scripts/04_analysis_offline.sh
  [数据采集]  另需 codex CLI + API key + 数据集快照。够用才能重跑 12.7 小时的 agent 执行

用法
----
python3 setup/check_env.py
python3 setup/check_env.py --collect   # 同时检查数据采集档的前置条件
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
    print(f"[{level}] {item}" + (f" — {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 10):
        record(OK, "Python", f"{s}（要求 >= 3.10）")
    else:
        record(BAD, "Python", f"{s}，但代码用了 3.10+ 语法（X | Y 类型标注）")


def check_numpy() -> None:
    if importlib.util.find_spec("numpy") is None:
        record(BAD, "numpy", "未安装 —— 部分分析脚本需要。pip install numpy")
        return
    import numpy as np

    record(OK, "numpy", np.__version__)


def check_package_imports() -> None:
    """确认 pilot 包可被导入（复现包完整性）。"""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import pilot.probes.task as T  # noqa: F401
        import pilot.harness.rollout  # noqa: F401
        import pilot.attribution.causal  # noqa: F401
        import pilot.uq.predictor  # noqa: F401

        record(OK, "pilot 包导入", "harness / attribution / uq / probes 均可导入")
    except Exception as e:  # noqa: BLE001
        record(BAD, "pilot 包导入", f"{type(e).__name__}: {e}")


def check_dataset() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from pilot.probes import task as T

    csv_path, projects = T.CSV_PATH, T.PROJECTS_ROOT
    src_csv = "SCAE_CSV" if os.environ.get("SCAE_CSV") else "默认"
    src_prj = "SCAE_PROJECTS" if os.environ.get("SCAE_PROJECTS") else "默认"

    if not csv_path.exists():
        record(
            BAD,
            f"instance_summary.csv（{src_csv}）",
            f"找不到 {csv_path}。跑 setup/prepare_dataset.py 生成，或 export SCAE_CSV=...",
        )
        return
    try:
        insts = T.load_instances()
    except Exception as e:  # noqa: BLE001
        record(BAD, "instance_summary.csv 解析", f"{type(e).__name__}: {e}")
        return

    n_gold0 = sum(1 for i in insts if not i.gold)
    multi = sum(1 for i in insts if len(i.gold) > 1)
    detail = f"{len(insts)} 实例，多文件 {multi}，无 gold {n_gold0}（{csv_path}）"
    record(OK if n_gold0 == 0 else WARN, f"instance_summary.csv（{src_csv}）", detail)

    if not projects.exists():
        record(
            WARN,
            f"仓库快照（{src_prj}）",
            f"找不到 {projects} —— 只跑离线分析可忽略；定位实验必需",
        )
        return
    present = sum(1 for i in insts if (projects / i.instance_id / ".git").exists())
    lvl = OK if present == len(insts) else WARN
    record(lvl, f"仓库快照（{src_prj}）", f"{present}/{len(insts)} 个实例有 git 快照（{projects}）")
    if present:
        # 抽查 base_commit，泄题是本项目最大的坑
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
                "base_commit 抽查",
                f"{len(bad)}/{len(sample)} 个快照 HEAD != base_commit（跑前会自动 reset，"
                f"但请确认没有 gold patch 叠加）",
            )
        else:
            record(OK, "base_commit 抽查", f"抽查 {len(sample)} 个，HEAD 均等于 base_commit")


def check_codex() -> None:
    exe = shutil.which(os.environ.get("SCAE_CODEX_BIN", "codex"))
    if not exe:
        record(BAD, "codex CLI", "不在 PATH 中 —— 数据采集必需（论文用 codex 0.145.0）")
        return
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
        ver = (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else "?"
        record(OK, "codex CLI", f"{ver}（{exe}）")
    except Exception as e:  # noqa: BLE001
        record(WARN, "codex CLI", f"存在但无法取版本: {e}")


def check_api_key() -> None:
    key = os.environ.get("IDEALAB_API_KEY")
    if key:
        record(OK, "IDEALAB_API_KEY", f"已设置（{len(key)} 字符）")
    else:
        record(BAD, "IDEALAB_API_KEY", "未设置 —— 数据采集与 judge 实验必需")


def check_git() -> None:
    exe = shutil.which("git")
    if not exe:
        record(BAD, "git", "不在 PATH 中 —— 仓库 reset 必需")
        return
    r = subprocess.run(["git", "--version"], capture_output=True, text=True)
    record(OK, "git", r.stdout.strip())


def check_runs_dir() -> None:
    runs = REPO_ROOT / "pilot" / "runs"
    if not runs.exists():
        record(WARN, "pilot/runs/", "不存在 —— 采集阶段会自动创建（复现包不携带录制数据）")
        return
    n = len(list(runs.glob("*")))
    record(OK if n else WARN, "pilot/runs/", f"{n} 个条目" if n else "空目录（尚未采集）")


def main() -> None:
    ap = argparse.ArgumentParser(description="SCAE 复现环境自检")
    ap.add_argument("--collect", action="store_true", help="同时检查数据采集档前置条件")
    args = ap.parse_args()

    print("=" * 66)
    print("SCAE 复现环境自检")
    print("=" * 66)
    print("\n--- 离线分析档（只需 python + numpy）---")
    check_python()
    check_numpy()
    check_package_imports()
    check_runs_dir()

    print("\n--- 数据集 ---")
    check_dataset()

    if args.collect:
        print("\n--- 数据采集档（重跑 agent 执行）---")
        check_git()
        check_codex()
        check_api_key()

    n_bad = sum(1 for lvl, _, _ in _results if lvl == BAD)
    n_warn = sum(1 for lvl, _, _ in _results if lvl == WARN)
    print("\n" + "=" * 66)
    print(f"结果：{len(_results)} 项检查，FAIL {n_bad}，warn {n_warn}")
    if n_bad:
        print("有 FAIL 项，先按上面的提示处理。见 REPRODUCE.md。")
    elif n_warn:
        print("无 FAIL。warn 项按你要跑的阶段决定是否需要处理。")
    else:
        print("全部通过。")
    print("=" * 66)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
