#!/usr/bin/env python3
"""从 SWE-bench-Verified 构建 SCAE 的数据集接口，并落地仓库快照。

复现包不携带数据。本脚本负责第 1 步：把 SWE-bench-Verified 转成
`pilot/probes/task.py` 期望的两样东西：

  1. instance_summary.csv    字段 instance_id, base_commit, problem_statement, patch_file
                             patch_file = gold patch 触及的文件路径（换行分隔）= 论文的 F*
  2. projects/<instance_id>/ 每实例一个 git 仓库快照，HEAD 停在 base_commit

用法
----
# A) 已有 SWE-bench-Verified 的 JSON（list of records，含 instance_id/base_commit/
#    problem_statement/patch 字段）
python3 setup/prepare_dataset.py --from-json /path/to/instance_questions.json

# B) 从 HuggingFace 拉取（需 `pip install datasets`）
python3 setup/prepare_dataset.py --from-hf

# C) 只写 CSV，不克隆仓库（约 900KB，秒级完成）
python3 setup/prepare_dataset.py --from-hf --skip-clone

# D) 克隆/校验仓库快照（磁盘约 20-40GB，耗时以小时计）
python3 setup/prepare_dataset.py --from-json data.json --clone --workers 4

# E) 只校验已有快照是否停在 base_commit（体检，不改动任何东西）
python3 setup/prepare_dataset.py --verify-only

为什么 gold 集要从 patch 里现算
------------------------------
论文的结果口径是 recall over F*（Appendix C.6），F* 的定义就是"reference patch 修改的文件
集合"。直接从 patch 的 unified diff 头解析，保证 F* 与论文定义逐字一致，而不依赖任何
下游数据集自带的字段。
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

# --- gold patch 的文件路径解析（与 SWE-bench 的 diff 格式一致） -------------------
# "diff --git a/<path> b/<path>"，路径含特殊字符时会被引号包裹
DIFF_GIT_RE = re.compile(
    r'^diff --git\s+(?:"a/(?P<qa>.+?)"|a/(?P<a>\S+))\s+(?:"b/(?P<qb>.+?)"|b/(?P<b>.+?))\s*$'
)
OLD_FILE_RE = re.compile(r'^--- (?:"?a/)?(?P<path>.+?)"?\s*$')
NEW_FILE_RE = re.compile(r'^\+\+\+ (?:"?b/)?(?P<path>.+?)"?\s*$')


def extract_diff_paths(patch: str) -> list[str]:
    """从 unified diff 中提取被修改的文件路径，保持出现顺序并去重。

    优先取 b/ 侧（新路径）；删除文件时 b 侧为 /dev/null，退回 a/ 侧。
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

    # 没有 "diff --git" 头时退回 ---/+++ 行
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


# --- 数据源 ---------------------------------------------------------------------
def load_from_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [data] if isinstance(data, dict) else data


def load_from_hf(split: str = "test") -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "缺少 `datasets` 包。请 `pip install datasets`，"
            "或改用 --from-json 指向本地 SWE-bench-Verified JSON。"
        )
    print(f"从 HuggingFace 拉取 {HF_DATASET} [{split}] ...")
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


# --- 仓库快照 -------------------------------------------------------------------
def repo_url(instance_id: str) -> str:
    """astropy__astropy-12907 -> https://github.com/astropy/astropy"""
    stem = instance_id.rsplit("-", 1)[0]
    if "__" not in stem:
        raise ValueError(f"无法从 instance_id 推出仓库名: {instance_id}")
    org, name = stem.split("__", 1)
    return f"https://github.com/{org}/{name}"


def _git(cwd: Path | None, *args: str, timeout: float = 1800) -> subprocess.CompletedProcess:
    cmd = ["git"] + (["-C", str(cwd)] if cwd else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def ensure_repo(instance_id: str, base_commit: str, projects: Path) -> tuple[str, str]:
    """确保 projects/<instance_id> 存在且 HEAD == base_commit。返回 (状态, 说明)。"""
    dest = projects / instance_id
    try:
        if not (dest / ".git").exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = repo_url(instance_id)
            p = _git(None, "clone", "--quiet", url, str(dest))
            if p.returncode != 0:
                return "clone_failed", p.stderr.strip()[:200]
        # 取到指定 commit（浅克隆或默认分支不含该 commit 时需要 fetch）
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
        return "timeout", "git 操作超时"
    except Exception as e:  # noqa: BLE001
        return "error", f"{type(e).__name__}: {e}"[:200]


def verify_repo(instance_id: str, base_commit: str, projects: Path) -> tuple[str, str]:
    """只读体检：HEAD 是否 == base_commit，工作区是否干净。"""
    dest = projects / instance_id
    if not (dest / ".git").exists():
        return "missing", "仓库快照不存在"
    head = _git(dest, "rev-parse", "HEAD").stdout.strip()
    if head != base_commit:
        return "wrong_commit", f"HEAD={head[:12]} != base={base_commit[:12]}"
    dirty = _git(dest, "status", "--porcelain").stdout.strip()
    if dirty:
        n = len(dirty.splitlines())
        return "dirty", f"工作区有 {n} 处未提交改动（gold patch 泄题风险）"
    return "ok", ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="构建 SCAE 数据集接口（instance_summary.csv + 仓库快照）"
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-json", type=Path, help="本地 SWE-bench-Verified JSON")
    src.add_argument("--from-hf", action="store_true", help=f"从 HuggingFace 拉取 {HF_DATASET}")
    ap.add_argument("--hf-split", default="test")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"输出 CSV（默认 {DEFAULT_CSV}）")
    ap.add_argument("--projects", type=Path, default=DEFAULT_PROJECTS)
    ap.add_argument("--clone", action="store_true", help="克隆/校正仓库快照（几十 GB，数小时）")
    ap.add_argument("--skip-clone", action="store_true", help="只写 CSV（默认行为）")
    ap.add_argument("--verify-only", action="store_true", help="只体检现有快照，不改动")
    ap.add_argument("--workers", type=int, default=4, help="并行 git 任务数")
    ap.add_argument("--limit", type=int, help="只处理前 N 个实例（调试用）")
    args = ap.parse_args()

    # --- 只体检 ---
    if args.verify_only:
        if not args.csv.exists():
            sys.exit(f"找不到 {args.csv}，先跑一次 --from-json/--from-hf 生成 CSV。")
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
                problems.append(f"  {r['instance_id']}: {st} — {msg}")
        print(f"\n体检 {len(rows)} 个实例（projects={args.projects}）:")
        for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
            print(f"  {k:14s} {v}")
        if problems:
            print(f"\n前 20 个问题实例:")
            for line in problems[:20]:
                print(line)
            print(
                "\n⚠️  非 ok 的实例会导致 agent 读到已修复代码（泄题）或重放不可比。"
                "\n   跑 --clone 修正，或见 docs/PITFALLS.md。"
            )
        sys.exit(0 if not problems else 1)

    if not args.from_json and not args.from_hf:
        ap.error("需指定数据源：--from-json <path> 或 --from-hf（或用 --verify-only 体检）")

    records = load_from_json(args.from_json) if args.from_json else load_from_hf(args.hf_split)
    if args.limit:
        records = records[: args.limit]

    info = write_csv(records, args.csv)
    print("\n=== instance_summary.csv ===")
    print(f"  实例总数        {info['n']}")
    print(f"  多文件实例      {info['multi_file']}")
    print(f"  未解析出 gold   {info['no_gold_path']}")
    print(f"  输出            {info['csv']}")
    if info["no_gold_path"]:
        print("  ⚠️  有实例未解析出 gold 路径，这些实例的 F* 为空，需从语料中剔除。")

    # --- 仓库快照 ---
    if not args.clone or args.skip_clone:
        print(
            f"\n未克隆仓库快照（默认跳过）。定位实验需要它们：\n"
            f"  python3 setup/prepare_dataset.py --from-{'json ' + str(args.from_json) if args.from_json else 'hf'} --clone\n"
            f"或把 SCAE_PROJECTS 指向已有快照目录。"
        )
        return

    with open(args.csv, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    print(f"\n克隆/校正 {len(rows)} 个仓库到 {args.projects}（workers={args.workers}）...")
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
                print(f"  [{i}/{len(rows)}] {iid}: {st} — {msg}")
            elif i % 25 == 0:
                print(f"  [{i}/{len(rows)}] ok={stats.get('ok', 0)}")
    print("\n=== 仓库快照 ===")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v}")
    print(
        "\n注意：定位实验每次跑任务前会再次强制 reset 到 base_commit"
        "（pilot/probes/task.py::reset_repo），这是防泄题的最后一道闸。"
    )


if __name__ == "__main__":
    main()
