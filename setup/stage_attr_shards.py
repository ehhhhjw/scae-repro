#!/usr/bin/env python3
"""为归因阶段准备分片目录。

背景
----
`pilot.harness.parallel --cmd attr` 不自己挑实例：它要求 `--out` 下已经存在
`shard*/`，每个 shard 里有 `results.jsonl`（该分片负责的定位结果）和 `rollouts/`
（可读到的 rollout JSONL）。原始实验里这步是手工做的，本脚本把它固定下来，
使第 2 阶段可复现。

做的事
------
1. 读定位阶段的结果（merged results.jsonl 或 shard*/results.jsonl）
2. 按 instance_id 去重，只保留 ok 且能找到 rollout 的记录
3. 按 repo 贪心分片（复用 pilot.harness.parallel.shard_by_repo，保证同一仓库
   目录同一时刻只被一个 worker 使用）
4. 每个 shard 写出 results.jsonl，并把 rollouts 软链（默认）或复制进去

用法
----
# 归因主样本（论文 D.10 的 70 episode，m=3，供 RQ5 judge 分析）
python3 setup/stage_attr_shards.py --locate pilot/runs/full_locate \
    --out pilot/runs/full_attr --workers 6

# m=8 子样本（论文 §5.6 的 190 估值点 / 38 episode，供 RQ4 步级功效分析）
python3 setup/stage_attr_shards.py --locate pilot/runs/full_locate \
    --out pilot/runs/v4prime --workers 6 --limit 38

注意
----
实例筛选（只留多文件实例）由 run_attribution 的 `--multifile-only` 完成，
本脚本不预筛，以免和论文口径产生第二处真相来源。
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
    """读定位结果，按 instance_id 去重（ok 记录优先）。"""
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
            # ok 记录优先；同为 ok 时保留步数更多的那条
            old = uniq.get(iid)
            if (
                old is None
                or (r.get("ok") and not old.get("ok"))
                or (r.get("ok") == old.get("ok") and (r.get("n_steps") or 0) > (old.get("n_steps") or 0))
            ):
                uniq[iid] = r
    return uniq


def find_rollouts(locate_dir: Path) -> dict[str, Path]:
    """收集 instance_id -> rollout 路径（取步数最全的 k0）。"""
    out: dict[str, Path] = {}
    for p in sorted(locate_dir.glob("shard*/rollouts/*.jsonl")) + sorted(
        locate_dir.glob("rollouts/*.jsonl")
    ):
        iid = p.name.split("__k")[0]
        # 同名多份时保留更大的（更完整）
        if iid not in out or p.stat().st_size > out[iid].stat().st_size:
            out[iid] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="准备归因分片目录")
    ap.add_argument("--locate", type=Path, default=REPO_ROOT / "pilot" / "runs" / "full_locate")
    ap.add_argument("--out", type=Path, required=True, help="归因输出根目录，如 pilot/runs/full_attr")
    ap.add_argument("--workers", type=int, default=6, help="分片数（= 并行 worker 数）")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 个多文件实例（m=8 子样本用）")
    ap.add_argument("--copy", action="store_true", help="复制 rollouts 而非软链（默认软链省磁盘）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 shard 目录")
    args = ap.parse_args()

    if not args.locate.exists():
        sys.exit(f"找不到定位结果目录 {args.locate}，先跑 scripts/01_locate.sh")

    uniq = load_locate(args.locate)
    rollouts = find_rollouts(args.locate)
    if not uniq:
        sys.exit(f"{args.locate} 下没读到任何定位结果。")
    if not rollouts:
        sys.exit(f"{args.locate} 下没找到 rollouts/*.jsonl —— 归因需要 rollout 才能重放前缀。")

    # 只保留 ok 且有 rollout 的
    usable = {i: r for i, r in uniq.items() if r.get("ok") and i in rollouts}
    dropped_notok = sum(1 for r in uniq.values() if not r.get("ok"))
    dropped_noroll = sum(1 for i, r in uniq.items() if r.get("ok") and i not in rollouts)

    # 多文件统计（供核对论文口径；实际筛选由 --multifile-only 做）
    try:
        gold = {i.instance_id: i.gold for i in T.load_instances()}
        multi = [i for i in usable if len(gold.get(i, ())) > 1]
    except Exception:  # noqa: BLE001
        multi = []

    ids = sorted(usable)
    if args.limit > 0:
        # 子样本按多文件优先取，保证 --multifile-only 之后还有足够 episode
        prefer = sorted(multi) if multi else ids
        ids = prefer[: args.limit]

    print(f"定位结果   {len(uniq)} 唯一实例")
    print(f"  可用     {len(usable)}（丢弃：非 ok {dropped_notok}，无 rollout {dropped_noroll}）")
    if multi:
        print(f"  多文件   {len(multi)}  ← 归因实际只跑这些（--multifile-only）")
    print(f"本次分片   {len(ids)} 实例 → {args.workers} 个 shard")

    args.out.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.out.glob("shard*"))
    if existing and not args.force:
        sys.exit(f"{args.out} 下已有 {len(existing)} 个 shard 目录。加 --force 覆盖。")
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
        # rollouts：归因要能按 instance 找到 rollout，链全量最省事也最安全
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
        print(f"  shard{i}: {len(sids)} 实例 → {sdir}")

    mode = "复制" if args.copy else "软链"
    print(f"\n完成：{len(shards)} 个 shard，{total_links} 个 rollout（{mode}）")
    print(f"下一步：bash scripts/02_attribution.sh  （或直接 python3 -m pilot.harness.parallel --cmd attr --out {args.out} ...）")


if __name__ == "__main__":
    main()
