"""并行执行编排 —— 全量实验的必需组件。

实测串行速度：单次 fork 续跑 ~200s（29 步任务）。全量归因 71 任务串行需 ~71h。
并行化后（W=6）可压到 ~12h。

隔离要点（每个 worker 必须完全独立，否则会互相污染）：
  - 独立 CODEX_HOME（含各自 config.toml）→ rollout 不混
  - 独立 AppServer 子进程 → JSON-RPC 不串
  - **仓库隔离**：多个 worker 同时 reset 同一个仓库会互相破坏。
    因此按 instance_id 分片，保证同一仓库目录同一时刻只被一个 worker 使用。
  - 独立结果文件，最后合并（避免并发写同一 jsonl）

用法：作为库被 run_full.py 调用；也可直接跑
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
    """按仓库分片：同一 instance 的仓库目录只归一个 shard。

    数据集里每个 instance_id 有自己独立的仓库目录（projects/<instance_id>），
    所以实际上不存在跨 instance 的目录竞争；但按 project 分组仍有好处：
    同一 project 的仓库大小相近，能让各 shard 负载更均衡。
    """
    by_proj: dict[str, list[str]] = {}
    for iid in instance_ids:
        by_proj.setdefault(iid.split("__")[0], []).append(iid)
    # 贪心装箱：按组大小降序，塞进当前最小的 shard
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
    """并行启动定位 runner，每 shard 一个进程。"""
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
        print(f"  shard{i}: {len(ids)} 实例 → PID {p.pid}", flush=True)
    return procs


def launch_attr_shards(
    runs_dirs: list[Path], out_root: Path, workers: int,
    model: str, provider: str, m: int, window_points: int,
    outcome: str, turn_timeout: float,
) -> list[subprocess.Popen]:
    """并行启动归因，每 shard 一个进程（输入已按 shard 切好的 results.jsonl）。"""
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
        print(f"[{label}] {len(procs)-len(alive)}/{len(procs)} shard 完成 "
              f"({(time.time()-t0)/60:.0f}min)", flush=True)
        time.sleep(poll)
    codes = [p.returncode for p in procs]
    print(f"[{label}] 全部完成，退出码 {codes}，用时 {(time.time()-t0)/60:.1f}min",
          flush=True)


def merge_jsonl(shard_dirs: list[Path], fname: str, out: Path) -> int:
    """合并各 shard 的 jsonl。"""
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
        print(f"全量定位：{len(ids)} 实例，{args.workers} 并行")
        procs = launch_locate_shards(ids, out_root, args.workers,
                                     args.model, args.model_provider,
                                     args.k, args.turn_timeout)
        wait_and_report(procs, "locate")
        shard_dirs = sorted(out_root.glob("shard*"))
        n = merge_jsonl(shard_dirs, "results.jsonl", out_root / "results.jsonl")
        print(f"合并 {n} 条结果 → {out_root/'results.jsonl'}")
    else:
        shard_dirs = sorted(out_root.glob("shard*"))
        print(f"全量归因：{len(shard_dirs)} shard")
        procs = launch_attr_shards(shard_dirs, out_root, args.workers,
                                   args.model, args.model_provider, args.m,
                                   args.window_points, args.outcome,
                                   args.turn_timeout)
        wait_and_report(procs, "attr")
        n = merge_jsonl(shard_dirs, "attribution.jsonl",
                        out_root / "attribution.jsonl")
        print(f"合并 {n} 条归因 → {out_root/'attribution.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
