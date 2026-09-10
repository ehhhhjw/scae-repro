"""按仓库隔离的构图 + 聚类编排器（通道 B 数据层）。

为什么需要这一层
----------------
上游两套工具的输出路径都是**硬编码常量**：
  AST_Tree_Graph:   src/split_file/const.py 中 VERSION="v9"，OUTPUT_DIR=Output/v9
  rag-python-tools: 产物固定命名 .output/cluster/v9_*.json
连续对多个仓库构图会**互相覆盖**（实测：astropy-7606 产物已被 12907 覆盖）。
499 仓库若不隔离则无法全量。

设计原则：**不修改上游代码**，在外层做"跑一个 → 校验 → 归档 → 下一个"，
已归档的仓库幂等跳过。

产物布局
--------
    pilot/runs/graphs/<instance_id>/
        entities.json              实体 + 关系（含 LineNumber，用于内容级归因）
        dependency_graph.json      文件级 import 图
        cluster_communities.json   社区级统计（cut_ratio / boundary_weight）
        cluster_entities.json      每实体 CommunityId
        cluster_report.json        modularity / noise
        meta.json                  仓库路径、耗时、归档校验
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.probes import task as T                          # noqa: E402

SCAE = Path(__file__).resolve().parents[2]
AST_DIR = SCAE / "tree-sitter-fold" / "AST_Tree_Graph"
AST_OUT = AST_DIR / "Output" / "v9"
CLUSTER_DIR = SCAE / "graph_leidan_fold" / "rag-python-tools"
CLUSTER_OUT = CLUSTER_DIR / ".output" / "cluster"
GRAPHS_ROOT = SCAE / "pilot" / "runs" / "graphs"

REQUIRED = ("entities.json", "dependency_graph.json")
CLUSTER_MAP = (("v9_communities.json", "cluster_communities.json"),
               ("v9_entities_clustered.json", "cluster_entities.json"),
               ("v9_cluster_report.json", "cluster_report.json"))


def archived(instance_id: str) -> bool:
    d = GRAPHS_ROOT / instance_id
    return all((d / f).exists() for f in REQUIRED)


def build_one(repo: Path, instance_id: str, do_cluster: bool = True,
              timeout: float = 1800.0) -> dict:
    out = GRAPHS_ROOT / instance_id
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {"instance_id": instance_id, "repo": str(repo)}
    t0 = time.time()

    try:
        p = subprocess.run(["./scripts/run_pipeline.sh", str(repo), "--no-neo4j"],
                           cwd=str(AST_DIR), capture_output=True, text=True,
                           timeout=timeout)
        meta["graph_rc"] = p.returncode
        if p.returncode != 0:
            meta["graph_stderr"] = p.stderr[-400:]
    except subprocess.TimeoutExpired:
        meta.update(graph_rc=-1, error=f"构图超时 {timeout}s")
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta
    meta["graph_sec"] = round(time.time() - t0, 1)

    # 校验产物确实属于本仓库 —— 防止把别的仓库数据归档过来
    dg = AST_OUT / "dependency_graph.json"
    if dg.exists():
        try:
            root = json.loads(dg.read_text()).get("root_dir", "")
            meta["root_dir_check"] = root
            if Path(root).resolve() != repo.resolve():
                meta["error"] = f"root_dir 不匹配: {root}"
                (out / "meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2))
                return meta
        except (json.JSONDecodeError, OSError) as e:
            meta["error"] = f"dependency_graph 解析失败: {e}"

    if do_cluster and "error" not in meta:
        t1 = time.time()
        py = CLUSTER_DIR / ".venv" / "bin" / "python"
        if py.exists():
            try:
                p2 = subprocess.run([str(py), "src/main.py", "--tool", "cluster"],
                                    cwd=str(CLUSTER_DIR), capture_output=True,
                                    text=True, timeout=timeout)
                meta["cluster_rc"] = p2.returncode
                if p2.returncode != 0:
                    meta["cluster_stderr"] = p2.stderr[-400:]
            except subprocess.TimeoutExpired:
                meta.update(cluster_rc=-1, cluster_error="聚类超时")
        else:
            meta.update(cluster_rc=-2, cluster_error=f"聚类 venv 缺失: {py}")
        meta["cluster_sec"] = round(time.time() - t1, 1)

    copied = {}
    for f in REQUIRED:
        src = AST_OUT / f
        if src.exists():
            shutil.copy(src, out / f)
            copied[f] = src.stat().st_size
    for src_name, dst_name in CLUSTER_MAP:
        src = CLUSTER_OUT / src_name
        if src.exists():
            shutil.copy(src, out / dst_name)
            copied[dst_name] = src.stat().st_size
    meta["archived"] = copied
    meta["total_sec"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--from-attr", default=None,
                    help="从归因结果取 instance 列表（优先做有归因数据的仓库）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cluster", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    insts = {i.instance_id: i for i in T.load_instances()}
    if args.instances:
        ids = [x for x in args.instances if x in insts]
    elif args.from_attr:
        ids = sorted({json.loads(l)["instance_id"]
                      for l in Path(args.from_attr).read_text().splitlines()
                      if l.strip() and not json.loads(l).get("error")}
                     & set(insts))
    else:
        ids = sorted(insts)
    if args.limit:
        ids = ids[: args.limit]

    todo = [i for i in ids if args.force or not archived(i)]
    print(f"目标 {len(ids)} 仓库，待处理 {len(todo)}（跳过已归档 {len(ids)-len(todo)}）")
    GRAPHS_ROOT.mkdir(parents=True, exist_ok=True)

    times, n_ok = [], 0
    for k, iid in enumerate(todo, 1):
        inst = insts[iid]
        if not inst.repo.is_dir():
            print(f"[{k}/{len(todo)}] {iid} 仓库不存在")
            continue
        m = build_one(inst.repo, iid, do_cluster=not args.no_cluster,
                      timeout=args.timeout)
        ok = "error" not in m and m.get("graph_rc") == 0
        n_ok += ok
        times.append(m.get("total_sec", 0))
        eta = sum(times) / len(times) * (len(todo) - k) / 60
        print(f"[{k}/{len(todo)}] {iid[:32]:<32} {'OK ' if ok else 'FAIL'} "
              f"g={m.get('graph_sec')}s c={m.get('cluster_sec')}s "
              f"ETA={eta:.0f}min {str(m.get('error',''))[:40]}", flush=True)
    if times:
        avg = sum(times) / len(times)
        print(f"\n成功 {n_ok}/{len(todo)}；平均 {avg:.1f}s/仓库 → "
              f"499 仓库约 {avg*499/3600:.1f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
