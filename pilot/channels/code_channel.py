r"""通道 B：代码关系 → U^obj。消费按仓库隔离的构图/聚类产物。零 API。

输入（pilot/runs/graphs/<instance_id>/）
--------------------------------------
  entities.json            实体 + Dependent 关系 + LineNumber
  dependency_graph.json    文件级 import 图
  cluster_entities.json    每实体 CommunityId
  cluster_communities.json 社区级 cut_ratio / boundary_weight

U^obj 的定义
-----------
agent 在第 t 步操作对象 obj_t（文件路径）。下一步对象的候选按结构分三层：

  L1 一跳邻居    dependency_graph 上与 obj_t 直接相连的文件
  L2 同社区      与 obj_t 同 CommunityId 的文件
  L3 跨社区      其余文件（按社区 boundary_weight 加权）

$U^{obj}_t = H\big(P(\text{obj}_{t+1} \mid \text{obj}_t, G)\big)$

其中 P 由三层的**结构先验**给出（不拟合、不看标签）：
    P(L1 内均匀) * w1 + P(L2 内均匀) * w2 + P(L3 内均匀) * w3
权重 w 反映"agent 倾向于就近移动"，由 E3 的实证支撑：
548 个 py→py 相邻步对中 62% 在一跳内、97% 在 3 跳内（中位距离 1）。

这样 U^obj 完全由代码结构决定，与成败标签无关 —— 这是绕开 94% 基线率的关键。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

SCAE = Path(__file__).resolve().parents[2]
GRAPHS = SCAE / "pilot" / "runs" / "graphs"

# 由 E3 实测的相邻步图距离分布定出的层权重（62% 一跳 / 中位 1 / 97% 三跳内）
W_L1, W_L2, W_L3 = 0.62, 0.30, 0.08


class CodeGraph:
    """单仓库的代码关系视图。"""

    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        d = GRAPHS / instance_id
        self.ok = (d / "dependency_graph.json").exists()
        self.edges: dict[str, set[str]] = defaultdict(set)
        self.files: set[str] = set()
        self.community: dict[str, int] = {}
        self.comm_files: dict[int, set[str]] = defaultdict(set)
        self.comm_stats: dict[int, dict] = {}
        if not self.ok:
            return
        self._load_dep(d / "dependency_graph.json")
        self._load_cluster(d / "cluster_entities.json", d / "cluster_communities.json")

    # ---------------- 载入 ----------------

    def _load_dep(self, p: Path) -> None:
        try:
            dg = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            self.ok = False
            return
        files = dg.get("files")
        if isinstance(files, dict):
            items = files.items()
        elif isinstance(files, list):
            items = [(f.get("path") or f.get("file") or "", f) for f in files]
        else:
            items = []
        for path, info in items:
            if not path:
                continue
            self.files.add(path)
            deps = []
            if isinstance(info, dict):
                for k in ("imports", "dependencies", "deps", "Dependent"):
                    v = info.get(k)
                    if isinstance(v, list):
                        deps += [str(x) for x in v]
                    elif isinstance(v, dict):
                        deps += [str(x) for x in v.keys()]
            for dep in deps:
                if dep and dep != path:
                    self.edges[path].add(dep)
                    self.edges[dep].add(path)      # 视为无向，反映"可达"
                    self.files.add(dep)

    def _load_cluster(self, ents: Path, comms: Path) -> None:
        if ents.exists():
            try:
                for e in json.loads(ents.read_text()):
                    root = e.get("Root") or ""
                    cid = e.get("CommunityId")
                    if root and cid is not None:
                        self.community[root] = cid
                        self.comm_files[cid].add(root)
                        self.files.add(root)
            except (OSError, json.JSONDecodeError):
                pass
        if comms.exists():
            try:
                for c in json.loads(comms.read_text()):
                    cid = c.get("community_id")
                    if cid is not None:
                        self.comm_stats[cid] = {
                            "file_count": c.get("file_count"),
                            "cut_ratio": c.get("cut_ratio"),
                            "boundary_weight": c.get("boundary_weight"),
                        }
            except (OSError, json.JSONDecodeError):
                pass

    # ---------------- 查询 ----------------

    @lru_cache(maxsize=4096)
    def neighbors(self, f: str) -> frozenset[str]:
        return frozenset(self.edges.get(f, set()))

    def same_community(self, f: str) -> frozenset[str]:
        cid = self.community.get(f)
        if cid is None:
            return frozenset()
        return frozenset(self.comm_files.get(cid, set()) - {f})

    def resolve(self, obj: str | None) -> str | None:
        """把 agent 的操作对象解析到图节点。

        V4 首轮诊断发现 U_obj 缺失 63%，主因不是数据缺失而是对象抽取过窄：
          - search 类动作只有 query 没有 path（67 个观测）
          - list_files 的目标是目录而非文件（62 个观测）
        这里补三条解析路径，把覆盖率提上去：
          1. 精确匹配文件
          2. 目录 → 该目录下所有文件的并集视为"目录节点"
          3. query 字符串 → 在符号名/文件名里做后缀匹配
        """
        if not obj or obj == "-" or not self.ok:
            return None
        if obj in self.files:
            return obj
        o = obj.lstrip("./").rstrip("/")
        if o in self.files:
            return o
        # 目录：返回一个虚拟目录节点（其邻域=目录下文件）
        if any(x.startswith(o + "/") for x in self.files):
            return f"dir:{o}"
        # query：后缀匹配文件名
        if o.startswith("q:"):
            q = o[2:].strip("'\"")
            if len(q) >= 4:
                hits = [x for x in self.files if q in x]
                if hits:
                    return sorted(hits)[0]
            return None
        base = o.split("/")[-1]
        if len(base) >= 4:
            hits = [x for x in self.files if x.endswith("/" + base) or x == base]
            if len(hits) == 1:
                return hits[0]
        return None

    def _dir_members(self, d: str) -> frozenset[str]:
        return frozenset(x for x in self.files if x.startswith(d + "/"))

    def u_obj_resolved(self, obj: str | None) -> dict:
        """先 resolve 再算 U_obj，支持目录节点。"""
        node = self.resolve(obj)
        if node is None:
            return {"H": None, "reason": "无法解析到图节点", "resolved": None}
        if node.startswith("dir:"):
            d = node[4:]
            members = self._dir_members(d)
            if not members:
                return {"H": None, "reason": "空目录", "resolved": node}
            # 目录节点：候选=目录内文件（L1）+ 同社区（L2）+ 其余（L3）
            n_all = len(self.files)
            L1 = members
            cids = {self.community.get(m) for m in members} - {None}
            L2 = frozenset().union(*[self.comm_files.get(c, set())
                                     for c in cids]) - L1 if cids else frozenset()
            L3n = max(0, n_all - len(L1) - len(L2))
            ps = []
            for size, w in ((len(L1), W_L1), (len(L2), W_L2), (L3n, W_L3)):
                if size > 0:
                    ps += [w / size] * size
            if not ps:
                return {"H": None, "reason": "无候选", "resolved": node}
            z = sum(ps)
            H = -sum((p / z) * math.log(p / z) for p in ps if p > 0)
            return {"H": round(H, 4), "n_l1": len(L1), "n_l2": len(L2),
                    "n_l3": L3n, "in_graph": True, "resolved": node,
                    "H_max": round(math.log(n_all), 4) if n_all > 1 else 0.0}
        r = self.u_obj(node)
        r["resolved"] = node
        return r

    def u_obj(self, f: str) -> dict:
        """结构不确定性：下一步对象分布的熵，由三层结构先验给出。

        返回 H 及三层规模，便于分解分析。H 越大表示"下一步可去的地方越多"。
        """
        if not self.ok or not self.files:
            return {"H": None, "reason": "无图数据"}
        n_all = len(self.files)
        L1 = self.neighbors(f)
        L2 = self.same_community(f) - L1
        L3n = max(0, n_all - len(L1) - len(L2) - 1)
        ps: list[float] = []
        for size, w in ((len(L1), W_L1), (len(L2), W_L2), (L3n, W_L3)):
            if size > 0:
                ps += [w / size] * size
        if not ps:
            return {"H": None, "reason": "对象不在图中或无候选",
                    "n_l1": len(L1), "n_l2": len(L2), "n_l3": L3n}
        z = sum(ps)
        H = -sum((p / z) * math.log(p / z) for p in ps if p > 0)
        return {"H": round(H, 4), "n_l1": len(L1), "n_l2": len(L2), "n_l3": L3n,
                "in_graph": f in self.files,
                "community": self.community.get(f),
                "H_max": round(math.log(n_all), 4) if n_all > 1 else 0.0}

    def graph_distance(self, a: str, b: str, max_hops: int = 3) -> int | None:
        if a == b:
            return 0
        frontier, seen = {a}, {a}
        for d in range(1, max_hops + 1):
            nxt: set[str] = set()
            for x in frontier:
                nxt |= self.edges.get(x, set())
            nxt -= seen
            if b in nxt:
                return d
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return None

    def summary(self) -> dict:
        return {"instance_id": self.instance_id, "ok": self.ok,
                "n_files": len(self.files),
                "n_edges": sum(len(v) for v in self.edges.values()) // 2,
                "n_communities": len(self.comm_files),
                "n_files_with_community": len(self.community)}


@lru_cache(maxsize=128)
def load_graph(instance_id: str) -> CodeGraph:
    return CodeGraph(instance_id)


def available() -> list[str]:
    if not GRAPHS.exists():
        return []
    return sorted(d.name for d in GRAPHS.iterdir()
                  if (d / "dependency_graph.json").exists())
