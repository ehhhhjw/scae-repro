r"""Channel B: code relations → U^obj. Consumes per-repository graph/cluster artifacts. Zero API.

Inputs (pilot/runs/graphs/<instance_id>/)
-----------------------------------------
  entities.json            entities + Dependent relations + LineNumber
  dependency_graph.json    file-level import graph
  cluster_entities.json    CommunityId per entity
  cluster_communities.json community-level cut_ratio / boundary_weight

Definition of U^obj
-------------------
At step t the agent operates on object obj_t (a file path). Next-step object candidates split structurally into three layers:

  L1 one-hop neighbors  files directly linked to obj_t in dependency_graph
  L2 same community     files sharing obj_t's CommunityId
  L3 cross-community    all remaining files (weighted by community boundary_weight)

$U^{obj}_t = H\big(P(\text{obj}_{t+1} \mid \text{obj}_t, G)\big)$

where P comes from the **structural prior** over the three layers (no fitting, no labels):
    P(uniform within L1) * w1 + P(uniform within L2) * w2 + P(uniform within L3) * w3
The weights w reflect "the agent tends to move nearby", supported empirically by E3:
of 548 adjacent py→py step pairs, 62% are within one hop and 97% within 3 hops (median distance 1).

U^obj is thus determined entirely by code structure, independent of the outcome label -- this is the key to sidestepping the 94% base rate.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

SCAE = Path(__file__).resolve().parents[2]
GRAPHS = SCAE / "pilot" / "runs" / "graphs"

# Layer weights fixed by the E3-measured adjacent-step graph distance distribution (62% one hop / median 1 / 97% within three hops)
W_L1, W_L2, W_L3 = 0.62, 0.30, 0.08


class CodeGraph:
    """Code-relation view of a single repository."""

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

    # ---------------- loading ----------------

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
                    self.edges[dep].add(path)      # treated as undirected, reflecting "reachability"
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

    # ---------------- queries ----------------

    @lru_cache(maxsize=4096)
    def neighbors(self, f: str) -> frozenset[str]:
        return frozenset(self.edges.get(f, set()))

    def same_community(self, f: str) -> frozenset[str]:
        cid = self.community.get(f)
        if cid is None:
            return frozenset()
        return frozenset(self.comm_files.get(cid, set()) - {f})

    def resolve(self, obj: str | None) -> str | None:
        """Resolve the agent's operated object to a graph node.

        The first V4 diagnostic found U_obj missing 63% of the time; the main cause was not missing data but overly narrow object extraction:
          - search actions carry only a query, no path (67 observations)
          - list_files targets a directory rather than a file (62 observations)
        Three resolution paths are added here to raise coverage:
          1. exact file match
          2. directory → the union of all files under it is treated as a "directory node"
          3. query string → suffix match against symbol names / file names
        """
        if not obj or obj == "-" or not self.ok:
            return None
        if obj in self.files:
            return obj
        o = obj.lstrip("./").rstrip("/")
        if o in self.files:
            return o
        # directory: return a virtual directory node (its neighborhood = files under it)
        if any(x.startswith(o + "/") for x in self.files):
            return f"dir:{o}"
        # query: suffix match on the file name
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
        """Resolve first, then compute U_obj; supports directory nodes."""
        node = self.resolve(obj)
        if node is None:
            return {"H": None, "reason": "cannot resolve to a graph node", "resolved": None}
        if node.startswith("dir:"):
            d = node[4:]
            members = self._dir_members(d)
            if not members:
                return {"H": None, "reason": "empty directory", "resolved": node}
            # directory node: candidates = files in the dir (L1) + same community (L2) + rest (L3)
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
                return {"H": None, "reason": "no candidates", "resolved": node}
            z = sum(ps)
            H = -sum((p / z) * math.log(p / z) for p in ps if p > 0)
            return {"H": round(H, 4), "n_l1": len(L1), "n_l2": len(L2),
                    "n_l3": L3n, "in_graph": True, "resolved": node,
                    "H_max": round(math.log(n_all), 4) if n_all > 1 else 0.0}
        r = self.u_obj(node)
        r["resolved"] = node
        return r

    def u_obj(self, f: str) -> dict:
        """Structural uncertainty: entropy of the next-object distribution, given by the three-layer structural prior.

        Returns H plus the three layer sizes, for decomposition analysis. Larger H means "more places the next step can go".
        """
        if not self.ok or not self.files:
            return {"H": None, "reason": "no graph data"}
        n_all = len(self.files)
        L1 = self.neighbors(f)
        L2 = self.same_community(f) - L1
        L3n = max(0, n_all - len(L1) - len(L2) - 1)
        ps: list[float] = []
        for size, w in ((len(L1), W_L1), (len(L2), W_L2), (L3n, W_L3)):
            if size > 0:
                ps += [w / size] * size
        if not ps:
            return {"H": None, "reason": "object not in graph or no candidates",
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
