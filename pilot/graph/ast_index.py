"""仓库 AST 索引 —— 把执行过程从「行为空间」搬到「语义空间」。

动机
----
前三次预测尝试（v0 / anytime / next-step）全部失败，根因同一个：所有特征都是
行为表层统计（步数、搜索占比、重复率），只知道 agent「动了多少」，不知道
「动对没动对」；动作桶用字符串路径，跨 repo 不可迁移（候选覆盖率仅 48%）。

AST 提供语义坐标系，补上三样缺失的东西：
  1. U_task 可直接测量：issue 提到的符号在仓库里有多少定义点、是否分散
  2. 动作可定位到依赖图节点 → 替代动作有合理来源（一跳邻居），跨 repo 可迁移
  3. 归因可下降到符号级 → 回答「哪一块内容」而非只到「哪一步」

为什么用内置 ast 而非 tree-sitter
---------------------------------
语料 499 个仓库全为 Python；内置 ast 给出等价的符号与依赖信息，且零外部依赖
（本环境未装 tree-sitter，且 rg 也不可用——见 scae_pitfalls）。接口按语言无关
方式设计，换 tree-sitter 只需替换 parse_file。

产出
----
RepoIndex:
  symbols:  name -> [(file, kind, lineno)]        符号定义点（含重名，重名即歧义）
  imports:  file -> set(module)                   文件级依赖
  file_syms: file -> set(name)                    文件定义的符号
  edges:    file -> set(file)                     由 import 解析出的文件级依赖边
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", ".tox", "build", "dist", ".eggs",
             "__pycache__", ".mypy_cache", ".pytest_cache"}


@dataclass
class RepoIndex:
    root: Path
    symbols: dict[str, list[tuple[str, str, int]]] = field(default_factory=lambda: defaultdict(list))
    imports: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    file_syms: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    n_files: int = 0
    n_parse_fail: int = 0

    # ---------------- 查询接口（供 UQ / 归因使用）----------------

    def definition_sites(self, name: str) -> list[tuple[str, str, int]]:
        """符号的所有定义点。len>1 即存在重名歧义。"""
        return self.symbols.get(name, [])

    def ambiguity(self, names: list[str]) -> dict:
        """一组符号的歧义度 —— U_task 的直接测量，而非残差倒推。

        n_defsites   总定义点数
        max_fanout   单个符号最多在几个文件里被定义（重名程度）
        n_files_hit  这些符号一共散落在多少个文件
        spread       命中文件在目录层面的分散度（不同顶层包数）
        """
        sites, files = 0, set()
        fanout = 0
        for nm in names:
            ds = self.definition_sites(nm)
            if not ds:
                continue
            sites += len(ds)
            fanout = max(fanout, len(ds))
            files.update(f for f, _, _ in ds)
        tops = {f.split("/", 1)[0] for f in files if f}
        return {"n_defsites": sites, "max_fanout": fanout,
                "n_files_hit": len(files), "spread": len(tops)}

    def neighbors(self, file: str, hops: int = 1) -> set[str]:
        """依赖图上的 k 跳邻居 —— 替代动作 a' 的合理来源。"""
        frontier, seen = {file}, {file}
        for _ in range(hops):
            nxt: set[str] = set()
            for f in frontier:
                nxt |= self.edges.get(f, set())
                nxt |= {g for g, outs in self.edges.items() if f in outs}
            nxt -= seen
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return seen - {file}

    def graph_distance(self, a: str, b: str, max_hops: int = 4) -> int | None:
        """两文件在依赖图上的最短跳数；None 表示 max_hops 内不连通。"""
        if a == b:
            return 0
        frontier, seen = {a}, {a}
        for d in range(1, max_hops + 1):
            nxt: set[str] = set()
            for f in frontier:
                nxt |= self.edges.get(f, set())
                nxt |= {g for g, outs in self.edges.items() if f in outs}
            nxt -= seen
            if b in nxt:
                return d
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return None


def _mod_to_file(mod: str, root: Path, all_files: set[str]) -> str | None:
    """把 import 的模块名解析为仓库内文件路径（尽力而为）。"""
    p = mod.replace(".", "/")
    for cand in (f"{p}.py", f"{p}/__init__.py"):
        if cand in all_files:
            return cand
    # 尾段匹配：a.b.c -> 任何 .../c.py
    tail = mod.rsplit(".", 1)[-1]
    hits = [f for f in all_files
            if f.endswith(f"/{tail}.py") or f == f"{tail}.py"]
    return hits[0] if len(hits) == 1 else None


def build_index(root: str | Path, max_files: int = 6000) -> RepoIndex:
    root = Path(root)
    idx = RepoIndex(root=root)
    files: list[tuple[str, Path]] = []
    for p in root.rglob("*.py"):
        if any(d in p.parts for d in SKIP_DIRS):
            continue
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            continue
        files.append((rel, p))
        if len(files) >= max_files:
            break
    all_files = {rel for rel, _ in files}
    idx.n_files = len(files)

    raw_imports: dict[str, set[str]] = defaultdict(set)
    for rel, p in files:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError, OSError):
            idx.n_parse_fail += 1
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                idx.symbols[node.name].append((rel, "func", node.lineno))
                idx.file_syms[rel].add(node.name)
            elif isinstance(node, ast.ClassDef):
                idx.symbols[node.name].append((rel, "class", node.lineno))
                idx.file_syms[rel].add(node.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    raw_imports[rel].add(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    raw_imports[rel].add(node.module)

    idx.imports = raw_imports
    for rel, mods in raw_imports.items():
        for m in mods:
            tgt = _mod_to_file(m, root, all_files)
            if tgt and tgt != rel:
                idx.edges[rel].add(tgt)
    return idx


def summarize(idx: RepoIndex) -> dict:
    dup = {k: len(v) for k, v in idx.symbols.items() if len(v) > 1}
    return {
        "n_files": idx.n_files,
        "n_parse_fail": idx.n_parse_fail,
        "n_symbols": len(idx.symbols),
        "n_duplicated_symbols": len(dup),
        "max_fanout": max(dup.values()) if dup else 0,
        "n_edges": sum(len(v) for v in idx.edges.values()),
        "n_files_with_edges": len(idx.edges),
    }


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ix = build_index(root)
    print(json.dumps(summarize(ix), ensure_ascii=False, indent=2))
