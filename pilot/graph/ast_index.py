"""Repository AST index -- moves execution out of "behavior space" into "semantic space".

Motivation
----------
The first three prediction attempts (v0 / anytime / next-step) all failed for the
same root cause: every feature was a surface behavioral statistic (step count,
search fraction, repeat rate), so we only knew "how much" the agent moved, not
"whether it moved to the right place"; action buckets keyed on string paths do
not transfer across repos (candidate coverage only 48%).

The AST supplies a semantic coordinate system, filling in the three missing pieces:
  1. U_task becomes directly measurable: how many definition sites the symbols
     mentioned in the issue have in the repo, and whether they are scattered
  2. Actions can be localized to dependency-graph nodes -> alternative actions
     have a principled source (one-hop neighbors) and transfer across repos
  3. Attribution can descend to the symbol level -> answers "which piece of
     content" instead of only "which step"

Why the builtin ast rather than tree-sitter
-------------------------------------------
All 499 repos in the corpus are Python; the builtin ast yields equivalent symbol
and dependency information with zero external dependencies (tree-sitter is not
installed in this environment, and rg is unavailable too -- see scae_pitfalls).
The interface is designed to be language-agnostic; switching to tree-sitter only
requires replacing parse_file.

Outputs
-------
RepoIndex:
  symbols:  name -> [(file, kind, lineno)]        definition sites (duplicate names = ambiguity)
  imports:  file -> set(module)                   file-level dependencies
  file_syms: file -> set(name)                    symbols defined by the file
  edges:    file -> set(file)                     file-level dependency edges resolved from imports
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

    # ---------------- Query interface (for UQ / attribution)----------------

    def definition_sites(self, name: str) -> list[tuple[str, str, int]]:
        """All definition sites of a symbol. len>1 means duplicate-name ambiguity."""
        return self.symbols.get(name, [])

    def ambiguity(self, names: list[str]) -> dict:
        """Ambiguity of a symbol set -- a direct measurement of U_task, not a residual inversion.

        n_defsites   total number of definition sites
        max_fanout   max number of files a single symbol is defined in (degree of name duplication)
        n_files_hit  how many files these symbols are scattered across in total
        spread       directory-level dispersion of the hit files (number of distinct top-level packages)
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
        """k-hop neighbors on the dependency graph -- a principled source for alternative actions a'."""
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
        """Shortest hop count between two files on the dependency graph; None = not connected within max_hops."""
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
    """Resolve an imported module name to a file path inside the repo (best effort)."""
    p = mod.replace(".", "/")
    for cand in (f"{p}.py", f"{p}/__init__.py"):
        if cand in all_files:
            return cand
    # Tail match: a.b.c -> any .../c.py
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
