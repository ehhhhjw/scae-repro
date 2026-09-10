"""codex 官方 ParsedCommand 分类的 Python 复刻 —— 通道 A 的状态空间。

依据
----
codex 内部枚举（codex-rs/protocol/src/parse_command.rs:9）把每个 shell 命令
官方地解析为 (工具, 操作对象)：

    Read      { cmd, name, path }          读取 → object = path
    ListFiles { cmd, path }                列目录 → object = path
    Search    { cmd, query, path }         搜索 → object = (query, path)
    Unknown   { cmd }                      其他

这正是用户设计中"每一步 = 工具 + 操作对象"的官方实现。我们据此定义状态空间，
而非从命令头反推（后者是重造轮子且依赖 shell 方言）。

先前基于命令头的 14 类分法结果（M2 top-1 63.7%、ppl 2.82）保留为稳健性对照，
用于说明结论不依赖分类粒度。
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

# 命令 → 类别的映射，对齐 codex 的解析意图
_READ_CMDS = {"cat", "sed", "head", "tail", "nl", "less", "more", "bat"}
_LIST_CMDS = {"ls", "find", "tree", "dir"}
_SEARCH_CMDS = {"grep", "rg", "egrep", "fgrep", "ag", "ack"}

_PATHISH = re.compile(r"[\w./\-]*/[\w./\-]+|\b[\w\-]+\.[a-zA-Z]{1,4}\b")
_ENVPREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


@dataclass
class Parsed:
    kind: str                 # read | list_files | search | unknown | <handler>
    path: str | None = None
    query: str | None = None
    raw_head: str = ""

    @property
    def obj(self) -> str:
        """操作对象的规范表示。search 用 path（若有）否则 query。"""
        if self.path:
            return self.path
        if self.query:
            return f"q:{self.query}"
        return "-"


def _strip_env(tokens: list[str]) -> list[str]:
    i = 0
    while i < len(tokens) and _ENVPREFIX.match(tokens[i]):
        i += 1
    return tokens[i:]


def _first_pathish(tokens: list[str]) -> str | None:
    for t in tokens[1:]:
        if t.startswith("-"):
            continue
        m = _PATHISH.search(t)
        if m:
            return m.group(0).lstrip("./")
    return None


def parse_segment(seg: str) -> Parsed:
    seg = seg.strip()
    if not seg:
        return Parsed(kind="unknown")
    try:
        toks = shlex.split(seg)
    except ValueError:
        toks = seg.split()
    toks = _strip_env(toks)
    if not toks:
        return Parsed(kind="unknown", raw_head="envset")
    head = toks[0].split("/")[-1]

    if head in _READ_CMDS:
        return Parsed(kind="read", path=_first_pathish(toks), raw_head=head)
    if head in _LIST_CMDS:
        return Parsed(kind="list_files", path=_first_pathish(toks), raw_head=head)
    if head in _SEARCH_CMDS:
        # 搜索：第一个非选项 token 视为 query，路径另取
        query = None
        for t in toks[1:]:
            if not t.startswith("-"):
                query = t
                break
        path = _first_pathish(toks[2:] if len(toks) > 2 else toks)
        return Parsed(kind="search", path=path, query=query, raw_head=head)
    return Parsed(kind="unknown", path=_first_pathish(toks), raw_head=head)


def parse_step(step) -> Parsed:
    """把一个 rollout step 解析为 (kind, obj)。

    非 shell 的 handler 级工具（apply_patch/plan/view_image/mcp/...）
    直接以 handler 名作为 kind —— 它们本身就是官方工具。
    """
    tool = (step.tool or "").strip()
    if tool and tool not in ("exec_command", "unified_exec", "shell", "local_shell"):
        return Parsed(kind=tool, raw_head=tool)

    a = step.args
    cmd = a.get("cmd") or a.get("command")
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    if not isinstance(cmd, str) or not cmd.strip():
        return Parsed(kind="unknown", raw_head=tool or "unknown")

    # 取管道首段作为主语义（codex 也按序解析，取首个有意义者）
    segs = [s for s in re.split(r"\|\||&&|\||;", cmd) if s.strip()]
    parsed = [parse_segment(s) for s in segs]
    for p in parsed:
        if p.kind != "unknown":
            return p
    return parsed[0] if parsed else Parsed(kind="unknown")


CATEGORIES = ["read", "list_files", "search", "unknown"]
