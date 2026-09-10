"""Python reimplementation of codex's official ParsedCommand taxonomy -- the state space of channel A.

Basis
-----
The codex internal enum (codex-rs/protocol/src/parse_command.rs:9) officially parses every shell command
into (tool, operand):

    Read      { cmd, name, path }          read → object = path
    ListFiles { cmd, path }                list directory → object = path
    Search    { cmd, query, path }         search → object = (query, path)
    Unknown   { cmd }                      other

This is exactly the official implementation of "each step = tool + operand" from the user's design. We
define the state space from it rather than reverse-engineering it from command heads (the latter reinvents
the wheel and depends on the shell dialect).

The earlier 14-class split based on command heads (M2 top-1 63.7%, ppl 2.82) is kept as a robustness
control, used to show the conclusions do not depend on classification granularity.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

# command → category mapping, aligned with codex's parsing intent
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
        """Canonical representation of the operand. search uses path (if any), else query."""
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
        # search: the first non-option token is taken as the query, the path taken separately
        query = None
        for t in toks[1:]:
            if not t.startswith("-"):
                query = t
                break
        path = _first_pathish(toks[2:] if len(toks) > 2 else toks)
        return Parsed(kind="search", path=path, query=query, raw_head=head)
    return Parsed(kind="unknown", path=_first_pathish(toks), raw_head=head)


def parse_step(step) -> Parsed:
    """Parse one rollout step into (kind, obj).

    Non-shell handler-level tools (apply_patch/plan/view_image/mcp/...) take the handler
    name directly as kind -- they are official tools in their own right.
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

    # take the first pipeline segment as the main semantics (codex also parses in order, taking the first meaningful one)
    segs = [s for s in re.split(r"\|\||&&|\||;", cmd) if s.strip()]
    parsed = [parse_segment(s) for s in segs]
    for p in parsed:
        if p.kind != "unknown":
            return p
    return parsed[0] if parsed else Parsed(kind="unknown")


CATEGORIES = ["read", "list_files", "search", "unknown"]
