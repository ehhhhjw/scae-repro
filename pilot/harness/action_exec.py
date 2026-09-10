"""Real execution of substitute actions -- the basis for low-entropy-step soft intervention in the
mixed estimator + the repair experiment.

substitute_action requires the **real execution result** of the substitute action (otherwise it is
not a faithful do()). All actions in localization tasks are read-only shell (grep/sed/ls/find/rg),
so they can safely be executed for real inside the repository.

This module provides:
  - exec_readonly_action: execute one read-only shell action inside the repository, return the real
    stdout (pinned by U_env)
  - oracle_action: build the "read the gold file" oracle substitute action (for the repair
    experiment / PN attribution)

Safety: only whitelisted commands are allowed (grep/rg/sed/ls/find/cat/head/tail/awk/wc), any write
operation is refused.
Output normalization: consistent with the determinism contract of tools_mcp (sorted, no mtime, fixed
truncation), which guarantees A3.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from pilot.harness.appserver import PINNED_ENV

# read-only command whitelist (low-entropy-step soft intervention and oracle actions only use these)
READONLY_CMDS = {"grep", "rg", "sed", "ls", "find", "cat", "head", "tail",
                 "awk", "wc", "sort", "uniq", "cut", "tr", "echo", "pwd", "true"}
# explicitly forbidden (anything that could write/delete/mutate state)
FORBIDDEN = {">", ">>", "rm", "mv", "cp", "mkdir", "touch", "tee", "chmod",
             "git", "python", "pip", "pytest", "make", "sh", "bash", "|dd"}


def is_readonly(cmd: str) -> bool:
    """Conservative check of whether a command is read-only. Any pipe/redirect that writes, or a
    non-whitelisted head command, is refused."""
    low = cmd.lower()
    for f in FORBIDDEN:
        if f in low.split() or f in (">", ">>") and f in cmd:
            return False
    if ">" in cmd or ">>" in cmd:
        return False
    # Check the head token of each segment (pipe-separated).
    # shlex.split is not safe on agent-generated commands: an unclosed quote or a trailing
    # backslash raises ValueError("No escaped character"). This must be tolerant -- a parse
    # failure means a conservative refusal.
    for seg in cmd.split("|"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            toks = shlex.split(seg)
        except ValueError:
            # cannot parse reliably ⇒ cannot confirm read-only ⇒ refuse (fail-closed)
            return False
        if not toks:
            continue
        head = Path(toks[0]).name
        if head not in READONLY_CMDS:
            return False
    return True


def exec_readonly_action(
    repo: str | Path,
    cmd: str,
    timeout: float = 30.0,
    max_output_chars: int = 20000,
) -> str:
    """Execute a read-only command inside the repository, return normalized stdout. Non-read-only
    commands are refused.

    Normalization: merge stdout+stderr, fixed-length truncation (consistent with the tool-surface
    contract); the environment uses PINNED_ENV (LC_ALL=C etc.) to keep replays consistent → supports A3.
    """
    if not is_readonly(cmd):
        return f"[REFUSED: non-readonly command] {cmd}"
    env = {**PINNED_ENV}
    import os
    full_env = {**os.environ, **env}
    try:
        p = subprocess.run(
            cmd, shell=True, cwd=str(Path(repo).resolve()),
            capture_output=True, text=True, timeout=timeout, env=full_env,
        )
        out = p.stdout
        if p.returncode != 0 and not out:
            out = f"[exit {p.returncode}] {p.stderr[:500]}"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s] {cmd}"
    if len(out) > max_output_chars:
        out = out[:max_output_chars] + "\n…[truncated]"
    return out


def read_file_action(path: str, tool: str = "exec_command") -> tuple[str, dict, str]:
    """Build the "read the head of the given file" action. The target is given explicitly by the caller."""
    cmd = f"sed -n '1,120p' {shlex.quote(path)}"
    return tool, {"cmd": cmd, "yield_time_ms": 10000, "max_output_tokens": 20000}, cmd


def oracle_action(inst, tool: str = "exec_command",
                  target: str | None = None) -> tuple[str, dict, str]:
    """Build the oracle substitute action: read the **specified** gold file.

    ⚠️ Design lesson: an early version always read sorted(gold)[0], but in 11/19 cases of the repair
    experiment the actually missed file was a **different** gold file -- repairing the wrong target
    made all three arms zero and the experiment uninformative.
    So target must be passed explicitly by the caller (usually that case's missed_files[0]).
    With target=None it falls back to sorted(gold)[0], only for scenarios without missed information.
    """
    if target is None:
        gold_files = sorted(inst.gold)
        target = gold_files[0] if gold_files else "."
    return read_file_action(target, tool)


def irrelevant_action(inst, idx, tool: str = "exec_command",
                      min_hops: int = 3) -> tuple[str, dict, str] | None:
    """Build the "read a file semantically unrelated to gold" action -- E1's content control arm.

    Unrelated = distance >= min_hops from every gold file on the AST dependency graph, or unreachable.
    Used to rule out the trivial explanation that "injecting any content works". idx is a RepoIndex.
    """
    gold = set(inst.gold)
    cands = [f for f in idx.file_syms.keys() if f not in gold]
    for f in sorted(cands):
        if "test" in f.lower():
            continue
        far = True
        for g in gold:
            d = idx.graph_distance(f, g, max_hops=min_hops)
            if d is not None and d < min_hops:
                far = False
                break
        if far:
            return read_file_action(f, tool)
    return None


def graph_neighbor_action(inst, current_target: str, tool: str = "exec_command") -> tuple[str, dict, str]:
    """Build a "graph-adjacent action": read another file in the same directory as the current
    target (a proxy for warming up the support set).

    Used as the non-oracle substitute action for low-entropy-step soft intervention (avoiding the
    optimistic bias of using only the oracle).
    Pilot simplification: same-directory sibling file; the full version uses one-hop neighbors of
    the tree-sitter dependency graph.
    """
    import os
    d = os.path.dirname(current_target) or "."
    cmd = f"ls {shlex.quote(d)} | head -40"
    return tool, {"cmd": cmd, "yield_time_ms": 10000, "max_output_tokens": 12000}, cmd
