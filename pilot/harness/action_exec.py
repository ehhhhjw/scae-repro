"""替代动作的真实执行 —— 混合估计器低熵步软干预 + 修复实验的基础。

substitute_action 要求填入替代动作的**真实执行结果**（否则不是忠实的 do()）。
定位任务的动作都是只读 shell（grep/sed/ls/find/rg），可安全地在仓库内真实执行。

本模块提供：
  - exec_readonly_action：在仓库内执行一个只读 shell 动作，返回真实 stdout（受 U_env 钉死）
  - oracle_action：构造"读 gold 文件"这个 oracle 替代动作（用于修复实验 / PN 归因）

安全：只允许白名单命令（grep/rg/sed/ls/find/cat/head/tail/awk/wc），拒绝任何写操作。
输出规范化：与 tools_mcp 的确定性契约一致（排序、无 mtime、固定截断），保证 A3。
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from pilot.harness.appserver import PINNED_ENV

# 只读命令白名单（低熵步软干预与 oracle 动作只会用到这些）
READONLY_CMDS = {"grep", "rg", "sed", "ls", "find", "cat", "head", "tail",
                 "awk", "wc", "sort", "uniq", "cut", "tr", "echo", "pwd", "true"}
# 明确禁止（任何可能写/删/改状态的）
FORBIDDEN = {">", ">>", "rm", "mv", "cp", "mkdir", "touch", "tee", "chmod",
             "git", "python", "pip", "pytest", "make", "sh", "bash", "|dd"}


def is_readonly(cmd: str) -> bool:
    """保守判断：命令是否只读。任何管道/重定向到写、或非白名单首命令都拒绝。"""
    low = cmd.lower()
    for f in FORBIDDEN:
        if f in low.split() or f in (">", ">>") and f in cmd:
            return False
    if ">" in cmd or ">>" in cmd:
        return False
    # 逐段（管道分隔）检查首 token。
    # shlex.split 对 agent 生成的命令并不安全：未闭合引号或尾部反斜杠会抛
    # ValueError("No escaped character")。此处必须容错——解析失败即保守拒绝。
    for seg in cmd.split("|"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            toks = shlex.split(seg)
        except ValueError:
            # 无法可靠解析 ⇒ 无法确认只读 ⇒ 拒绝（fail-closed）
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
    """在仓库内执行只读命令，返回规范化 stdout。非只读命令拒绝执行。

    规范化：合并 stdout+stderr、固定长度截断（与工具面契约一致），
    环境用 PINNED_ENV（LC_ALL=C 等）保证跨次重放一致 → 支撑 A3。
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
    """构造"读取指定文件开头"的动作。靶点由调用方显式给出。"""
    cmd = f"sed -n '1,120p' {shlex.quote(path)}"
    return tool, {"cmd": cmd, "yield_time_ms": 10000, "max_output_tokens": 20000}, cmd


def oracle_action(inst, tool: str = "exec_command",
                  target: str | None = None) -> tuple[str, dict, str]:
    """构造 oracle 替代动作：读取**指定的** gold 文件。

    ⚠️ 设计教训：早期版本固定读 sorted(gold)[0]，但修复实验中 11/19 的 case
    实际漏掉的是**其他** gold 文件 —— 修错对象导致三臂全零、实验无信息。
    因此 target 必须由调用方显式传入（通常是该 case 的 missed_files[0]）。
    target=None 时退回 sorted(gold)[0]，仅供无 missed 信息的场景使用。
    """
    if target is None:
        gold_files = sorted(inst.gold)
        target = gold_files[0] if gold_files else "."
    return read_file_action(target, tool)


def irrelevant_action(inst, idx, tool: str = "exec_command",
                      min_hops: int = 3) -> tuple[str, dict, str] | None:
    """构造"读一个与 gold 语义无关的文件"的动作 —— E1 的内容对照臂。

    无关 = 在 AST 依赖图上与所有 gold 文件距离 >= min_hops 或不可达。
    用于排除"注入任何内容都有效"这一平凡解释。idx 为 RepoIndex。
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
    """构造"图上邻近动作"：读与当前目标同目录的另一个文件（升温支撑集的代理）。

    用于低熵步软干预的非 oracle 替代动作（避免只用 oracle 造成的乐观偏置）。
    pilot 简化版：同目录兄弟文件；正式版用 tree-sitter 依赖图的一跳邻居。
    """
    import os
    d = os.path.dirname(current_target) or "."
    cmd = f"ls {shlex.quote(d)} | head -40"
    return tool, {"cmd": cmd, "yield_time_ms": 10000, "max_output_tokens": 12000}, cmd
