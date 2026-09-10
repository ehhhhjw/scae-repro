"""Read/write/truncate/rewrite codex rollout JSONL -- the substrate of SCAE prefix replay.

File format (fixed by codex-rs/protocol/src/protocol.rs)
--------------------------------------------------------
One `RolloutLine` per line:
    {"timestamp": "...", "ordinal": <u64?>, "type": <variant>, "payload": <...>}
`type` comes from `RolloutItem` (#[serde(tag="type", content="payload", rename_all="snake_case")]):
    session_meta | response_item | inter_agent_communication
    | inter_agent_communication_metadata | compacted | turn_context | world_state | event_msg

The payload of a `response_item` is a Responses API item; the parts we care about:
    actions:      function_call / local_shell_call / custom_tool_call
    observations: function_call_output / local_shell_call_output / custom_tool_call_output

Invariant (truncation must preserve it, or the model side errors on a dangling tool call)
-----------------------------------------------------------------------------------------
Every function_call must be immediately followed by its paired *_output. The two truncation
operators in this module guarantee that by construction, with `assert_no_dangling_calls` as a
backstop check.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ACTION_TYPES = {"function_call", "local_shell_call", "custom_tool_call"}
OUTPUT_TYPES = {"function_call_output", "local_shell_call_output", "custom_tool_call_output"}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return items


def dump(items: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path


def line_type(item: dict) -> str:
    return item.get("type", "")


def payload(item: dict) -> dict:
    p = item.get("payload")
    return p if isinstance(p, dict) else {}


def payload_type(item: dict) -> str:
    return payload(item).get("type", "")


# ---------------------------------------------------------------------------
# Step index
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One execution unit u_t = <tool, object, ...>, i.e. A_t and O_t in the AE-SCM."""

    idx: int                      # 1-based step index t
    call_line: int                # index of the action within items
    out_line: int | None          # index of the paired output (may be missing: interrupted turn)
    tool: str                     # tool name
    args_raw: str                 # raw argument string
    args: dict = field(default_factory=dict)   # parsed arguments (empty on failure)
    call_id: str | None = None
    output_text: str = ""

    @property
    def complete(self) -> bool:
        return self.out_line is not None


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    except json.JSONDecodeError:
        return {}


def _output_text(p: dict) -> str:
    out = p.get("output")
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        # some variants are {"content": "...", "success": bool}
        for key in ("content", "text", "stdout"):
            if isinstance(out.get(key), str):
                return out[key]
        return json.dumps(out, ensure_ascii=False)
    if isinstance(out, list):
        return "\n".join(
            seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in out
        )
    return ""


def steps(items: list[dict]) -> list[Step]:
    """Extract the ordered step list. Pairing rule: prefer call_id, fall back to "next output"."""
    out_by_call: dict[str, int] = {}
    for i, it in enumerate(items):
        if line_type(it) == "response_item" and payload_type(it) in OUTPUT_TYPES:
            cid = payload(it).get("call_id") or payload(it).get("id")
            if cid:
                out_by_call.setdefault(str(cid), i)

    result: list[Step] = []
    t = 0
    for i, it in enumerate(items):
        if line_type(it) != "response_item" or payload_type(it) not in ACTION_TYPES:
            continue
        p = payload(it)
        cid = p.get("call_id") or p.get("id")
        out_line = out_by_call.get(str(cid)) if cid else None
        if out_line is None:
            for j in range(i + 1, len(items)):
                if line_type(items[j]) == "response_item" and payload_type(items[j]) in OUTPUT_TYPES:
                    out_line = j
                    break
        t += 1
        raw_args = p.get("arguments", p.get("input", ""))
        result.append(
            Step(
                idx=t,
                call_line=i,
                out_line=out_line,
                tool=str(p.get("name") or p.get("tool_name") or payload_type(it)),
                args_raw=raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False),
                args=_parse_args(raw_args),
                call_id=str(cid) if cid else None,
                output_text=_output_text(payload(items[out_line])) if out_line is not None else "",
            )
        )
    return result


# ---------------------------------------------------------------------------
# Truncation and rewriting (intervention operators)
# ---------------------------------------------------------------------------

def truncate_before_step(items: list[dict], k: int) -> list[dict]:
    """Keep everything before step k (excluding step k's action and output). Estimates v_{k-1}."""
    st = steps(items)
    if not 1 <= k <= len(st):
        raise IndexError(f"k={k} out of range, {len(st)} steps total")
    cut = st[k - 1].call_line
    return _renumber(items[:cut])


def truncate_including_step(items: list[dict], k: int) -> list[dict]:
    """Keep through step k (including its action and output). Estimates v_k."""
    st = steps(items)
    if not 1 <= k <= len(st):
        raise IndexError(f"k={k} out of range, {len(st)} steps total")
    s = st[k - 1]
    if s.out_line is None:
        raise ValueError(f"step {k} has no paired output; truncating would leave a dangling tool call")
    cut = max(s.call_line, s.out_line) + 1
    return _renumber(items[:cut])


def substitute_action(
    items: list[dict],
    k: int,
    new_tool: str,
    new_args: dict,
    new_output: str,
) -> list[dict]:
    """Replace step k with (new_tool, new_args) and its **actual execution result** new_output.

    ⚠️ new_output must come from really executing new_args. Filling in a fabricated output is not
    a faithful do(); it is an intervention on the observation -- the causal semantics of the two
    are entirely different.

    Returns: the prefix (including the replaced step k); all later steps are dropped (the agent
    continues on its own).
    """
    st = steps(items)
    if not 1 <= k <= len(st):
        raise IndexError(f"k={k} out of range, {len(st)} steps total")
    s = st[k - 1]
    prefix = items[: s.call_line]
    call_id = s.call_id or f"scae_{k}"
    tmpl_call = dict(items[s.call_line])
    tmpl_call["payload"] = {
        "type": "function_call",
        "name": new_tool,
        "arguments": json.dumps(new_args, ensure_ascii=False, sort_keys=True),
        "call_id": call_id,
    }
    tmpl_out = dict(items[s.out_line]) if s.out_line is not None else dict(items[s.call_line])
    tmpl_out["payload"] = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": new_output,
    }
    return _renumber([*prefix, tmpl_call, tmpl_out])


def mask_output_lines(
    items: list[dict],
    k: int,
    targets: Iterable[str],
    mode: str = "drop",
) -> tuple[list[dict], dict]:
    """**do(observation)**: mask the lines mentioning `targets` in step k's output, action unchanged.

    Causal-semantics difference from `substitute_action` (do not conflate them)
    ---------------------------------------------------------------------------
    `substitute_action` is do(action): swap the action and fill in the **actually executed** result.
    This operator is do(observation): action, arguments and environment all stay put, only **what
    the agent sees** changes.
    It is well-defined under the AE-SCM (observations are nodes) and corresponds to the
    **information term** of Theorem 1; it measures the contribution of **this one information
    path** -- i.e. a path-specific effect.

    It **deliberately** does not block "the file can be discovered by other means": if the agent
    later finds the file again via ls/grep, that is exactly the redundancy we want to measure. So
    the effect of this operator is that of "removing a single provenance edge", not that of
    "making the file unreachable".

    ⚠️ Internal validity: masking changes output length and content distribution. The experiment
    therefore must carry a **placebo arm** (masking the same number of non-target lines), or
    "information removed" cannot be told apart from "output perturbed".

    Parameters
    ----------
    mode="drop"     delete the matched lines outright (equivalent to "grep matched nothing here")
    mode="placebo"  same semantics as drop, but the caller passes non-target paths, as a control
    Returns
    -------
    (rewritten items prefix, diagnostics). The prefix is kept up to step k's output; later steps
    are dropped (the agent continues on its own), same as `truncate_including_step`.
    """
    st = steps(items)
    if not 1 <= k <= len(st):
        raise IndexError(f"k={k} out of range, {len(st)} steps total")
    s = st[k - 1]
    if s.out_line is None:
        raise ValueError(f"step {k} has no paired output, cannot mask")
    tgts = [t for t in targets if t]
    if not tgts:
        raise ValueError("targets is empty")

    p_out = payload(items[s.out_line])
    orig = _output_text(p_out)
    lines = orig.split("\n")
    kept, dropped = [], []
    for ln in lines:
        if any(t in ln for t in tgts):
            dropped.append(ln)
        else:
            kept.append(ln)
    if not dropped:
        raise ValueError(f"step {k} output has no line matching {tgts[:2]}..., masking is a no-op")

    new_items = [dict(it) for it in items[: s.out_line + 1]]
    np_ = dict(payload(new_items[s.out_line]))
    np_["output"] = "\n".join(kept)
    new_items[s.out_line] = {**new_items[s.out_line], "payload": np_}
    diag = {
        "mode": mode, "step": k, "n_lines_total": len(lines),
        "n_lines_dropped": len(dropped),
        "chars_before": len(orig), "chars_after": len("\n".join(kept)),
        "targets": tgts[:8],
        "dropped_sample": dropped[:3],
    }
    out = _renumber(new_items)
    assert_no_dangling_calls(out)
    return out, diag


def count_output_lines_matching(items: list[dict], k: int,
                               targets: Iterable[str]) -> int:
    """Lines in step k's output matching targets -- lets the placebo arm balance masked lines."""
    st = steps(items)
    if not 1 <= k <= len(st) or st[k - 1].out_line is None:
        return 0
    txt = _output_text(payload(items[st[k - 1].out_line]))
    tg = [t for t in targets if t]
    return sum(1 for ln in txt.split("\n") if any(t in ln for t in tg))


def _renumber(items: list[dict]) -> list[dict]:
    """Renumber ordinal. Conservatively keep the original timestamp (not part of visible history)."""
    out = []
    n = 0
    for it in items:
        it = dict(it)
        if "ordinal" in it:
            n += 1
            it["ordinal"] = n
        out.append(it)
    return out


def assert_no_dangling_calls(items: list[dict]) -> None:
    """Post-truncation invariant check: every action must have a paired output."""
    pending: list[str] = []
    for it in items:
        if line_type(it) != "response_item":
            continue
        pt = payload_type(it)
        if pt in ACTION_TYPES:
            pending.append(str(payload(it).get("call_id") or payload(it).get("id") or "?"))
        elif pt in OUTPUT_TYPES:
            cid = str(payload(it).get("call_id") or payload(it).get("id") or "?")
            if cid in pending:
                pending.remove(cid)
            elif pending:
                pending.pop(0)
    if pending:
        raise ValueError(f"truncated result contains {len(pending)} dangling tool call(s): {pending}")


# ---------------------------------------------------------------------------
# Compaction detection (Proposition 4)
# ---------------------------------------------------------------------------

def compaction_lines(items: list[dict]) -> list[int]:
    """Line indices of all compaction events. Non-empty means assumption A1 fails on this trace."""
    return [i for i, it in enumerate(items) if line_type(it) == "compacted"]


# ---------------------------------------------------------------------------
# Action bucketing: map continuous tool calls onto a discrete categorical support
# ---------------------------------------------------------------------------

_PATHISH = re.compile(r"[\w./\-]*/[\w./\-]+|\b[\w\-]+\.(?:py|md|txt|cfg|toml|json|ya?ml)\b")


def action_bucket(step: Step, repo_root: str | Path | None = None) -> tuple[str, str]:
    """(tool, target) -- the equivalence class of A_t in the AE-SCM.

    This is the minimal usable version of §3.7 "recovering effective overlap with the code
    graph": collapse free-text arguments into a (tool, target node) pair, so the action space
    becomes enumerable, frequency-countable and entropy-computable.
    """
    a = step.args
    for key in ("path", "file", "file_path", "target", "dir", "pattern", "query"):
        if isinstance(a.get(key), str) and a[key].strip():
            return (step.tool, _norm_target(a[key], repo_root))

    # shell-like: take the command name + the first path-looking argument
    cmd = a.get("command")
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    text = cmd if isinstance(cmd, str) else step.args_raw
    if text:
        head = text.strip().split()[0] if text.strip() else step.tool
        m = _PATHISH.search(text)
        return (f"{step.tool}:{Path(head).name}", _norm_target(m.group(0), repo_root) if m else "-")
    return (step.tool, "-")


def _norm_target(raw: str, repo_root: str | Path | None) -> str:
    s = raw.strip().strip("'\"")
    if repo_root:
        try:
            s = str(Path(s).resolve().relative_to(Path(repo_root).resolve()))
        except (ValueError, OSError):
            pass
    s = s.lstrip("./")
    return s or "-"


def trace_fingerprint(items: list[dict], repo_root: str | Path | None = None) -> str:
    """Trace fingerprint, for dedup and for telling whether two rollouts took the same path."""
    seq = "|".join(f"{t}#{o}" for t, o in (action_bucket(s, repo_root) for s in steps(items)))
    return hashlib.sha256(seq.encode()).hexdigest()[:16]
