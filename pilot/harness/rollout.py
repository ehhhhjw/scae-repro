"""读/写/截断/改写 codex 的 rollout JSONL —— SCAE 前缀重放的底层。

文件格式（由 codex-rs/protocol/src/protocol.rs 确定）
---------------------------------------------------
每行一个 `RolloutLine`：
    {"timestamp": "...", "ordinal": <u64?>, "type": <variant>, "payload": <...>}
`type` 来自 `RolloutItem`（#[serde(tag="type", content="payload", rename_all="snake_case")]）：
    session_meta | response_item | inter_agent_communication
    | inter_agent_communication_metadata | compacted | turn_context | world_state | event_msg

其中 `response_item` 的 payload 是 Responses API 的 item，我们关心：
    动作：function_call / local_shell_call / custom_tool_call
    观测：function_call_output / local_shell_call_output / custom_tool_call_output

不变式（截断必须维持，否则模型侧会因悬空 tool call 报错）
--------------------------------------------------------
任何 function_call 必须紧随其配对的 *_output。本模块的两个截断算子按构造保证这一点，
并由 `assert_no_dangling_calls` 兜底检查。
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
                raise ValueError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
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
# 步骤索引
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """一个执行单元 u_t = <tool, object, ...>，即 AE-SCM 里的 A_t 与 O_t。"""

    idx: int                      # 1-based 步序 t
    call_line: int                # 动作在 items 中的下标
    out_line: int | None          # 配对输出的下标（可能缺失：被中断的 turn）
    tool: str                     # 工具名
    args_raw: str                 # 原始参数字符串
    args: dict = field(default_factory=dict)   # 解析后的参数（失败则空）
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
        # 某些变体是 {"content": "...", "success": bool}
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
    """抽取有序步骤列表。配对规则：优先 call_id，退化为"下一个 output"。"""
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
# 截断与改写（干预算子）
# ---------------------------------------------------------------------------

def truncate_before_step(items: list[dict], k: int) -> list[dict]:
    """保留到第 k 步之前（不含第 k 步的动作与输出）。用于估计 v_{k-1}。"""
    st = steps(items)
    if not 1 <= k <= len(st):
        raise IndexError(f"k={k} 越界，共 {len(st)} 步")
    cut = st[k - 1].call_line
    return _renumber(items[:cut])


def truncate_including_step(items: list[dict], k: int) -> list[dict]:
    """保留到第 k 步（含其动作与输出）。用于估计 v_k。"""
    st = steps(items)
    if not 1 <= k <= len(st):
        raise IndexError(f"k={k} 越界，共 {len(st)} 步")
    s = st[k - 1]
    if s.out_line is None:
        raise ValueError(f"第 {k} 步没有配对输出，截断会留下悬空 tool call")
    cut = max(s.call_line, s.out_line) + 1
    return _renumber(items[:cut])


def substitute_action(
    items: list[dict],
    k: int,
    new_tool: str,
    new_args: dict,
    new_output: str,
) -> list[dict]:
    """把第 k 步替换为 (new_tool, new_args) 及其**真实执行结果** new_output。

    ⚠️ new_output 必须由我们真正执行 new_args 得到。填入编造的输出就不是忠实的 do()，
    而是在干预观测——两者的因果语义完全不同。

    返回：前缀（含被替换的第 k 步），后续步骤一律丢弃（由 agent 自行续写）。
    """
    st = steps(items)
    if not 1 <= k <= len(st):
        raise IndexError(f"k={k} 越界，共 {len(st)} 步")
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
    """**do(observation)**：把第 k 步输出里提到 `targets` 的行掩蔽掉，动作与参数不变。

    与 `substitute_action` 的因果语义差别（务必分清）
    ------------------------------------------------
    `substitute_action` 是 do(action)：换动作、并填入**真实执行**的结果。
    本算子是 do(observation)：动作、参数、环境全不动，只改 agent **看到的内容**。
    它在 AE-SCM 下是良定义的（观测是节点），对应 Theorem 1 的**信息项**，
    测的是**这一条信息路径**的贡献 —— 即 path-specific effect。

    它**故意**不阻断"文件可由别的途径被发现"：若 agent 后续用 ls/grep 又找到该文件，
    那正是我们要测的冗余。因此本算子的效应是"移除单条溯源边"的效应，
    不是"使文件不可达"的效应。

    ⚠️ 内部效度：掩蔽会改变输出长度与内容分布。因此实验必须带**安慰剂臂**
    （掩蔽等量的非目标行），否则无法区分"信息被移除"与"输出被扰动"。

    参数
    ----
    mode="drop"     直接删掉命中的行（等价于"grep 没匹配到这里"）
    mode="placebo"  语义同 drop，但由调用方传入非目标路径，用作对照
    返回
    ----
    (改写后的 items 前缀, 诊断信息)。前缀保留到第 k 步的输出为止，
    后续步骤丢弃（由 agent 自行续写），与 `truncate_including_step` 一致。
    """
    st = steps(items)
    if not 1 <= k <= len(st):
        raise IndexError(f"k={k} 越界，共 {len(st)} 步")
    s = st[k - 1]
    if s.out_line is None:
        raise ValueError(f"第 {k} 步没有配对输出，无法掩蔽")
    tgts = [t for t in targets if t]
    if not tgts:
        raise ValueError("targets 为空")

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
        raise ValueError(f"第 {k} 步输出里没有命中 {tgts[:2]}… 的行，掩蔽无操作")

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
    """第 k 步输出里命中 targets 的行数 —— 供安慰剂臂配平掩蔽行数。"""
    st = steps(items)
    if not 1 <= k <= len(st) or st[k - 1].out_line is None:
        return 0
    txt = _output_text(payload(items[st[k - 1].out_line]))
    tg = [t for t in targets if t]
    return sum(1 for ln in txt.split("\n") if any(t in ln for t in tg))


def _renumber(items: list[dict]) -> list[dict]:
    """重排 ordinal。保守起见保留原 timestamp（它不参与模型可见历史）。"""
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
    """截断后的不变式检查：每个动作都必须有配对输出。"""
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
        raise ValueError(f"截断结果含 {len(pending)} 个悬空 tool call: {pending}")


# ---------------------------------------------------------------------------
# 压缩检测（命题 4）
# ---------------------------------------------------------------------------

def compaction_lines(items: list[dict]) -> list[int]:
    """返回所有压缩事件所在行下标。非空即意味着假设 A1 在该轨迹上失效。"""
    return [i for i, it in enumerate(items) if line_type(it) == "compacted"]


# ---------------------------------------------------------------------------
# 动作分桶：把连续的工具调用映射到离散的 categorical 支撑集
# ---------------------------------------------------------------------------

_PATHISH = re.compile(r"[\w./\-]*/[\w./\-]+|\b[\w\-]+\.(?:py|md|txt|cfg|toml|json|ya?ml)\b")


def action_bucket(step: Step, repo_root: str | Path | None = None) -> tuple[str, str]:
    """(tool, target) —— AE-SCM 中 A_t 的等价类。

    这是 §3.7「用代码图恢复有效 overlap」的最小可用版本：把自由文本参数
    收敛到 (工具, 目标节点) 二元组，使动作空间可枚举、可统计频率、可算熵。
    """
    a = step.args
    for key in ("path", "file", "file_path", "target", "dir", "pattern", "query"):
        if isinstance(a.get(key), str) and a[key].strip():
            return (step.tool, _norm_target(a[key], repo_root))

    # shell 类：取命令名 + 第一个像路径的参数
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
    """轨迹指纹，用于去重与判断两次 rollout 是否走了同一条路径。"""
    seq = "|".join(f"{t}#{o}" for t, o in (action_bucket(s, repo_root) for s in steps(items)))
    return hashlib.sha256(seq.encode()).hexdigest()[:16]
