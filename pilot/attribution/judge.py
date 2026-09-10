"""full-trace LLM-as-judge 归因 —— C3 中被攻击的基线（复现主流实践）。

judge 看到**完整轨迹（动作+观测）+ 最终结果**，被要求指出 top-3 根因步。
这正是我们要证明其有碰撞偏差的对象：它条件化了后继步骤与结果（$A_t$ 的后代），
理论预测其归因会系统性偏向靠后步骤（§3.5）。

另设 prefix-only judge（只给前缀、不给后继与结果）作为隔离对照：
若 full-trace judge 的步位显著晚于 prefix-only judge，碰撞偏差就被单独摘出来了。

judge 走同一个 idealab chat/responses 端点，用一次性 turn 完成（不需要工具）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot.harness import rollout as R                                   # noqa: E402
from pilot.harness.appserver import AppServer                            # noqa: E402
from pilot.probes import task as T                                       # noqa: E402

JUDGE_INSTRUCTIONS = """\
You are analyzing the execution trace of a code-localization agent. The agent was asked to \
identify which file(s) must be modified to resolve an issue. You are given the ordered list \
of the agent's steps (each step is a tool call and its observed output){outcome_clause}.

Your job: identify which single step was MOST responsible for the final localization outcome \
— the step that most determined whether the agent found the correct file(s). Also give your \
top-3 in order.

Answer ONLY with a JSON object on the last line, no prose after it:
{{"top1_step": <int>, "top3_steps": [<int>, <int>, <int>], "reason": "<one sentence>"}}

Step indices are 1-based, matching the numbering in the trace below.
"""

OUTCOME_CLAUSE_FULL = ", and the final outcome (whether localization succeeded)"
OUTCOME_CLAUSE_PREFIX = ""

_JSON_RE = re.compile(r"\{[^{}]*\"top1_step\"[^{}]*\}", re.DOTALL)


def render_trace(steps: list, max_out_chars: int = 400, upto: int | None = None) -> str:
    """把步骤序列渲染成 judge 可读的文本。upto=只渲染前 N 步（prefix-only 用）。"""
    lines = []
    seq = steps[:upto] if upto else steps
    for s in seq:
        out = (s.output_text or "").replace("\n", " ")
        if len(out) > max_out_chars:
            out = out[:max_out_chars] + "…"
        args = s.args_raw.replace("\n", " ")
        if len(args) > 200:
            args = args[:200] + "…"
        lines.append(f"Step {s.idx}: tool={s.tool} args={args}\n  -> output: {out}")
    return "\n".join(lines)


def _parse_judge(text: str, n_steps: int) -> dict | None:
    m = _JSON_RE.findall(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m[-1])
    except json.JSONDecodeError:
        return None
    t1 = obj.get("top1_step")
    if not isinstance(t1, int) or not (1 <= t1 <= n_steps):
        return None
    top3 = [x for x in obj.get("top3_steps", []) if isinstance(x, int) and 1 <= x <= n_steps]
    return {"top1_step": t1, "top3_steps": top3 or [t1], "reason": obj.get("reason", "")}


def judge_attribution(
    srv: AppServer,
    base_items: list[dict],
    inst: T.Instance,
    outcome_Y: int,
    model: str | None,
    provider: str | None,
    mode: str = "full",          # "full" | "prefix"
    prefix_upto: int | None = None,
    turn_timeout: float = 300.0,
) -> dict | None:
    """让 judge 对一条 trace 做根因归因。mode=prefix 时只给前 prefix_upto 步、不给结果。"""
    steps = R.steps(base_items)
    n = len(steps)
    if n == 0:
        return None

    if mode == "full":
        instr = JUDGE_INSTRUCTIONS.format(outcome_clause=OUTCOME_CLAUSE_FULL)
        trace = render_trace(steps)
        outcome_line = (
            f"\nFinal outcome: localization {'SUCCEEDED' if outcome_Y else 'FAILED'}.\n"
        )
    else:
        instr = JUDGE_INSTRUCTIONS.format(outcome_clause=OUTCOME_CLAUSE_PREFIX)
        trace = render_trace(steps, upto=prefix_upto)
        outcome_line = "\n(You are given only the first part of the trace; outcome unknown.)\n"

    prompt = (
        f"Issue:\n{inst.problem_statement[:2000]}\n\n"
        f"Agent trace:\n{trace}\n{outcome_line}\n"
        f"Which step was most responsible? Respond with the JSON object."
    )

    # judge 不需要工具/仓库，用一次性 thread 完成
    started = srv.thread_start(
        cwd=inst.repo, model=model, model_provider=provider,
        sandbox="read-only", approval_policy="never",
        base_instructions=instr, timeout=120,
    )
    tid = started["thread"]["id"]
    run = srv.turn_run(tid, prompt, timeout=turn_timeout)
    text = T.final_message_text(run["items"])
    parsed = _parse_judge(text, n)
    if parsed is None:
        return {"mode": mode, "n_steps": n, "parsed": False, "raw": text[-300:]}
    return {
        "mode": mode,
        "n_steps": n,
        "parsed": True,
        "top1_step": parsed["top1_step"],
        "top1_pos": round(parsed["top1_step"] / n, 4),
        "top3_steps": parsed["top3_steps"],
        "reason": parsed["reason"],
    }
